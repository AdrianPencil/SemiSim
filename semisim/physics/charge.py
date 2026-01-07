# semisim/physics/charge.py
"""
Charge aggregation (SI units) for Poisson / Continuity coupling.

Assembles volumetric charge density and interface sheet charges from:
    - Free carriers:   rho_car   = q (p - n)
    - Dopants:         rho_dop   = q (N_D^+ - N_A^-)
    - Bulk traps:      rho_trap  (from traps.bulk_trap_charge)
    - Polarization:    rho_pol   (from polarization.compute_polarization)
    - Extra volumetric:rho_extra (user-specified)
Sheets at interfaces:
    sigma_total = sigma_pol + sigma_it + sigma_fix (+ user sheets)

Optionally returns d rho / d phi for Newton (frozen default uses carriers only).

Integration patterns
--------------------
1) **Recommended with current Poisson** (you already compute carriers internally):
    - Keep Poisson's built-in q(p-n) + (N_D - N_A) term.
    - Set PoissonSetup.rho_extra_Cm3 = rho_trap + rho_pol (+ other)
    - Set PoissonSetup.sheet_nodes / sheet_sigma from this module.
    - This avoids double counting and keeps clean Jacobians (since Poisson computes dn/dphi, dp/dphi).

2) **Full RHS outside Poisson**:
    - Use this module's rho_total and d_rho_d_phi, pass rho_extra=0 and ND=NA=0 to Poisson,
      and modify Poisson to accept externally assembled total rho. (Advanced; not required now.)

All arrays are float64 & C-contiguous. Shapes follow the node grid (N,).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import numpy as np

from .carriers.statistics import carriers_3d, derivatives_3d
from .carriers.bands import BandParams, band_edges_from_potential
from .carriers.intrinsic import ionized_donors, ionized_acceptors
from .traps import (
    BulkTrapSet, InterfaceTrapSet,
    BulkTrapInputs, bulk_trap_charge,
    InterfaceTrapInputs, interface_trap_sheet,
)
from .polarization import PolarizationResult

# ---- constants (SI) ----
Q = 1.602176634e-19

__all__ = [
    "DopingSpec",
    "SheetsFixed",
    "ChargeInputs",
    "ChargeOptions",
    "ChargeComponents",
    "ChargeSheets",
    "ChargeResult",
    "consolidate_sheets",
    "assemble_charge",
]


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------


@dataclass(slots=True)
class DopingSpec:
    """
    Doping specification at nodes.

    Provide ND, NA in [1/m^3]. For incomplete ionization, provide energy levels
    and degeneracies; the ionized portions N_D^+, N_A^- will be computed.

    If E_D_J / E_A_J are None, dopants are treated as fully ionized.
    """
    ND_m3: np.ndarray
    NA_m3: np.ndarray
    E_D_J: Optional[float] = None     # donor level (J) absolute
    E_A_J: Optional[float] = None     # acceptor level (J) absolute
    g_D: float = 2.0
    g_A: float = 4.0


@dataclass(slots=True)
class SheetsFixed:
    """
    Fixed/process sheets explicitly provided (e.g., Qf at oxide/semiconductor).
    """
    nodes: np.ndarray          # int indices
    sigma_Cm2: np.ndarray      # C/m^2


@dataclass(slots=True)
class ChargeInputs:
    """
    Inputs required to assemble charge terms.

    Required:
      - z-grid sized fields: (length N) E_C, E_V either by (phi + bands) or provided directly.
      - Either provide (n,p) or (mu,T,bands,stats) to compute carriers.

    Optional:
      - Doping (with or without incomplete ionization).
      - Bulk traps set and/or interface traps set.
      - Polarization result (bulk + sheets).
      - Extra volumetric or sheet sources.

    Notes:
      - If E_C_J/E_V_J are not provided, they will be computed from (phi, bands).
    """
    # Grid / bands / electrostatics
    phi_V: Optional[np.ndarray]                 # electrostatic potential [V] (for E_C/E_V if needed)
    bands: Optional[BandParams]                 # reference bands (used with phi_V)
    E_C_J: Optional[np.ndarray] = None          # explicit conduction band edge [J]
    E_V_J: Optional[np.ndarray] = None          # explicit valence band edge [J]

    # Carriers
    n_m3: Optional[np.ndarray] = None
    p_m3: Optional[np.ndarray] = None
    mu_J: Optional[float] = None
    T_K: Optional[float] = None
    me_rel: Optional[float | np.ndarray] = None
    mh_rel: Optional[float | np.ndarray] = None
    gvc: float = 1.0
    gvv: float = 1.0
    stats: Literal["FD", "MB"] = "FD"
    exp_clip: float = 60.0

    # Doping
    doping: Optional[DopingSpec] = None

    # Traps
    bulk_traps: Optional[BulkTrapSet] = None
    interface_traps: Optional[InterfaceTrapSet] = None

    # Polarization (precomputed)
    polarization: Optional[PolarizationResult] = None

    # Extra sources
    rho_extra_Cm3: Optional[np.ndarray] = None
    extra_sheets: Optional[SheetsFixed] = None


@dataclass(slots=True)
class ChargeOptions:
    """
    Options controlling assembly and linearization.
    """
    # Which contributions to include in volumetric sum
    include_carriers: bool = True
    include_dopants: bool = True
    include_bulk_traps: bool = True
    include_polarization_bulk: bool = True
    include_extra_vol: bool = True

    # Sheet contributions
    include_polarization_sheets: bool = True
    include_interface_trap_sheets: bool = True
    include_fixed_sheets: bool = True

    # Linearization model for d rho / d phi
    # "frozen"  : carriers only (q(dp/dφ - dn/dφ)); robust & fast
    # "full"    : placeholder hook (currently same as "frozen"; extend later)
    linearization: Literal["frozen", "full"] = "frozen"


@dataclass(slots=True)
class ChargeComponents:
    """
    Individual volumetric components (diagnostics).
    """
    rho_car_Cm3: np.ndarray
    rho_dop_Cm3: np.ndarray
    rho_trap_Cm3: np.ndarray
    rho_pol_Cm3: np.ndarray
    rho_extra_Cm3: np.ndarray


@dataclass(slots=True)
class ChargeSheets:
    nodes: np.ndarray
    sigma_Cm2: np.ndarray


@dataclass(slots=True)
class ChargeResult:
    """
    Final aggregated charge.
    """
    rho_total_Cm3: np.ndarray
    drho_dphi_C_per_m3V: np.ndarray     # derivative w.r.t φ [C/(m^3 V)]
    components: ChargeComponents
    sheets: ChargeSheets

    # Convenience
    def as_dict(self) -> dict:
        return {
            "rho_total_Cm3": self.rho_total_Cm3,
            "drho_dphi": self.drho_dphi_C_per_m3V,
            "sheets_nodes": self.sheets.nodes,
            "sheets_sigma": self.sheets.sigma_Cm2,
        }


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _c64(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def consolidate_sheets(nodes: np.ndarray, sigma_Cm2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sum sheet charges that land on the same node.
    """
    if nodes.size == 0:
        return nodes, sigma_Cm2
    nodes = np.asarray(nodes, dtype=int)
    sigma = _c64(sigma_Cm2)
    uniq = np.unique(nodes)
    out_nodes = []
    out_sigma = []
    for i in uniq:
        sel = nodes == i
        out_nodes.append(int(i))
        out_sigma.append(float(np.sum(sigma[sel])))
    return np.asarray(out_nodes, dtype=int), np.asarray(out_sigma, dtype=np.float64)


# ---------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------


def assemble_charge(inp: ChargeInputs, opt: ChargeOptions) -> ChargeResult:
    """
    Assemble volumetric and sheet charges (and dρ/dφ) on the node grid.

    Returns
    -------
    ChargeResult
    """
    # ---- band edges
    if inp.E_C_J is None or inp.E_V_J is None:
        if inp.phi_V is None or inp.bands is None:
            raise ValueError("Need either (E_C_J,E_V_J) or (phi_V + bands) to get band edges")
        E_C, E_V = band_edges_from_potential(_c64(inp.phi_V), inp.bands)
    else:
        E_C = _c64(inp.E_C_J)
        E_V = _c64(inp.E_V_J)

    N = E_C.size

    # ---- carriers (n,p)
    if inp.n_m3 is None or inp.p_m3 is None:
        if inp.mu_J is None or inp.T_K is None or inp.me_rel is None or inp.mh_rel is None:
            raise ValueError("Need (n,p) or (mu_J, T_K, me_rel, mh_rel) to compute carriers")
        n, p = carriers_3d(
            E_C, E_V, float(inp.mu_J), float(inp.T_K),
            me_rel=inp.me_rel, mh_rel=inp.mh_rel,
            gvc=inp.gvc, gvv=inp.gvv, stats=inp.stats, exp_clip=inp.exp_clip,
        )
    else:
        n, p = _c64(inp.n_m3), _c64(inp.p_m3)

    rho_car = Q * (p - n) if opt.include_carriers else np.zeros(N, dtype=np.float64)

    # ---- dopants (ionized or full)
    rho_dop = np.zeros(N, dtype=np.float64)
    if opt.include_dopants and inp.doping is not None:
        ND = _c64(inp.doping.ND_m3)
        NA = _c64(inp.doping.NA_m3)
        if inp.doping.E_D_J is None and inp.doping.E_A_J is None:
            NDp = ND
            NAm = NA
        else:
            if inp.mu_J is None or inp.T_K is None:
                raise ValueError("Incomplete ionization needs (mu_J, T_K)")
            NDp = ND if inp.doping.E_D_J is None else ionized_donors(inp.doping.E_D_J, float(inp.mu_J), float(inp.T_K), ND, g_D=inp.doping.g_D)
            NAm = NA if inp.doping.E_A_J is None else ionized_acceptors(inp.doping.E_A_J, float(inp.mu_J), float(inp.T_K), NA, g_A=inp.doping.g_A)
        rho_dop = Q * (NDp - NAm)

    # ---- bulk traps
    rho_trap = np.zeros(N, dtype=np.float64)
    if opt.include_bulk_traps and inp.bulk_traps is not None:
        if inp.T_K is None or inp.me_rel is None or inp.mh_rel is None:
            raise ValueError("Bulk traps need (T_K, me_rel, mh_rel)")
        bulk_res = bulk_trap_charge(
            BulkTrapInputs(
                E_C_J=E_C, E_V_J=E_V, n_m3=n, p_m3=p, T_K=float(inp.T_K),
                me_rel=inp.me_rel, mh_rel=inp.mh_rel,
                traps=inp.bulk_traps, gvc=inp.gvc, gvv=inp.gvv,
            )
        )
        rho_trap = _c64(bulk_res.rho_trap_Cm3)

    # ---- polarization bulk
    rho_pol = np.zeros(N, dtype=np.float64)
    if opt.include_polarization_bulk and inp.polarization is not None:
        rho_pol = _c64(inp.polarization.rho_pol_Cm3)

    # ---- extra volumetric
    rho_extra = _c64(inp.rho_extra_Cm3) if (opt.include_extra_vol and inp.rho_extra_Cm3 is not None) else np.zeros(N, dtype=np.float64)

    # ---- total volumetric
    rho_total = rho_car + rho_dop + rho_trap + rho_pol + rho_extra

    # ---- sheets: polarization + interface traps + fixed
    nodes_list: list[int] = []
    sigma_list: list[float] = []

    if opt.include_polarization_sheets and inp.polarization is not None:
        nodes_arr = np.asarray(inp.polarization.sheet_nodes, dtype=int)
        if nodes_arr.size > 0:
            sig_all = _c64(inp.polarization.sigma_pol_Cm2)
            # Use only the entries that correspond to the reported sheet nodes
            sig_sel = sig_all[nodes_arr]
            nodes_list.extend([int(i) for i in nodes_arr])
            sigma_list.extend([float(s) for s in sig_sel])

    if opt.include_interface_trap_sheets and inp.interface_traps is not None:
        if inp.T_K is None or inp.me_rel is None or inp.mh_rel is None:
            raise ValueError("Interface traps need (T_K, me_rel, mh_rel)")
        # Evaluate at *all interface nodes present* in the set; if you want a subset, pass that instead.
        if len(inp.interface_traps.discrete) > 0:
            idxs = np.array([it.node_index for it in inp.interface_traps.discrete], dtype=int)
        elif len(inp.interface_traps.spectra) > 0:
            idxs = np.array([sp.node_index for sp in inp.interface_traps.spectra], dtype=int)
        else:
            idxs = np.zeros(0, dtype=int)

        if idxs.size > 0:
            itres = interface_trap_sheet(
                InterfaceTrapInputs(
                    node_indices=idxs,
                    E_C_J=E_C, E_V_J=E_V,
                    n_m3=n, p_m3=p,
                    T_K=float(inp.T_K),
                    me_rel=inp.me_rel, mh_rel=inp.mh_rel,
                    traps=inp.interface_traps,
                    gvc=inp.gvc, gvv=inp.gvv,
                )
            )
            nodes_list.extend([int(i) for i in itres.sheet_nodes])
            sigma_list.extend([float(s) for s in itres.sigma_it_Cm2])

    if opt.include_fixed_sheets and inp.extra_sheets is not None:
        nodes_list.extend([int(i) for i in inp.extra_sheets.nodes])
        sigma_list.extend([float(s) for s in inp.extra_sheets.sigma_Cm2])

    if len(nodes_list) > 0:
        sheet_nodes, sheet_sigma = consolidate_sheets(np.asarray(nodes_list, dtype=int), np.asarray(sigma_list, dtype=np.float64))
    else:
        sheet_nodes = np.zeros(0, dtype=int)
        sheet_sigma = np.zeros(0, dtype=np.float64)

    # ---- derivative d rho / d phi (for Poisson Jacobian)
    if inp.mu_J is not None and inp.T_K is not None and inp.me_rel is not None and inp.mh_rel is not None:
        dn_dphi = np.zeros(N, dtype=np.float64)
        dp_dphi = np.zeros(N, dtype=np.float64)
        if opt.include_carriers:
            dn_dphi, dp_dphi, _, _ = derivatives_3d(
                E_C, E_V, float(inp.mu_J), float(inp.T_K),
                me_rel=inp.me_rel, mh_rel=inp.mh_rel,
                gvc=inp.gvc, gvv=inp.gvv, stats=inp.stats, exp_clip=inp.exp_clip,
            )
        # Frozen: only carriers contribute
        drho_dphi = Q * (dp_dphi - dn_dphi)

        if opt.linearization == "full":
            # Hooks for adding dopant/trap/polarization derivatives later.
            # For now, same as "frozen" to keep numerics robust.
            pass
    else:
        drho_dphi = np.zeros(N, dtype=np.float64)

    components = ChargeComponents(
        rho_car_Cm3=_c64(rho_car),
        rho_dop_Cm3=_c64(rho_dop),
        rho_trap_Cm3=_c64(rho_trap),
        rho_pol_Cm3=_c64(rho_pol),
        rho_extra_Cm3=_c64(rho_extra),
    )
    sheets = ChargeSheets(nodes=sheet_nodes, sigma_Cm2=sheet_sigma)

    return ChargeResult(
        rho_total_Cm3=_c64(rho_total),
        drho_dphi_C_per_m3V=_c64(drho_dphi),
        components=components,
        sheets=sheets,
    )