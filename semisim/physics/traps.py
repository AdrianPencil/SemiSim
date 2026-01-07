# semisim/physics/traps.py
"""
Trap physics: occupancies, capture/emission, and trap charge (bulk & interface).

Outputs:
    - Bulk:  rho_trap_Cm3(z) = q [ N_D^+(z) - N_A^-(z) ]  (C/m^3)
    - Interface: sigma_it_Cm2 = q [ N_D^+ - N_A^- ]       (C/m^2)
      with: N_D^+ = N_D (1 - f_D),  N_A^- = N_A f_A.

Steady-state occupancy (Shockley-Read-Hall kinetics):
    f* = (c_n n + e_p) / (c_n n + c_p p + e_n + e_p),
    with e_n = c_n n1,  e_p = c_p p1,
         n1 = N_c exp(-(E_C - E_t)/kT),  p1 = N_v exp(-(E_t - E_V)/kT).

API (stable):
    TrapKind, EnergyRef
    BulkTrap, InterfaceTrap, InterfaceSpectrum
    BulkTrapSet, InterfaceTrapSet
    n1_p1_from_bands(E_C, E_V, E_t, T, me_rel, mh_rel, gvc=1.0, gvv=1.0) -> (n1, p1)
    occupancy_steady(n, p, n1, p1, cn, cp) -> f_star
    bulk_trap_charge(setup) -> BulkTrapResult
    interface_trap_sheet(setup) -> InterfaceTrapResult

Notes:
- Recombination rate U_SRH will live in recombination.py; here we expose n1,p1,f*.
- Energies accepted as:
    E_t_rel_J (relative to ref: "Ec" or "Ev"): E_t = E_C - E_t_rel (Ec) or E_V + E_t_rel (Ev).
"""
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence, Tuple

import numpy as np

from .carriers.statistics import Nc_3d, Nv_3d

# ---- constants (SI) ----
Q = 1.602176634e-19     # C
K_B = 1.380649e-23      # J/K

__all__ = [
    "TrapKind",
    "EnergyRef",
    "BulkTrap",
    "InterfaceTrap",
    "InterfaceSpectrum",
    "BulkTrapSet",
    "InterfaceTrapSet",
    "n1_p1_from_bands",
    "occupancy_steady",
    "bulk_trap_charge",
    "interface_trap_sheet",
]


# ---------------------------------------------------------------------
# Types & dataclasses
# ---------------------------------------------------------------------

TrapKind = Literal["donor", "acceptor"]
EnergyRef = Literal["Ec", "Ev"]


@dataclass(slots=True)
class BulkTrap:
    """
    Discrete bulk trap (volumetric).

    Parameters
    ----------
    kind : "donor" or "acceptor"
        Donor-like (positive when empty) or acceptor-like (negative when filled).
    Erel_J : float
        Trap energy relative to reference edge [J], see `ref`.
    ref : "Ec" or "Ev"
        If "Ec": E_t = E_C - Erel     (Erel > 0 is below Ec).
        If "Ev": E_t = E_V + Erel     (Erel > 0 is above Ev).
    Nt_m3 : float
        Concentration [1/m^3].
    sigma_n_m2, sigma_p_m2 : float
        Capture cross-sections [m^2].
    label : str
        Optional label for identification.
    """
    kind: TrapKind
    Erel_J: float
    ref: EnergyRef
    Nt_m3: float
    sigma_n_m2: float
    sigma_p_m2: float
    label: str = ""


@dataclass(slots=True)
class InterfaceTrap:
    """
    Discrete interface trap (areal).

    Parameters
    ----------
    node_index : int
        Geometry node index where the interface is mapped (delta source).
    kind, Erel_J, ref, sigma_n_m2, sigma_p_m2 : see BulkTrap docs
    Nit_m2 : float
        Areal density [1/m^2] for the discrete level.
    """
    node_index: int
    kind: TrapKind
    Erel_J: float
    ref: EnergyRef
    Nit_m2: float
    sigma_n_m2: float
    sigma_p_m2: float
    label: str = ""


@dataclass(slots=True)
class InterfaceSpectrum:
    """
    Continuous interface state spectrum Dit(E) on an energy grid.

    Parameters
    ----------
    node_index : int
        Geometry node index for this interface.
    ref : "Ec" or "Ev"
        Energy reference for Erel_J grid.
    Erel_J : np.ndarray
        Energy grid relative to ref [J]; shape (M,).
    Dit_perJ_m2 : np.ndarray
        Density of states per energy [1/(m^2 J)], shape (M,).
    sigma_n_m2, sigma_p_m2 : float or arrays (shape (M,))
        Capture cross-sections [m^2].
    """
    node_index: int
    ref: EnergyRef
    Erel_J: np.ndarray
    Dit_perJ_m2: np.ndarray
    sigma_n_m2: float | np.ndarray
    sigma_p_m2: float | np.ndarray
    label: str = ""


@dataclass(slots=True)
class BulkTrapSet:
    traps: Sequence[BulkTrap]


@dataclass(slots=True)
class InterfaceTrapSet:
    discrete: Sequence[InterfaceTrap] = ()
    spectra: Sequence[InterfaceSpectrum] = ()


# Results

@dataclass(slots=True)
class BulkTrapResult:
    rho_trap_Cm3: np.ndarray        # [C/m^3], shape (N,)
    f_star_per_level: list[np.ndarray]  # each array shape (N,)


@dataclass(slots=True)
class InterfaceTrapResult:
    sheet_nodes: np.ndarray         # int indices
    sigma_it_Cm2: np.ndarray        # [C/m^2], same length as sheet_nodes
    f_star_levels: list[float]      # occupancies for discrete levels (scalars at that interface)
    meta: dict                      # bookkeeping (e.g., per-spectrum integrals)


# ---------------------------------------------------------------------
# Core formulas
# ---------------------------------------------------------------------


def _as_c64(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def _E_t_from_rel(E_C_J: np.ndarray, E_V_J: np.ndarray, Erel_J: float | np.ndarray, ref: EnergyRef) -> np.ndarray:
    Erel = _as_c64(Erel_J)
    if ref == "Ec":
        return _as_c64(E_C_J) - Erel
    else:
        return _as_c64(E_V_J) + Erel


def n1_p1_from_bands(
    E_C_J: np.ndarray,
    E_V_J: np.ndarray,
    E_t_J: np.ndarray,
    T_K: float | np.ndarray,
    *,
    me_rel: float | np.ndarray,
    mh_rel: float | np.ndarray,
    gvc: float = 1.0,
    gvv: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SRH helper densities:
        n1 = N_c exp(-(E_C - E_t)/kT),  p1 = N_v exp(-(E_t - E_V)/kT).
    """
    T = _as_c64(T_K)
    Nc = Nc_3d(T, me_rel, g_s=2.0, g_v=gvc)
    Nv = Nv_3d(T, mh_rel, g_s=2.0, g_v=gvv)
    n1 = Nc * np.exp(-(E_C_J - E_t_J) / (K_B * T))
    p1 = Nv * np.exp(-(E_t_J - E_V_J) / (K_B * T))
    return n1, p1


def occupancy_steady(
    n_m3: np.ndarray,
    p_m3: np.ndarray,
    n1_m3: np.ndarray,
    p1_m3: np.ndarray,
    c_n_m3_per_s: np.ndarray,
    c_p_m3_per_s: np.ndarray,
) -> np.ndarray:
    """
    Steady-state SRH occupancy:
        f* = (c_n n + e_p) / (c_n n + c_p p + e_n + e_p),
    where e_n = c_n n1, e_p = c_p p1.
    """
    cn = _as_c64(c_n_m3_per_s)
    cp = _as_c64(c_p_m3_per_s)
    n = _as_c64(n_m3)
    p = _as_c64(p_m3)
    n1 = _as_c64(n1_m3)
    p1 = _as_c64(p1_m3)

    en = cn * n1
    ep = cp * p1
    denom = cn * (n + n1) + cp * (p + p1)
    denom = np.where(denom == 0.0, 1e-30, denom)
    return (cn * n + ep) / denom


# ---------------------------------------------------------------------
# Bulk traps (volumetric)
# ---------------------------------------------------------------------


@dataclass(slots=True)
class BulkTrapInputs:
    """
    Inputs to evaluate bulk trap charge on a 1D grid (shape N).
    """
    E_C_J: np.ndarray
    E_V_J: np.ndarray
    n_m3: np.ndarray
    p_m3: np.ndarray
    T_K: float
    me_rel: float | np.ndarray
    mh_rel: float | np.ndarray
    traps: BulkTrapSet
    gvc: float = 1.0
    gvv: float = 1.0


def bulk_trap_charge(inp: BulkTrapInputs) -> BulkTrapResult:
    """
    Compute volumetric trap charge:
        rho_trap = q [ sum_D N_D (1 - f_D) - sum_A N_A f_A ]  (C/m^3).
    Also returns f* per level for diagnostics.
    """
    N = inp.n_m3.size
    rho = np.zeros(N, dtype=np.float64)
    f_levels: list[np.ndarray] = []

    for tr in inp.traps.traps:
        # Absolute trap energy
        E_t = _E_t_from_rel(inp.E_C_J, inp.E_V_J, tr.Erel_J, tr.ref)

        # SRH parameters
        # capture coefficients c = σ * v_th; we don't model velocity here,
        # so treat σ as an *effective* capture coefficient [m^3/s] placeholder
        # (tune numerically later).
        cn = np.full(N, float(tr.sigma_n_m2), dtype=np.float64)
        cp = np.full(N, float(tr.sigma_p_m2), dtype=np.float64)

        # n1, p1
        n1, p1 = n1_p1_from_bands(
            inp.E_C_J, inp.E_V_J, E_t, inp.T_K,
            me_rel=inp.me_rel, mh_rel=inp.mh_rel, gvc=inp.gvc, gvv=inp.gvv,
        )

        fstar = occupancy_steady(inp.n_m3, inp.p_m3, n1, p1, cn, cp)
        f_levels.append(fstar)

        if tr.kind == "donor":
            ND_plus = float(tr.Nt_m3) * (1.0 - fstar)   # donor positive when empty
            rho += Q * ND_plus
        else:  # "acceptor"
            NA_minus = float(tr.Nt_m3) * fstar          # acceptor negative when filled
            rho += -Q * NA_minus

    return BulkTrapResult(rho_trap_Cm3=np.ascontiguousarray(rho), f_star_per_level=f_levels)


# ---------------------------------------------------------------------
# Interface traps (sheet)
# ---------------------------------------------------------------------


@dataclass(slots=True)
class InterfaceTrapInputs:
    """
    Inputs for interface trap sheet calculations at one or more interfaces.

    Provide local band edges and carrier densities *at the interface node indices*.
    """
    node_indices: np.ndarray           # shape (K,), interface node indices
    E_C_J: np.ndarray                  # shape (N,)
    E_V_J: np.ndarray                  # shape (N,)
    n_m3: np.ndarray                   # shape (N,)
    p_m3: np.ndarray                   # shape (N,)
    T_K: float
    me_rel: float | np.ndarray
    mh_rel: float | np.ndarray
    traps: InterfaceTrapSet
    gvc: float = 1.0
    gvv: float = 1.0


def interface_trap_sheet(inp: InterfaceTrapInputs) -> InterfaceTrapResult:
    """
    Compute interface sheet charge:
        sigma_it = q [ sum_D N_D^+ - sum_A N_A^- ]  (C/m^2),
    for both discrete levels and spectra at provided interface nodes.
    """
    # Prepare outputs
    node_sigma: dict[int, float] = {}
    fstar_levels: list[float] = []
    meta: dict = {"spectra_integrals": []}

    # Helper to accumulate on a node
    def add_sigma(idx: int, val: float) -> None:
        node_sigma[idx] = node_sigma.get(idx, 0.0) + float(val)

    # ---- Discrete interface levels
    for it in inp.traps.discrete:
        idx = int(it.node_index)
        # Local values at node
        Ec = float(inp.E_C_J[idx])
        Ev = float(inp.E_V_J[idx])
        n = float(inp.n_m3[idx])
        p = float(inp.p_m3[idx])

        # Absolute trap energy
        Et = Ec - it.Erel_J if it.ref == "Ec" else Ev + it.Erel_J

        # n1,p1 at this node
        Nc = float(Nc_3d(inp.T_K, inp.me_rel, g_s=2.0, g_v=inp.gvc))
        Nv = float(Nv_3d(inp.T_K, inp.mh_rel, g_s=2.0, g_v=inp.gvv))
        n1 = Nc * np.exp(-(Ec - Et) / (K_B * inp.T_K))
        p1 = Nv * np.exp(-(Et - Ev) / (K_B * inp.T_K))

        # Effective capture coefficients (placeholder; see bulk)
        cn = float(it.sigma_n_m2)
        cp = float(it.sigma_p_m2)

        denom = cn * (n + n1) + cp * (p + p1)
        denom = denom if denom != 0.0 else 1e-30
        fstar = (cn * n + cp * p1) / denom
        fstar_levels.append(float(fstar))

        if it.kind == "donor":
            ND_plus = it.Nit_m2 * (1.0 - fstar)
            add_sigma(idx, +Q * ND_plus)
        else:
            NA_minus = it.Nit_m2 * fstar
            add_sigma(idx, -Q * NA_minus)

    # ---- Spectral Dit(E) (energy grid integration, per spectrum)
    for spec in inp.traps.spectra:
        idx = int(spec.node_index)
        Ec = float(inp.E_C_J[idx])
        Ev = float(inp.E_V_J[idx])
        n = float(inp.n_m3[idx])
        p = float(inp.p_m3[idx])

        Erel = _as_c64(spec.Erel_J)              # shape (M,)
        Dit = _as_c64(spec.Dit_perJ_m2)          # [1/(m^2 J)]
        # Promote sigma arrays if needed
        sig_n = _as_c64(spec.sigma_n_m2) if np.ndim(spec.sigma_n_m2) else float(spec.sigma_n_m2) * np.ones_like(Erel)
        sig_p = _as_c64(spec.sigma_p_m2) if np.ndim(spec.sigma_p_m2) else float(spec.sigma_p_m2) * np.ones_like(Erel)

        # Absolute energies along grid
        Et = (Ec - Erel) if spec.ref == "Ec" else (Ev + Erel)

        # n1(E), p1(E) on grid (use scalar Nc,Nv at T)
        Nc = float(Nc_3d(inp.T_K, inp.me_rel, g_s=2.0, g_v=inp.gvc))
        Nv = float(Nv_3d(inp.T_K, inp.mh_rel, g_s=2.0, g_v=inp.gvv))
        n1 = Nc * np.exp(-(Ec - Et) / (K_B * inp.T_K))
        p1 = Nv * np.exp(-(Et - Ev) / (K_B * inp.T_K))

        cn = sig_n
        cp = sig_p
        denom = cn * (n + n1) + cp * (p + p1)
        denom = np.where(denom == 0.0, 1e-30, denom)
        fstar = (cn * n + cp * p1) / denom

        # Charge contribution per energy:
        # Split spectrum into donor-like part above midgap and acceptor-like below?
        # Here we treat Dit as *neutral spectrum* requiring a sign model.
        # Minimal pragmatic approach:
        #   - If ref == "Ec": we assume states measured from Ec are donor-like (positive when empty).
        #   - If ref == "Ev": we assume acceptor-like (negative when filled).
        if spec.ref == "Ec":
            ND_plus_E = Dit * (1.0 - fstar)     # [1/m^2/J]
            sigma_E = Q * ND_plus_E             # [C/m^2/J]
        else:  # "Ev"
            NA_minus_E = Dit * fstar
            sigma_E = -Q * NA_minus_E

        # Integrate over energy grid by trapezoid (Erel in J)
        dE = np.diff(Erel)
        sigma_sum = float(np.trapz(sigma_E, Erel))
        add_sigma(idx, sigma_sum)

        meta["spectra_integrals"].append(
            {"node": idx, "sigma_Cm2": sigma_sum, "label": spec.label}
        )

    # Consolidate outputs
    if node_sigma:
        nodes = np.array(sorted(node_sigma.keys()), dtype=int)
        sigma = np.array([node_sigma[i] for i in nodes], dtype=np.float64)
    else:
        nodes = np.zeros(0, dtype=int)
        sigma = np.zeros(0, dtype=np.float64)

    return InterfaceTrapResult(
        sheet_nodes=nodes,
        sigma_it_Cm2=sigma,
        f_star_levels=fstar_levels,
        meta=meta,
    )

#from semisim.physics.traps import BulkTrap, BulkTrapSet, BulkTrapInputs, bulk_trap_charge

#Bulk_set = BulkTrapSet(traps=[
    #BulkTrap(kind="donor",    Erel_J=0.20*1.602e-19, ref="Ec", Nt_m3=1e22, sigma_n_m2=1e-19, sigma_p_m2=1e-20, label="D1"),
    #BulkTrap(kind="acceptor", Erel_J=0.30*1.602e-19, ref="Ev", Nt_m3=5e21,  sigma_n_m2=1e-20, sigma_p_m2=1e-19, label="A1"),
#])

#bulk_res = bulk_trap_charge(BulkTrapInputs(
    #E_C_J=E_C, E_V_J=E_V, n_m3=n, p_m3=p, T_K=T,
    #me_rel=mat.me_dos_rel, mh_rel=mat.mh_dos_rel, traps=bulk_set,
#))
#rho_extra = rho_pol + bulk_res.rho_trap_Cm3   # add polarization bulk charge as well
#from semisim.physics.traps import InterfaceTrap, InterfaceTrapSet, InterfaceTrapInputs, interface_trap_sheet

#it_set = InterfaceTrapSet(
    #discrete=[InterfaceTrap(node_index=idx_if, kind="acceptor", ref="Ev",
                            #Erel_J=0.15*1.602e-19, Nit_m2=5e16, sigma_n_m2=1e-19, sigma_p_m2=1e-19, label="Pb")],
    #spectra=[],
#)

#it_res = interface_trap_sheet(InterfaceTrapInputs(
    #node_indices=np.array([idx_if], dtype=int),
    #E_C_J=E_C, E_V_J=E_V, n_m3=n, p_m3=p, T_K=T,
    #me_rel=mat.me_dos_rel, mh_rel=mat.mh_dos_rel, traps=it_set,
#))

# Merge with polarization sheets (and fixed), then pass to PoissonSetup
#sheet_nodes = np.concatenate([pol.sheet_nodes, it_res.sheet_nodes])
#sheet_sigma = np.concatenate([pol.sigma_pol_Cm2, it_res.sigma_it_Cm2])
