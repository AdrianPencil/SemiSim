# semisim/physics/recombination.py
"""
Recombination–generation models (SI units, vectorized).

Included:
  • Bulk SRH from discrete trap levels (uses traps.BulkTrapSet).
  • Radiative recombination: U_B = B (n p - n_i^2).
  • Auger recombination: U_C = (C_n n + C_p p) (n p - n_i^2).
  • Interface SRH (per area) from discrete interface traps (traps.InterfaceTrapSet),
    and an effective surface-velocity (SRV) model with (s_n, s_p).

All routines return both the **rate** and **derivatives** w.r.t. (n, p) for Newton/AC.

Conventions
-----------
- Bulk rates U are volumetric [1/m^3/s].
- Interface rates U_s are areal [1/m^2/s] (to be used in boundary currents).
- Effective capture coefficients c_n, c_p are in [m^3/s] (consistent with traps.py).
- n_i^2 is usually MB intrinsic square: n_i^2 = N_c N_v exp(-E_g/kT).

Public API (stable):
    # Helpers
    ni2_MB(Eg_J, T_K, me_rel, mh_rel, gvc=1.0, gvv=1.0) -> np.ndarray

    # Bulk
    BulkSRHInputs, BulkSRHResult, srh_bulk_from_traps(inputs)
    RadiativeInputs, bulk_radiative(inputs)
    AugerInputs,     bulk_auger(inputs)

    # Interface
    InterfaceSRHInputs, InterfaceSRHResult, srh_interface_from_traps(inputs)
    SRVInputs, SRVResult, srh_interface_srv(inputs)
"""
from dataclasses import dataclass
from typing import Sequence, Tuple
from typing import Dict, Callable

import numpy as np

from .traps import (
    BulkTrapSet,
    InterfaceTrapSet,
    _E_t_from_rel,               # reuse internal helper from traps
    n1_p1_from_bands,
)
from .carriers.statistics import Nc_3d, Nv_3d

# ---- constants (SI) ----
Q = 1.602176634e-19     # C
K_B = 1.380649e-23      # J/K

__all__ = [
    "ni2_MB",
    "BulkSRHInputs",
    "BulkSRHResult",
    "srh_bulk_from_traps",
    "RadiativeInputs",
    "bulk_radiative",
    "AugerInputs",
    "bulk_auger",
    "InterfaceSRHInputs",
    "InterfaceSRHResult",
    "srh_interface_from_traps",
    "SRVInputs",
    "SRVResult",
    "srh_interface_srv",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _c64(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def ni2_MB(
    Eg_J: np.ndarray,
    T_K: float | np.ndarray,
    me_rel: float | np.ndarray,
    mh_rel: float | np.ndarray,
    gvc: float = 1.0,
    gvv: float = 1.0,
) -> np.ndarray:
    """
    Intrinsic square n_i^2 (MB), used in radiative/Auger numerators:

        n_i^2 = N_c N_v exp(-E_g / kT).
    """
    T = _c64(T_K)
    Nc = Nc_3d(T, me_rel, g_s=2.0, g_v=gvc)
    Nv = Nv_3d(T, mh_rel, g_s=2.0, g_v=gvv)
    Eg = _c64(Eg_J)
    return Nc * Nv * np.exp(-Eg / (K_B * T))


# ---------------------------------------------------------------------
# Bulk SRH from discrete traps (volumetric)
# ---------------------------------------------------------------------


@dataclass(slots=True)
class BulkSRHInputs:
    """
    Inputs for bulk SRH on a 1D node grid (shape N).
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


@dataclass(slots=True)
class BulkSRHResult:
    U_srh: np.ndarray           # [1/m^3/s]
    dU_dn: np.ndarray           # [m^3/s]
    dU_dp: np.ndarray           # [m^3/s]


def _srh_single_level_bulk(
    n: np.ndarray,
    p: np.ndarray,
    n1: np.ndarray,
    p1: np.ndarray,
    cn: np.ndarray,
    cp: np.ndarray,
    Nt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One discrete SRH level (bulk), vectorized.

    U = (cn cp Nt) * (n p - n1 p1) / D,
    D = cn (n + n1) + cp (p + p1).

    Derivatives:
        ∂U/∂n = (cn cp Nt) * ( p D - (n p - n1 p1) cn ) / D^2
        ∂U/∂p = (cn cp Nt) * ( n D - (n p - n1 p1) cp ) / D^2
    """
    A = n * p - n1 * p1
    D = cn * (n + n1) + cp * (p + p1)
    D2 = np.where(D == 0.0, 1e-30, D) ** 2
    pref = cn * cp * Nt
    U = pref * A / np.where(D == 0.0, 1e-30, D)
    dU_dn = pref * (p * D - A * cn) / D2
    dU_dp = pref * (n * D - A * cp) / D2
    return U, dU_dn, dU_dp


def srh_bulk_from_traps(inp: BulkSRHInputs) -> BulkSRHResult:
    """
    Sum SRH contributions from all discrete bulk traps in 'traps'.
    """
    zN = inp.n_m3.size
    U = np.zeros(zN, dtype=np.float64)
    dUn = np.zeros(zN, dtype=np.float64)
    dUp = np.zeros(zN, dtype=np.float64)

    for tr in inp.traps.traps:
        # Level energy
        E_t = _E_t_from_rel(inp.E_C_J, inp.E_V_J, tr.Erel_J, tr.ref)
        # SRH helper densities
        n1, p1 = n1_p1_from_bands(
            inp.E_C_J, inp.E_V_J, E_t, inp.T_K,
            me_rel=inp.me_rel, mh_rel=inp.mh_rel, gvc=inp.gvc, gvv=inp.gvv,
        )
        # Effective capture coeffs (m^3/s) — consistent with traps.py usage
        cn = np.full(zN, float(tr.sigma_n_m2), dtype=np.float64)
        cp = np.full(zN, float(tr.sigma_p_m2), dtype=np.float64)

        Uj, dUnj, dUpj = _srh_single_level_bulk(inp.n_m3, inp.p_m3, n1, p1, cn, cp, float(tr.Nt_m3))
        U += Uj
        dUn += dUnj
        dUp += dUpj

    return BulkSRHResult(U_srh=_c64(U), dU_dn=_c64(dUn), dU_dp=_c64(dUp))


# ---------------------------------------------------------------------
# Radiative recombination (bulk)
# ---------------------------------------------------------------------


@dataclass(slots=True)
class RadiativeInputs:
    n_m3: np.ndarray
    p_m3: np.ndarray
    B_m3_per_s: float | np.ndarray     # radiative coefficient B [m^3/s]
    ni2_m6: float | np.ndarray         # n_i^2 [1/m^6]


def bulk_radiative(inp: RadiativeInputs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    U_B = B (n p - n_i^2).
    Derivatives:
        ∂U/∂n = B p,   ∂U/∂p = B n.
    """
    n = _c64(inp.n_m3)
    p = _c64(inp.p_m3)
    B = _c64(inp.B_m3_per_s)
    ni2 = _c64(inp.ni2_m6)

    G = n * p - ni2
    U = B * G
    dU_dn = B * p
    dU_dp = B * n
    return U, dU_dn, dU_dp


# ---------------------------------------------------------------------
# Auger recombination (bulk)
# ---------------------------------------------------------------------


@dataclass(slots=True)
class AugerInputs:
    n_m3: np.ndarray
    p_m3: np.ndarray
    Cn_m6_per_s: float | np.ndarray    # [m^6/s]
    Cp_m6_per_s: float | np.ndarray    # [m^6/s]
    ni2_m6: float | np.ndarray         # n_i^2 [1/m^6]


def bulk_auger(inp: AugerInputs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    U_C = (C_n n + C_p p) (n p - n_i^2).
    Derivatives (product rule):
        ∂U/∂n = (C_n n + C_p p) p + C_n (n p - n_i^2)
        ∂U/∂p = (C_n n + C_p p) n + C_p (n p - n_i^2)
    """
    n = _c64(inp.n_m3)
    p = _c64(inp.p_m3)
    Cn = _c64(inp.Cn_m6_per_s)
    Cp = _c64(inp.Cp_m6_per_s)
    ni2 = _c64(inp.ni2_m6)

    F = Cn * n + Cp * p
    G = n * p - ni2
    U = F * G
    dU_dn = F * p + Cn * G
    dU_dp = F * n + Cp * G
    return U, dU_dn, dU_dp


# ---------------------------------------------------------------------
# Interface SRH (areal, at interface nodes)
# ---------------------------------------------------------------------


@dataclass(slots=True)
class InterfaceSRHInputs:
    """
    Interface SRH from discrete interface traps (sums over levels/spectra).

    node_indices : indices of interface nodes to evaluate.
    E_C_J, E_V_J : band edges arrays (for n1,p1).
    n_m3, p_m3   : carrier densities on the grid.
    traps        : InterfaceTrapSet (discrete levels and/or spectra).
    """
    node_indices: np.ndarray
    E_C_J: np.ndarray
    E_V_J: np.ndarray
    n_m3: np.ndarray
    p_m3: np.ndarray
    T_K: float
    me_rel: float | np.ndarray
    mh_rel: float | np.ndarray
    traps: InterfaceTrapSet
    gvc: float = 1.0
    gvv: float = 1.0


@dataclass(slots=True)
class InterfaceSRHResult:
    nodes: np.ndarray                # node indices (echo)
    U_s: np.ndarray                  # per-area rate [1/m^2/s] at each node
    dUs_dn: np.ndarray               # derivative wrt n at that node [m/s]
    dUs_dp: np.ndarray               # derivative wrt p at that node [m/s]
    meta: dict


def _srh_single_level_surface(
    n: float,
    p: float,
    n1: float,
    p1: float,
    cn: float,
    cp: float,
    Nit_m2: float,
) -> Tuple[float, float, float]:
    """
    Interface SRH for a single *areal* level:
        U_s = (cn cp N_it) * (n p - n1 p1) / D,
        D   = cn (n + n1) + cp (p + p1),
    with derivatives like the bulk case (units adjusted).
    """
    A = n * p - n1 * p1
    D = cn * (n + n1) + cp * (p + p1)
    if D == 0.0:
        D = 1e-30
    pref = cn * cp * Nit_m2
    U = pref * A / D
    dU_dn = pref * (p * D - A * cn) / (D * D)
    dU_dp = pref * (n * D - A * cp) / (D * D)
    return float(U), float(dU_dn), float(dU_dp)


def srh_interface_from_traps(inp: InterfaceSRHInputs) -> InterfaceSRHResult:
    """
    Sum interface SRH (areal) from discrete levels and spectra.
    Spectra are integrated over energy as in traps.interface_trap_sheet.
    """
    nodes = np.asarray(inp.node_indices, dtype=int)
    U_s = np.zeros_like(nodes, dtype=np.float64)
    dUs_dn = np.zeros_like(nodes, dtype=np.float64)
    dUs_dp = np.zeros_like(nodes, dtype=np.float64)
    meta: dict = {"spectra": []}

    # Map node -> position in output arrays
    node_pos = {int(i): k for k, i in enumerate(nodes)}

    # --- Discrete interface levels
    for it in inp.traps.discrete:
        idx = int(it.node_index)
        if idx not in node_pos:
            continue
        k = node_pos[idx]
        n = float(inp.n_m3[idx])
        p = float(inp.p_m3[idx])
        Ec = float(inp.E_C_J[idx])
        Ev = float(inp.E_V_J[idx])
        Et = Ec - it.Erel_J if it.ref == "Ec" else Ev + it.Erel_J

        # n1,p1 at node
        Nc = float(Nc_3d(inp.T_K, inp.me_rel, g_s=2.0, g_v=inp.gvc))
        Nv = float(Nv_3d(inp.T_K, inp.mh_rel, g_s=2.0, g_v=inp.gvv))
        n1 = Nc * np.exp(-(Ec - Et) / (K_B * inp.T_K))
        p1 = Nv * np.exp(-(Et - Ev) / (K_B * inp.T_K))

        cn = float(it.sigma_n_m2)
        cp = float(it.sigma_p_m2)

        Us, dnd, dpd = _srh_single_level_surface(n, p, n1, p1, cn, cp, float(it.Nit_m2))
        U_s[k] += Us
        dUs_dn[k] += dnd
        dUs_dp[k] += dpd

    # --- Spectral Dit(E) (integrate per spectrum)
    for spec in inp.traps.spectra:
        idx = int(spec.node_index)
        if idx not in node_pos:
            continue
        k = node_pos[idx]
        n = float(inp.n_m3[idx])
        p = float(inp.p_m3[idx])
        Ec = float(inp.E_C_J[idx])
        Ev = float(inp.E_V_J[idx])

        Erel = _c64(spec.Erel_J)            # (M,)
        Dit = _c64(spec.Dit_perJ_m2)        # [1/(m^2 J)]
        sig_n = _c64(spec.sigma_n_m2) if np.ndim(spec.sigma_n_m2) else float(spec.sigma_n_m2) * np.ones_like(Erel)
        sig_p = _c64(spec.sigma_p_m2) if np.ndim(spec.sigma_p_m2) else float(spec.sigma_p_m2) * np.ones_like(Erel)

        Et = (Ec - Erel) if spec.ref == "Ec" else (Ev + Erel)

        Nc = float(Nc_3d(inp.T_K, inp.me_rel, g_s=2.0, g_v=inp.gvc))
        Nv = float(Nv_3d(inp.T_K, inp.mh_rel, g_s=2.0, g_v=inp.gvv))
        n1 = Nc * np.exp(-(Ec - Et) / (K_B * inp.T_K))
        p1 = Nv * np.exp(-(Et - Ev) / (K_B * inp.T_K))

        # Energy-resolved SRH pieces (no derivative integration shortcuts here—do it pointwise)
        cn = sig_n
        cp = sig_p
        A = n * p - n1 * p1
        D = cn * (n + n1) + cp * (p + p1)
        D = np.where(D == 0.0, 1e-30, D)
        Us_E = (cn * cp * Dit) * A / D                 # [1/m^2/s/J]
        dUs_dn_E = (cn * cp * Dit) * (p * D - A * cn) / (D * D)
        dUs_dp_E = (cn * cp * Dit) * (n * D - A * cp) / (D * D)

        Us = float(np.trapz(Us_E, Erel))
        dUs_dn_k = float(np.trapz(dUs_dn_E, Erel))
        dUs_dp_k = float(np.trapz(dUs_dp_E, Erel))

        U_s[k] += Us
        dUs_dn[k] += dUs_dn_k
        dUs_dp[k] += dUs_dp_k

        meta["spectra"].append({"node": idx, "Us_Cm2s": Us})

    return InterfaceSRHResult(
        nodes=nodes, U_s=_c64(U_s), dUs_dn=_c64(dUs_dn), dUs_dp=_c64(dUs_dp), meta=meta
    )


# ---------------------------------------------------------------------
# Interface SRH via effective surface recombination velocities (SRV)
# ---------------------------------------------------------------------


@dataclass(slots=True)
class SRVInputs:
    """
    Effective interface SRH using SRVs (s_n, s_p) at interface nodes.

    This models a single effective trap 'centroid' via n1, p1. If you want to
    account for a detailed Dit(E), prefer srh_interface_from_traps().
    """
    node_indices: np.ndarray
    E_C_J: np.ndarray
    E_V_J: np.ndarray
    n_m3: np.ndarray
    p_m3: np.ndarray
    T_K: float
    me_rel: float | np.ndarray
    mh_rel: float | np.ndarray
    s_n_m_per_s: float | np.ndarray     # [m/s]
    s_p_m_per_s: float | np.ndarray     # [m/s]
    Erel_J: float                       # trap centroid relative to ref [J]
    ref: str                            # "Ec" or "Ev"
    gvc: float = 1.0
    gvv: float = 1.0


@dataclass(slots=True)
class SRVResult:
    nodes: np.ndarray
    U_s: np.ndarray
    dUs_dn: np.ndarray
    dUs_dp: np.ndarray


def srh_interface_srv(inp: SRVInputs) -> SRVResult:
    """
    SRH surface rate per area using effective SRVs (s_n, s_p):

        U_s = (s_n s_p) (n p - n1 p1) / D,
        D   = s_n (n + n1) + s_p (p + p1),

    evaluated at the provided interface node indices.
    """
    nodes = np.asarray(inp.node_indices, dtype=int)
    U_s = np.zeros_like(nodes, dtype=np.float64)
    dUs_dn = np.zeros_like(nodes, dtype=np.float64)
    dUs_dp = np.zeros_like(nodes, dtype=np.float64)

    s_n = np.asarray(inp.s_n_m_per_s, dtype=np.float64)
    s_p = np.asarray(inp.s_p_m_per_s, dtype=np.float64)
    if s_n.ndim == 0:
        s_n = np.full_like(U_s, float(s_n))
    if s_p.ndim == 0:
        s_p = np.full_like(U_s, float(s_p))

    for k, idx in enumerate(nodes):
        n = float(inp.n_m3[idx])
        p = float(inp.p_m3[idx])
        Ec = float(inp.E_C_J[idx])
        Ev = float(inp.E_V_J[idx])
        Et = Ec - inp.Erel_J if inp.ref == "Ec" else Ev + inp.Erel_J

        Nc = float(Nc_3d(inp.T_K, inp.me_rel, g_s=2.0, g_v=inp.gvc))
        Nv = float(Nv_3d(inp.T_K, inp.mh_rel, g_s=2.0, g_v=inp.gvv))
        n1 = Nc * np.exp(-(Ec - Et) / (K_B * inp.T_K))
        p1 = Nv * np.exp(-(Et - Ev) / (K_B * inp.T_K))

        A = n * p - n1 * p1
        D = s_n[k] * (n + n1) + s_p[k] * (p + p1)
        if D == 0.0:
            D = 1e-30
        Us = (s_n[k] * s_p[k]) * A / D
        dnd = (s_n[k] * s_p[k]) * (p * D - A * s_n[k]) / (D * D)
        dpd = (s_n[k] * s_p[k]) * (n * D - A * s_p[k]) / (D * D)

        U_s[k] = Us
        dUs_dn[k] = dnd
        dUs_dp[k] = dpd

    return SRVResult(nodes=nodes, U_s=_c64(U_s), dUs_dn=_c64(dUs_dn), dUs_dp=_c64(dUs_dp))


#how to use (comment out:)

from semisim.physics.recombination import (
    BulkSRHInputs, srh_bulk_from_traps,
    RadiativeInputs, bulk_radiative,
    AugerInputs, bulk_auger, ni2_MB
)

# SRH from your discrete bulk traps:
srh_res = srh_bulk_from_traps(BulkSRHInputs(
    E_C_J=E_C, E_V_J=E_V, n_m3=n, p_m3=p, T_K=T,
    me_rel=mat.me_dos_rel, mh_rel=mat.mh_dos_rel,
    traps=bulk_trap_set
))

# Radiative & Auger:
ni2 = ni2_MB(mat.Eg_J, T, mat.me_dos_rel, mat.mh_dos_rel)
U_B, dB_dn, dB_dp = bulk_radiative(RadiativeInputs(n, p, B_m3_per_s=B0, ni2_m6=ni2))
U_C, dC_dn, dC_dp = bulk_auger(AugerInputs(n, p, Cn_m6_per_s=Cn0, Cp_m6_per_s=Cp0, ni2_m6=ni2))

U_total = srh_res.U_srh + U_B + U_C
dU_dn = srh_res.dU_dn + dB_dn + dC_dn
dU_dp = srh_res.dU_dp + dB_dp + dC_dp

from semisim.physics.recombination import InterfaceSRHInputs, srh_interface_from_traps

iface_res = srh_interface_from_traps(InterfaceSRHInputs(
    node_indices=np.array([idx_if]),
    E_C_J=E_C, E_V_J=E_V, n_m3=n, p_m3=p, T_K=T,
    me_rel=mat.me_dos_rel, mh_rel=mat.mh_dos_rel,
    traps=it_set
))
# Boundary flux density to continuity:  J_rec = q * U_s  (with sign by convention).

from semisim.physics.recombination import SRVInputs, srh_interface_srv

srv = srh_interface_srv(SRVInputs(
    node_indices=np.array([idx_if]),
    E_C_J=E_C, E_V_J=E_V, n_m3=n, p_m3=p, T_K=T,
    me_rel=mat.me_dos_rel, mh_rel=mat.mh_dos_rel,
    s_n_m_per_s=1e3, s_p_m_per_s=1e3, Erel_J=0.3*Q, ref="Ec"
))

# ---------------------------------------------------------------------
# Bulk SRH with lifetimes (τn, τp) and a small config → evaluator builder
# ---------------------------------------------------------------------

@dataclass(slots=True)
class BulkSRHTauInputs:
    E_C_J: np.ndarray
    E_V_J: np.ndarray
    n_m3: np.ndarray
    p_m3: np.ndarray
    T_K: float
    me_rel: float | np.ndarray
    mh_rel: float | np.ndarray
    tau_n_s: float
    tau_p_s: float
    gvc: float = 1.0
    gvv: float = 1.0
    # E_t specified via midgap offset (Et - midgap) if desired
    Et_from_midgap_J: float = 0.0   # default midgap

def _srh_tau_eval(
    n: np.ndarray, p: np.ndarray, n1: np.ndarray, p1: np.ndarray, tau_n: float, tau_p: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    τ-based SRH:
        U = (n p - n1 p1) / (τ_p (n + n1) + τ_n (p + p1))
    """
    n = _c64(n); p = _c64(p); n1 = _c64(n1); p1 = _c64(p1)
    tau_n = float(tau_n); tau_p = float(tau_p)
    A = n * p - n1 * p1
    D = tau_p * (n + n1) + tau_n * (p + p1)
    D = np.where(D == 0.0, 1e-30, D)
    U = A / D
    D2 = D * D
    dU_dn = (p * D - A * tau_p) / D2
    dU_dp = (n * D - A * tau_n) / D2
    return U, dU_dn, dU_dp

def srh_bulk_tau(inp: BulkSRHTauInputs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SRH with lifetimes (no explicit trap set). n1, p1 taken from bands + Et.
    """
    Ec, Ev = _c64(inp.E_C_J), _c64(inp.E_V_J)
    Et_mid = 0.5 * (Ec + Ev) + float(inp.Et_from_midgap_J)
    n1, p1 = n1_p1_from_bands(
        Ec, Ev, Et_mid, float(inp.T_K),
        me_rel=inp.me_rel, mh_rel=inp.mh_rel, gvc=inp.gvc, gvv=inp.gvv
    )
    return _srh_tau_eval(inp.n_m3, inp.p_m3, n1, p1, float(inp.tau_n_s), float(inp.tau_p_s))

# ---------------------------------------------------------------------
# Builder: config → callable U(n,p), dU/dn(n,p), dU/dp(n,p)
# ---------------------------------------------------------------------
def build_recomb_evaluator_from_config(
    cfg: Dict,
    *,
    E_C_J: np.ndarray,
    E_V_J: np.ndarray,
    T_K: float,
    me_rel: float,
    mh_rel: float,
    gvc: float = 1.0,
    gvv: float = 1.0,
) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compose SRH(τ) + Radiative + Auger according to cfg.
    Expected cfg shape (any section may be omitted/disabled):
      {
        "srh": {
          "enable": true,
          "tau_n_s": 1e-7, "tau_p_s": 1e-7,
          "Et_from_midgap_eV": 0.0
        },
        "radiative": { "enable": false, "B_m3_per_s": 1e-16 },
        "auger":     { "enable": false, "Cn_m6_per_s": 2.8e-31, "Cp_m6_per_s": 0.99e-31 }
      }
    """
    srh_cfg = dict(cfg.get("srh", {}) or {})
    use_srh = bool(srh_cfg.get("enable", False))
    tau_n = float(srh_cfg.get("tau_n_s", 0.0))
    tau_p = float(srh_cfg.get("tau_p_s", 0.0))
    Et_off_J = float(srh_cfg.get("Et_from_midgap_eV", 0.0)) * Q

    rad_cfg = dict(cfg.get("radiative", {}) or {})
    use_rad = bool(rad_cfg.get("enable", False))
    B = float(rad_cfg.get("B_m3_per_s", 0.0))

    aug_cfg = dict(cfg.get("auger", {}) or {})
    use_aug = bool(aug_cfg.get("enable", False))
    Cn = float(aug_cfg.get("Cn_m6_per_s", 0.0))
    Cp = float(aug_cfg.get("Cp_m6_per_s", 0.0))

    Ec = _c64(E_C_J); Ev = _c64(E_V_J)
    Eg = Ec - Ev
    ni2 = ni2_MB(Eg, float(T_K), float(me_rel), float(mh_rel), gvc=float(gvc), gvv=float(gvv))

    # Precompute SRH helpers if enabled
    if use_srh and (tau_n > 0.0 or tau_p > 0.0):
        Et_mid = 0.5 * (Ec + Ev) + Et_off_J
        n1, p1 = n1_p1_from_bands(Ec, Ev, Et_mid, float(T_K),
                                  me_rel=float(me_rel), mh_rel=float(mh_rel), gvc=float(gvc), gvv=float(gvv))
    else:
        n1 = p1 = None

    def _eval(n: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = n.size
        U = np.zeros(N, dtype=np.float64)
        dUn = np.zeros(N, dtype=np.float64)
        dUp = np.zeros(N, dtype=np.float64)

        if use_srh and (tau_n > 0.0 or tau_p > 0.0):
            Us, dnd, dpd = _srh_tau_eval(n, p, n1, p1, tau_n, tau_p)
            U += Us; dUn += dnd; dUp += dpd

        if use_rad and (B != 0.0):
            Ub, dnd, dpd = bulk_radiative(RadiativeInputs(n, p, B_m3_per_s=B, ni2_m6=ni2))
            U += Ub; dUn += dnd; dUp += dpd

        if use_aug and (Cn != 0.0 or Cp != 0.0):
            Uc, dnd, dpd = bulk_auger(AugerInputs(n, p, Cn_m6_per_s=Cn, Cp_m6_per_s=Cp, ni2_m6=ni2))
            U += Uc; dUn += dnd; dUp += dpd

        return U, dUn, dUp

    return _eval

