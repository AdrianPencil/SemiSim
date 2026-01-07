# semisim/physics/carriers/statistics.py
"""
Carrier statistics utilities (bulk 3D + optional 2D sheets).

- SI units throughout.
- Vectorized over numpy arrays.
- Degeneracy-aware (Fermi–Dirac) with piecewise asymptotics:
    * MB tail for eta << 0
    * Sommerfeld expansion for eta >> 0
    * Smooth blend in between

Public API (stable):
    thermal_voltage(T)
    Nc_3d(T, me_dos_rel, g_s=2.0, g_v=1.0)
    Nv_3d(T, mh_dos_rel, g_s=2.0, g_v=1.0)

    carriers_3d(Ec_J, Ev_J, mu_J, T, *, me_rel, mh_rel,
                gvc=1.0, gvv=1.0, stats="FD", exp_clip=60.0)
    derivatives_3d(Ec_J, Ev_J, mu_J, T, *, me_rel, mh_rel,
                   gvc=1.0, gvv=1.0, stats="FD", exp_clip=60.0)

    sheet_carriers_2d(Esub_J, mu_J, T, *, m2d_rel, g_s=2.0, g_v=1.0)
    fermi_dirac_F12(eta), fermi_dirac_Fm12(eta)
"""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

# Prefer a single constants home for consistency
from ...utils.constants import K_B, Q, HBAR, M0
PI = np.pi
SQRT_PI = np.sqrt(np.pi)

__all__ = [
    "thermal_voltage",
    "Nc_3d",
    "Nv_3d",
    "fermi_dirac_F12",
    "fermi_dirac_Fm12",
    "carriers_3d",
    "derivatives_3d",
    "sheet_carriers_2d",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _c64(x):
    """Contiguous float64 array."""
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))

def _assert_finite(name: str, *arrs: np.ndarray) -> None:
    for i, a in enumerate(arrs, 1):
        if not np.all(np.isfinite(a)):
            bad = np.where(~np.isfinite(a))[0]
            raise FloatingPointError(
                f"{name}: NaN/Inf detected in array #{i}; first bad index={bad[0] if bad.size else 'n/a'}."
            )

def _validate_stats_inputs(Ec_J, Ev_J, mu_J, T, me_rel, mh_rel) -> None:
    Ec_J = np.asarray(Ec_J, dtype=np.float64)
    Ev_J = np.asarray(Ev_J, dtype=np.float64)
    mu_J = np.asarray(mu_J, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    me_rel = np.asarray(me_rel, dtype=np.float64)
    mh_rel = np.asarray(mh_rel, dtype=np.float64)

    # Finite checks
    if not np.all(np.isfinite(Ec_J)):
        idx = np.where(~np.isfinite(Ec_J))[0][:5]
        raise ValueError(f"E_C has non-finite values at indices {idx.tolist()}.")
    if not np.all(np.isfinite(Ev_J)):
        idx = np.where(~np.isfinite(Ev_J))[0][:5]
        raise ValueError(f"E_V has non-finite values at indices {idx.tolist()}.")
    if not np.all(np.isfinite(mu_J)):
        raise ValueError("mu_J has non-finite values.")
    if not np.all(np.isfinite(T)):
        raise ValueError("Temperature T has non-finite values.")
    if not np.all(np.isfinite(me_rel)):
        idx = np.where(~np.isfinite(me_rel))[0][:5]
        raise ValueError(f"me_dos_rel has non-finite values at indices {idx.tolist()}.")
    if not np.all(np.isfinite(mh_rel)):
        idx = np.where(~np.isfinite(mh_rel))[0][:5]
        raise ValueError(f"mh_dos_rel has non-finite values at indices {idx.tolist()}.")

    # Physical sanity
    if np.any(T <= 0.0):
        tmin = float(np.min(T))
        raise ValueError(f"T must be > 0 K (got min {tmin}).")
    if np.any(me_rel <= 0.0):
        idx = np.where(me_rel <= 0.0)[0][:5]
        mn = float(np.min(me_rel))
        raise ValueError(f"me_dos_rel must be > 0 everywhere (min {mn}); bad indices {idx.tolist()}.")
    if np.any(mh_rel <= 0.0):
        idx = np.where(mh_rel <= 0.0)[0][:5]
        mn = float(np.min(mh_rel))
        raise ValueError(f"mh_dos_rel must be > 0 everywhere (min {mn}); bad indices {idx.tolist()}.")

def _exp_safe(x: np.ndarray, clip: float = 60.0) -> np.ndarray:
    """exp(x) with symmetric clipping to avoid overflow/underflow."""
    return np.exp(np.clip(x, -clip, clip))


def thermal_voltage(T: float | np.ndarray) -> np.ndarray:
    """Thermal voltage V_T = k_B T / q [V]."""
    T = _c64(T)
    return (K_B * T) / Q


def _warn_bad_mu(mu_J) -> None:
    """Soft guard to catch wrong μ units (expects Joules ~1e-19 J)."""
    try:
        mu = float(np.asarray(mu_J).ravel()[0])
        if mu != 0.0 and not (1e-22 < abs(mu) < 1e-17):
            print(f"[warn] mu_J looks wrong (J): {mu:.3e} (expected ~1e-19 J)")
    except Exception:
        print("[warn] mu_J not parseable as float — expected Joules ~1e-19 J.")


# ---------------------------------------------------------------------
# Effective 3D DOS
# ---------------------------------------------------------------------
def Nc_3d(
    T: float | np.ndarray,
    me_dos_rel: float | np.ndarray,
    g_s: float = 2.0,
    g_v: float = 1.0,
) -> np.ndarray:
    """
    Effective conduction-band DOS per volume [1/m^3] at temperature T.
    Nc = g_s g_v (2π m* k_B T / h^2)^{3/2}
    """
    T = _c64(T)
    me = _c64(me_dos_rel) * M0
    pref = g_s * g_v * (2.0 * PI * K_B / ( (2.0 * PI * HBAR) ** 2)) ** 1.5  # since h = 2πħ
    return pref * (me ** 1.5) * (T ** 1.5)


def Nv_3d(
    T: float | np.ndarray,
    mh_dos_rel: float | np.ndarray,
    g_s: float = 2.0,
    g_v: float = 1.0,
) -> np.ndarray:
    """Effective valence-band DOS per volume [1/m^3] at temperature T."""
    T = _c64(T)
    mh = _c64(mh_dos_rel) * M0
    pref = g_s * g_v * (2.0 * PI * K_B / ( (2.0 * PI * HBAR) ** 2)) ** 1.5
    return pref * (mh ** 1.5) * (T ** 1.5)


# ---------------------------------------------------------------------
# Fermi–Dirac integrals (orders 1/2 and -1/2): piecewise approximations
# ---------------------------------------------------------------------
def _blend(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Blend two arrays: (1 - w) * a + w * b; w in [0, 1]."""
    return (1.0 - w) * a + w * b


def _logistic(x: np.ndarray, k: float = 0.7) -> np.ndarray:
    """Smooth transition weight ~ 1/(1+exp(-k x))."""
    arg = -k * np.clip(x, -100, 100)
    return 1.0 / (1.0 + np.exp(arg))


def fermi_dirac_F12(eta: np.ndarray | float) -> np.ndarray:
    """Approximate complete FD integral F_{1/2}(eta) (semiconductor normalization)."""
    eta = _c64(eta)
    # MB tail
    F_mb = _exp_safe(eta)
    # Sommerfeld head
    ep = np.maximum(eta, 0.0)
    ep = np.where(ep < 1e-12, 1e-12, ep)
    A0 = 2.0 / (3.0 * SQRT_PI)
    F_som = A0 * (ep ** 1.5) * (1.0 + (PI**2) / (8.0 * ep**2) + (7.0 * PI**4) / (640.0 * ep**4))
    # Blend
    w = _logistic(eta)
    return _blend(F_mb, F_som, w)


def fermi_dirac_Fm12(eta: np.ndarray | float) -> np.ndarray:
    """Approximate complete FD integral F_{-1/2}(eta)."""
    eta = _c64(eta)
    F_mb = _exp_safe(eta)
    ep = np.maximum(eta, 0.0)
    ep = np.where(ep < 1e-12, 1e-12, ep)
    A0 = 1.0 / SQRT_PI
    F_som = A0 * (ep ** 0.5) * (1.0 - (PI**2) / (24.0 * ep**2) - (7.0 * PI**4) / (3840.0 * ep**4))
    w = _logistic(eta)
    return _blend(F_mb, F_som, w)


# ---------------------------------------------------------------------
# Bulk 3D carriers (FD/MB) and derivatives
# ---------------------------------------------------------------------
def _eta_n(Ec_J: np.ndarray, mu_J: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (mu_J - Ec_J) / (K_B * T)


def _eta_p(Ev_J: np.ndarray, mu_J: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (Ev_J - mu_J) / (K_B * T)


def carriers_3d(
    Ec_J: np.ndarray,
    Ev_J: np.ndarray,
    mu_J: np.ndarray,
    T: float | np.ndarray,
    *,
    me_rel: float | np.ndarray,
    mh_rel: float | np.ndarray,
    gvc: float = 1.0,
    gvv: float = 1.0,
    stats: Literal["FD", "MB"] = "FD",
    exp_clip: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    3D bulk carrier densities n,p [m^-3].
    Raises a clear ValueError when inputs are unphysical (e.g., zero DOS masses).
    """
    _validate_stats_inputs(Ec_J, Ev_J, mu_J, T, me_rel, mh_rel)

    Ec_J = _c64(Ec_J)
    Ev_J = _c64(Ev_J)
    mu_J = _c64(mu_J)
    T = _c64(T)

    Nc = Nc_3d(T, me_rel, g_s=2.0, g_v=gvc)
    Nv = Nv_3d(T, mh_rel, g_s=2.0, g_v=gvv)

    if stats == "MB":
        n = Nc * _exp_safe((mu_J - Ec_J) / (K_B * T), clip=exp_clip)
        p = Nv * _exp_safe((Ev_J - mu_J) / (K_B * T), clip=exp_clip)
        _assert_finite("carriers_3d(MB)", n, p)
        return _c64(n), _c64(p)

    # FD
    eta_n = (mu_J - Ec_J) / (K_B * T)
    eta_p = (Ev_J - mu_J) / (K_B * T)
    n = Nc * fermi_dirac_F12(eta_n)
    p = Nv * fermi_dirac_F12(eta_p)

    _assert_finite("carriers_3d(FD)", n, p)
    return _c64(n), _c64(p)


def derivatives_3d(
    Ec_J: np.ndarray,
    Ev_J: np.ndarray,
    mu_J: np.ndarray,
    T: float | np.ndarray,
    *,
    me_rel: float | np.ndarray,
    mh_rel: float | np.ndarray,
    gvc: float = 1.0,
    gvv: float = 1.0,
    stats: Literal["FD", "MB"] = "FD",
    exp_clip: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Derivatives for Newton/Jacobians: dn/dphi, dp/dphi, dn/dT, dp/dT.
    Raises clear errors if inputs are unphysical.
    """
    _validate_stats_inputs(Ec_J, Ev_J, mu_J, T, me_rel, mh_rel)

    Ec_J = _c64(Ec_J)
    Ev_J = _c64(Ev_J)
    mu_J = _c64(mu_J)
    T = _c64(T)

    Nc = Nc_3d(T, me_rel, g_s=2.0, g_v=gvc)
    Nv = Nv_3d(T, mh_rel, g_s=2.0, g_v=gvv)

    if stats == "MB":
        eta_n = (mu_J - Ec_J) / (K_B * T)
        eta_p = (Ev_J - mu_J) / (K_B * T)
        en = _exp_safe(eta_n, clip=exp_clip)
        ep = _exp_safe(eta_p, clip=exp_clip)
        n = Nc * en
        p = Nv * ep
        dn_dphi = (Q / (K_B * T)) * n
        dp_dphi = -(Q / (K_B * T)) * p
        dNc_dT = (1.5 / T) * Nc
        dNv_dT = (1.5 / T) * Nv
        dn_dT = dNc_dT * en + Nc * en * (-eta_n / T)
        dp_dT = dNv_dT * ep + Nv * ep * (-eta_p / T)
        _assert_finite("derivatives_3d(MB)", dn_dphi, dp_dphi, dn_dT, dp_dT)
        return _c64(dn_dphi), _c64(dp_dphi), _c64(dn_dT), _c64(dp_dT)

    # FD
    eta_n = (mu_J - Ec_J) / (K_B * T)
    eta_p = (Ev_J - mu_J) / (K_B * T)
    F12_n = fermi_dirac_F12(eta_n)
    F12_p = fermi_dirac_F12(eta_p)
    Fm12_n = fermi_dirac_Fm12(eta_n)
    Fm12_p = fermi_dirac_Fm12(eta_p)

    dn_dphi = (Q / (K_B * T)) * Nc * Fm12_n
    dp_dphi = -(Q / (K_B * T)) * Nv * Fm12_p
    dNc_dT = (1.5 / T) * Nc
    dNv_dT = (1.5 / T) * Nv
    dn_dT = dNc_dT * F12_n + Nc * Fm12_n * (-eta_n / T)
    dp_dT = dNv_dT * F12_p + Nv * Fm12_p * (-eta_p / T)

    _assert_finite("derivatives_3d(FD)", dn_dphi, dp_dphi, dn_dT, dp_dT)
    return _c64(dn_dphi), _c64(dp_dphi), _c64(dn_dT), _c64(dp_dT)

# ---------------------------------------------------------------------
# 2D sheet statistics (single subband or list of subbands)
# ---------------------------------------------------------------------
def sheet_carriers_2d(
    Esub_J: np.ndarray,
    mu_J: float | np.ndarray,
    T: float | np.ndarray,
    *,
    m2d_rel: float | np.ndarray,
    g_s: float = 2.0,
    g_v: float = 1.0,
    exp_clip: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    2D electron gas per subband with Fermi–Dirac statistics.

    Esub_J : (..., Nsub) subband minima [J] (absolute, e.g., E_C(node)+ΔE_n)
    Returns (ns [1/m^2], d_ns_deta [1/m^2]); use dη/dφ=+q/(kT) for Jacobian.
    """
    Esub_J = _c64(Esub_J)
    mu = _c64(mu_J)
    T = _c64(T)
    m2d = _c64(m2d_rel) * M0

    kT = K_B * T
    eta = (np.expand_dims(mu, axis=-1) - Esub_J) / np.expand_dims(kT, axis=-1)

    # 2D DOS: g2D = g_s g_v m*/(π ħ^2)  [1/(J·m^2)], multiply by kT → [1/m^2]
    g2d = (float(g_s) * float(g_v)) * m2d / (np.pi * HBAR * HBAR)

    eta_clip = np.clip(eta, -float(exp_clip), float(exp_clip))
    e_eta = np.exp(eta_clip)
    log1p = np.log1p(e_eta)                     # ln(1 + e^η)
    occ = 1.0 / (1.0 + np.exp(-eta_clip))       # logistic(η)

    ns = g2d * kT * np.sum(log1p, axis=-1)      # [1/m^2]
    d_ns_deta = g2d * kT * np.sum(occ, axis=-1) # [1/m^2] per unit-η

    return _c64(ns), _c64(d_ns_deta)
