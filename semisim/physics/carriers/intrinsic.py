# semisim/physics/carriers/intrinsic.py
"""
Intrinsic references and incomplete ionization helpers (SI units).

- Minimal, reusable utilities for diagnostics and model seeding.
- MB (Maxwell–Boltzmann) formulas for E_i and n_i (good baseline).
- Optional incomplete ionization for dopants (freeze-out).

Public API:
    intrinsic_level_MB(E_C_J, E_V_J, T, me_rel, mh_rel, gvc=1.0, gvv=1.0) -> E_i
    intrinsic_density_MB(Eg_J, T, me_rel, mh_rel, gvc=1.0, gvv=1.0) -> n_i
    ionized_donors(E_D_J, mu_J, T, N_D, g_D=2.0) -> N_D_plus
    ionized_acceptors(E_A_J, mu_J, T, N_A, g_A=4.0) -> N_A_minus

Notes
-----
- E_C = E_C^0 - q phi, E_V = E_V^0 - q phi should be provided by bands module.
- For degenerate regimes, use carriers/statistics.py for n,p and Jacobians; this
  module is primarily for references/initialization and low-T doping freeze-out.
"""

from __future__ import annotations

import numpy as np

from semisim.physics.carriers.statistics import Nc_3d, Nv_3d

# ---- constants (SI) ----
Q = 1.602176634e-19    # C
K_B = 1.380649e-23     # J/K

__all__ = [
    "intrinsic_level_MB",
    "intrinsic_density_MB",
    "ionized_donors",
    "ionized_acceptors",
]


def intrinsic_level_MB(
    E_C_J: np.ndarray,
    E_V_J: np.ndarray,
    T: float | np.ndarray,
    *,
    me_rel: float | np.ndarray,
    mh_rel: float | np.ndarray,
    gvc: float = 1.0,
    gvv: float = 1.0,
) -> np.ndarray:
    """
    Intrinsic Fermi level (J) under MB approximation:
        E_i = (E_C + E_V)/2 + (k_B T / 2) * ln(N_v / N_c).
    """
    T = np.asarray(T, dtype=np.float64)
    Nc = Nc_3d(T, me_rel, g_s=2.0, g_v=gvc)
    Nv = Nv_3d(T, mh_rel, g_s=2.0, g_v=gvv)
    midgap = 0.5 * (np.asarray(E_C_J, dtype=np.float64) + np.asarray(E_V_J, dtype=np.float64))
    Ei = midgap + 0.5 * K_B * T * np.log(Nv / Nc)
    return np.ascontiguousarray(Ei)


def intrinsic_density_MB(
    Eg_J: np.ndarray,
    T: float | np.ndarray,
    *,
    me_rel: float | np.ndarray,
    mh_rel: float | np.ndarray,
    gvc: float = 1.0,
    gvv: float = 1.0,
) -> np.ndarray:
    """
    Intrinsic density (1/m^3) under MB approximation:
        n_i = sqrt(N_c N_v) * exp(-E_g / (2 k_B T)).
    """
    T = np.asarray(T, dtype=np.float64)
    Nc = Nc_3d(T, me_rel, g_s=2.0, g_v=gvc)
    Nv = Nv_3d(T, mh_rel, g_s=2.0, g_v=gvv)
    Eg = np.asarray(Eg_J, dtype=np.float64)
    return np.sqrt(Nc * Nv) * np.exp(-Eg / (2.0 * K_B * T))


def ionized_donors(
    E_D_J: np.ndarray | float,
    mu_J: np.ndarray | float,
    T: float | np.ndarray,
    N_D: np.ndarray | float,
    g_D: float = 2.0,
) -> np.ndarray:
    """
    Incomplete ionization for donors (shallow level E_D, degeneracy g_D):
        f_D = 1 / (1 + g_D * exp((E_D - mu)/kT)),  N_D^+ = N_D * (1 - f_D).
    """
    T = np.asarray(T, dtype=np.float64)
    E_D = np.asarray(E_D_J, dtype=np.float64)
    mu = np.asarray(mu_J, dtype=np.float64)
    N_D = np.asarray(N_D, dtype=np.float64)
    eta = (E_D - mu) / (K_B * T)
    fD = 1.0 / (1.0 + g_D * np.exp(eta))
    return np.ascontiguousarray(N_D * (1.0 - fD))


def ionized_acceptors(
    E_A_J: np.ndarray | float,
    mu_J: np.ndarray | float,
    T: float | np.ndarray,
    N_A: np.ndarray | float,
    g_A: float = 4.0,
) -> np.ndarray:
    """
    Incomplete ionization for acceptors (level E_A, degeneracy g_A):
        f_A = 1 / (1 + g_A * exp((mu - E_A)/kT)),  N_A^- = N_A * f_A.
    """
    T = np.asarray(T, dtype=np.float64)
    E_A = np.asarray(E_A_J, dtype=np.float64)
    mu = np.asarray(mu_J, dtype=np.float64)
    N_A = np.asarray(N_A, dtype=np.float64)
    eta = (mu - E_A) / (K_B * T)
    fA = 1.0 / (1.0 + g_A * np.exp(eta))
    return np.ascontiguousarray(N_A * fA)
