"""
semisim/discretization/fluxes.py

Scharfetter–Gummel (SG) flux utilities for 1-D drift–diffusion.
Provides numerically stable Bernoulli functions and per-face electron/hole
fluxes with a clear sign convention.

Sign convention (1-D mesh):
- Nodes increase with index i → i+1 (left → right).
- Face i+1/2 is between nodes i (left) and i+1 (right).
- Electrostatic potential: phi [V]; thermal voltage: V_T = kT/q [V].
- For electrons:
    y = (phi_R - phi_L) / V_T
    Jn_{i+1/2} = (q * mu_n * V_T / dx) * ( n_L * B(+y) - n_R * B(-y) )
- For holes:
    Jp_{i+1/2} = (q * mu_p * V_T / dx) * ( p_R * B(+y) - p_L * B(-y) )

At thermal equilibrium with constant quasi-Fermi levels, these give zero flux.

NOTE: Mobility mu and density n, p are evaluated at nodes.
      dx is the cell length between nodes (right - left).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


Q = 1.602176634e-19  # Coulomb


def bern(x: np.ndarray | float) -> np.ndarray | float:
    """
    Numerically stable Bernoulli function:
        B(x) = x / (exp(x) - 1)
    with series expansion for small |x|.
    Returns array-like with dtype float64.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x_arr)

    # |x| small: use series B(x) ≈ 1 - x/2 + x^2/12 - x^4/720 ...
    small = np.abs(x_arr) < 1.0e-4
    xs = x_arr[small]
    out[small] = 1.0 - xs / 2.0 + xs * xs / 12.0 - (xs ** 4) / 720.0

    # |x| large: direct formula
    big = ~small
    xb = x_arr[big]
    out[big] = xb / (np.exp(xb) - 1.0)

    # Return scalar if scalar input
    return out if isinstance(x, np.ndarray) else float(out)


def bern_pair(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (B(y), B(-y)) using stable identities. Uses:
      B(-y) = exp(y) * B(y)
    to avoid extra exponentials where possible.
    """
    By = bern(y)
    Bm = np.exp(y) * By  # B(-y) = e^{y} B(y)
    return By, Bm


def sg_flux_n(
    nL: float | np.ndarray,
    nR: float | np.ndarray,
    phiL: float | np.ndarray,
    phiR: float | np.ndarray,
    mu_n: float | np.ndarray,
    V_T: float,
    dx: float | np.ndarray,
) -> np.ndarray:
    """
    Electron SG face flux Jn_{i+1/2} from node values (left L, right R).

    Parameters
    ----------
    nL, nR : node electron densities [m^-3]
    phiL, phiR : node potentials [V]
    mu_n : electron mobility [m^2/(V·s)]
    V_T : thermal voltage [V]
    dx : cell length [m]

    Returns
    -------
    Jn_face : electron current density at face [A/m^2]
    """
    nL = np.asarray(nL, dtype=np.float64)
    nR = np.asarray(nR, dtype=np.float64)
    phiL = np.asarray(phiL, dtype=np.float64)
    phiR = np.asarray(phiR, dtype=np.float64)
    mu_n = np.asarray(mu_n, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)

    y = (phiR - phiL) / float(V_T)
    By, Bm = bern_pair(y)
    return (Q * mu_n * float(V_T) / dx) * (nL * By - nR * Bm)


def sg_flux_p(
    pL: float | np.ndarray,
    pR: float | np.ndarray,
    phiL: float | np.ndarray,
    phiR: float | np.ndarray,
    mu_p: float | np.ndarray,
    V_T: float,
    dx: float | np.ndarray,
) -> np.ndarray:
    """
    Hole SG face flux Jp_{i+1/2} from node values (left L, right R).

    Using the convention:
      Jp_{i+1/2} = (q * mu_p * V_T / dx) * ( p_R * B(+y) - p_L * B(-y) )
    with y = (phi_R - phi_L)/V_T (same y as electrons).
    """
    pL = np.asarray(pL, dtype=np.float64)
    pR = np.asarray(pR, dtype=np.float64)
    phiL = np.asarray(phiL, dtype=np.float64)
    phiR = np.asarray(phiR, dtype=np.float64)
    mu_p = np.asarray(mu_p, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)

    y = (phiR - phiL) / float(V_T)
    By, Bm = bern_pair(y)
    return (Q * mu_p * float(V_T) / dx) * (pR * By - pL * Bm)


def faces_from_nodes(arr: np.ndarray) -> np.ndarray:
    """Simple arithmetic average from nodes to faces (i,i+1) → i+1/2."""
    arr = np.asarray(arr, dtype=np.float64)
    return 0.5 * (arr[1:] + arr[:-1])


def divergence_from_faces(Jf: np.ndarray, Vi: np.ndarray) -> np.ndarray:
    """
    Compute divergence of face fluxes on nodes using control volumes Vi.
    Assumes 1-D with Jf shape (N-1,) and Vi shape (N,).
    Returns array shape (N,) with zero at boundaries (to be set by BCs).
    """
    N = Vi.size
    div = np.zeros(N, dtype=np.float64)
    div[1:-1] = (Jf[1:] - Jf[:-1]) / Vi[1:-1]
    return div
