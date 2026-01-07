"""
semisim/solver/subsolvers/continuity_linear.py

Carrier "linear" sub-solver for Gummel with fixed electrostatic potential φ.
This module updates carriers and (optionally) computes SG currents, but keeps
things simple and robust by avoiding a big coupled linear solve at first:
- With φ fixed, compute n, p from band edges and statistics (MB/FD).
- Optionally compute SG face fluxes Jn, Jp for diagnostics.
- Apply carrier masks (e.g., metals/oxides → zero carriers).
- Provide a hook to extend to a true tridiagonal solve for quasi-Fermi vars.

This keeps the first Gummel version stable and easy to wire. Later, you can
upgrade `solve_electrons/solve_holes` to do a real SG-based linear solve for
ψ_n, ψ_p with your chosen boundary conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from semisim.discretization.fluxes import (
    faces_from_nodes,
    sg_flux_n,
    sg_flux_p,
    divergence_from_faces,
)

# Physics helpers (adjust paths if your repo differs)
try:
    from semisim.physics.carriers.bands import band_edges_from_potential  # type: ignore
except Exception:  # pragma: no cover
    from bands import band_edges_from_potential  # fallback

# Corrected version
try:
    from semisim.physics.carriers.statistics import (
        carriers_3d,
    )
except Exception:
    from semisim.physics.carriers.statistics import carriers_3d # fallback for local dev


Q = 1.602176634e-19  # C
K_B = 1.380649e-23   # J/K


@dataclass
class CarrierSolveResult:
    n: np.ndarray
    p: np.ndarray
    Jn_faces: Optional[np.ndarray]
    Jp_faces: Optional[np.ndarray]
    divJn: Optional[np.ndarray]
    divJp: Optional[np.ndarray]


def _thermal_voltage(T_K: float) -> float:
    return float(K_B * float(T_K) / Q)


def _apply_mask(x: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return x
    y = x.copy()
    y[~mask.astype(bool)] = 0.0
    return y


def solve_electrons(
    *,
    phi: np.ndarray,
    z: np.ndarray,
    mu_n: np.ndarray | float,
    bands: object,
    mat: object,
    mu_J: float,
    T_K: float,
    gvc: float,
    stats: str,
    exp_clip: float,
    carrier_mask: Optional[np.ndarray],
    compute_currents: bool = True,
    debug: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Update electron density for fixed φ using statistics (MB/FD). Optionally
    compute SG face currents for diagnostics (not used for the update itself).
    """
    V_T = _thermal_voltage(T_K)
    E_C, E_V = band_edges_from_potential(phi, bands)

    # Node electron density from statistics (no continuity linear solve yet)
    n = carriers_3d(
        E_C,
        E_V,
        mu_J,
        T_K,
        me_rel=mat.me_dos_rel,
        mh_rel=mat.mh_dos_rel,
        gvc=gvc,
        gvv=1.0,  # not used for electrons
        stats=stats,
        exp_clip=exp_clip,
    )[0]  # returns (n, p); we take n

    n = _apply_mask(n, carrier_mask)

    if not compute_currents:
        return n, None, None

    # SG face currents (diagnostics)
    dx = np.diff(z)
    mu_n_nodes = np.asarray(mu_n, dtype=np.float64) if np.ndim(mu_n) else float(mu_n) * np.ones_like(phi)
    mu_n_faces = faces_from_nodes(mu_n_nodes)

    Jn_faces = sg_flux_n(
        nL=n[:-1],
        nR=n[1:],
        phiL=phi[:-1],
        phiR=phi[1:],
        mu_n=mu_n_faces,
        V_T=V_T,
        dx=dx,
    )
    divJn = divergence_from_faces(Jn_faces, Vi=np.ones_like(phi))  # Vi unused for pure diagnostic

    if debug:
        print(
            f"[ContinuityLinear:e-] max n={float(np.max(n)):.3e} m^-3 | "
            f"max|Jn|={float(np.max(np.abs(Jn_faces))):.3e} A/m^2"
        )

    return n, Jn_faces, divJn


def solve_holes(
    *,
    phi: np.ndarray,
    z: np.ndarray,
    mu_p: np.ndarray | float,
    bands: object,
    mat: object,
    mu_J: float,
    T_K: float,
    gvv: float,
    stats: str,
    exp_clip: float,
    carrier_mask: Optional[np.ndarray],
    compute_currents: bool = True,
    debug: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Update hole density for fixed φ using statistics (MB/FD). Optionally
    compute SG face currents for diagnostics (not used for the update itself).
    """
    V_T = _thermal_voltage(T_K)
    E_C, E_V = band_edges_from_potential(phi, bands)

    # Node hole density from statistics
    p = carriers_3d(
        E_C,
        E_V,
        mu_J,
        T_K,
        me_rel=mat.me_dos_rel,
        mh_rel=mat.mh_dos_rel,
        gvc=1.0,  # not used for holes
        gvv=gvv,
        stats=stats,
        exp_clip=exp_clip,
    )[1]  # returns (n, p); we take p

    p = _apply_mask(p, carrier_mask)

    if not compute_currents:
        return p, None, None

    # SG face currents (diagnostics)
    dx = np.diff(z)
    mu_p_nodes = np.asarray(mu_p, dtype=np.float64) if np.ndim(mu_p) else float(mu_p) * np.ones_like(phi)
    mu_p_faces = faces_from_nodes(mu_p_nodes)

    Jp_faces = sg_flux_p(
        pL=p[:-1],
        pR=p[1:],
        phiL=phi[:-1],
        phiR=phi[1:],
        mu_p=mu_p_faces,
        V_T=V_T,
        dx=dx,
    )
    divJp = divergence_from_faces(Jp_faces, Vi=np.ones_like(phi))

    if debug:
        print(
            f"[ContinuityLinear:h+] max p={float(np.max(p)):.3e} m^-3 | "
            f"max|Jp|={float(np.max(np.abs(Jp_faces))):.3e} A/m^2"
        )

    return p, Jp_faces, divJp


def solve_carriers_fixed_phi(
    *,
    phi: np.ndarray,
    z: np.ndarray,
    mu_n: np.ndarray | float,
    mu_p: np.ndarray | float,
    bands: object,
    mat: object,
    mu_J: float,
    T_K: float,
    gvc: float,
    gvv: float,
    stats: str,
    exp_clip: float,
    carrier_mask: Optional[np.ndarray],
    compute_currents: bool = True,
    debug: bool = False,
) -> CarrierSolveResult:
    """
    Convenience wrapper: update both n and p for fixed φ and (optionally)
    compute SG face currents for diagnostics.
    """
    n, Jn_faces, divJn = solve_electrons(
        phi=phi,
        z=z,
        mu_n=mu_n,
        bands=bands,
        mat=mat,
        mu_J=mu_J,
        T_K=T_K,
        gvc=gvc,
        stats=stats,
        exp_clip=exp_clip,
        carrier_mask=carrier_mask,
        compute_currents=compute_currents,
        debug=debug,
    )

    p, Jp_faces, divJp = solve_holes(
        phi=phi,
        z=z,
        mu_p=mu_p,
        bands=bands,
        mat=mat,
        mu_J=mu_J,
        T_K=T_K,
        gvv=gvv,
        stats=stats,
        exp_clip=exp_clip,
        carrier_mask=carrier_mask,
        compute_currents=compute_currents,
        debug=debug,
    )

    return CarrierSolveResult(n=n, p=p, Jn_faces=Jn_faces, Jp_faces=Jp_faces, divJn=divJn, divJp=divJp)
