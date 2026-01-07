"""
semisim/solver/subsolvers/poisson_linear.py

Linear Poisson sub-solver used by Gummel:
- Treats carriers and 2DEG sheet as *frozen* inputs.
- Assembles the electrostatic tridiagonal Jacobian for phi.
- Solves J · dphi = -res using Thomas algorithm with simple damping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class PoissonLinearResult:
    phi: np.ndarray
    residual: np.ndarray
    iters: int
    accepted: bool


def _solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Thomas algorithm for tridiagonal systems."""
    n = b.size
    ac = a.copy()
    bc = b.copy()
    cc = c.copy()
    dc = d.copy()

    for i in range(1, n):
        if bc[i - 1] == 0.0:
            raise ZeroDivisionError("Zero diagonal encountered (forward elim).")
        m = ac[i] / bc[i - 1]
        bc[i] -= m * cc[i - 1]
        dc[i] -= m * dc[i - 1]

    x = np.zeros_like(dc)
    if bc[-1] == 0.0:
        raise ZeroDivisionError("Zero diagonal encountered (back solve).")
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        if bc[i] == 0.0:
            raise ZeroDivisionError("Zero diagonal encountered.")
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]
    return x

# --- helpers to read the new PoissonBC API safely --------------------
def _is_dirichlet_left(bc: object) -> bool:
    kind = getattr(bc, "left_kind", "dirichlet")
    return str(kind).lower() == "dirichlet"

def _is_dirichlet_right(bc: object) -> bool:
    kind = getattr(bc, "right_kind", "dirichlet")
    return str(kind).lower() == "dirichlet"

def _phi_left(bc: object) -> float:
    return float(getattr(bc, "left_value", 0.0))

def _phi_right(bc: object) -> float:
    return float(getattr(bc, "right_value", 0.0))

def assemble_electrostatic_tridiagonal(
    alpha_f: np.ndarray,
    Vi: np.ndarray,
    bc: object,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = Vi.size
    a = np.zeros(N, dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)
    c = np.zeros(N, dtype=np.float64)

    a[1:-1] = -alpha_f[:-1] / Vi[1:-1]
    c[1:-1] = -alpha_f[1:] / Vi[1:-1]
    b[1:-1] = (alpha_f[:-1] + alpha_f[1:]) / Vi[1:-1]

    # Left boundary
    if _is_dirichlet_left(bc):
        b[0] = 1.0; a[0] = 0.0; c[0] = 0.0
    else:
        # Natural/Neumann: residual[0] handled in residual function
        b[0] += alpha_f[0] / Vi[0]
        c[0] += -alpha_f[0] / Vi[0]

    # Right boundary
    if _is_dirichlet_right(bc):
        b[-1] = 1.0; a[-1] = 0.0; c[-1] = 0.0
    else:
        b[-1] += alpha_f[-1] / Vi[-1]
        a[-1] += -alpha_f[-1] / Vi[-1]

    return a, b, c

def residual_from_phi_fixed_carriers(
    phi: np.ndarray,
    alpha_f: np.ndarray,
    Vi: np.ndarray,
    ND: np.ndarray,
    NA: np.ndarray,
    rho_extra: np.ndarray,
    sigma_node: np.ndarray,
    n_fixed: Optional[np.ndarray] = None,
    p_fixed: Optional[np.ndarray] = None,
    sigma2d_fixed: Optional[np.ndarray] = None,
    bc: object = None,
) -> np.ndarray:
    """
    Residual for Poisson with *fixed* n, p, and optional sigma_2D arrays.
    """
    Q = 1.602176634e-19
    N = Vi.size

    if n_fixed is None:
        n_fixed = np.zeros(N)
    if p_fixed is None:
        p_fixed = np.zeros(N)
    if sigma2d_fixed is None:
        sigma2d_fixed = np.zeros(N)

    # volumetric and sheet charge contributions
    rho_vol = Q * (p_fixed - n_fixed + ND - NA) + rho_extra
    sigma_total = sigma_node + sigma2d_fixed

    # face flux and interior residuals
    F = -alpha_f * (phi[1:] - phi[:-1])  # face flux
    resid = np.zeros(N, dtype=np.float64)
    resid[1:-1] = (F[1:] - F[:-1]) / Vi[1:-1] + rho_vol[1:-1] + sigma_total[1:-1] / Vi[1:-1]

    # Left boundary
    if _is_dirichlet_left(bc):
        resid[0] = phi[0] - _phi_left(bc)
    else:
        # natural/Neumann-like row
        resid[0] = +F[0] / Vi[0] + rho_vol[0] + sigma_total[0] / Vi[0]

    # Right boundary
    if _is_dirichlet_right(bc):
        resid[-1] = phi[-1] - _phi_right(bc)
    else:
        resid[-1] = -F[-1] / Vi[-1] + rho_vol[-1] + sigma_total[-1] / Vi[-1]

    return resid

def project_dirichlet(phi: np.ndarray, bc: object) -> None:
    if _is_dirichlet_left(bc):
        phi[0] = _phi_left(bc)
    if _is_dirichlet_right(bc):
        phi[-1] = _phi_right(bc)


def solve(
    *,
    phi0: np.ndarray,
    alpha_f: np.ndarray,
    Vi: np.ndarray,
    ND: np.ndarray,
    NA: np.ndarray,
    rho_extra: np.ndarray,
    sigma_node: np.ndarray,
    n_fixed: Optional[np.ndarray],
    p_fixed: Optional[np.ndarray],
    sigma2d_fixed: Optional[np.ndarray],
    bc: object,
    damping_init: float = 1.0,
    max_step_volts: float = 0.2,
    min_damping: float = 1e-12,
    max_iters: int = 20,
    print_every: int = 1,
    debug: bool = False,
) -> PoissonLinearResult:
    """
    Solve the electrostatic Poisson subproblem with *fixed* carriers.
    Returns updated phi after at most `max_iters` (typically 1–3 is enough).
    """
    phi = np.asarray(phi0, dtype=np.float64).copy()
    project_dirichlet(phi, bc)

    a, b, c = assemble_electrostatic_tridiagonal(alpha_f, Vi, bc)

    resid = residual_from_phi_fixed_carriers(
        phi, alpha_f, Vi, ND, NA, rho_extra, sigma_node, n_fixed, p_fixed, sigma2d_fixed, bc
    )
    res_inf = float(np.linalg.norm(resid, ord=np.inf))
    damping = float(damping_init)

    if debug:
        print(
            f"[PoissonLinear] start | ||res||_inf={res_inf:.3e} | "
            f"φ∈[{phi.min():+.3e},{phi.max():+.3e}] V | damping={damping:.2e}"
        )

    accepted = False
    it = 0
    for it in range(1, int(max_iters) + 1):
        delta = _solve_tridiagonal(a, b, c, -resid)

        # Clamp update to avoid overshoot
        delta = np.clip(delta, -float(max_step_volts), +float(max_step_volts))

        local = damping
        while local >= float(min_damping):
            phi_trial = phi + local * delta
            project_dirichlet(phi_trial, bc)

            resid_t = residual_from_phi_fixed_carriers(
                phi_trial, alpha_f, Vi, ND, NA, rho_extra, sigma_node, n_fixed, p_fixed, sigma2d_fixed, bc
            )
            res_inf_t = float(np.linalg.norm(resid_t, ord=np.inf))

            if res_inf_t < res_inf:
                phi = phi_trial
                resid = resid_t
                res_inf = res_inf_t
                accepted = True
                damping = min(1.0, local * 1.25)
                break
            local *= 0.5

        if debug and (it % int(print_every) == 0):
            print(
                f"[PoissonLinear] iter {it:02d} | ||res||_inf={res_inf:.3e} | "
                f"damping={damping:.2e} | max|Δφ|={float(np.max(np.abs(delta))):.3e} V | "
                f"φ∈[{phi.min():+.3e},{phi.max():+.3e}] V"
            )

        if accepted:
            break

    return PoissonLinearResult(phi=phi, residual=resid, iters=it, accepted=accepted)
