# semisim/solver/newton.py
# Newton–Raphson nonlinear solver for the PoissonProblem.
# Uses a custom tridiagonal solver (Thomas) to avoid SciPy deps.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from semisim.utils import diagnostics as diag

@dataclass
class SolveResult:
    x: np.ndarray           # final phi
    residual: np.ndarray    # final residual vector
    iters: int
    converged: bool


# ---- numerics ---------------------------------------------------------------


def _solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve tridiagonal system Ax=d, where A has subdiag a, diag b, superdiag c.
    Overwrites temporary copies of a,b,c,d (Thomas algorithm). O(N).
    """
    n = b.size
    ac = a.copy()
    bc = b.copy()
    cc = c.copy()
    dc = d.copy()

    # Forward elimination
    for i in range(1, n):
        if bc[i - 1] == 0.0:
            raise ZeroDivisionError("Zero diagonal encountered in tridiagonal solve.")
        m = ac[i] / bc[i - 1]
        bc[i] -= m * cc[i - 1]
        dc[i] -= m * dc[i - 1]

    # Back substitution
    x = np.zeros_like(dc)
    if bc[-1] == 0.0:
        raise ZeroDivisionError("Zero diagonal encountered in tridiagonal solve (last row).")
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        if bc[i] == 0.0:
            raise ZeroDivisionError("Zero diagonal encountered in tridiagonal solve.")
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]
    return x


def _inf_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, ord=np.inf))


# ---- solver -----------------------------------------------------------------


def solve(problem, options: Dict[str, Any] | None = None) -> SolveResult:
    """Newton solve for PoissonProblem using full Jacobian (with ∂ρ/∂φ terms).

    Parameters
    ----------
    problem : object with API:
        - x0: np.ndarray
        - residual(phi, for_newton=True) -> np.ndarray
        - jacobian(phi,  for_newton=True) -> (a,b,c) tridiagonal
        - project_dirichlet(phi) -> None
    options : dict
        Keys (all optional):
          - max_iters (int, default 50)
          - tol_res_inf (float, default 1e-6)
          - damping_init (float, default 1.0)
          - min_damping (float, default 1e-12)
          - max_step_volts (float, default 0.25)
          - print_every (int, default 1)
          - debug (bool, default False)
    """
    opts = dict(
        max_iters=50,
        tol_res_inf=1e-6,
        damping_init=1.0,
        min_damping=1e-12,
        max_step_volts=0.25,
        print_every=1,
        debug=getattr(problem, "debug", False),
    )
    if options:
        opts.update(options)

    use_scaled = getattr(problem, "scales", None) is not None
    Vstar = float(getattr(problem.scales, "V_scale", 1.0)) if use_scaled else 1.0
    Rstar = float(getattr(problem.scales, "R_scale", 1.0)) if use_scaled else 1.0

    def _project_scaled_inplace(x: np.ndarray) -> None:
        if not use_scaled:
            problem.project_dirichlet(x)
        else:
            phi_tmp = Vstar * x
            problem.project_dirichlet(phi_tmp)
            np.divide(phi_tmp, Vstar, out=x)

    # Initial guess
    x = (problem.x0_scaled().copy() if use_scaled else problem.x0.copy())
    _project_scaled_inplace(x)

    damping = float(opts["damping_init"])
    converged = False

    # Initial residual (scaled or physical)
    resid = (problem.residual_scaled(x, for_newton=True) if use_scaled
             else problem.residual(x, for_newton=True))
    res_inf = _inf_norm(resid)

    if opts["debug"]:
        phi_phys = Vstar * x
        diag.log_solver_start(
            solver="Newton", res_inf=res_inf,
            phi_min=float(phi_phys.min()), phi_max=float(phi_phys.max()),
            damping=float(damping),
        )

    for it in range(1, int(opts["max_iters"]) + 1):
        a, b, c = (problem.jacobian_scaled(x, for_newton=True) if use_scaled
                   else problem.jacobian(x, for_newton=True))
        delta = _solve_tridiagonal(a, b, c, -resid)

        # Clamp |Δφ| in volts
        max_step_V = float(opts["max_step_volts"])
        if use_scaled:
            delta_phi = Vstar * delta
            np.clip(delta_phi, -max_step_V, +max_step_V, out=delta_phi)
            delta = delta_phi / Vstar
        else:
            np.clip(delta, -max_step_V, +max_step_V, out=delta)

        # Backtracking line-search
        accepted = False
        local_damp = damping
        while local_damp >= float(opts["min_damping"]):
            x_trial = x + local_damp * delta
            _project_scaled_inplace(x_trial)

            resid_t = (problem.residual_scaled(x_trial, for_newton=True) if use_scaled
                       else problem.residual(x_trial, for_newton=True))
            res_inf_t = _inf_norm(resid_t)

            if res_inf_t < res_inf:  # accept
                x = x_trial
                resid = resid_t
                res_inf = res_inf_t
                accepted = True
                damping = min(1.0, local_damp * 1.25)
                break

            local_damp *= 0.5

        if not accepted:
            if opts["debug"]:
                diag.log_solver_backtrack_fail(
                    solver="Newton", it=int(it), res_inf=res_inf,
                    min_damping=float(opts["min_damping"]),
                )
            break

        if opts["debug"] and (it % int(opts["print_every"]) == 0):
            max_dphi = float(np.max(np.abs((Vstar * delta))))
            phi_phys = Vstar * x
            diag.log_solver_iter(
                solver="Newton", it=int(it), res_inf=res_inf,
                damping=float(damping), max_dphi=max_dphi,
                phi_min=float(phi_phys.min()), phi_max=float(phi_phys.max()),
            )

        if res_inf < float(opts["tol_res_inf"]):
            converged = True
            break

    if opts["debug"]:
        diag.log_convergence_summary(
            solver="Newton", converged=bool(converged),
            iters=int(it), res_inf=_inf_norm(resid),
        )

    phi_out = (Vstar * x) if use_scaled else x
    resid_out = (Rstar * resid) if use_scaled else resid
    return SolveResult(x=phi_out, residual=resid_out, iters=it, converged=converged)
