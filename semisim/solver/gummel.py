# semisim/solver/gummel.py
# Fixed-point (Gummel) solver for the PoissonProblem.
# Uses electrostatic Jacobian only (carriers treated as fixed in each outer step).

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


def _solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Thomas algorithm (see newton.py for comments)."""
    n = b.size
    ac = a.copy()
    bc = b.copy()
    cc = c.copy()
    dc = d.copy()

    for i in range(1, n):
        if bc[i - 1] == 0.0:
            raise ZeroDivisionError("Zero diagonal encountered in tridiagonal solve.")
        m = ac[i] / bc[i - 1]
        bc[i] -= m * cc[i - 1]
        dc[i] -= m * dc[i - 1]

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


def solve(problem, options: Dict[str, Any] | None = None) -> SolveResult:
    """Gummel fixed-point for PoissonProblem.

    Treats carriers as *fixed* within each outer iteration; i.e., uses the
    electrostatic Jacobian only (no ∂ρ/∂φ terms). This is typically more robust
    than raw Newton on first contact.

    Parameters
    ----------
    problem : object with API:
        - x0: np.ndarray
        - residual(phi, for_newton=False) -> np.ndarray
        - jacobian(phi,  for_newton=False) -> (a,b,c) tridiagonal (electrostatic only)
        - project_dirichlet(phi) -> None
    options : dict
        Keys (all optional):
          - max_iters (int, default 120)
          - tol_res_inf (float, default 1e-6)
          - damping_init (float, default 1.0)
          - min_damping (float, default 1e-12)
          - max_step_volts (float, default 0.20)
          - print_every (int, default 1)
          - debug (bool, default False)
    """
    opts = dict(
        max_iters=120,
        tol_res_inf=1e-6,
        damping_init=1.0,
        min_damping=1e-12,
        max_step_volts=0.20,
        print_every=1,
        debug=getattr(problem, "debug", False),
    )
    if options:
        opts.update(options)

    use_scaled = getattr(problem, "scales", None) is not None
    Vstar = float(getattr(problem.scales, "V_scale", 1.0)) if use_scaled else 1.0
    Rstar = float(getattr(problem.scales, "R_scale", 1.0)) if use_scaled else 1.0

    def _project_scaled_inplace(x: np.ndarray) -> None:
        """Project Dirichlet in physical volts, then map back to scaled x."""
        if not use_scaled:
            problem.project_dirichlet(x)
        else:
            phi_tmp = Vstar * x
            problem.project_dirichlet(phi_tmp)
            np.divide(phi_tmp, Vstar, out=x)

    # Initial guess (scaled or physical)
    if use_scaled:
        x = problem.x0_scaled().copy()
    else:
        x = problem.x0.copy()
    _project_scaled_inplace(x)

    # NEW: refresh carriers for this potential (fixed-φ continuity step)
    if hasattr(problem, "update_carriers_fixed_phi"):
        if use_scaled:
            problem.update_carriers_fixed_phi(Vstar * x, compute_currents=False)
        else:
            problem.update_carriers_fixed_phi(x, compute_currents=False)

    damping = float(opts["damping_init"])
    converged = False

    # Initial residual
    if use_scaled:
        resid = problem.residual_scaled(x, for_newton=False)
    else:
        resid = problem.residual(x, for_newton=False)
    res_inf = _inf_norm(resid)

    if opts["debug"]:
        # Report φ-range in physical volts, even if solving in scaled coords
        phi_phys = Vstar * x
        diag.log_solver_start(
            solver="Gummel", res_inf=res_inf,
            phi_min=float(phi_phys.min()), phi_max=float(phi_phys.max()),
            damping=float(damping),
        )

    for it in range(1, int(opts["max_iters"]) + 1):
        # Jacobian (scaled or physical)
        if use_scaled:
            a, b, c = problem.jacobian_scaled(x, for_newton=False)
        else:
            a, b, c = problem.jacobian(x, for_newton=False)

        delta = _solve_tridiagonal(a, b, c, -resid)

        # Step control: clamp in *volts*
        max_step_V = float(opts["max_step_volts"])
        if use_scaled:
            delta_phi = Vstar * delta
            np.clip(delta_phi, -max_step_V, +max_step_V, out=delta_phi)
            delta = delta_phi / Vstar
        else:
            np.clip(delta, -max_step_V, +max_step_V, out=delta)

        accepted = False
        local_damp = damping
        while local_damp >= float(opts["min_damping"]):
            x_trial = x + local_damp * delta
            _project_scaled_inplace(x_trial)

            # NEW: refresh carriers for the trial potential before assembling residual
            if hasattr(problem, "update_carriers_fixed_phi"):
                if use_scaled:
                    problem.update_carriers_fixed_phi(Vstar * x_trial, compute_currents=False)
                else:
                    problem.update_carriers_fixed_phi(x_trial, compute_currents=False)

            # Recompute residual with updated carriers (fixed-point sense)
            if use_scaled:
                resid_t = problem.residual_scaled(x_trial, for_newton=False)
            else:
                resid_t = problem.residual(x_trial, for_newton=False)
            res_inf_t = _inf_norm(resid_t)

            if res_inf_t < res_inf:
                x = x_trial
                resid = resid_t
                res_inf = res_inf_t
                accepted = True
                damping = min(1.0, local_damp * 1.25)
                break

            local_damp *= 0.5  # backtrack

        if not accepted:
            if opts["debug"]:
                diag.log_solver_backtrack_fail(
                    solver="Gummel", it=int(it), res_inf=res_inf,
                    min_damping=float(opts["min_damping"]),
                )
            break

        if opts["debug"] and (it % int(opts["print_every"]) == 0):
            max_dphi = float(np.max(np.abs((Vstar * delta))))  # in volts
            phi_phys = Vstar * x
            diag.log_solver_iter(
                solver="Gummel", it=int(it), res_inf=res_inf,
                damping=float(damping), max_dphi=max_dphi,
                phi_min=float(phi_phys.min()), phi_max=float(phi_phys.max()),
            )

        if res_inf < float(opts["tol_res_inf"]):
            converged = True
            break

    if opts["debug"]:
        diag.log_convergence_summary(
            solver="Gummel", converged=bool(converged),
            iters=int(it), res_inf=_inf_norm(resid),
        )

    # Return φ and residual in *physical* units
    phi_out = (Vstar * x) if use_scaled else x
    resid_out = (Rstar * resid) if use_scaled else resid
    return SolveResult(x=phi_out, residual=resid_out, iters=it, converged=converged)
