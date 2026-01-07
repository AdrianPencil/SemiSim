"""
semisim/solver/homotopy.py

Simple homotopy controller for turning on 'hard' physics gradually (e.g. carriers,
FD stats, 2DEG). It manages a scale parameter s ∈ (0, 1], adapts it based on the
observed residual drop, and provides a clean API for your workflow.

Typical loop:
    h = HomotopyController(start=1e-4, stop=1.0)
    while h.active:
        s = h.current
        sol = run_step_with_scale(s)  # your callback elsewhere
        h.update(res0=sol.res0, res1=sol.res1, converged=sol.converged)

When residual improves sufficiently, s is increased; otherwise reduced.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Callable, Sequence
import numpy as np

@dataclass
class HomotopyController:
    start: float = 1e-4
    stop: float = 1.0
    grow: float = 2.5           # multiplicative growth when successful
    shrink: float = 0.5         # multiplicative shrink when failing
    min_step: float = 1e-6      # do not go below this scale
    success_factor: float = 5.0 # require residual drop by this factor
    max_inner_fails: int = 2    # how many consecutive fails before shrinking
    verbose: bool = True

    # internal
    current: float = None
    _fails: int = 0
    active: bool = True

    def __post_init__(self) -> None:
        if self.current is None:
            self.current = float(self.start)
        self.active = True
        self._fails = 0

    def update(self, *, res0: float, res1: float, converged: bool) -> None:
        """
        Update controller with result of a solve at the current scale.
        res0 : initial residual at this scale
        res1 : final residual at this scale
        converged : whether the sub-solve reported convergence
        """
        improved = (res1 < max(res0 / self.success_factor, 1e-300)) and converged

        if improved:
            # Success: increase s
            self._fails = 0
            next_s = min(self.current * self.grow, self.stop)
            if self.verbose:
                print(f"[homotopy] success at s={self.current:g} → next {next_s:g} "
                      f"(||res|| {res0:.3e} → {res1:.3e})")
            self.current = next_s
            if self.current >= self.stop:
                self.active = False
        else:
            # Failure: try again (a couple of times) then shrink
            self._fails += 1
            if self._fails > self.max_inner_fails:
                next_s = max(self.current * self.shrink, self.min_step)
                if self.verbose:
                    print(f"[homotopy] failure at s={self.current:g} → shrink to {next_s:g} "
                          f"(||res|| {res0:.3e} → {res1:.3e})")
                self.current = next_s
                self._fails = 0
            else:
                if self.verbose:
                    print(f"[homotopy] retry at same s={self.current:g} "
                          f"(attempt {self._fails}/{self.max_inner_fails})")

    def reset(self, *, s: Optional[float] = None) -> None:
        """Reset controller to start or custom s."""
        self.current = float(self.start if s is None else s)
        self._fails = 0
        self.active = True
        if self.verbose:
            print(f"[homotopy] reset to s={self.current:g}")

# -----------------------------------------------------------------------------
# Adaptive, physics-aware homotopy runner for Poisson
# -----------------------------------------------------------------------------
def run_adaptive_scales_poisson(
    *,
    base_setup,
    phi_init: np.ndarray,
    s0: float = 1e-4,
    s_stop: float = 1.0,
    # --- policy thresholds (can be surfaced via config later) ---
    carriers_on_min_s: float = 3e-4,
    debye_cells_threshold: float = 5.0,
    charge_ratio_threshold: float = 1e-3,
    fd_eta_threshold: float = 2.0,
    hemt_s_min: float = 0.30,
    n2d_min_cm2: float = 1e10,
    band_align_offset_J: float = 0.0,
    # --- solver selection ---
    solver_mode: str = "hybrid",           # "gummel" | "newton" | "hybrid"
    newton_switch_scaled_tol: float = 1e-2,  # switch to Newton when ||r||_inf/R* <= tol
    # --- step controller ---
    grow_factor: float = 1.5,
    shrink_factor: float = 0.5,
    accept_ratio: float = 0.80,
    reject_ratio: float = 0.95,
    max_retries: int = 2,
    # --- plumbing ---
    solve_fn: Optional[Callable] = None,
    debug: bool = True,
):
    """
    Adaptive charge-scale loop with physics triggers:
      - carriers on: Debye/ratio triggers (or s >= carriers_on_min_s),
      - MB→FD: degeneracy test via η = (μ-E_C)/kT or (E_V-μ)/kT,
      - 2DEG on: band alignment or ns_2D threshold (or s >= hemt_s_min).

    Returns: list of per-step Poisson results; last one is the final.
    """
    # Lazy imports to avoid cycles
    if solve_fn is None:
        from semisim.physics.poisson import solve_poisson_1d as _solve
    else:
        _solve = solve_fn
    from semisim.physics.carriers.bands import band_edges_from_potential
    from semisim.physics.carriers.statistics import carriers_3d, K_B, Q
    from semisim.physics.interfaces import hemt_2deg_sigma_and_jacobian as _hemt_sigma_jac
    try:
        from semisim.utils import diagnostics as diag
    except Exception:
        class _Bare:
            @staticmethod
            def log_homotopy_step(**kw):  # fallback
                if debug:
                    print("[homotopy]", kw)
        diag = _Bare()

    # Helpers
    def _norm_inf(x: np.ndarray) -> float:
        return float(np.linalg.norm(x, ord=np.inf))

    def _estimate_carriers(phi, stats_mode):
        Ec, Ev = band_edges_from_potential(phi, base_setup.bands)
        n, p = carriers_3d(Ec, Ev, base_setup.mu_J, base_setup.T_K,
                           me_rel=base_setup.mat.me_dos_rel,
                           mh_rel=base_setup.mat.mh_dos_rel,
                           stats=stats_mode)
        return n, p, Ec, Ev

    def _lambda_debye(n_guess):
        # λ_D = sqrt(ε kT / (q^2 n)), use eps from material fields
        eps = np.asarray(base_setup.mat.eps, dtype=np.float64)
        kT = K_B * float(base_setup.T_K)
        n_safe = np.maximum(n_guess, 1.0)  # avoid zero
        return np.sqrt(eps * kT / (Q * Q * n_safe))

    def _dz_min():
        z = base_setup.geom.z
        return float(np.min(np.diff(z)))

    def _free_vs_fixed_ratio(n_guess, p_guess):
        rho_free = Q * (p_guess - n_guess)
        rho_fixed = np.asarray(base_setup.rho_extra_Cm3, dtype=np.float64) if base_setup.rho_extra_Cm3 is not None else np.zeros_like(rho_free)
        denom = np.max(np.abs(rho_fixed)) if rho_fixed.size else 0.0
        if denom <= 0.0:
            return np.inf  # no fixed charge → treat as large ratio
        return float(np.max(np.abs(rho_free)) / denom)

    def _should_enable_2deg(Ec):
        if base_setup.hemt2deg is None:
            return False
        # Band-alignment trigger
        nodes = np.asarray(base_setup.hemt2deg.nodes, dtype=int)
        if nodes.size == 0:
            return False
        # For the first node, test minimal subband
        Esub_min = np.min(Ec[nodes] + np.atleast_1d(base_setup.hemt2deg.Erel_J))
        if (Esub_min - float(base_setup.mu_J)) <= float(band_align_offset_J):
            return True
        # ns_2D trigger (cheap: compute sigma and divide)
        sigma2d, _ = _hemt_sigma_jac(Ec, params=base_setup.hemt2deg,
                                     mu_J=float(base_setup.mu_J),
                                     T_K=float(base_setup.T_K))
        ns_node = -sigma2d / Q
        return (np.max(ns_node) >= float(n2d_min_cm2) * 1e4)  # [cm^-2] → [m^-2]

    # Initialize
    s = float(s0)
    phi_curr = np.ascontiguousarray(phi_init, dtype=np.float64)
    results = []
    prev_res_norm = None
    retries = 0

    while True:
        # Physics flags by triggers
        # Start in MB for diagnostics; we may upgrade to FD by degeneracy
        n_guess, p_guess, Ec, Ev = _estimate_carriers(phi_curr, stats_mode="MB")
        lam = _lambda_debye(n_guess)
        lam_min = float(np.min(lam))
        enable_carriers = (s >= carriers_on_min_s) or \
                          (lam_min <= debye_cells_threshold * _dz_min()) or \
                          (_free_vs_fixed_ratio(n_guess, p_guess) >= charge_ratio_threshold)

        # FD when degenerate anywhere
        kT = K_B * float(base_setup.T_K)
        eta_n = (float(base_setup.mu_J) - Ec) / kT
        eta_p = (Ev - float(base_setup.mu_J)) / kT
        max_eta = float(np.max([np.max(eta_n), np.max(eta_p)]))
        stats_mode = "FD" if max_eta >= fd_eta_threshold else "MB"

        # 2DEG by band alignment / ns trigger / s fallback
        enable_2deg = (s >= hemt_s_min) or _should_enable_2deg(Ec)

        # Build the per-step setup (we will select the solver below)
        setup_s = replace(
            base_setup,
            rho_extra_Cm3=(None if base_setup.rho_extra_Cm3 is None else s * base_setup.rho_extra_Cm3),
            sheet_sigma_Cm2=(None if base_setup.sheet_sigma_Cm2 is None else s * base_setup.sheet_sigma_Cm2),
            include_carriers=enable_carriers,
            stats=stats_mode,
            hemt2deg=(base_setup.hemt2deg if enable_2deg else None),
            phi_guess=phi_curr,
        )

        # --- choose solver per policy -----------------------------------
        # Estimate an R_scale (see Fix #14) to work with *scaled* threshold
        # R* ≈ q * max(|ND|,|NA|,|rho_extra|/q, 1e20). If no data, keep 1.0.
        Qe = Q
        dens_from_rho = np.max(np.abs(setup_s.rho_extra_Cm3)) / Qe if setup_s.rho_extra_Cm3 is not None else 0.0
        dens_candidates = [
            float(np.max(np.abs(setup_s.ND_m3))) if setup_s.ND_m3 is not None and setup_s.ND_m3.size else 0.0,
            float(np.max(np.abs(setup_s.NA_m3))) if setup_s.NA_m3 is not None and setup_s.NA_m3.size else 0.0,
            float(dens_from_rho),
            1e20,
        ]
        N_ref_est = max(dens_candidates)
        R_scale_est = Qe * N_ref_est

        # default: Gummel early, Newton later
        chosen_solver = "gummel"
        if solver_mode.lower() == "gummel":
            chosen_solver = "gummel"
        elif solver_mode.lower() == "newton":
            chosen_solver = "newton"
        else:
            # hybrid: if previous *scaled* residual small enough, prefer Newton
            if prev_res_norm is not None and R_scale_est > 0.0:
                prev_scaled = prev_res_norm / R_scale_est
                if prev_scaled <= newton_switch_scaled_tol:
                    chosen_solver = "newton"

        # try chosen solver; on failure and hybrid mode, fallback to Gummel once
        setup_s = replace(setup_s, nonlinear_solver=chosen_solver)
        res = _solve(setup_s)
        res_norm = _norm_inf(res.resid)

        if solver_mode.lower() == "hybrid" and (not res.converged) and (chosen_solver == "newton"):
            # graceful fallback in-place
            setup_s = replace(setup_s, nonlinear_solver="gummel", phi_guess=res.phi)
            res = _solve(setup_s)
            res_norm = _norm_inf(res.resid)
            chosen_solver = "gummel"

        # Acceptance logic
        accepted = True
        if prev_res_norm is not None:
            drop = res_norm / (prev_res_norm + 1e-300)
            if (not res.converged) or (drop > reject_ratio):
                accepted = False

        # Log step
        diag.log_homotopy_step(
            s=s, accepted=bool(accepted),
            converged=bool(res.converged),
            res_norm=res_norm, prev_res_norm=(prev_res_norm if prev_res_norm is not None else np.nan),
            enable_carriers=bool(enable_carriers),
            stats_mode=stats_mode,
            enable_2deg=bool(enable_2deg),
            lam_min=lam_min, dz_min=_dz_min(), max_eta=max_eta,
            solver=chosen_solver,
        )

        if accepted:
            results.append(res)
            phi_curr = res.phi
            prev_res_norm = res_norm
            retries = 0
            if s >= s_stop:
                break
            s = min(s * grow_factor, s_stop)
        else:
            retries += 1
            if retries > max_retries:
                # Shrink and reset retries
                s = max(s * shrink_factor, 1e-8)
                retries = 0
            # No results append; try again at new/old s

    return results
