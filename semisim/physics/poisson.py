# semisim/physics/poisson.py
"""
1-D Poisson driver (MOS/MES/HEMT; variable ε, fixed + dynamic sheets).

This module is *assembly + dispatch only*:
- Prepares geometry/material arrays
- Builds a discrete Poisson problem (no iteration here)
- Dispatches to external solvers (linear / Gummel / Newton)
- Returns final state (phi, carriers, bands, residual, etc.)

Strong form (1D):
    d/dz( ε(z) dφ/dz ) = -ρ_vol(z) - Σ_j σ_j δ(z - z_j)

Finite-volume residual at node i (node-centered CV of size V_i):
    R_i = (F_{i+1/2} - F_{i-1/2})/V_i + ρ_vol,i + σ_i/V_i = 0
with face flux F_{i+1/2} = - ε_{i+1/2} [φ_{i+1} - φ_i] / Δz_{i+1/2}.
"""
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np

# Types from your codebase
from ..geometry.builder import Geometry1D, MaterialFields
from .carriers.bands import BandParams
from ..physics.interfaces import HEMT2DEGParams  # unified in interfaces.py
from ..boundaries.electrical import Contact
from ..physics.traps import BulkTrapSet, InterfaceTrapSet

__all__ = ["PoissonBC", "PoissonSetup", "PoissonResult", "solve_poisson_1d"]


# ----------------------------- Data containers ----------------------------- #
@dataclass(slots=True)
class PoissonBC:
    """Electrostatic boundary conditions."""
    # kinds: "dirichlet" or "neumann"
    left_kind: str = "dirichlet"
    left_value: float = 0.0
    right_kind: str = "dirichlet"
    right_value: float = 0.0


@dataclass(slots=True)
class PoissonSetup:
    """Inputs to assemble and solve the nonlinear Poisson problem."""
    geom: Geometry1D
    mat: MaterialFields
    bands: BandParams
    bc: PoissonBC
    T_K: float
    mu_J: float
    stats: Literal["FD", "MB"] = "FD"
    gvc: float = 1.0
    gvv: float = 1.0

    ND_m3: Optional[np.ndarray] = None
    NA_m3: Optional[np.ndarray] = None
    rho_extra_Cm3: Optional[np.ndarray] = None

    # Fixed/interface sheets (e.g., polarization, fixed interface charge)
    sheet_nodes: Optional[np.ndarray] = None
    sheet_sigma_Cm2: Optional[np.ndarray] = None

    # HEMT 2DEG (dynamic sheet at interfaces; optional)
    hemt2deg: Optional[HEMT2DEGParams] = None

    # Traps (optional): evaluated dynamically in residual
    bulk_traps: Optional[BulkTrapSet] = None
    interface_traps: Optional[InterfaceTrapSet] = None

    # Optional carrier-transport contacts (used by continuity solvers)
    contact_left: Optional[Contact] = None
    contact_right: Optional[Contact] = None

    # Nonlinear controls
    include_carriers: bool = True
    exp_clip: float = 60.0
    phi_guess: Optional[np.ndarray] = None
    carrier_mask: Optional[np.ndarray] = None

    # External solver selection / options
    nonlinear_solver: Optional[Literal["linear", "gummel", "newton"]] = None
    solver_options: Optional[Dict[str, Any]] = None

    # ---- Full continuity (Fix #6) -----------------------------------------
    enable_full_continuity: bool = False
    continuity_options: Optional[Dict[str, Any]] = None

    # ---- Mobility Model (Fix #7) -------------------------------------------
    transport: Optional[Dict[str, Any]] = None

    # ---- Recombination (Fix #8) -------------------------------------------
    # Model + parameters for bulk recombination: SRH(τ), Radiative, Auger.
    recomb: Optional[Dict[str, Any]] = None

    # Debug knobs
    debug: bool = False

@dataclass(slots=True)
class PoissonResult:
    """Result container (float64, C-contiguous arrays)."""
    z: np.ndarray
    phi: np.ndarray
    n: np.ndarray
    p: np.ndarray
    E_C_J: np.ndarray
    E_V_J: np.ndarray
    resid: np.ndarray
    iters: int
    converged: bool

    def as_matrix(self) -> np.ndarray:
        return np.column_stack((self.z, self.phi, self.n, self.p, self.E_C_J, self.E_V_J))


# ----------------------------- Internal helpers ---------------------------- #
def _c64(x: np.ndarray | float) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


def _warn_bad_mu(mu_J: float) -> None:
    """Soft guard to catch incorrect μ units (expects Joules ~1e-19 J)."""
    try:
        mu = float(mu_J)
        if mu != 0.0 and not (1e-22 < abs(mu) < 1e-17):
            print(f"[warn] mu_J looks wrong (J): {mu:.3e} (expected ~1e-19 J)")
    except Exception:
        print("[warn] mu_J not parseable as float — expected Joules ~1e-19 J.")


def _face_eps_harmonic_safe(eps_n: np.ndarray) -> np.ndarray:
    """
    ε_face = 2 εL εR / (εL + εR), robustly:
      - only divide where denom is finite and > 0
      - fall back to max(εL, εR) otherwise
      - floor tiny/non-positive values
    """
    epL = _c64(eps_n[:-1])
    epR = _c64(eps_n[1:])
    denom = epL + epR
    hm = np.empty_like(epL)
    mask = np.isfinite(denom) & (denom > 0.0)
    hm[~mask] = np.maximum(epL[~mask], epR[~mask])
    hm[mask] = (2.0 * epL[mask] * epR[mask]) / denom[mask]
    hm = np.nan_to_num(hm, nan=np.maximum(epL, epR), posinf=np.maximum(epL, epR), neginf=0.0)
    hm = np.where(hm <= 0.0, np.maximum(np.maximum(epL, epR), 1e-30), hm)
    return hm


def _make_dz_Vi_safe(z: np.ndarray, Vi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (dz_face_safe, Vi_safe) with strictly positive, finite values.
    - dz_face_safe = diff(z), but any non-finite or ≤0 replaced by a tiny floor.
    - Vi_safe = original Vi where valid, otherwise rebuilt from z as CV sizes.
    """
    z = _c64(z)
    Vi_in = _c64(Vi)
    N = z.size
    if N < 2:
        raise ValueError("Geometry needs at least 2 nodes.")

    dz_raw = np.diff(z)  # (N-1,)
    mask_dz = np.isfinite(dz_raw) & (dz_raw > 0.0)
    dz_face = np.where(mask_dz, dz_raw, 1e-12)

    Vi_from_z = np.empty(N, dtype=np.float64)
    Vi_from_z[0] = 0.5 * dz_face[0]
    Vi_from_z[-1] = 0.5 * dz_face[-1]
    if N > 2:
        Vi_from_z[1:-1] = 0.5 * (dz_face[:-1] + dz_face[1:])

    Vi_ok = np.isfinite(Vi_in) & (Vi_in > 0.0)
    Vi_safe = np.where(Vi_ok, Vi_in, Vi_from_z)
    Vi_safe = np.where(Vi_safe > 0.0, Vi_safe, 1e-18)
    return dz_face, Vi_safe


# --------------------------------- Driver ---------------------------------- #
def solve_poisson_1d(setup: PoissonSetup) -> PoissonResult:
    """
    1-D Poisson driver (modular):
      - prepares geometry/material arrays
      - builds a discrete Poisson problem (no solving here)
      - dispatches to external solver (linear / Gummel / Newton)
      - returns final state
    """
    _warn_bad_mu(setup.mu_J)

    if setup.debug:
        print(
            "[Poisson] include_carriers="
            f"{setup.include_carriers}, stats={setup.stats}, "
            f"mu_J={float(setup.mu_J):.3e} J, T={setup.T_K} K, "
            f"sheet_mode={'node-aligned' if setup.sheet_nodes is None else 'indexed'}"
        )

    # --- Geometry & material prep ---
    z = _c64(setup.geom.z)
    Vi_in = _c64(setup.geom.Vi)
    eps_n = _c64(setup.mat.eps)

    dz_f, Vi = _make_dz_Vi_safe(z, Vi_in)
    N = z.size
    if N < 3:
        raise ValueError("Geometry must have at least 3 nodes.")
    if dz_f.size != N - 1:
        raise ValueError("geom.dz must have shape (N-1,) with face spacings.")
    eps_f = _c64(_face_eps_harmonic_safe(eps_n))
    alpha_f = _c64(eps_f / dz_f)  # face coeffs

    # Doping & extra charge
    ND = _c64(np.zeros(N) if setup.ND_m3 is None else setup.ND_m3)
    NA = _c64(np.zeros(N) if setup.NA_m3 is None else setup.NA_m3)
    rho_extra = _c64(np.zeros(N) if setup.rho_extra_Cm3 is None else setup.rho_extra_Cm3)
    ND, NA, rho_extra = np.nan_to_num(ND), np.nan_to_num(NA), np.nan_to_num(rho_extra)

    # Sheet charges (node-aligned or indexed)
    sigma_node = _c64(np.zeros(N))
    if setup.sheet_sigma_Cm2 is not None:
        sig = _c64(setup.sheet_sigma_Cm2)
        if setup.sheet_nodes is None:
            if sig.shape == (N,):
                sigma_node += sig
            else:
                raise ValueError("sheet_sigma_Cm2 without sheet_nodes must be length N.")
        else:
            idx = np.asarray(setup.sheet_nodes, dtype=np.int64)
            if idx.size != sig.size:
                raise ValueError("sheet_nodes and sheet_sigma_Cm2 length mismatch.")
            np.add.at(sigma_node, idx, sig)

    # Initial potential (respect Dirichlet at ends if present)
    if setup.phi_guess is not None and setup.phi_guess.shape == z.shape:
        phi0 = _c64(setup.phi_guess.copy())
    else:
        phi0 = _c64(np.zeros_like(z))

    if setup.bc.left_kind.lower() == "dirichlet":
        phi0[0] = float(setup.bc.left_value)
    if setup.bc.right_kind.lower() == "dirichlet":
        phi0[-1] = float(setup.bc.right_value)

    # Optional carrier mask (e.g., only in semiconductors)
    carrier_mask = (
        np.ones(N, dtype=bool)
        if setup.carrier_mask is None
        else np.asarray(setup.carrier_mask, dtype=bool)
    )
    
    # --- Build discrete problem via assembler (no solving here) ---
    # Expected to return a PoissonProblem with:
    #   - x0 (initial phi), residual(phi), jacobian(phi) [optional],
    #   - project_bc(phi) (enforce Dirichlet), state_from_phi(phi)
    from semisim.discretization.assemble import build_poisson_problem

    problem = build_poisson_problem(
        z=z,
        Vi=Vi,
        alpha_f=alpha_f,
        eps_f=eps_f,
        ND=ND,
        NA=NA,
        rho_extra=rho_extra,
        sigma_node=sigma_node,
        bands=setup.bands,
        mat=setup.mat,
        stats=setup.stats,
        mu_J=setup.mu_J,
        T_K=setup.T_K,
        gvc=setup.gvc,
        gvv=setup.gvv,
        hemt2deg=setup.hemt2deg,
        include_carriers=setup.include_carriers,
        exp_clip=setup.exp_clip,
        bc=setup.bc,
        carrier_mask=carrier_mask,
        phi0=phi0,
        debug=setup.debug,
        # NEW
        bulk_traps=setup.bulk_traps,
        interface_traps=setup.interface_traps,
    )

    # Make contacts available to continuity sub-solvers via the problem
    if setup.contact_left is not None:
        setattr(problem, "contact_left", setup.contact_left)
    if setup.contact_right is not None:
        setattr(problem, "contact_right", setup.contact_right)

    # Full continuity control (Fix #6)
    if bool(getattr(setup, "enable_full_continuity", False)):
        setattr(problem, "enable_full_continuity", True)
    if getattr(setup, "continuity_options", None) is not None:
        setattr(problem, "continuity_options", dict(setup.continuity_options))

    # ---- Fix #7: transport config (Matthiessen mobility) ----------------
    if hasattr(setup, "transport") and getattr(setup, "transport") is not None:
        setattr(problem, "transport_config", dict(setup.transport))

    # ---- Fix #8: recombination model ------------------------------------
    if hasattr(setup, "recomb") and getattr(setup, "recomb") is not None:
        setattr(problem, "recomb_config", dict(setup.recomb))

    # --- Dispatch to external solver (linear / Gummel / Newton) ---
    solver_name = (setup.nonlinear_solver or "").lower()
    solver_options = (setup.solver_options or {}).copy()

    if solver_name == "":
        solver_name = "gummel" if setup.include_carriers else "linear"

    if solver_name == "linear":
        # Electrostatic one-shot with frozen charges
        from semisim.solver.subsolvers.poisson_linear import solve as poisson_linear_solve
        sol_lin = poisson_linear_solve(
            phi0=problem.x0,
            alpha_f=alpha_f,
            Vi=Vi,
            ND=ND,
            NA=NA,
            rho_extra=rho_extra,
            sigma_node=sigma_node,
            n_fixed=None,
            p_fixed=None,
            sigma2d_fixed=None,
            bc=setup.bc,
            **{k: v for k, v in solver_options.items() if k in {
                "damping_init", "max_step_volts", "min_damping", "max_iters", "print_every", "debug"
            }},
        )
        phi = _c64(sol_lin.phi)
        resid = _c64(sol_lin.residual)
        iters = int(sol_lin.iters)
        converged = bool(sol_lin.accepted)
    elif solver_name == "gummel":
        from semisim.solver.gummel import solve as gummel_solve
        sol = gummel_solve(problem, options=solver_options)
        phi = _c64(sol.x)
        resid = _c64(sol.residual)
        iters = int(sol.iters)
        converged = bool(sol.converged)
    elif solver_name == "newton":
        from semisim.solver.newton import solve as newton_solve
        sol = newton_solve(problem, options=solver_options)
        phi = _c64(sol.x)
        resid = _c64(sol.residual)
        iters = int(sol.iters)
        converged = bool(sol.converged)
    else:
        raise ValueError(f"Unknown nonlinear_solver='{solver_name}'")

    # --- Final state from converged potential ---
    state = problem.state_from_phi(phi)
    E_C = _c64(state.E_C_J)
    E_V = _c64(state.E_V_J)
    n = _c64(state.n)
    p = _c64(state.p)

    return PoissonResult(
        z=z,
        phi=phi,
        n=n,
        p=p,
        E_C_J=E_C,
        E_V_J=E_V,
        resid=resid,
        iters=iters,
        converged=converged,
    )
