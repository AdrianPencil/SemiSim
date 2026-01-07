# semisim/physics/continuity.py
"""
1D drift–diffusion continuity (steady DC) with Scharfetter–Gummel fluxes.

This module advances carriers (n, p) on a fixed electrostatic potential φ(z).
Gummel-style decoupled Newton: solve electrons with p frozen, then holes with
n frozen, iterating until convergence.

New in this version
-------------------
- Optional **carrier_mask** to disable carriers in non-semis (metals/oxides).
- Optional **scaling hooks** via semisim.utils.scaling (safe fallback).
- Clearer, guarded line-search and positivity floors.
- Small debug prints gated by ContinuityOptions.debug.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from ..boundaries.electrical import Contact, eval_contact_flux
# Use shared SG numerics (Bernoulli + node→edge averaging) from discretization core
from ..discretization.fluxes import bern as _sg_bern, faces_from_nodes

# ---- constants (SI) ----
Q = 1.602176634e-19   # C
K_B = 1.380649e-23    # J/K


__all__ = [
    "ContinuityInputs",
    "ContinuityOptions",
    "ContinuityResult",
    "assemble_residual_jacobian_n",
    "assemble_residual_jacobian_p",
    "solve_continuity_1d",
]


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------


@dataclass(slots=True)
class ContinuityInputs:
    """
    Fixed inputs for continuity solve on a 1D mesh.
    """
    z_m: np.ndarray                 # (N,) node coordinates [m]
    phi_V: np.ndarray               # (N,) electrostatic potential [V]
    T_K: float                      # temperature [K]

    # Carrier state (initial guess / updated in-place by solver)
    n_m3: np.ndarray                # (N,) electrons [1/m^3]
    p_m3: np.ndarray                # (N,) holes [1/m^3]

    # Transport coefficients at nodes
    mu_n_m2Vs: np.ndarray           # (N,) electron mobility [m^2/V/s]
    mu_p_m2Vs: np.ndarray           # (N,) hole mobility [m^2/V/s]
    # Optional diffusion coefficients at nodes; if None, use MB Einstein D=μ V_T
    Dn_m2s: Optional[np.ndarray] = None
    Dp_m2s: Optional[np.ndarray] = None

    # Band edges (for boundary equilibria & recomb models, if needed by callback)
    E_C_J: Optional[np.ndarray] = None
    E_V_J: Optional[np.ndarray] = None

    # Effective masses for boundary models (if used by contacts / recomb)
    me_rel: Optional[float] = None
    mh_rel: Optional[float] = None
    gvc: float = 1.0
    gvv: float = 1.0

    # Boundary contacts (left, right)
    contact_left: Optional[Contact] = None
    contact_right: Optional[Contact] = None

    # Optional: restrict carriers to semiconductors only (False → carriers off)
    carrier_mask: Optional[np.ndarray] = None


@dataclass(slots=True)
class ContinuityOptions:
    """
    Solver and discretization options.
    """
    max_iters: int = 50
    tol_abs: float = 1e-6               # absolute residual norm (A/m^2 on flux balance)
    tol_rel: float = 1e-6               # relative update norm
    damping_init: float = 1.0           # line-search initial damping in (0,1]
    damping_min: float = 1e-4
    use_arithmetic_edge_avg: bool = True
    enforce_positivity_floor: float = 1e0   # small floor for n,p to avoid zero (1/m^3)
    debug: bool = False
    debug_every: int = 1


@dataclass(slots=True)
class ContinuityResult:
    n_m3: np.ndarray
    p_m3: np.ndarray
    iters: int
    converged: bool
    res_norm_n: float
    res_norm_p: float


# ---------------------------------------------------------------------
# Numerics helpers
# ---------------------------------------------------------------------


def _c64(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def _nz(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return np.where(x == 0.0, eps, x)


def _bernoulli(x: np.ndarray) -> np.ndarray:
    """
    Delegate to the shared SG Bernoulli from discretization/fluxes.py
    so all modules use the exact same stable evaluation.
    """
    return _sg_bern(_c64(x))

def _cell_widths(z: np.ndarray) -> np.ndarray:
    """
    Control-volume widths centered at nodes:
        Δz_i = 0.5*(z_{i+1} - z_{i-1}); endpoints use one-sided.
    """
    dz = np.empty_like(z)
    dz[0] = z[1] - z[0]
    dz[-1] = z[-1] - z[-2]
    dz[1:-1] = 0.5 * (z[2:] - z[:-2])
    return dz


def _solve_tridiag(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Thomas algorithm for a tridiagonal system:
        a[i] x[i-1] + b[i] x[i] + c[i] x[i+1] = d[i]
    with a[0]=0, c[-1]=0.
    """
    n = b.size
    ac = np.zeros_like(a)
    bc = np.zeros_like(b)
    cc = np.zeros_like(c)
    dc = np.zeros_like(d)

    bc[0] = b[0]
    cc[0] = c[0]
    dc[0] = d[0]

    for i in range(1, n):
        m = ac[i] = a[i] / _nz(bc[i - 1])
        bc[i] = b[i] - m * cc[i - 1]
        dc[i] = d[i] - m * dc[i - 1]

    x = np.zeros_like(d)
    x[-1] = dc[-1] / _nz(bc[-1])
    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / _nz(bc[i])
    return x


# ---------------------------------------------------------------------
# Flux & residual assembly (electrons / holes)
# ---------------------------------------------------------------------


def _edge_flux_electron(
    n: np.ndarray, phi: np.ndarray, D_edge: np.ndarray, dz_edge: np.ndarray, V_T: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Electron SG edge flux and partial derivatives wrt (n_i, n_{i+1}).
    Returns: Jn_e (N-1,), dJn_dn_i (N-1,), dJn_dn_ip1 (N-1,)
    """
    psi = (phi[1:] - phi[:-1]) / V_T
    Bp = _bernoulli(psi)
    Bm = _bernoulli(-psi)  # equals exp(psi)*Bp but computed stably
    alpha = Q * D_edge / _nz(dz_edge)
    J = alpha * (n[1:] * Bp - n[:-1] * Bm)
    d_i = -alpha * Bm
    d_ip1 = alpha * Bp
    return J, d_i, d_ip1


def _edge_flux_hole(
    p: np.ndarray, phi: np.ndarray, D_edge: np.ndarray, dz_edge: np.ndarray, V_T: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hole SG edge flux and partial derivatives wrt (p_i, p_{i+1}).
    Returns: Jp_e (N-1,), dJp_dp_i (N-1,), dJp_dp_ip1 (N-1,)
    """
    psi = (phi[1:] - phi[:-1]) / V_T
    Bp = _bernoulli(psi)
    Bm = _bernoulli(-psi)
    alpha = Q * D_edge / _nz(dz_edge)
    J = alpha * (p[:-1] * Bp - p[1:] * Bm)
    d_i = alpha * Bp
    d_ip1 = -alpha * Bm
    return J, d_i, d_ip1


def assemble_residual_jacobian_n(
    z: np.ndarray,
    phi: np.ndarray,
    T_K: float,
    n: np.ndarray,
    p_frozen: np.ndarray,
    mu_n: np.ndarray,
    Dn_opt: Optional[np.ndarray],
    U: np.ndarray,
    dU_dn: np.ndarray,
    contact_left: Optional[Contact],
    contact_right: Optional[Contact],
    *,
    E_C_J: Optional[np.ndarray] = None,
    E_V_J: Optional[np.ndarray] = None,
    me_rel: Optional[float] = None,
    mh_rel: Optional[float] = None,
    gvc: float = 1.0,
    gvv: float = 1.0,
    carrier_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble electron residual Rn and Jacobian tridiagonal bands (a,b,c):

        Rn[i] = J_{i+1/2} - J_{i-1/2} - q U_i Δz_i  = 0
    """
    z = _c64(z)
    phi = _c64(phi)
    n = _c64(n)
    p_frozen = _c64(p_frozen)
    U = _c64(U)
    dU_dn = _c64(dU_dn)
    V_T = K_B * float(T_K) / Q

    if carrier_mask is not None:
        mask = np.asarray(carrier_mask, dtype=bool)
        n = n.copy()
        p_frozen = p_frozen.copy()
        n[~mask] = 0.0
        p_frozen[~mask] = 0.0

    dz_edge = z[1:] - z[:-1]
    dz_cell = _cell_widths(z)

    # Diffusion coefficients on nodes (MB Einstein by default), then edges
    if Dn_opt is None:
        Dn = _c64(mu_n) * V_T
    else:
        Dn = _c64(Dn_opt)
    D_e = faces_from_nodes(Dn)

    # Edge fluxes & partials
    J_e, dJ_di, dJ_dip1 = _edge_flux_electron(n, phi, D_e, dz_edge, V_T)

    # Residual init: interior nodes
    N = z.size
    R = np.zeros(N, dtype=np.float64)
    a = np.zeros(N, dtype=np.float64)  # lower diag (i-1)
    b = np.zeros(N, dtype=np.float64)  # diag (i)
    c = np.zeros(N, dtype=np.float64)  # upper diag (i+1)

    # Interior contributions
    R[1:-1] += J_e[1:] - J_e[:-1]
    b[1:-1] = (-dJ_di[1:]) - (dJ_dip1[:-1])
    a[1:-1] = dJ_di[:-1]
    c[1:-1] = dJ_dip1[1:]

    # Recombination sink term: - q U Δz  (U depends on n: add -q Δz dU_dn on diag)
    R -= Q * U * dz_cell
    b -= Q * dU_dn * dz_cell

    # Boundary faces: left i=0
    if contact_left is not None:
        Jn_out, _, dJn_dn, _, _, _ = eval_contact_flux(
            kind=contact_left.kind,
            E_C_J=float(E_C_J[0]) if E_C_J is not None else 0.0,
            E_V_J=float(E_V_J[0]) if E_V_J is not None else 0.0,
            n_m3=float(n[0]),
            p_m3=float(p_frozen[0]),
            T_K=float(T_K),
            me_rel=float(me_rel) if me_rel is not None else 1.0,
            mh_rel=float(mh_rel) if mh_rel is not None else 1.0,
            gvc=float(gvc),
            gvv=float(gvv),
            ohmic=contact_left.model if contact_left.kind == "ohmic" else None,
            schottky=contact_left.model if contact_left.kind == "schottky" else None,
        )
        J_face = -Jn_out
        R[0] += J_e[0] - J_face
        b[0] += -dJ_di[0] + dJn_dn  # minus right-edge term, plus contact derivative
        c[0] += dJ_dip1[0]
    else:
        R[0] += J_e[0]
        b[0] += -dJ_di[0]
        c[0] += dJ_dip1[0]

    # Right boundary i=N-1
    if contact_right is not None:
        Jn_out, _, dJn_dn, _, _, _ = eval_contact_flux(
            kind=contact_right.kind,
            E_C_J=float(E_C_J[-1]) if E_C_J is not None else 0.0,
            E_V_J=float(E_V_J[-1]) if E_V_J is not None else 0.0,
            n_m3=float(n[-1]),
            p_m3=float(p_frozen[-1]),
            T_K=float(T_K),
            me_rel=float(me_rel) if me_rel is not None else 1.0,
            mh_rel=float(mh_rel) if mh_rel is not None else 1.0,
            gvc=float(gvc),
            gvv=float(gvv),
            ohmic=contact_right.model if contact_right.kind == "ohmic" else None,
            schottky=contact_right.model if contact_right.kind == "schottky" else None,
        )
        J_face = +Jn_out
        R[-1] += J_face - J_e[-1]
        b[-1] += dJn_dn - dJ_dip1[-1]
        a[-1] += dJ_di[-1]
    else:
        R[-1] += -J_e[-1]
        b[-1] += -dJ_dip1[-1]
        a[-1] += dJ_di[-1]

    # Masking: hard-pin masked nodes by making J large diagonal to keep n≈0
    if carrier_mask is not None:
        mask = np.asarray(carrier_mask, dtype=bool)
        pin = ~mask
        if np.any(pin):
            big = 1e12  # A/m^2-level stiffness to force δn≈0 there
            b[pin] += big
            # also remove coupling through edges bordering pinned nodes
            a[pin] = 0.0
            c[pin] = 0.0
            # pull residual toward zero there
            R[pin] += big * (0.0 - n[pin])

    return R, a, b, c


def assemble_residual_jacobian_p(
    z: np.ndarray,
    phi: np.ndarray,
    T_K: float,
    p: np.ndarray,
    n_frozen: np.ndarray,
    mu_p: np.ndarray,
    Dp_opt: Optional[np.ndarray],
    U: np.ndarray,
    dU_dp: np.ndarray,
    contact_left: Optional[Contact],
    contact_right: Optional[Contact],
    *,
    E_C_J: Optional[np.ndarray] = None,
    E_V_J: Optional[np.ndarray] = None,
    me_rel: Optional[float] = None,
    mh_rel: Optional[float] = None,
    gvc: float = 1.0,
    gvv: float = 1.0,
    carrier_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble hole residual Rp and Jacobian tridiagonal bands (a,b,c):

        Rp[i] = Jp_{i+1/2} - Jp_{i-1/2} + q U_i Δz_i  = 0
    """
    z = _c64(z)
    phi = _c64(phi)
    p = _c64(p)
    n_frozen = _c64(n_frozen)
    U = _c64(U)
    dU_dp = _c64(dU_dp)
    V_T = K_B * float(T_K) / Q

    if carrier_mask is not None:
        mask = np.asarray(carrier_mask, dtype=bool)
        p = p.copy()
        n_frozen = n_frozen.copy()
        p[~mask] = 0.0
        n_frozen[~mask] = 0.0

    dz_edge = z[1:] - z[:-1]
    dz_cell = _cell_widths(z)

    if Dp_opt is None:
        Dp = _c64(mu_p) * V_T
    else:
        Dp = _c64(Dp_opt)
    D_e = faces_from_nodes(Dp)

    J_e, dJ_di, dJ_dip1 = _edge_flux_hole(p, phi, D_e, dz_edge, V_T)

    N = z.size
    R = np.zeros(N, dtype=np.float64)
    a = np.zeros(N, dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)
    c = np.zeros(N, dtype=np.float64)

    R[1:-1] += J_e[1:] - J_e[:-1]
    b[1:-1] = (-dJ_di[1:]) - (dJ_dip1[:-1])
    a[1:-1] = dJ_di[:-1]
    c[1:-1] = dJ_dip1[1:]

    # Note the sign: + q U Δz
    R += Q * U * dz_cell
    b += Q * dU_dp * dz_cell

    # Left boundary
    if contact_left is not None:
        _, Jp_out, _, _, _, dJp_dp = eval_contact_flux(
            kind=contact_left.kind,
            E_C_J=float(E_C_J[0]) if E_C_J is not None else 0.0,
            E_V_J=float(E_V_J[0]) if E_V_J is not None else 0.0,
            n_m3=float(n_frozen[0]),
            p_m3=float(p[0]),
            T_K=float(T_K),
            me_rel=float(me_rel) if me_rel is not None else 1.0,
            mh_rel=float(mh_rel) if mh_rel is not None else 1.0,
            gvc=float(gvc),
            gvv=float(gvv),
            ohmic=contact_left.model if contact_left.kind == "ohmic" else None,
            schottky=contact_left.model if contact_left.kind == "schottky" else None,
        )
        J_face = -Jp_out
        R[0] += J_e[0] - J_face
        b[0] += -dJ_di[0] + dJp_dp
        c[0] += dJ_dip1[0]
    else:
        R[0] += J_e[0]
        b[0] += -dJ_di[0]
        c[0] += dJ_dip1[0]

    # Right boundary
    if contact_right is not None:
        _, Jp_out, _, _, _, dJp_dp = eval_contact_flux(
            kind=contact_right.kind,
            E_C_J=float(E_C_J[-1]) if E_C_J is not None else 0.0,
            E_V_J=float(E_V_J[-1]) if E_V_J is not None else 0.0,
            n_m3=float(n_frozen[-1]),
            p_m3=float(p[-1]),
            T_K=float(T_K),
            me_rel=float(me_rel) if me_rel is not None else 1.0,
            mh_rel=float(mh_rel) if mh_rel is not None else 1.0,
            gvc=float(gvc),
            gvv=float(gvv),
            ohmic=contact_right.model if contact_right.kind == "ohmic" else None,
            schottky=contact_right.model if contact_right.kind == "schottky" else None,
        )
        J_face = +Jp_out
        R[-1] += J_face - J_e[-1]
        b[-1] += dJp_dp - dJ_dip1[-1]
        a[-1] += dJ_di[-1]
    else:
        R[-1] += -J_e[-1]
        b[-1] += -dJ_dip1[-1]
        a[-1] += dJ_di[-1]

    # Masking (pin p≈0 where carriers disabled)
    if carrier_mask is not None:
        mask = np.asarray(carrier_mask, dtype=bool)
        pin = ~mask
        if np.any(pin):
            big = 1e12
            b[pin] += big
            a[pin] = 0.0
            c[pin] = 0.0
            R[pin] += big * (0.0 - p[pin])

    return R, a, b, c


# ---------------------------------------------------------------------
# Solver (Gummel decoupled Newton with damping)
# ---------------------------------------------------------------------


RecombEval = Callable[
    [np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]
# Signature: U(n,p), dU/dn(n,p), dU/dp(n,p)  -> arrays (N,)


def _apply_floor(x: np.ndarray, floor: float) -> np.ndarray:
    return np.maximum(x, floor)


def solve_continuity_1d(
    inp: ContinuityInputs,
    recomb_eval: RecombEval,
    opt: ContinuityOptions = ContinuityOptions(),
) -> ContinuityResult:
    """
    Solve steady 1D continuity for (n,p) with SG fluxes and given φ(z).
    """
    z = _c64(inp.z_m)
    phi = _c64(inp.phi_V)
    T = float(inp.T_K)
    n = _c64(inp.n_m3).copy()
    p = _c64(inp.p_m3).copy()
    mu_n = _c64(inp.mu_n_m2Vs)
    mu_p = _c64(inp.mu_p_m2Vs)
    Dn = None if inp.Dn_m2s is None else _c64(inp.Dn_m2s)
    Dp = None if inp.Dp_m2s is None else _c64(inp.Dp_m2s)
    mask = None if inp.carrier_mask is None else np.asarray(inp.carrier_mask, dtype=bool)

    if opt.debug:
        print(f"[Continuity] start | T={T:.1f} K | V_T={K_B*T/Q:.3e} V | "
              f"n0~{float(np.max(n)):.3e} m^-3 p0~{float(np.max(p)):.3e} m^-3")

    # Iterate Gummel
    converged = False
    res_norm_n = np.inf
    res_norm_p = np.inf
    for it in range(1, opt.max_iters + 1):
        # Clamp to positivity (avoid underflows in SG)
        n = _apply_floor(n, opt.enforce_positivity_floor)
        p = _apply_floor(p, opt.enforce_positivity_floor)
        if mask is not None:
            n[~mask] = 0.0
            p[~mask] = 0.0

        # Recombination at current state
        U, dU_dn, dU_dp = recomb_eval(n, p)

        # --- Electron step (p frozen)
        Rn, an, bn, cn = assemble_residual_jacobian_n(
            z, phi, T, n, p, mu_n, Dn, U, dU_dn,
            inp.contact_left, inp.contact_right,
            E_C_J=inp.E_C_J, E_V_J=inp.E_V_J,
            me_rel=inp.me_rel, mh_rel=inp.mh_rel,
            gvc=inp.gvc, gvv=inp.gvv,
            carrier_mask=mask,
        )
        dn = _solve_tridiag(an, bn, cn, -Rn)

        # Line search / damping
        damp = opt.damping_init
        base_norm = float(np.linalg.norm(Rn, ord=np.inf))
        n_trial = _apply_floor(n + damp * dn, opt.enforce_positivity_floor)
        if mask is not None:
            n_trial[~mask] = 0.0

        Rn_trial, _, _, _ = assemble_residual_jacobian_n(
            z, phi, T, n_trial, p, mu_n, Dn, U, dU_dn,
            inp.contact_left, inp.contact_right,
            E_C_J=inp.E_C_J, E_V_J=inp.E_V_J,
            me_rel=inp.me_rel, mh_rel=inp.mh_rel,
            gvc=inp.gvc, gvv=inp.gvv,
            carrier_mask=mask,
        )
        while float(np.linalg.norm(Rn_trial, ord=np.inf)) > base_norm and damp > opt.damping_min:
            damp *= 0.5
            n_trial = _apply_floor(n + damp * dn, opt.enforce_positivity_floor)
            if mask is not None:
                n_trial[~mask] = 0.0
            Rn_trial, _, _, _ = assemble_residual_jacobian_n(
                z, phi, T, n_trial, p, mu_n, Dn, U, dU_dn,
                inp.contact_left, inp.contact_right,
                E_C_J=inp.E_C_J, E_V_J=inp.E_V_J,
                me_rel=inp.me_rel, mh_rel=inp.mh_rel,
                gvc=inp.gvc, gvv=inp.gvv,
                carrier_mask=mask,
            )

        n = n_trial
        res_norm_n = float(np.linalg.norm(Rn_trial, ord=np.inf))

        # Update recombination (n changed)
        U, dU_dn, dU_dp = recomb_eval(n, p)

        # --- Hole step (n frozen)
        Rp, ap, bp, cp = assemble_residual_jacobian_p(
            z, phi, T, p, n, mu_p, Dp, U, dU_dp,
            inp.contact_left, inp.contact_right,
            E_C_J=inp.E_C_J, E_V_J=inp.E_V_J,
            me_rel=inp.me_rel, mh_rel=inp.mh_rel,
            gvc=inp.gvc, gvv=inp.gvv,
            carrier_mask=mask,
        )
        dp = _solve_tridiag(ap, bp, cp, -Rp)

        damp = opt.damping_init
        base_norm = float(np.linalg.norm(Rp, ord=np.inf))
        p_trial = _apply_floor(p + damp * dp, opt.enforce_positivity_floor)
        if mask is not None:
            p_trial[~mask] = 0.0

        Rp_trial, _, _, _ = assemble_residual_jacobian_p(
            z, phi, T, p_trial, n, mu_p, Dp, U, dU_dp,
            inp.contact_left, inp.contact_right,
            E_C_J=inp.E_C_J, E_V_J=inp.E_V_J,
            me_rel=inp.me_rel, mh_rel=inp.mh_rel,
            gvc=inp.gvc, gvv=inp.gvv,
            carrier_mask=mask,
        )
        while float(np.linalg.norm(Rp_trial, ord=np.inf)) > base_norm and damp > opt.damping_min:
            damp *= 0.5
            p_trial = _apply_floor(p + damp * dp, opt.enforce_positivity_floor)
            if mask is not None:
                p_trial[~mask] = 0.0
            Rp_trial, _, _, _ = assemble_residual_jacobian_p(
                z, phi, T, p_trial, n, mu_p, Dp, U, dU_dp,
                inp.contact_left, inp.contact_right,
                E_C_J=inp.E_C_J, E_V_J=inp.E_V_J,
                me_rel=inp.me_rel, mh_rel=inp.mh_rel,
                gvc=inp.gvc, gvv=inp.gvv,
                carrier_mask=mask,
            )

        p = p_trial
        res_norm_p = float(np.linalg.norm(Rp_trial, ord=np.inf))

        # Convergence checks
        upd_rel_n = float(np.linalg.norm(dn, ord=np.inf) / _nz(np.linalg.norm(n, ord=np.inf)))
        upd_rel_p = float(np.linalg.norm(dp, ord=np.inf) / _nz(np.linalg.norm(p, ord=np.inf)))

        if opt.debug and (it % max(1, opt.debug_every) == 0):
            print(f"[Continuity] iter {it:02d} | ||R_n||∞={res_norm_n:.3e} "
                  f"||R_p||∞={res_norm_p:.3e} | upd_rel n/p = {upd_rel_n:.2e}/{upd_rel_p:.2e}")

        if max(res_norm_n, res_norm_p) < opt.tol_abs and max(upd_rel_n, upd_rel_p) < opt.tol_rel:
            converged = True
            return ContinuityResult(
                n_m3=_c64(n), p_m3=_c64(p), iters=it, converged=True,
                res_norm_n=res_norm_n, res_norm_p=res_norm_p
            )

    return ContinuityResult(
        n_m3=_c64(n), p_m3=_c64(p), iters=opt.max_iters, converged=converged,
        res_norm_n=res_norm_n, res_norm_p=res_norm_p
)
