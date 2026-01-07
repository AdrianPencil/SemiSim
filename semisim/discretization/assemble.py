# semisim/discretization/assemble.py
# PEP-8, no external deps beyond numpy. Provides a thin, testable problem object
# for Poisson in 1-D with optional carrier coupling and 2DEG.
#
# New in this version
# -------------------
# - Stable handling for carrier masks (metals/oxides → carriers off).
# - Optional scaling hooks via semisim.utils.scaling (safe fallback).
# - Residual/Jacobian split kept clean; extra helpers to access scaled forms.

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import numpy as np

# ---- constants (SI) ----
Q = 1.602176634e-19    # C
K_B = 1.380649e-23     # J/K

# ---- Optional imports (project-internal) -----------------------------------
try:
    from semisim.physics.carriers.bands import band_edges_from_potential  # type: ignore
except Exception:  # pragma: no cover
    from bands import band_edges_from_potential  # fallback for local dev

try:
    from semisim.physics.carriers.statistics import (
        carriers_3d,
        derivatives_3d,
    )
except Exception:  # pragma: no cover
    from semisim.physics.carriers.statistics import (
        carriers_3d,
        derivatives_3d,
    )

# 2DEG interface model (single source of truth)
from semisim.physics.interfaces import hemt_2deg_sigma_and_jacobian
_HAS_2DEG = True

# Traps (bulk + interface)
try:
    from semisim.physics.traps import (
        BulkTrapSet, BulkTrapInputs, bulk_trap_charge,
        InterfaceTrapSet, InterfaceTrapInputs, interface_trap_sheet,
    )
    _HAS_TRAPS = True
except Exception:  # pragma: no cover
    _HAS_TRAPS = False

# Optional scaling utilities
try:
    from semisim.utils.scaling import PoissonScales, compute_poisson_scales  # type: ignore
    _HAS_SCALING = True
except Exception:  # pragma: no cover
    _HAS_SCALING = False


def _to_f64(x) -> np.ndarray:
    """Return a C-contiguous float64 ndarray without copying if possible."""
    a = np.asarray(x, dtype=np.float64)
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return a


def _inf_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, ord=np.inf))


class State(NamedTuple):
    E_C_J: np.ndarray
    E_V_J: np.ndarray
    n: np.ndarray
    p: np.ndarray


@dataclass
class PoissonProblem:
    """Discrete 1-D Poisson(+carriers) problem used by Newton/Gummel solvers.

    This object is *stateless* with respect to iteration, but holds all inputs
    needed to assemble residuals and Jacobians for a given potential phi.
    """

    # Geometry / discretization
    z: np.ndarray  # node coordinates, shape (N,)
    Vi: np.ndarray  # control volume per node, shape (N,)
    alpha_f: np.ndarray  # ε_face / dz_face, shape (N-1,)
    eps_f: np.ndarray  # face permittivity (for debug), shape (N-1,)

    # Physics
    ND: np.ndarray
    NA: np.ndarray
    rho_extra: np.ndarray  # extra fixed volumetric charge [C/m^3]
    sigma_node: np.ndarray  # node-applied sheet charge [C/m^2]
    bands: object
    mat: object
    stats: str
    mu_J: float
    T_K: float
    gvc: float
    gvv: float
    include_carriers: bool
    exp_clip: float
    hemt2deg: Optional[object] = None  # parameters object for 2DEG, can be None

    # Optional traps (bulk + interface)
    bulk_traps: Optional["BulkTrapSet"] = None
    interface_traps: Optional["InterfaceTrapSet"] = None

    # Boundary conditions (left Dirichlet always; right Dirichlet or "natural")
    bc: object = None

    # Optional mask to restrict carriers to semiconductor nodes
    carrier_mask: Optional[np.ndarray] = None

    # Initial guess
    x0: np.ndarray = None

    # Debug mode
    debug: bool = False

    # Optional scales (non-dimensional form); may be None if scaling disabled
    scales: Optional[object] = None  # PoissonScales

    # -------------------------- high-level API -------------------------------

    def project_dirichlet(self, phi: np.ndarray) -> None:
        """Enforce Dirichlet values at boundaries in-place (PoissonBC API)."""
        if getattr(self.bc, "left_kind", "dirichlet").lower() == "dirichlet":
            phi[0] = float(self.bc.left_value)
        if getattr(self.bc, "right_kind", "dirichlet").lower() == "dirichlet":
            phi[-1] = float(self.bc.right_value)

    def update_carriers_fixed_phi(self, phi: np.ndarray, *, compute_currents: bool = False) -> None:
        """
        Update self._carriers_override = (n, p) for the given potential phi.

        Order of preference:
        1) Full continuity (Fix #6) if enabled and module present.
        2) Linear continuity (legacy helper).
        3) Statistics-only fallback.
        """
        n: np.ndarray
        p: np.ndarray

        # --- Try full continuity first (Fix #6) --------------------------
        if bool(getattr(self, "enable_full_continuity", False)):
            try:
                from semisim.physics.continuity import (
                    ContinuityInputs, ContinuityOptions, solve_continuity_1d
                )  # type: ignore
                from semisim.physics.transport import (
                    TransportInputs, build_kernel_from_config, mobility_from_kernel, diffusion_from_mobility
                )  # type: ignore
                from semisim.physics.recombination import (
                    build_recomb_evaluator_from_config,
                )  # type: ignore

                # --- Mobility via Matthiessen (Fix #7) --------------------
                tr_cfg = getattr(self, "transport_config", None) or {}
                kernel = build_kernel_from_config(tr_cfg)

                # Initial guess from statistics (needed for degeneracy, masks)
                E_C0, E_V0 = band_edges_from_potential(_to_f64(phi), self.bands)
                n0, p0 = carriers_3d(
                    E_C0, E_V0, self.mu_J, self.T_K,
                    me_rel=self.mat.me_dos_rel, mh_rel=self.mat.mh_dos_rel,
                    gvc=self.gvc, gvv=self.gvv, stats=self.stats, exp_clip=self.exp_clip,
                )
                if self.carrier_mask is not None:
                    mask = np.asarray(self.carrier_mask, dtype=bool)
                    n0[~mask] = 0.0; p0[~mask] = 0.0

                # Build transport inputs (no explicit Nd/Na yet → zeros; x_alloy unknown → NaN)
                Nd = np.zeros_like(self.z); Na = np.zeros_like(self.z)
                x_alloy = np.full_like(self.z, np.nan, dtype=np.float64)
                tinp = TransportInputs(
                    T_K=float(self.T_K),
                    eps_r=_to_f64(self.mat.eps_r),
                    n_m3=_to_f64(n0), p_m3=_to_f64(p0),
                    Nd_m3=_to_f64(Nd), Na_m3=_to_f64(Na),
                    x_alloy=_to_f64(x_alloy),
                    z_m=_to_f64(self.z),
                )
                mu_n, mu_p = mobility_from_kernel(tinp, kernel)

                # Degenerate Einstein correction using η ≈ (μ_J - E_C)/kT, (E_V - μ_J)/kT
                V_T = float(self.T_K) * K_B / Q
                eta_n = (float(self.mu_J) - E_C0) / (K_B * float(self.T_K))
                eta_p = (E_V0 - float(self.mu_J)) / (K_B * float(self.T_K))
                Dn = diffusion_from_mobility(mu_n, self.T_K, eta_n)
                Dp = diffusion_from_mobility(mu_p, self.T_K, eta_p)

                # Initial guess from statistics
                E_C0, E_V0 = band_edges_from_potential(_to_f64(phi), self.bands)
                n0, p0 = carriers_3d(
                    E_C0, E_V0, self.mu_J, self.T_K,
                    me_rel=self.mat.me_dos_rel, mh_rel=self.mat.mh_dos_rel,
                    gvc=self.gvc, gvv=self.gvv, stats=self.stats, exp_clip=self.exp_clip,
                )
                if self.carrier_mask is not None:
                    mask = np.asarray(self.carrier_mask, dtype=bool)
                    n0[~mask] = 0.0; p0[~mask] = 0.0

                # Continuity inputs/options
                opt_dict = dict(getattr(self, "continuity_options", {}) or {})
                opt = ContinuityOptions(
                    max_iters=int(opt_dict.get("max_iters", 50)),
                    tol_abs=float(opt_dict.get("tol_abs", 1e-6)),
                    tol_rel=float(opt_dict.get("tol_rel", 1e-6)),
                    damping_init=float(opt_dict.get("damping_init", 1.0)),
                    damping_min=float(opt_dict.get("damping_min", 1e-4)),
                    debug=bool(opt_dict.get("debug", getattr(self, "debug", False))),
                    debug_every=int(opt_dict.get("debug_every", 1)),
                )

                inp = ContinuityInputs(
                    z_m=self.z,
                    phi_V=_to_f64(phi),
                    T_K=float(self.T_K),
                    n_m3=_to_f64(n0),
                    p_m3=_to_f64(p0),
                    mu_n_m2Vs=_to_f64(mu_n),
                    mu_p_m2Vs=_to_f64(mu_p),
                    Dn_m2s=_to_f64(Dn),
                    Dp_m2s=_to_f64(Dp),
                    E_C_J=_to_f64(E_C0),
                    E_V_J=_to_f64(E_V0),
                    me_rel=float(self.mat.me_dos_rel),
                    mh_rel=float(self.mat.mh_dos_rel),
                    gvc=float(self.gvc),
                    gvv=float(self.gvv),
                    contact_left=getattr(self, "contact_left", None),
                    contact_right=getattr(self, "contact_right", None),
                    carrier_mask=self.carrier_mask,
                )

                # ---- Fix #8: build recombination evaluator from config
                rcfg = getattr(self, "recomb_config", None) or {}
                recomb_eval = build_recomb_evaluator_from_config(
                    rcfg,
                    E_C_J=_to_f64(E_C0),
                    E_V_J=_to_f64(E_V0),
                    T_K=float(self.T_K),
                    me_rel=float(self.mat.me_dos_rel),
                    mh_rel=float(self.mat.mh_dos_rel),
                    gvc=float(self.gvc),
                    gvv=float(self.gvv),
                )

                res = solve_continuity_1d(inp, recomb_eval, opt=opt)

                n = res.n_m3
                p = res.p_m3
                self._carriers_override = (_to_f64(n), _to_f64(p))
                return
            except Exception:
                # Fall through to linear/statistics path
                pass

        try:
            from semisim.solver.subsolvers.continuity_linear import solve_electrons, solve_holes  # type: ignore
            from semisim.physics.transport import (
                TransportInputs, build_kernel_from_config, mobility_from_kernel
            )  # type: ignore

            # Use transport kernel for μ (diagnostics path)
            tr_cfg = getattr(self, "transport_config", None) or {}
            kernel = build_kernel_from_config(tr_cfg)

            E_C0, E_V0 = band_edges_from_potential(_to_f64(phi), self.bands)
            n0, p0 = carriers_3d(
                E_C0, E_V0, self.mu_J, self.T_K,
                me_rel=self.mat.me_dos_rel, mh_rel=self.mat.mh_dos_rel,
                gvc=self.gvc, gvv=self.gvv, stats=self.stats, exp_clip=self.exp_clip,
            )
            if self.carrier_mask is not None:
                mask = np.asarray(self.carrier_mask, dtype=bool)
                n0[~mask] = 0.0; p0[~mask] = 0.0

            Nd = np.zeros_like(self.z); Na = np.zeros_like(self.z)
            x_alloy = np.full_like(self.z, np.nan, dtype=np.float64)
            tinp = TransportInputs(
                T_K=float(self.T_K), eps_r=_to_f64(self.mat.eps_r),
                n_m3=_to_f64(n0), p_m3=_to_f64(p0),
                Nd_m3=_to_f64(Nd), Na_m3=_to_f64(Na),
                x_alloy=_to_f64(x_alloy), z_m=_to_f64(self.z),
            )
            mu_n, mu_p = mobility_from_kernel(tinp, kernel)

            n, _, _ = solve_electrons(
                phi=_to_f64(phi), z=self.z, mu_n=_to_f64(mu_n),
                bands=self.bands, mat=self.mat, mu_J=self.mu_J, T_K=self.T_K,
                gvc=self.gvc, stats=self.stats, exp_clip=self.exp_clip,
                carrier_mask=self.carrier_mask, compute_currents=bool(compute_currents),
                debug=bool(self.debug),
            )
            p, _, _ = solve_holes(
                phi=_to_f64(phi), z=self.z, mu_p=_to_f64(mu_p),
                bands=self.bands, mat=self.mat, mu_J=self.mu_J, T_K=self.T_K,
                gvv=self.gvv, stats=self.stats, exp_clip=self.exp_clip,
                carrier_mask=self.carrier_mask, compute_currents=bool(compute_currents),
                debug=bool(self.debug),
            )
        except Exception:
            # --- Finally: statistics-only fallback ------------------------
            E_C, E_V = band_edges_from_potential(_to_f64(phi), self.bands)
            n, p = carriers_3d(
                E_C, E_V, self.mu_J, self.T_K,
                me_rel=self.mat.me_dos_rel, mh_rel=self.mat.mh_dos_rel,
                gvc=self.gvc, gvv=self.gvv, stats=self.stats, exp_clip=self.exp_clip,
            )
            if self.carrier_mask is not None:
                mask = np.asarray(self.carrier_mask, dtype=bool)
                n = n.copy(); p = p.copy()
                n[~mask] = 0.0; p[~mask] = 0.0

        self._carriers_override = (_to_f64(n), _to_f64(p))

    # ---- physical residual/J (original API used by solvers) ----
    def residual(self, phi: np.ndarray, for_newton: bool = True) -> np.ndarray:
        """Assemble physical residual for current potential phi."""
        N = self.z.size
        Vi = self.Vi
        alpha_f = self.alpha_f

        # Bands and carriers
        E_C, E_V = band_edges_from_potential(_to_f64(phi), self.bands)

        if self.include_carriers:
            # NEW: allow an externally-updated (n,p) when not using the Newton path
            use_override = (not for_newton) and hasattr(self, "_carriers_override") and (self._carriers_override is not None)
            if use_override:
                n, p = self._carriers_override
            else:
                n, p = carriers_3d(
                    E_C,
                    E_V,
                    self.mu_J,
                    self.T_K,
                    me_rel=self.mat.me_dos_rel,
                    mh_rel=self.mat.mh_dos_rel,
                    gvc=self.gvc,
                    gvv=self.gvv,
                    stats=self.stats,
                    exp_clip=self.exp_clip,
                )

            if self.carrier_mask is not None:
                mask = np.asarray(self.carrier_mask, dtype=bool)
                n = n.copy()
                p = p.copy()
                n[~mask] = 0.0
                p[~mask] = 0.0
        else:
            N = self.z.size
            n = np.zeros(N)
            p = np.zeros(N)

        # 2DEG sheet (dynamic), if available
        sigma2d = np.zeros(N)
        if self.include_carriers and self.hemt2deg is not None and _HAS_2DEG:
            sigma2d, _ = hemt_2deg_sigma_and_jacobian(
                E_C,
                params=self.hemt2deg,
                mu_J=self.mu_J,
                T_K=self.T_K,
                exp_clip=self.exp_clip,
            )

        # ---- Traps (dynamic, evaluated at current φ; Jacobian contribution is
        #      neglected for robustness — Gummel/homotopy handles the nonlinearity)
        rho_trap = np.zeros(N, dtype=np.float64)
        sigma_it_node = np.zeros(N, dtype=np.float64)
        if _HAS_TRAPS:
            # Volumetric bulk traps
            if getattr(self, "bulk_traps", None) is not None:
                try:
                    bt_inp = BulkTrapInputs(
                        E_C_J=_to_f64(E_C),
                        E_V_J=_to_f64(E_V),
                        n_m3=_to_f64(n),
                        p_m3=_to_f64(p),
                        T_K=float(self.T_K),
                        me_rel=float(self.mat.me_dos_rel),
                        mh_rel=float(self.mat.mh_dos_rel),
                        traps=self.bulk_traps,
                        gvc=float(self.gvc),
                        gvv=float(self.gvv),
                    )
                    bt_res = bulk_trap_charge(bt_inp)
                    rho_trap = _to_f64(bt_res.rho_trap_Cm3)
                except Exception:
                    pass  # remain zeros if anything goes wrong

            # Interface trap sheets
            if getattr(self, "interface_traps", None) is not None:
                try:
                    # Collect interface node indices from the set
                    if len(self.interface_traps.discrete) > 0:
                        idxs = np.array([it.node_index for it in self.interface_traps.discrete], dtype=int)
                    elif len(self.interface_traps.spectra) > 0:
                        idxs = np.array([sp.node_index for sp in self.interface_traps.spectra], dtype=int)
                    else:
                        idxs = np.zeros(0, dtype=int)

                    if idxs.size > 0:
                        it_inp = InterfaceTrapInputs(
                            node_indices=idxs,
                            E_C_J=_to_f64(E_C),
                            E_V_J=_to_f64(E_V),
                            n_m3=_to_f64(n),
                            p_m3=_to_f64(p),
                            T_K=float(self.T_K),
                            me_rel=float(self.mat.me_dos_rel),
                            mh_rel=float(self.mat.mh_dos_rel),
                            traps=self.interface_traps,
                            gvc=float(self.gvc),
                            gvv=float(self.gvv),
                        )
                        it_res = interface_trap_sheet(it_inp)
                        if it_res.sheet_nodes.size > 0:
                            np.add.at(sigma_it_node, it_res.sheet_nodes, _to_f64(it_res.sigma_it_Cm2))
                except Exception:
                    pass  # keep zeros
        # ------------------------------------------------------------------

        # Total volumetric charge and face fluxes
        Q = 1.602176634e-19  # coulomb
        rho_vol = _to_f64(Q * (p - n + self.ND - self.NA) + self.rho_extra + rho_trap)
        F = _to_f64(-alpha_f * (phi[1:] - phi[:-1]))  # face flux

        sigma_total = _to_f64(self.sigma_node + sigma2d + sigma_it_node)

        resid = np.zeros(N, dtype=np.float64)
        resid[1:-1] = (F[1:] - F[:-1]) / Vi[1:-1] + rho_vol[1:-1] + sigma_total[1:-1] / Vi[1:-1]

        # --- Boundary rows in PoissonBC API ---
        left_kind  = getattr(self.bc, "left_kind",  "dirichlet").lower()
        right_kind = getattr(self.bc, "right_kind", "dirichlet").lower()
        left_val   = float(getattr(self.bc, "left_value",  0.0))
        right_val  = float(getattr(self.bc, "right_value", 0.0))

        dz = np.diff(self.z)
        dz0  = dz[0]
        dzn1 = dz[-1]

        if left_kind == "dirichlet":
            resid[0] = phi[0] - left_val
        else:  # neumann: (phi[1]-phi[0])/dz0 = left_val
            resid[0] = (phi[1] - phi[0]) - left_val * dz0

        if right_kind == "dirichlet":
            resid[-1] = phi[-1] - right_val
        else:  # neumann: (phi[-1]-phi[-2])/dzn1 = right_val
            resid[-1] = (phi[-1] - phi[-2]) - right_val * dzn1

        return resid

    def jacobian(self, phi: np.ndarray, for_newton: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assemble physical tridiagonal Jacobian J for current phi.

        Returns (a, b, c) where:
          a[i] is sub-diagonal J[i, i-1]
          b[i] is diagonal     J[i, i]
          c[i] is super-diag   J[i, i+1]
        """
        N = self.z.size
        Vi = self.Vi
        alpha_f = self.alpha_f

        a = np.zeros(N, dtype=np.float64)
        b = np.zeros(N, dtype=np.float64)
        c = np.zeros(N, dtype=np.float64)

        # Baseline electrostatic (div(ε grad φ)) terms
        a[1:-1] = -alpha_f[:-1] / Vi[1:-1]
        c[1:-1] = -alpha_f[1:] / Vi[1:-1]
        b[1:-1] = (alpha_f[:-1] + alpha_f[1:]) / Vi[1:-1]

        if for_newton and self.include_carriers:
            # Add dynamic charge derivatives ∂ρ/∂φ
            E_C, E_V = band_edges_from_potential(_to_f64(phi), self.bands)

            dn_dphi = np.zeros(N)
            dp_dphi = np.zeros(N)
            dn_dphi, dp_dphi, _, _ = derivatives_3d(
                E_C,
                E_V,
                self.mu_J,
                self.T_K,
                me_rel=self.mat.me_dos_rel,
                mh_rel=self.mat.mh_dos_rel,
                gvc=self.gvc,
                gvv=self.gvv,
                stats=self.stats,
                exp_clip=self.exp_clip,
            )
            if self.carrier_mask is not None:
                mask = np.asarray(self.carrier_mask, dtype=bool)
                dn_dphi = dn_dphi.copy()
                dp_dphi = dp_dphi.copy()
                dn_dphi[~mask] = 0.0
                dp_dphi[~mask] = 0.0

            Q = 1.602176634e-19
            b[1:-1] += Q * (dp_dphi[1:-1] - dn_dphi[1:-1])

            if self.hemt2deg is not None and _HAS_2DEG:
                _, dsigma2d_dphi = hemt_2deg_sigma_and_jacobian(
                    E_C,
                    params=self.hemt2deg,
                    mu_J=self.mu_J,
                    T_K=self.T_K,
                    exp_clip=self.exp_clip,
                )
                if hasattr(self.hemt2deg, "nodes"):
                    idx = np.asarray(self.hemt2deg.nodes, dtype=np.int64)
                    b[idx] += dsigma2d_dphi[idx] / Vi[idx]
                else:
                    b += dsigma2d_dphi / Vi

        left_kind  = getattr(self.bc, "left_kind",  "dirichlet").lower()
        right_kind = getattr(self.bc, "right_kind", "dirichlet").lower()

        dz = np.diff(self.z)
        dz0  = dz[0]
        dzn1 = dz[-1]

        if left_kind == "dirichlet":
            b[0] = 1.0
            a[0] = 0.0
            c[0] = 0.0
        else:
            b[0] = -1.0
            c[0] = +1.0
            a[0] = 0.0

        if right_kind == "dirichlet":
            b[-1] = 1.0
            a[-1] = 0.0
            c[-1] = 0.0
        else:
            a[-1] = -1.0
            b[-1] = +1.0
            c[-1] = 0.0

        return a, b, c

    # ---- convenience API for top-level driver --------------------------------

    def state_from_phi(self, phi: np.ndarray) -> State:
        E_C, E_V = band_edges_from_potential(_to_f64(phi), self.bands)
        if self.include_carriers:
            n, p = carriers_3d(
                E_C,
                E_V,
                self.mu_J,
                self.T_K,
                me_rel=self.mat.me_dos_rel,
                mh_rel=self.mat.mh_dos_rel,
                gvc=self.gvc,
                gvv=self.gvv,
                stats=self.stats,
                exp_clip=self.exp_clip,
            )
            if self.carrier_mask is not None:
                mask = np.asarray(self.carrier_mask, dtype=bool)
                n = n.copy()
                p = p.copy()
                n[~mask] = 0.0
                p[~mask] = 0.0
        else:
            N = self.z.size
            n = np.zeros(N)
            p = np.zeros(N)
        return State(E_C_J=_to_f64(E_C), E_V_J=_to_f64(E_V), n=_to_f64(n), p=_to_f64(p))

    # ---- optional scaled interface (for solvers that opt-in) -----------------

    def x0_scaled(self) -> np.ndarray:
        """Initial guess in scaled coordinates (if scaling available)."""
        if self.scales is None:
            return self.x0
        return _to_f64(self.x0 / float(self.scales.V_scale))

    def residual_scaled(self, x_scaled: np.ndarray, for_newton: bool = True) -> np.ndarray:
        """Residual in scaled coords: R̃(x̃) = R(V* x̃) / R*."""
        if self.scales is None:
            return self.residual(x_scaled, for_newton=for_newton)
        Vstar = float(self.scales.V_scale)
        Rstar = float(self.scales.R_scale)  # residual scale
        phi = _to_f64(x_scaled) * Vstar
        return _to_f64(self.residual(phi, for_newton=for_newton) / Rstar)

    def jacobian_scaled(self, x_scaled: np.ndarray, for_newton: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Jacobian in scaled coords: J̃ = (V*/R*) J."""
        if self.scales is None:
            return self.jacobian(x_scaled, for_newton=for_newton)
        Vstar = float(self.scales.V_scale)
        Rstar = float(self.scales.R_scale)
        phi = _to_f64(x_scaled) * Vstar
        a, b, c = self.jacobian(phi, for_newton=for_newton)
        return _to_f64((Vstar / Rstar) * a), _to_f64((Vstar / Rstar) * b), _to_f64((Vstar / Rstar) * c)


# ---- Public builder ---------------------------------------------------------

def build_poisson_problem(
    *,
    z: np.ndarray,
    Vi: np.ndarray,
    alpha_f: np.ndarray,
    eps_f: np.ndarray,
    ND: np.ndarray,
    NA: np.ndarray,
    rho_extra: np.ndarray,
    sigma_node: np.ndarray,
    bands: object,
    mat: object,
    stats: str,
    mu_J: float,
    T_K: float,
    gvc: float,
    gvv: float,
    hemt2deg: Optional[object],
    include_carriers: bool,
    exp_clip: float,
    bc: object,
    carrier_mask: Optional[np.ndarray],
    phi0: np.ndarray,
    debug: bool,
    # NEW (optional)
    bulk_traps: Optional["BulkTrapSet"] = None,
    interface_traps: Optional["InterfaceTrapSet"] = None,
) -> PoissonProblem:
    """Create a ready-to-solve PoissonProblem instance (no iteration here)."""
    problem = PoissonProblem(
        z=_to_f64(z),
        Vi=_to_f64(Vi),
        alpha_f=_to_f64(alpha_f),
        eps_f=_to_f64(eps_f),
        ND=_to_f64(ND),
        NA=_to_f64(NA),
        rho_extra=_to_f64(rho_extra),
        sigma_node=_to_f64(sigma_node),
        bands=bands,
        mat=mat,
        stats=str(stats),
        mu_J=float(mu_J),
        T_K=float(T_K),
        gvc=float(gvc),
        gvv=float(gvv),
        include_carriers=bool(include_carriers),
        exp_clip=float(exp_clip),
        hemt2deg=hemt2deg,
        bc=bc,
        carrier_mask=(None if carrier_mask is None else np.asarray(carrier_mask, dtype=bool)),
        x0=_to_f64(phi0),
        debug=bool(debug),
        scales=None,
    )

    # Thread traps into the problem (may be None)
    if bulk_traps is not None:
        problem.bulk_traps = bulk_traps
    if interface_traps is not None:
        problem.interface_traps = interface_traps

    # Optional scaling: compute default scales if the module is present
    if _HAS_SCALING:
        try:
            problem.scales = compute_poisson_scales(
                z=problem.z,
                Vi=problem.Vi,
                eps_face=problem.eps_f,
                ND=problem.ND,
                NA=problem.NA,
                rho_extra=problem.rho_extra,
                T_K=problem.T_K,
            )
        except Exception:
            # Leave scaling disabled if anything goes wrong
            problem.scales = None

    if problem.debug:
        emn = float(np.min(problem.eps_f))
        emx = float(np.max(problem.eps_f))
        print(f"[Assemble] N={problem.z.size} | eps_f range [{emn:.3e}, {emx:.3e}] "
              f"| include_carriers={include_carriers} | mask={'yes' if carrier_mask is not None else 'no'} "
              f"| scaling={'on' if problem.scales is not None else 'off'}")

    return problem
