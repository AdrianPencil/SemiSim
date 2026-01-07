# semisim/models/pn_junction.py
"""
PN-junction (Si) demo using LayerStack + Poisson driver.

- Two equal Si slabs (p | n)
- Band edges from affinity model: E_C = E_C^0 - q φ, E_V = E_V^0 - q φ
- Dirichlet–Neumann BCs (left reference, right zero-field)
- Depletion-approximation seed for φ for robust convergence

Run:
    python -m semisim pn --help
    python -m semisim pn --ND 1e22 --NA 1e22 --Lp 2e-6 --Ln 2e-6 --N 1201
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import argparse
import numpy as np

from semisim.physics.poisson import (
    PoissonBC,
    PoissonSetup,
    PoissonResult,
    solve_poisson_1d,
)
from semisim.physics.carriers.bands import (
    BandParams,
    build_band_params_from_fields,
    intrinsic_density_MB,
)
from semisim.geometry.builder import (
    LayerSpec,
    StackSpec,
    MeshSpec,
    build_geometry,
    resolve_material_fields,
    attach_stack,
    list_interfaces_array,
)
from semisim.postprocess.visualization import plot_band_diagram
from semisim.physics.carriers.intrinsic import intrinsic_density_MB

# ---- constants (SI) ----
Q = 1.602176634e-19        # C
K_B = 1.380649e-23         # J/K

__all__ = [
    "PNJunctionParams",
    "PNResult",
    "solve_pn_equilibrium",
    "build_pn_geometry",          # NEW
    "doping_arrays_from_geom",    # NEW
    "main",
]


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class PNJunctionParams:
    """Inputs for a "textbook" 1-D silicon PN junction (two equal Si slabs)."""
    ND_m3: float = 5e22          # 5e16 cm^-3
    NA_m3: float = 5e22          # 5e16 cm^-3
    Lp_m: float = 0.5e-6
    Ln_m: float = 0.5e-6
    N: int = 401
    T_K: float = 300.0
    stats: Literal["MB", "FD"] = "MB"
    solver: Literal["gummel", "newton"] = "gummel"

    # Optional override if you want to fix μ (absolute energy, J)
    mu_J: Optional[float] = None

    # Debug prints
    debug: bool = False


@dataclass(slots=True)
class PNResult:
    """Outputs suitable for band-diagram plotting and analysis."""
    z_m: np.ndarray
    phi_V: np.ndarray
    E_C_J: np.ndarray
    E_V_J: np.ndarray
    n_m3: np.ndarray
    p_m3: np.ndarray
    mu_J: float
    converged: bool
    iters: int


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _c64(x) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


def _estimate_mu_quasineutral(
    mat, bands: BandParams, ND: float, NA: float, T_K: float
) -> float:
    """
    Pick μ so that far-field n≈ND (n-side) and p≈NA (p-side) under MB stats.
    Uses symmetric Nc, Nv estimate from n_i and Eg.
    """
    T = float(T_K)
    kT = K_B * T
    ni = float(
        np.mean(
            intrinsic_density_MB(
                mat.Eg_J, T, me_rel=mat.me_dos_rel, mh_rel=mat.mh_dos_rel
            )
        )
    )
    Eg = float(np.mean(mat.Eg_J))
    Nc = ni * np.exp(+0.5 * Eg / kT)
    Nv = ni * np.exp(+0.5 * Eg / kT)

    Ec_R = float(bands.E_C0_J[-1])  # rightmost node (n side)
    Ev_L = float(bands.E_V0_J[0])   # leftmost node  (p side)

    mu_n = Ec_R + kT * np.log(max(ND, 1.0) / Nc)
    mu_p = Ev_L - kT * np.log(max(NA, 1.0) / Nv)
    return 0.5 * (mu_n + mu_p)


def _depletion_seed_phi(
    z: np.ndarray,
    j0: int,
    ND: float,
    NA: float,
    eps_mean: float,
    T_K: float,
    mat,
) -> np.ndarray:
    """
    Classic abrupt-junction depletion seed (piecewise parabolic), φ(z0)=0.
    This is only a *seed*; final φ is found by the nonlinear solve.
    """
    T = float(T_K)
    VT = K_B * T / Q
    # Use n_i from material fields for consistency
    ni = float(
        np.mean(
            intrinsic_density_MB(
                mat.Eg_J, T, me_rel=mat.me_dos_rel, mh_rel=mat.mh_dos_rel
            )
        )
    )
    Vbi = VT * np.log((max(ND, 1.0) * max(NA, 1.0)) / (ni * ni))

    # Total depletion width (1D)
    # W = sqrt( 2 ε_s / q * (1/ND + 1/NA) * Vbi )
    eps = float(eps_mean)
    W = np.sqrt(2.0 * eps / Q * (1.0 / max(ND, 1.0) + 1.0 / max(NA, 1.0)) * Vbi)

    # Split per side (charge neutrality): x_n : x_p = NA : ND
    xn = W * NA / (ND + NA)
    xp = W * ND / (ND + NA)

    z0 = float(z[j0])
    zl, zr = z0 - xp, z0 + xn

    phi = np.zeros_like(z, dtype=np.float64)
    left = z < z0
    right = ~left

    # Parabolic in depletion, flat outside (seed only)
    phi[left] = -Vbi + 0.5 * Q * ND / eps * np.clip(z[left] - zl, 0.0, xp) ** 2
    phi[right] = +Vbi - 0.5 * Q * NA / eps * np.clip(zr - z[right], 0.0, xn) ** 2

    # Shift so φ(z0)=0 exactly
    phi -= phi[j0]
    return phi


# -----------------------------------------------------------------------------
# Main solve
# -----------------------------------------------------------------------------

def solve_pn_equilibrium(params: PNJunctionParams) -> PNResult:
    """
    Build a Si PN junction (two equal Si layers) and solve equilibrium Poisson.
    - Geometry/materials via LayerStack & database
    - Doping assigned by layer (left=p, right=n)
    """
    # --- LayerStack: two equal Si slabs (p | n) ---
    layers = [
        LayerSpec(name="p-Si", role="semiconductor", thickness=float(params.Lp_m), material="Si"),
        LayerSpec(name="n-Si", role="semiconductor", thickness=float(params.Ln_m), material="Si"),
    ]
    stack = StackSpec(layers=layers, T=float(params.T_K))
    mesh = MeshSpec(N_total=int(params.N))

    geom = build_geometry(stack, mesh)
    geom = attach_stack(geom, stack)

    # Per-node material fields (from materials database)
    mat = resolve_material_fields(geom)

    # Bands (affinity reference) with proper layer_id and interfaces
    bands = build_band_params_from_fields(
        material=mat,
        layer_id=geom.layer_id,
        interfaces=list_interfaces_array(geom),
        model="affinity",
    )

    z = geom.z
    N = z.size
    j0 = int(np.where(geom.layer_id == 1)[0][0])  # first node of the right (n) layer

    # Doping arrays by layer: left=p, right=n
    ND_arr = np.zeros(N, dtype=np.float64)
    NA_arr = np.zeros(N, dtype=np.float64)
    NA_arr[geom.layer_id == 0] = float(params.NA_m3)
    ND_arr[geom.layer_id == 1] = float(params.ND_m3)

    # Fermi level μ: estimate from far-field dopings if not provided
    mu_J = (
        _estimate_mu_quasineutral(mat, bands, params.ND_m3, params.NA_m3, params.T_K)
        if params.mu_J is None
        else float(params.mu_J)
    )

    # Boundary conditions (Ohmic far-field contacts, equilibrium):
    # Dirichlet–Dirichlet with ±½·V_bi so bands stay continuous for Si|Si.
    #   V_bi = V_T * ln(NA * ND / ni^2), with V_T = k_B T / q
    T = float(params.T_K)
    V_T = K_B * T / Q

    # intrinsic n_i from material fields (MB is fine at 300 K)
    ni = float(np.mean(intrinsic_density_MB(Eg_J=mat.Eg_J, T=T, me_rel=mat.me_dos_rel, mh_rel=mat.mh_dos_rel)))

    ND = max(float(params.ND_m3), 1.0)
    NA = max(float(params.NA_m3), 1.0)
    V_bi = V_T * np.log((NA * ND) / (ni * ni))

    bc = PoissonBC(
        left_kind="dirichlet",  left_value=-0.5 * V_bi,   # p-contact
        right_kind="dirichlet", right_value=+0.5 * V_bi,   # n-contact
    )


    # Depletion-approximation seed around the interface
    phi_seed = _depletion_seed_phi(
        z=z,
        j0=j0,
        ND=float(params.ND_m3),
        NA=float(params.NA_m3),
        eps_mean=float(np.mean(mat.eps)),
        T_K=float(params.T_K),
        mat=mat,
    )

    # Assemble and solve
    setup = PoissonSetup(
        geom=geom,
        mat=mat,
        bands=bands,
        bc=bc,
        T_K=float(params.T_K),
        mu_J=float(mu_J),
        stats=str(params.stats),
        gvc=1.0,
        gvv=1.0,
        ND_m3=_c64(ND_arr),
        NA_m3=_c64(NA_arr),
        rho_extra_Cm3=None,
        sheet_nodes=None,
        sheet_sigma_Cm2=None,
        hemt2deg=None,
        bulk_traps=None,
        interface_traps=None,
        contact_left=None,
        contact_right=None,
        include_carriers=True,
        exp_clip=60.0,
        phi_guess=_c64(phi_seed),
        carrier_mask=np.ones(N, dtype=bool),
        nonlinear_solver=("gummel" if params.solver == "gummel" else "newton"),
        solver_options=dict(
            max_iters=150 if params.solver == "gummel" else 80,
            tol_res_inf=1e-8,
            damping_init=1.0,
            max_step_volts=0.25,
            debug=bool(params.debug),
            print_every=1,
        ),
        enable_full_continuity=False,
        continuity_options=None,
        transport=None,
        recomb=None,
        debug=bool(params.debug),
    )

    sol: PoissonResult = solve_poisson_1d(setup)

    return PNResult(
        z_m=_c64(sol.z),
        phi_V=_c64(sol.phi),
        E_C_J=_c64(sol.E_C_J),
        E_V_J=_c64(sol.E_V_J),
        n_m3=_c64(sol.n),
        p_m3=_c64(sol.p),
        mu_J=float(mu_J),
        converged=bool(sol.converged),
        iters=int(sol.iters),
    )

def build_pn_geometry(par: PNJunctionParams):
    """Geometry+materials only (no physics). Two Si layers, p then n."""
    layers = [
        LayerSpec(name="p-Si", role="semiconductor", thickness=par.Lp_m, material="Si"),
        LayerSpec(name="n-Si", role="semiconductor", thickness=par.Ln_m, material="Si"),
    ]
    stack = StackSpec(layers=layers, T=par.T_K)

    mesh = MeshSpec(
        N_total=par.N,
        refine_interfaces=True, stretch_ratio=1.0, stretch_cells=0
    )

    geom = build_geometry(stack, mesh)
    attach_stack(geom, stack)                 # so materials can be resolved
    mat = resolve_material_fields(geom)
    return geom, mat

def doping_arrays_from_geom(geom, par: PNJunctionParams):
    """Piecewise-constant doping by layer: acceptors in layer 0, donors in layer 1."""
    N = geom.z.size
    ND = np.zeros(N, dtype=np.float64)
    NA = np.zeros(N, dtype=np.float64)
    ND[geom.layer_id == 1] = float(par.ND_m3)
    NA[geom.layer_id == 0] = float(par.NA_m3)
    return ND, NA
# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Silicon PN junction — equilibrium band diagram")
    p.add_argument("--ND", type=float, default=1e22, help="Donor conc on n-side [1/m^3]")
    p.add_argument("--NA", type=float, default=1e22, help="Acceptor conc on p-side [1/m^3]")
    p.add_argument("--Lp", type=float, default=2.0e-6, help="Left (p-side) length [m]")
    p.add_argument("--Ln", type=float, default=2.0e-6, help="Right (n-side) length [m]")
    p.add_argument("--N", type=int, default=1201, help="Total grid nodes (>=3)")
    p.add_argument("--T", type=float, default=300.0, help="Temperature [K]")
    p.add_argument("--stats", choices=["MB", "FD"], default="MB", help="Carrier statistics")
    p.add_argument("--solver", choices=["gummel", "newton"], default="gummel", help="Nonlinear solver")
    p.add_argument("--debug", action="store_true", help="Verbose solver prints")
    p.add_argument("--csv", default="pn_equilibrium.csv", help="CSV output path")
    p.add_argument("--png", default="pn_band_diagram.png", help="PNG plot output path")
    p.add_argument("--title", default="PN Junction (Si) — Equilibrium", help="Plot title")
    return p


def main() -> None:
    ap = _build_parser().parse_args()

    params = PNJunctionParams(
        ND_m3=ap.ND,
        NA_m3=ap.NA,
        Lp_m=ap.Lp,
        Ln_m=ap.Ln,
        N=ap.N,
        T_K=ap.T,
        stats=str(ap.stats),
        solver=str(ap.solver),
        debug=bool(ap.debug),
    )

    res = solve_pn_equilibrium(params)

    # CSV
    arr = np.column_stack([res.z_m, res.phi_V, res.E_C_J, res.E_V_J, res.n_m3, res.p_m3])
    header = "z_m,phi_V,E_C_J,E_V_J,n_m3,p_m3"
    np.savetxt(ap.csv, arr, delimiter=",", header=header, comments="")
    print(f"[ok] wrote {ap.csv}  (converged={res.converged}, iters={res.iters})")

    # Plot
    fig, ax = plot_band_diagram(res.z_m, res.E_C_J, res.E_V_J, res.mu_J, title=str(ap.title))
    fig.savefig(ap.png, dpi=180)
    print(f"[ok] wrote {ap.png}")


if __name__ == "__main__":
    main()
