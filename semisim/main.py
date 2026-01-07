# semisim/main.py
"""
SemiSim main entrypoint.

Default subcommand: pn
Usage examples:
    python -m semisim
    python -m semisim pn --help
    python -m semisim pn --ND 1e22 --NA 5e21 --solver newton --stats FD
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import sys
import numpy as np

from .models.pn_junction import (
    PNJunctionParams,
    solve_pn_equilibrium,
    build_pn_geometry,
    doping_arrays_from_geom,
)
from .postprocess.visualization import plot_band_diagram
from .postprocess.structure import plot_device_structure

__all__ = ["main"]


# ------------------------------ PN subcommand -------------------------------


@dataclass(slots=True)
class _PNArgs:
    ND_m3: float
    NA_m3: float
    Lp_m: float
    Ln_m: float
    N: int
    T_K: float
    stats: str
    solver: str
    debug: bool
    csv_out: str
    png_out: str
    title: str
    structure_png_out: str
    no_structure: bool


def _add_pn_subparser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "pn", help="Silicon PN junction (equilibrium band diagram)"
    )
    p.add_argument(
        "--ND", type=float, default=1e22 * 0.1, help="Donor conc on n-side [1/m^3]"
        # e.g. 1e22 m^-3 ≈ 1e16 cm^-3
    )
    p.add_argument(
        "--NA", type=float, default=5e21, help="Acceptor conc on p-side [1/m^3]"
    )
    p.add_argument("--Lp", type=float, default=1.5e-6, help="Left (p-side) length [m]")
    p.add_argument("--Ln", type=float, default=1.5e-6, help="Right (n-side) length [m]")
    p.add_argument("--N", type=int, default=601, help="Total grid nodes (>=3)")
    p.add_argument("--T", type=float, default=300.0, help="Temperature [K]")
    p.add_argument(
        "--stats", choices=["MB", "FD"], default="MB", help="Carrier statistics"
    )
    p.add_argument(
        "--solver", choices=["gummel", "newton"], default="gummel",
        help="Nonlinear solver"
    )
    p.add_argument("--debug", action="store_true", help="Verbose solver prints")
    p.add_argument("--csv", default="pn_equilibrium.csv", help="CSV output path")
    p.add_argument("--png", default="pn_band_diagram.png", help="PNG plot output path")
    p.add_argument(
        "--title", default="PN Junction (Si) — Equilibrium", help="Plot title"
    )
    # structure / figure-1 style output
    p.add_argument(
        "--structure-png", default="pn_structure.png",
        help="PNG output path for device structure (Figure-1 style)"
    )
    p.add_argument(
        "--no-structure", action="store_true",
        help="Skip writing the device-structure figure"
    )
    p.set_defaults(cmd="pn")
    return p


def _run_pn(args: _PNArgs) -> None:
    # ---- geometry-only figure (Figure-1 style) ----
    if not args.no_structure:
        geom, _mat = build_pn_geometry(
            PNJunctionParams(
                ND_m3=args.ND_m3,
                NA_m3=args.NA_m3,
                Lp_m=args.Lp_m,
                Ln_m=args.Ln_m,
                N=args.N,
                T_K=args.T_K,
            )
        )
        ND, NA = doping_arrays_from_geom(geom, PNJunctionParams(
            ND_m3=args.ND_m3, NA_m3=args.NA_m3, Lp_m=args.Lp_m,
            Ln_m=args.Ln_m, N=args.N, T_K=args.T_K
        ))
        figS, _ = plot_device_structure(geom, ND, NA)
        figS.savefig(args.structure_png_out, dpi=180)
        print(f"[ok] wrote {args.structure_png_out}")

    # ---- equilibrium PN solve + band diagram ----
    params = PNJunctionParams(
        ND_m3=args.ND_m3,
        NA_m3=args.NA_m3,
        Lp_m=args.Lp_m,
        Ln_m=args.Ln_m,
        N=args.N,
        T_K=args.T_K,
        # the following exist in your pn_junction module's params
        # (ignored if not used by the equilibrium path)
    )
    # pass through optional choices if your PNJunctionParams accepts them
    try:
        params.stats = args.stats
        params.solver = args.solver
        params.debug = args.debug
    except Exception:
        # keep compatibility if PNJunctionParams doesn't have these fields
        pass

    res = solve_pn_equilibrium(params)

    # CSV
    arr = np.column_stack(
        [res.z_m, res.phi_V, res.E_C_J, res.E_V_J, res.n_m3, res.p_m3]
    )
    header = "z_m,phi_V,E_C_J,E_V_J,n_m3,p_m3"
    np.savetxt(args.csv_out, arr, delimiter=",", header=header, comments="")
    print(
        f"[ok] wrote {args.csv_out}  "
        f"(converged={getattr(res, 'converged', None)}, iters={getattr(res, 'iters', None)})"
    )

    # Plot band diagram
    fig, _ax = plot_band_diagram(
        res.z_m, res.E_C_J, res.E_V_J, res.mu_J, title=args.title
    )
    fig.savefig(args.png_out, dpi=180)
    print(f"[ok] wrote {args.png_out}")


# --------------------------------- main() ------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SemiSim — minimal device demos")
    sub = parser.add_subparsers(dest="cmd")

    pn_parser = _add_pn_subparser(sub)

    # If no subcommand given, default to 'pn' with defaults
    if len(sys.argv) == 1:
        ns = pn_parser.parse_args([])
        args = _PNArgs(
            ND_m3=ns.ND,
            NA_m3=ns.NA,
            Lp_m=ns.Lp,
            Ln_m=ns.Ln,
            N=ns.N,
            T_K=ns.T,
            stats=str(ns.stats),
            solver=str(ns.solver),
            debug=bool(ns.debug),
            csv_out=str(ns.csv),
            png_out=str(ns.png),
            title=str(ns.title),
            structure_png_out=str(ns.structure_png),
            no_structure=bool(ns.no_structure),
        )
        _run_pn(args)
        return

    ns = parser.parse_args()
    if ns.cmd == "pn":
        args = _PNArgs(
            ND_m3=ns.ND,
            NA_m3=ns.NA,
            Lp_m=ns.Lp,
            Ln_m=ns.Ln,
            N=ns.N,
            T_K=ns.T,
            stats=str(ns.stats),
            solver=str(ns.solver),
            debug=bool(ns.debug),
            csv_out=str(ns.csv),
            png_out=str(ns.png),
            title=str(ns.title),
            structure_png_out=str(ns.structure_png),
            no_structure=bool(ns.no_structure),
        )
        _run_pn(args)
        return

    parser.error("Unknown command (try: pn)")


if __name__ == "__main__":
    main()
