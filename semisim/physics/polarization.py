# semisim/physics/polarization.py
"""
Polarization charge for wurtzite stacks (1D along c-axis).

Computes:
  - volumetric polarization charge:   rho_pol(z)  = - dPz/dz   [C/m^3]
  - interface sheet charge at z_i:    sigma_pol =  ΔPz|_i      [C/m^2]

where Pz = P_sp + P_pz and (for biaxial strain) e_zz ≈ -2 (c13/c33) e_xx.

This implementation is robust to missing fields: if a field is absent in MaterialFields,
it defaults to zeros (i.e., no polarization from that mechanism). It also accepts
either a structured interfaces array with fields ('z','left','right') or a plain Nx3 array.

Public API:
    PolarizationSetup
    PolarizationResult
    compute_polarization(setup)

Polarization charge for wurtzite stacks (1D along c-axis).

Computes either:
  - volumetric bound charge:  rho_pol = -dPz/dz                [C/m^3]
  - interface sheet charge:   sigma_pol(node_j) = ΔPz|interface [C/m^2]

ON DOUBLE COUNTING
-----------------------
You should NOT use both the volumetric term and explicit sheets at the same
time unless you know exactly how your control volumes handle the jump.
By default we use **sheets_only** to avoid double counting.

2DEG COUPLING
-------------
If a separate 2DEG model will contribute a dynamic sheet at specific nodes,
you can suppress the fixed polarization sheet at those nodes to avoid
double-counting:
    setup.hemt2deg_nodes = np.array([j_interface, ...])
    setup.suppress_when_2deg = True  # default
"""
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from ..geometry.builder import Geometry1D, MaterialFields

__all__ = ["PolarizationSetup", "PolarizationResult", "compute_polarization"]


# --------------------------------- data ---------------------------------- #
@dataclass(slots=True)
class PolarizationSetup:
    geom: Geometry1D
    mat: MaterialFields
    orientation: Literal["+c", "-c"] = "+c"     # sign of c-axis
    mode: Literal["sheets_only", "volume_only", "both"] = "sheets_only"
    # If 2DEG will add a dynamic sheet at these nodes, suppress fixed σ_pol there
    hemt2deg_nodes: Optional[np.ndarray] = None
    suppress_when_2deg: bool = True
    debug: bool = False


@dataclass(slots=True)
class PolarizationResult:
    sheet_nodes: np.ndarray        # (M,) interface node indices (int)
    sigma_pol_Cm2: np.ndarray      # (N,) node-aligned sheets [C/m^2] (zero except at interfaces)
    rho_pol_Cm3: np.ndarray        # (N,) volumetric charge [C/m^3]
    Pz_C_per_m2: np.ndarray        # (N,) total polarization along c-axis [C/m^2]


# -------------------------------- utils ---------------------------------- #
def _c64(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


def _get_field(obj, name: str, N: int, default: float) -> np.ndarray:
    """Get obj.name as float64 vector of length N; sanitize NaNs/Infs."""
    arr = getattr(obj, name, None)
    if arr is None:
        out = np.full(N, default, dtype=np.float64)
    else:
        out = np.asarray(arr, dtype=np.float64)
        if out.shape != (N,):
            out = np.resize(out, (N,))
    if not np.all(np.isfinite(out)):
        print(f"[pol] WARNING: field {name} contains NaN/Inf; sanitizing.")
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


# -------------------------------- main ----------------------------------- #
def compute_polarization(setup: PolarizationSetup) -> PolarizationResult:
    geom, mat = setup.geom, setup.mat
    z = _c64(geom.z)
    N = z.size
    if N < 2:
        raise ValueError("Geometry must have at least 2 nodes for polarization.")

    sgn = +1.0 if setup.orientation == "+c" else -1.0

    # --- material fields (metals/oxides should already be zeros in MaterialFields) ---
    Psp = _get_field(mat, "Psp_C_per_m2", N, 0.0)
    e31 = _get_field(mat, "e31_C_per_m2", N, 0.0)
    e33 = _get_field(mat, "e33_C_per_m2", N, 0.0)
    c13 = _get_field(mat, "c13_Pa", N, 0.0)
    c33 = _get_field(mat, "c33_Pa", N, 1.0)  # 1.0 avoids div0 outside wurtzite
    exx = _get_field(mat, "strain_xx", N, 0.0)

    ezz = getattr(mat, "strain_zz", None)
    if ezz is None:
        ratio = np.divide(c13, c33, out=np.zeros_like(c13), where=(c33 != 0.0))
        ezz = -2.0 * ratio * exx
    else:
        ezz = _c64(ezz)
        if ezz.shape != (N,):
            ezz = np.resize(ezz, (N,))
        if not np.all(np.isfinite(ezz)):
            print("[pol] WARNING: field strain_zz contains NaN/Inf; sanitizing.")
            ezz = np.nan_to_num(ezz, nan=0.0, posinf=0.0, neginf=0.0)

    # --- total polarization along c-axis ---
    Ppz = 2.0 * e31 * exx + e33 * ezz
    Pz = sgn * (Psp + Ppz)  # [C/m^2]

    # --- compute requested charge representation(s) ---
    rho_pol = np.zeros(N, dtype=np.float64)
    sigma_pol = np.zeros(N, dtype=np.float64)
    sheet_nodes: list[int] = []

    if setup.mode in ("volume_only", "both"):
        # ρ = -dPz/dz (finite-volume compatible gradient)
        dPdz = np.gradient(Pz, z, edge_order=2)
        rho_pol = -dPdz

    if setup.mode in ("sheets_only", "both"):
        # Node-aligned sheets at geometric interfaces: σ = ΔPz across interface
        if hasattr(geom, "interfaces"):
            zi = geom.interfaces["z"]
            for k in range(zi.size):
                # nearest node to interface position
                j = int(np.argmin(np.abs(z - zi[k])))
                j = int(np.clip(j, 1, N - 1))
                # jump taken as right - left value at the node pair
                sigma_pol[j] += Pz[j] - Pz[j - 1]
                sheet_nodes.append(j)

    # --- optional suppression at 2DEG nodes to avoid double counting ---
    if setup.suppress_when_2deg and setup.hemt2deg_nodes is not None and sheet_nodes:
        suppress = np.asarray(setup.hemt2deg_nodes, dtype=int).ravel()
        suppress = suppress[(suppress >= 0) & (suppress < N)]
        if suppress.size:
            if setup.debug:
                uniq = np.unique(suppress)
                print(f"[pol] suppressing σ_pol at 2DEG nodes {uniq.tolist()} (avoid double-counting).")
            sigma_pol[suppress] = 0.0

    sheet_nodes_arr = np.asarray(sheet_nodes, dtype=int) if sheet_nodes else np.zeros(0, dtype=int)

    if setup.debug:
        Pmin, Pmax = float(np.min(Pz)), float(np.max(Pz))
        print(f"[pol] Pz∈[{Pmin:+.3e},{Pmax:+.3e}] C/m^2 | mode={setup.mode} | "
              f"σ≠0 count={int(np.count_nonzero(sigma_pol))} | any ρ?={bool(np.any(rho_pol))}")

    return PolarizationResult(
        sheet_nodes=sheet_nodes_arr,
        sigma_pol_Cm2=sigma_pol,
        rho_pol_Cm3=rho_pol,
        Pz_C_per_m2=Pz,
    )
