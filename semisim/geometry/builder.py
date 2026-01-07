# semisim/geometry/builder.py
"""
1D geometry builder for layered stacks (MOS/MES/HEMT-friendly).

- SI units throughout.
- Constructs a node-centered 1D mesh over a z-stack of layers.
- Supports constant or linear alloy grading per layer (e.g., Al_x Ga_{1-x} N).
- Emits region/layer maps, interface table, cell metrics (for FV/SG stencils).
- Optional T-resolved material fields via materials/database.py.

Public API (stable):
    LayerSpec
    CompositionSpec
    StackSpec
    MeshSpec
    Geometry1D
    MaterialFields

    build_geometry(stack: StackSpec, mesh: MeshSpec) -> Geometry1D
    resolve_material_fields(geom: Geometry1D) -> MaterialFields
    list_interfaces(geom: Geometry1D) -> np.ndarray

Notes
-----
- No physics here: only geometry + property lookup. Polarization sheets etc.
  are handled in physics/polarization.py.
- 'role' gates downstream behavior (e.g., mask semiconductor vs oxide vs metal).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence, Tuple
import numpy as np
from ..materials.database import get_material, get_alloy

# ---- SI constants ----
EPS0 = 8.8541878128e-12
from ..utils.constants import Q

__all__ = [
    "LayerSpec", "CompositionSpec", "StackSpec", "MeshSpec", "Geometry1D",
    "MaterialFields", "build_geometry", "resolve_material_fields",
    "list_interfaces", "attach_stack", "list_interfaces_array"
]

# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


RoleT = Literal["semiconductor", "oxide", "metal", "void"]


@dataclass(slots=True)
class CompositionSpec:
    """
    Composition profile within a layer.

    mode="constant": use x0 (0..1).
    mode="linear":   linear grade from x0 at z_in (layer start) to x1 at z_out (layer end).
    """
    mode: Literal["constant", "linear"] = "constant"
    x0: float = 0.0
    x1: float = 0.0  # only used in "linear"


@dataclass(slots=True)
class LayerSpec:
    """
    One layer in the 1D stack.

    Attributes
    ----------
    name : str
        Human-readable label.
    role : RoleT
        Semantic role; downstream masks rely on this.
    thickness : float
        Layer thickness [m], must be > 0.
    material : str
        Base material key, e.g., "GaN", "AlN". For alloys, set 'system' and 'comp'.
    system : Optional[str]
        Alloy system key, e.g., "AlGaN" (see materials/database.py). If None, 'material' is used.
    comp : Optional[CompositionSpec]
        Composition spec for alloys. Ignored for pure materials.
    """
    name: str
    role: RoleT
    thickness: float
    material: str
    system: Optional[str] = None
    comp: Optional[CompositionSpec] = None


@dataclass(slots=True)
class StackSpec:
    """
    Full stack definition and temperature context.

    Attributes
    ----------
    layers : Sequence[LayerSpec]
        From z=0 (surface) to z=L (bulk).
    T : float
        Temperature [K].
    """
    layers: Sequence[LayerSpec]
    T: float = 300.0


@dataclass(slots=True)
class MeshSpec:
    """
    Mesh controls for node-centered 1D grid.

    Choose one of:
      - N_total: total nodes for whole stack (>= 3).
      - hz_max:  max node spacing [m] (applied per layer).
      - per_layer_N: nodes per layer (len == len(layers)).

    Optionally grade near interfaces using geometric stretching.
    """
    N_total: Optional[int] = None
    hz_max: Optional[float] = None
    per_layer_N: Optional[Sequence[int]] = None

    # Interface refinement (stretching ratio r >= 1):
    refine_interfaces: bool = True
    stretch_ratio: float = 1.0  # r=1 means uniform
    stretch_cells: int = 0      # number of cells each side of an interface to stretch


@dataclass(slots=True)
class Geometry1D:
    """
    Geometry result (node-centered).

    Arrays are float64, C-contiguous where applicable.
    """
    T: float

    # Mesh
    z: np.ndarray          # shape (N,), node coordinates [m], ascending
    dz: np.ndarray         # shape (N-1,), edge lengths [m]
    Vi: np.ndarray         # shape (N,), control-volume lengths [m] (1D volumes)

    # Layer / region maps
    layer_id: np.ndarray   # shape (N,), node -> layer index
    role_id: np.ndarray    # shape (N,), enum: 0=semi,1=oxide,2=metal,3=void
    region_id: np.ndarray  # shape (N,), contiguous integers per role change (optional)

    # Composition (for alloys only; NaN where not applicable)
    x: np.ndarray          # shape (N,), mole fraction of 'system' first endmember

    # Interface table: (z_i, left_layer_idx, right_layer_idx)
    interfaces: np.ndarray  # shape (n_ifaces, 3), float64 for z_i, int for indices (view/packed)


    # Convenience masks
    mask_semiconductor: np.ndarray  # shape (N,), bool
    mask_oxide: np.ndarray          # shape (N,), bool
    mask_metal: np.ndarray          # shape (N,), bool

    _layers: tuple = ()
    # ---------------------


@dataclass(slots=True)
class MaterialFields:
    """
    Per-node material fields resolved at Temperature T.

    Units are SI; eps = eps_r * EPS0.
    """
    eps_r: np.ndarray       # relative permittivity (-)
    eps: np.ndarray         # absolute permittivity [F/m]
    Eg_J: np.ndarray        # band gap [J]
    chi_J: np.ndarray       # electron affinity [J]
    me_dos_rel: np.ndarray  # electron DOS mass / m0
    mh_dos_rel: np.ndarray  # hole DOS mass / m0

    # Wurtzite piezo/spontaneous (zeros for non-wurtzite)
    e31_C_per_m2: np.ndarray
    e33_C_per_m2: np.ndarray
    c13_Pa: np.ndarray
    c33_Pa: np.ndarray
    Psp_C_per_m2: np.ndarray


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------


def _role_to_id(role: RoleT) -> int:
    if role == "semiconductor":
        return 0
    if role == "oxide":
        return 1
    if role == "metal":
        return 2
    return 3  # void


def _as_c64(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def build_geometry(stack: StackSpec, mesh: MeshSpec) -> Geometry1D:
    """Construct a node-centered 1D geometry from a layered stack."""
    layers = list(stack.layers)
    if len(layers) < 1:
        raise ValueError("stack must contain at least one LayerSpec")

    # Cumulate layer thicknesses and interface positions
    t = np.array([ly.thickness for ly in layers], dtype=np.float64)
    if np.any(t <= 0.0):
        raise ValueError("layer thickness must be positive")
    z_ifaces = np.concatenate(([0.0], np.cumsum(t)))
    L = float(z_ifaces[-1])

    # Decide node counts per layer
    if mesh.per_layer_N is not None:
        perN = np.array(mesh.per_layer_N, dtype=int)
        if perN.size != len(layers):
            raise ValueError("per_layer_N length must match number of layers")
        if np.any(perN < 2):
            raise ValueError("each layer must have at least 2 nodes")
    elif mesh.hz_max is not None:
        perN = np.maximum(2, np.ceil(t / float(mesh.hz_max)).astype(int))
    elif mesh.N_total is not None:
        # Proportional allocation
        w = t / np.sum(t)
        perN = np.maximum(2, np.floor(w * mesh.N_total).astype(int))
        # Fix rounding to ensure sum(perN) ~ N_total while keeping >=2
        deficit = mesh.N_total - int(np.sum(perN))
        order = np.argsort(-(w - perN / np.sum(perN + 1e-12)))
        k = 0
        while deficit > 0:
            perN[order[k % len(layers)]] += 1
            deficit -= 1
            k += 1
    else:
        raise ValueError("provide one of: per_layer_N, hz_max, or N_total")

    # Build z-grid per layer (uniform within layer by default)
    z_nodes: list[np.ndarray] = []
    for i, Ni in enumerate(perN):
        z0, z1 = z_ifaces[i], z_ifaces[i + 1]
        z_i = np.linspace(z0, z1, int(Ni), endpoint=False)  # drop last to avoid duplicate at interface
        if mesh.refine_interfaces and mesh.stretch_ratio > 1.0 and mesh.stretch_cells > 0:
            # Optional: simple symmetric geometric stretching around both ends of the layer
            # Construct interior nodes by stretching; keep endpoints fixed
            Ni_int = max(int(Ni) - 2, 0)
            if Ni_int > 0:
                r = float(mesh.stretch_ratio)
                m = int(mesh.stretch_cells)
                # left cluster
                n_left = min(m, Ni_int)
                # right cluster
                n_right = min(m, Ni_int - n_left)
                n_mid = Ni_int - n_left - n_right
                # geometric partitions
                left = z0 + (z1 - z0) * (1 - r ** -np.arange(1, n_left + 1)) / (1 - r ** -max(n_left, 1))
                right = z1 - (z1 - z0) * (1 - r ** -np.arange(1, n_right + 1)) / (1 - r ** -max(n_right, 1))
                mid = np.linspace(left[-1] if n_left else z0, right[0] if n_right else z1, n_mid + 2)[1:-1] if n_mid > 0 else np.array([], dtype=np.float64)
                z_i = np.concatenate(([z0], left, mid, right, [z1]))
        else:
            z_i = np.linspace(z0, z1, int(Ni), endpoint=True)
        # drop first node for i>0 to avoid duplicates across layers
        if i > 0:
            z_i = z_i[1:]
        z_nodes.append(z_i)

    z = _as_c64(np.concatenate(z_nodes))
    if not np.all(np.diff(z) > 0):
        raise RuntimeError("non-monotonic z grid constructed")

    # Metrics
    dz = _as_c64(np.diff(z))
    # control volumes (1D): Vi[i] = 0.5*(z[i+1]-z[i-1]), with endpoints halved
    Vi = np.empty_like(z)
    Vi[0] = 0.5 * dz[0]
    Vi[-1] = 0.5 * dz[-1]
    Vi[1:-1] = 0.5 * (dz[1:] + dz[:-1])

    # Node-to-layer assignment (half-open on right, except last layer is closed)
    layer_id = np.empty_like(z, dtype=int)
    layer_bounds = list(zip(z_ifaces[:-1], z_ifaces[1:]))

    # use a tiny tolerance relative to span to avoid FP edge cases
    tol = 1e-18

    for i, (z0, z1) in enumerate(layer_bounds):
        if i < len(layer_bounds) - 1:
            # [z0, z1)
            idx = np.where((z >= z0 - tol) & (z < z1 - tol))[0]
        else:
            # last layer: [z_last0, z_last1]
            idx = np.where((z >= z0 - tol) & (z <= z1 + tol))[0]
        layer_id[idx] = i


    # Role map
    role_id = np.array([_role_to_id(layers[i].role) for i in layer_id], dtype=int)

    # Region id (optional: increments when role changes)
    region_id = np.zeros_like(role_id)
    r = 0
    region_id[0] = r
    for i in range(1, region_id.size):
        if role_id[i] != role_id[i - 1]:
            r += 1
        region_id[i] = r

    # Composition profile x(z) (NaN if not alloy layer)
    x = np.full_like(z, np.nan, dtype=np.float64)
    for i, layer in enumerate(layers):
        if layer.system is None or layer.comp is None:
            continue
        z0, z1 = layer_bounds[i]
        idx = np.where((z >= z0 - 1e-18) & (z <= z1 + 1e-18))[0]
        if layer.comp.mode == "constant":
            x[idx] = float(layer.comp.x0)
        elif layer.comp.mode == "linear":
            xi = float(layer.comp.x0)
            xo = float(layer.comp.x1)
            xi = max(0.0, min(1.0, xi))
            xo = max(0.0, min(1.0, xo))
            x[idx] = xi + (xo - xi) * (z[idx] - z0) / (z1 - z0)
        else:
            raise ValueError(f"unknown composition mode: {layer.comp.mode!r}")

    # Interface table (internal interfaces only: between layers i-1 -> i)
    ifaces = []
    for i in range(1, len(layers)):  # 1 .. len(layers)-1
        zi = float(z_ifaces[i])
        ifaces.append((zi, i - 1, i))

    iface_dtype = np.dtype([("z", np.float64), ("left", np.int64), ("right", np.int64)])
    interfaces = np.array(ifaces, dtype=iface_dtype)
    if interfaces.size == 0:
        interfaces = np.array([], dtype=iface_dtype)


    # Masks
    mask_semiconductor = role_id == _role_to_id("semiconductor")
    mask_oxide = role_id == _role_to_id("oxide")
    mask_metal = role_id == _role_to_id("metal")

    return Geometry1D(
        T=float(stack.T),
        z=z,
        dz=dz,
        Vi=Vi,
        layer_id=layer_id,
        role_id=role_id,
        region_id=region_id,
        x=x,
        interfaces=interfaces,
        mask_semiconductor=mask_semiconductor,
        mask_oxide=mask_oxide,
        mask_metal=mask_metal,
    )

# ---------------------------------------------------------------------
# Convenience: attach stack to geometry (so materials can be resolved)
# ---------------------------------------------------------------------
def attach_stack(geom: Geometry1D, stack: StackSpec) -> Geometry1D:
    """
    Attach the original StackSpec to a Geometry1D instance so
    resolve_material_fields() can access LayerSpec details.
    """
    setattr(geom, "_layers", tuple(stack.layers))
    return geom

# ---------------------------------------------------------------------
# Material field resolution
# ---------------------------------------------------------------------


def _resolve_layer_props(
    layer: LayerSpec, T: float, x_value: Optional[float]
):
    """Helper: pick get_material or get_alloy depending on layer definition."""
    if layer.system is None:
        return get_material(layer.material, T)
    # alloy
    xv = 0.0 if x_value is None or np.isnan(x_value) else float(x_value)
    return get_alloy(layer.system, xv, T)

def resolve_material_fields(geom: Geometry1D) -> MaterialFields:
    """
    Resolve per-node material fields from geometry+materials database.
    This version is aware of layer roles and correctly handles alloys.
    """
    N = geom.z.size
    # Initialize with safe, non-zero defaults
    eps_r = np.ones(N, dtype=np.float64)
    Eg_J = np.ones(N, dtype=np.float64) * 5 * Q  # Default to large gap
    chi_J = np.zeros(N, dtype=np.float64)
    me_dos_rel = np.ones(N, dtype=np.float64)
    mh_dos_rel = np.ones(N, dtype=np.float64)
    e31 = np.zeros(N, dtype=np.float64)
    e33 = np.zeros(N, dtype=np.float64)
    c13 = np.zeros(N, dtype=np.float64)
    c33 = np.ones(N, dtype=np.float64)
    Psp = np.zeros(N, dtype=np.float64)

    if not hasattr(geom, "_layers"):
        raise AttributeError("Geometry1D lacks layer specifications. Call `attach_stack` first.")

    layers: Sequence[LayerSpec] = getattr(geom, "_layers")

    for i, ly in enumerate(layers):
        idx = np.where(geom.layer_id == i)[0]
        if idx.size == 0:
            continue

        # Non-semiconductor layers: fill basic fields, leave piezo/spontaneous at zeros
        if ly.role != "semiconductor":
            props = get_material(ly.material, geom.T)
            eps_r[idx] = props.eps_r
            Eg_J[idx] = props.Eg_J
            chi_J[idx] = props.chi_J
            me_dos_rel[idx] = props.me_dos_rel
            mh_dos_rel[idx] = props.mh_dos_rel
            continue

        # Semiconductors
        if ly.system is not None and ly.comp is not None:
            # Alloy
            if ly.comp.mode == "constant":
                props = get_alloy(ly.system, ly.comp.x0, geom.T)
                eps_r[idx] = props.eps_r
                Eg_J[idx] = props.Eg_J
                chi_J[idx] = props.chi_J
                me_dos_rel[idx] = props.me_dos_rel
                mh_dos_rel[idx] = props.mh_dos_rel
                e31[idx] = props.e31_C_per_m2
                e33[idx] = props.e33_C_per_m2
                c13[idx] = props.c13_Pa
                c33[idx] = props.c33_Pa
                Psp[idx] = props.Psp_C_per_m2
            else:
                # Graded alloy: per-node mixing using geom.x
                for j in idx:
                    props = get_alloy(ly.system, geom.x[j], geom.T)
                    eps_r[j] = props.eps_r
                    Eg_J[j] = props.Eg_J
                    chi_J[j] = props.chi_J
                    me_dos_rel[j] = props.me_dos_rel
                    mh_dos_rel[j] = props.mh_dos_rel
                    e31[j] = props.e31_C_per_m2
                    e33[j] = props.e33_C_per_m2
                    c13[j] = props.c13_Pa
                    c33[j] = props.c33_Pa
                    Psp[j] = props.Psp_C_per_m2
        else:
            # Pure material
            props = get_material(ly.material, geom.T)
            eps_r[idx] = props.eps_r
            Eg_J[idx] = props.Eg_J
            chi_J[idx] = props.chi_J
            me_dos_rel[idx] = props.me_dos_rel
            mh_dos_rel[idx] = props.mh_dos_rel
            e31[idx] = props.e31_C_per_m2
            e33[idx] = props.e33_C_per_m2
            c13[idx] = props.c13_Pa
            c33[idx] = props.c33_Pa
            Psp[idx] = props.Psp_C_per_m2

    return MaterialFields(
        eps_r=_as_c64(eps_r),
        eps=_as_c64(eps_r * EPS0),
        Eg_J=_as_c64(Eg_J),
        chi_J=_as_c64(chi_J),
        me_dos_rel=_as_c64(me_dos_rel),
        mh_dos_rel=_as_c64(mh_dos_rel),
        e31_C_per_m2=_as_c64(e31),
        e33_C_per_m2=_as_c64(e33),
        c13_Pa=_as_c64(c13),
        c33_Pa=_as_c64(c33),
        Psp_C_per_m2=_as_c64(Psp),
    )



def list_interfaces(geom: Geometry1D) -> np.ndarray:
    """Return a copy of the interface table (z_i, left_layer, right_layer)."""
    return np.array(geom.interfaces, copy=True)


# ---------------------------------------------------------------------
# Convenience: attach stack to geometry (so materials can be resolved)
# ---------------------------------------------------------------------


def attach_stack(geom: Geometry1D, stack: StackSpec) -> Geometry1D:
    """
    Attach the original StackSpec to a Geometry1D instance so
    resolve_material_fields() can access LayerSpec details.

    This is a lightweight side-car to keep Geometry1D minimal by default.
    """
    setattr(geom, "_layers", tuple(stack.layers))
    return geom

def list_interfaces_array(geom: Geometry1D) -> np.ndarray:
    """Return interfaces as a (n_ifaces, 3) float64 array [z, left, right]."""
    if geom.interfaces.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    out = np.empty((geom.interfaces.size, 3), dtype=np.float64)
    out[:, 0] = geom.interfaces["z"]
    out[:, 1] = geom.interfaces["left"].astype(np.float64)
    out[:, 2] = geom.interfaces["right"].astype(np.float64)
    return out

