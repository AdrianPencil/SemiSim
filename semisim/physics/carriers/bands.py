# semisim/physics/carriers/bands.py
"""
Band-edge utilities and alignment across 1D stacks.

- SI units throughout.
- Two reference-band models:
    (1) 'affinity'   : E_C^0 = -chi,  E_V^0 = E_C^0 - Eg        (vacuum-level common)
    (2) 'ratio'      : enforce ΔE_C = Q_c * ΔE_g at interfaces  (per-interface ΔE partition)
- Produces reference band edges E_C^0(z), E_V^0(z) (no electrostatic shift),
  and final E_C(z), E_V(z) given potential φ(z):  E_C = E_C^0 - q φ,  E_V = E_V^0 - q φ.
- Provides intrinsic level E_i(z) and intrinsic density n_i(z) for diagnostics.

Public API (stable):
    BandParams
    build_band_params_from_fields(material, layer_id, interfaces,
                                  model="affinity", Qc=None, base_layer=0)
    band_edges_from_potential(phi, params) -> (E_C, E_V)
    intrinsic_level(E_C, E_V, T, me_rel, mh_rel) -> E_i
    intrinsic_density_MB(Eg_J, T, me_rel, mh_rel) -> n_i   # MB approximation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import numpy as np

from semisim.geometry.builder import MaterialFields
from .statistics import Nc_3d, Nv_3d

# ---- constants (SI) ----
Q = 1.602176634e-19    # C
K_B = 1.380649e-23     # J/K

__all__ = [
    "BandParams",
    "build_band_params_from_fields",
    "band_edges_from_potential",
    "intrinsic_level",
    "intrinsic_density_MB",
]


# ---------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------


@dataclass(slots=True)
class BandParams:
    """
    Reference band edges and materials context.

    E_C^0, E_V^0: 'electrostatics-free' edges. Apply potential via E_{C,V} = E_{C,V}^0 - q φ.
    """
    E_C0_J: np.ndarray          # shape (N,), conduction band edge at φ=0 [J]
    E_V0_J: np.ndarray          # shape (N,), valence band edge at φ=0 [J]

    Eg_J: np.ndarray            # band gap [J]
    chi_J: np.ndarray           # electron affinity [J]

    # Optional bookkeeping
    model: Literal["affinity", "ratio"]
    Qc_interfaces: Optional[np.ndarray]  # per-interface Qc used if model == "ratio"
    base_layer: int                      # layer index taken as reference in ratio model


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------


def _affinity_reference(chi_J: np.ndarray, Eg_J: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Electron-affinity model: vacuum level is global; E_C^0 = -chi, E_V^0 = E_C^0 - Eg.
    """
    E_C0 = -np.asarray(chi_J, dtype=np.float64)
    E_V0 = E_C0 - np.asarray(Eg_J, dtype=np.float64)
    return E_C0, E_V0


def _ratio_reference(
    Eg_J: np.ndarray,
    layer_id: np.ndarray,
    interfaces: np.ndarray,
    *,
    Qc: float | Sequence[float] = 0.7,
    base_layer: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Conduction-band-offset ratio model:
        Enforce at each interface i (left layer L, right layer R):
            ΔE_C(i) = Q_c(i) * ΔE_g(i),
            ΔE_V(i) = (1 - Q_c(i)) * ΔE_g(i),
        where ΔE_g(i) = E_{g,R} - E_{g,L} (note: sign matters for which side is wider gap).

    Implementation:
        - Build a piecewise constant E_C^0 shift per layer such that across interface
          the jump equals ΔE_C(i).
        - Set the base layer's E_C^0 reference to zero. All other layers referenced to it.
        - E_V^0 = E_C^0 - E_g layerwise.

    Returns:
        E_C0_J, E_V0_J, Qc_vec
    """
    Eg = np.asarray(Eg_J, dtype=np.float64)
    lid = np.asarray(layer_id, dtype=int)
    ifaces = np.asarray(interfaces, dtype=np.float64)

    n_layers = int(lid.max()) + 1
    # Qc per interface
    if np.isscalar(Qc):
        Qc_vec = np.full(ifaces.shape[0], float(Qc), dtype=np.float64)
    else:
        Qc_vec = np.asarray(Qc, dtype=np.float64)
        if Qc_vec.size != ifaces.shape[0]:
            raise ValueError("Qc sequence length must equal number of interfaces")

    # Layerwise constants S[L]: reference E_C^0 shift in each layer
    S = np.zeros(n_layers, dtype=np.float64)
    # Propagate from base_layer to others using interface constraints
    # Build adjacency from interfaces
    # interfaces rows: (z_i, L_idx, R_idx)
    # We'll do a simple flood fill; stack neighbors with required Δ across each edge.
    visited = np.zeros(n_layers, dtype=bool)
    S[base_layer] = 0.0
    stack = [base_layer]
    # Precompute per-interface ΔEg and mapping edges -> indices
    L_idx = ifaces[:, 1].astype(int)
    R_idx = ifaces[:, 2].astype(int)

    dEg = np.zeros(ifaces.shape[0], dtype=np.float64)
    for k in range(ifaces.shape[0]):
        # Choose representative Eg per layer via any node belonging to layer (constant within layer)
        # We'll take the mean Eg across nodes of that layer to be robust.
        # (Assumes piecewise-constant Eg inside each layer.)
        # Left/right layer masks
        # Note: For large arrays, computing masks inside loop is heavier; here layers are few.
        dEg[k] = float(np.nanmean(Eg[lid == R_idx[k]]) - np.nanmean(Eg[lid == L_idx[k]]))

    # BFS/DFS
    while stack:
        L = stack.pop()
        visited[L] = True
        # Explore interfaces touching L
        touch = np.where((L_idx == L) | (R_idx == L))[0]
        for k in touch:
            iL, iR = int(L_idx[k]), int(R_idx[k])
            # Determine neighbor R (the other side)
            R = iR if iL == L else iL
            if visited[R]:
                continue
            # Required jump across interface (from L to R):
            dEc_req = Qc_vec[k] * dEg[k]  # could be negative
            # If crossing from L->R, we want S[R] - S[L] = dEc_req
            S[R] = S[L] + dEc_req
            stack.append(R)

    # Layerwise constants -> nodal arrays
    E_C0 = np.array([S[i] for i in lid], dtype=np.float64)
    E_V0 = E_C0 - Eg
    return E_C0, E_V0, Qc_vec


def build_band_params_from_fields(
    material: MaterialFields,
    layer_id: np.ndarray,
    interfaces: np.ndarray,
    *,
    model: Literal["affinity", "ratio"] = "affinity",
    Qc: Optional[float | Sequence[float]] = None,
    base_layer: int = 0,
) -> BandParams:
    """
    Build reference band edges (no electrostatic shift) from material fields.

    Parameters
    ----------
    material : MaterialFields
        Output of geometry.resolve_material_fields().
    layer_id : np.ndarray
        Per-node layer index (from Geometry1D).
    interfaces : np.ndarray
        Interface table with columns (z_i, left_layer_idx, right_layer_idx).
    model : "affinity" or "ratio"
        Reference-band model (see module docstring).
    Qc : float or sequence
        Conduction-band offset ratio per interface (used only if model=="ratio").
        If scalar, applied to all interfaces.
    base_layer : int
        Reference layer for the ratio model (E_C^0 set to zero in this layer).

    Returns
    -------
    BandParams
    """
    Eg = np.asarray(material.Eg_J, dtype=np.float64)
    chi = np.asarray(material.chi_J, dtype=np.float64)

    if model == "affinity":
        E_C0, E_V0 = _affinity_reference(chi, Eg)
        Qc_vec = None
    elif model == "ratio":
        if interfaces.size == 0:
            raise ValueError("ratio model requires non-empty interfaces")
            # Guard: if Eg is essentially constant (e.g., Si|Si homojunction),
            # there is no physical band offset. Fall back to affinity reference
            # to avoid introducing an artificial step at the junction.
        if np.allclose(Eg_J.max(), Eg_J.min(), rtol=0.0,
                        atol=1e-6 * float(np.abs(Eg_J).max() + 1.0)):
            E_C0, E_V0 = _affinity_reference(chi_J, Eg_J)
            Qc_vec = None
        else:
            E_C0, E_V0, Qc_vec = _ratio_reference(
                Eg_J, layer_id, interfaces,
                Qc=Qc if Qc is not None else 0.7, base_layer=base_layer
            )
    else:
        raise ValueError(f"unknown band reference model: {model!r}")

    return BandParams(
        E_C0_J=np.ascontiguousarray(E_C0, dtype=np.float64),
        E_V0_J=np.ascontiguousarray(E_V0, dtype=np.float64),
        Eg_J=np.ascontiguousarray(Eg, dtype=np.float64),
        chi_J=np.ascontiguousarray(chi, dtype=np.float64),
        model=model,
        Qc_interfaces=None if model == "affinity" else np.ascontiguousarray(Qc_vec, dtype=np.float64),
        base_layer=int(base_layer),
    )


# ---------------------------------------------------------------------
# Edges under electrostatic potential
# ---------------------------------------------------------------------


def band_edges_from_potential(
    phi_V: np.ndarray,
    params: BandParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply electrostatic shift to reference bands:

        E_C = E_C^0 - q φ,   E_V = E_V^0 - q φ.

    Parameters
    ----------
    phi_V : np.ndarray
        Electrostatic potential [V], shape (N,).
    params : BandParams
        Reference bands and materials context.

    Returns
    -------
    E_C_J, E_V_J : arrays [J] shaped like phi_V
    """
    phi = np.asarray(phi_V, dtype=np.float64)
    E_C = params.E_C0_J - Q * phi
    E_V = params.E_V0_J - Q * phi
    return np.ascontiguousarray(E_C), np.ascontiguousarray(E_V)


# ---------------------------------------------------------------------
# Intrinsic level and intrinsic density (diagnostics & init)
# ---------------------------------------------------------------------


def intrinsic_level(
    E_C_J: np.ndarray,
    E_V_J: np.ndarray,
    T: float | np.ndarray,
    *,
    me_rel: float | np.ndarray,
    mh_rel: float | np.ndarray,
    gvc: float = 1.0,
    gvv: float = 1.0,
) -> np.ndarray:
    """
    Intrinsic Fermi level E_i (J) in the **MB** approximation:

        E_i = (E_C + E_V)/2 + (k_B T / 2) ln(N_v / N_c).

    Useful for initialization and diagnostics; for strong degeneracy use carriers module directly.
    """
    T = np.asarray(T, dtype=np.float64)
    Nc = Nc_3d(T, me_rel, g_s=2.0, g_v=gvc)
    Nv = Nv_3d(T, mh_rel, g_s=2.0, g_v=gvv)
    midgap = 0.5 * (np.asarray(E_C_J, dtype=np.float64) + np.asarray(E_V_J, dtype=np.float64))
    Ei = midgap + 0.5 * K_B * T * np.log(Nv / Nc)
    return np.ascontiguousarray(Ei)


def intrinsic_density_MB(
    Eg_J: np.ndarray,
    T: float | np.ndarray,
    *,
    me_rel: float | np.ndarray,
    mh_rel: float | np.ndarray,
    gvc: float = 1.0,
    gvv: float = 1.0,
) -> np.ndarray:
    """
    Intrinsic carrier density n_i (1/m^3), **MB approximation**:

        n_i = sqrt(N_c N_v) * exp(-E_g / (2 k_B T)).

    This is convenient for sanity checks; the simulator should still use FD stats generally.
    """
    T = np.asarray(T, dtype=np.float64)
    Nc = Nc_3d(T, me_rel, g_s=2.0, g_v=gvc)
    Nv = Nv_3d(T, mh_rel, g_s=2.0, g_v=gvv)
    Eg = np.asarray(Eg_J, dtype=np.float64)
    return np.sqrt(Nc * Nv) * np.exp(-Eg / (2.0 * K_B * T))
