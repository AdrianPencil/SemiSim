# semisim/physics/interfaces.py
"""Interface charge utilities (fixed sheets + dynamic 2DEG for HEMT).

This module centralizes interface-localized sheet charges used by Poisson:
- Fixed sheets (e.g., polarization, fixed interface charge, processed charges)
- Dynamic 2DEG sheets at semiconductor interfaces (e.g., AlGaN/GaN HEMT)

Design notes
------------
* Stateless, functional helpers so Poisson can call them every Newton step.
* No dependency on geometry internals: callers pass interface node indices.
* The 2DEG API returns both the sheet sigma and its diagonal Jacobian term.

Units
-----
- sigma_*  : [C/m^2]
- E*, mu_J : [J]
- T_K      : [K]
"""
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

from .carriers.statistics import sheet_carriers_2d

# ---- physical constants (SI) ----
Q = 1.602176634e-19  # C
K_B = 1.380649e-23   # J/K

__all__ = [
    "InterfaceSheet",
    "compose_sigma_from_sheets",
    "HEMT2DEGParams",
    "hemt_2deg_sigma",
    "hemt_2deg_sigma_and_jacobian",
]


def _c64(a: np.ndarray | float | Iterable[float]) -> np.ndarray:
    """Ensure float64, C-contiguous ndarray."""
    return np.ascontiguousarray(a, dtype=np.float64)


# ---------------------------------------------------------------------
# Fixed sheet charges
# ---------------------------------------------------------------------
@dataclass(slots=True)
class InterfaceSheet:
    """A named fixed sheet charge applied at specific interface nodes.

    Attributes
    ----------
    nodes : (M,) int ndarray
        Node indices where the sheet applies (interface-centered nodes).
    sigma_Cm2 : (M,) float64 ndarray
        Sheet charge density at those nodes [C/m^2]. Sign convention:
        electrons -> negative, holes -> positive.
    name : str
        Descriptive name (e.g., "polarization", "fixed_charge", "oxide_trap").
    """

    nodes: np.ndarray
    sigma_Cm2: np.ndarray
    name: str = "sheet"

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=np.int64)
        self.sigma_Cm2 = _c64(self.sigma_Cm2)
        if self.nodes.ndim != 1 or self.sigma_Cm2.ndim != 1:
            raise ValueError("nodes and sigma_Cm2 must be 1D arrays.")
        if self.nodes.size != self.sigma_Cm2.size:
            raise ValueError("nodes and sigma_Cm2 must have the same length.")


def compose_sigma_from_sheets(
    n_nodes: int, sheets: Sequence[InterfaceSheet] | None
) -> np.ndarray:
    """Accumulate all fixed interface sheets into a node-aligned array.

    Parameters
    ----------
    n_nodes : int
        Total number of grid nodes.
    sheets : sequence of InterfaceSheet or None

    Returns
    -------
    sigma_node : (n_nodes,) ndarray
        Node-localized sheet charges [C/m^2], zeros where no sheet is present.
    """
    sigma_node = np.zeros(int(n_nodes), dtype=np.float64)
    if not sheets:
        return sigma_node
    for sh in sheets:
        if sh.nodes.size == 0:
            continue
        # Sum if multiple sheets touch the same node
        np.add.at(sigma_node, sh.nodes, sh.sigma_Cm2)
    return sigma_node


# ---------------------------------------------------------------------
# Dynamic 2DEG sheets (HEMT)
# ---------------------------------------------------------------------
@dataclass(slots=True)
class HEMT2DEGParams:
    """Minimal parameters for a HEMT 2DEG sheet.

    Attributes
    ----------
    nodes : (M,) ndarray of int
        Interface node indices where 2DEG may form (e.g., AlGaN/GaN).
    Erel_J : float or sequence of float
        Subband energy offsets ΔE_n [J] *relative to local E_C(node)*.
        Use a list for multiple subbands; scalar for a single subband.
        For more advanced models (e.g., Fang–Howard), caller should update
        ΔE_n externally each Newton step based on the local field.
    m2d_rel : float
        DOS effective mass ratio m*/m0 for 2D subbands (electrons).
    g_s, g_v : float
        Spin and valley degeneracy factors (defaults 2, 1).
    """
    nodes: np.ndarray
    Erel_J: float | Sequence[float]
    m2d_rel: float
    g_s: float = 2.0
    g_v: float = 1.0


def hemt_2deg_sigma(
    Ec_J: np.ndarray,
    *,
    params: HEMT2DEGParams,
    mu_J: float,
    T_K: float,
    exp_clip: float = 60.0,
) -> np.ndarray:
    """Compute the dynamic 2DEG sheet sigma at interface nodes.

    Parameters
    ----------
    Ec_J : (N,) ndarray
        Conduction band edge vs node [J].
    params : HEMT2DEGParams
        2DEG configuration (nodes, subband offsets, m*/m0, degeneracy).
    mu_J : float
        Fermi level [J].
    T_K : float
        Temperature [K].
    exp_clip : float
        Exponent clipping for numerical stability.

    Returns
    -------
    sigma2d_node : (N,) ndarray
        Node-aligned sheet charge [C/m^2] (zeros elsewhere).
    """
    N = int(Ec_J.size)
    sigma2d_node = np.zeros(N, dtype=np.float64)

    nodes = np.asarray(params.nodes, dtype=np.int64)
    if nodes.size == 0:
        return sigma2d_node

    dE = np.atleast_1d(_c64(params.Erel_J))  # (Nsub,)
    # Build subband minima Esub for each interface node: E_C(node) + ΔE_n
    Esub = np.stack([Ec_J[n] + dE for n in nodes], axis=0)  # (M, Nsub)

    ns, _ = sheet_carriers_2d(
        Esub_J=Esub,
        mu_J=float(mu_J),
        T=float(T_K),
        m2d_rel=float(params.m2d_rel),
        g_s=float(params.g_s),
        g_v=float(params.g_v),
        exp_clip=float(exp_clip),
    )
    # Electrons => negative sheet charge
    sigma2d = -Q * ns  # (M,)
    np.add.at(sigma2d_node, nodes, sigma2d)
    return sigma2d_node


def hemt_2deg_sigma_and_jacobian(
    Ec_J: np.ndarray,
    *,
    params: HEMT2DEGParams,
    mu_J: float,
    T_K: float,
    exp_clip: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2DEG sheet sigma and its diagonal Jacobian wrt potential.

    The derivative uses:
        eta = (mu - Esub) / (kT)
        ns = g2D * kT * sum_n ln(1 + exp(eta_n))
        d ns / d eta = g2D * kT * sum_n logistic(eta_n)
        d eta / d phi = +q / (kT),     d Esub / d phi = -q  (via E_C = E_C0 - q(phi-phi_b))

    Therefore:
        d sigma / d phi = -q * d ns/d eta * (q / (kT))

    Returns
    -------
    sigma2d_node : (N,) ndarray
        Node-aligned 2DEG sheet [C/m^2].
    dsigma_dphi_diag : (N,) ndarray
        Diagonal contribution [C/(m^2·V)] to add to Poisson Jacobian at nodes.
    """
    N = int(Ec_J.size)
    sigma2d_node = np.zeros(N, dtype=np.float64)
    dsigma_dphi_diag = np.zeros(N, dtype=np.float64)

    nodes = np.asarray(params.nodes, dtype=np.int64)
    if nodes.size == 0:
        return sigma2d_node, dsigma_dphi_diag

    dE = np.atleast_1d(_c64(params.Erel_J))
    Esub = np.stack([Ec_J[n] + dE for n in nodes], axis=0)  # (M, Nsub)

    ns, d_ns_deta = sheet_carriers_2d(
        Esub_J=Esub,
        mu_J=float(mu_J),
        T=float(T_K),
        m2d_rel=float(params.m2d_rel),
        g_s=float(params.g_s),
        g_v=float(params.g_v),
        exp_clip=float(exp_clip),
    )

    sigma2d = -Q * ns
    np.add.at(sigma2d_node, nodes, sigma2d)

    # d sigma / d phi (per interface node) — all subbands accounted for in d_ns_deta
    dsigma = -(Q) * d_ns_deta * (Q / (K_B * float(T_K)))  # [C/(m^2·V)]
    np.add.at(dsigma_dphi_diag, nodes, dsigma)

    return sigma2d_node, dsigma_dphi_diag
