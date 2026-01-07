# semisim/boundaries/electrical.py
"""
Electrical boundary conditions for drift–diffusion (1D, SI units).

This module provides **contact models** for the carrier continuity equations:
    - Ohmic (quasi-Fermi pinning) via surface recombination velocities (SRV).
    - Schottky (thermionic emission limited) via effective emission velocities.
    - Insulating / symmetry (zero carrier flux).

All BCs are returned as **outward-normal current densities** at the boundary:
    J_n^out, J_p^out  [A/m^2],
together with their **Jacobians** w.r.t. the local state (n, p):
    ∂J_n^out/∂n, ∂J_n^out/∂p, ∂J_p^out/∂n, ∂J_p^out/∂p.

Your continuity residual can then use the same face-divergence pattern as Poisson:
    R_i ← (F_{i+1/2} - F_{i-1/2})/V_i + ...,
with the *boundary face flux* F at a domain boundary taken from J^out at that boundary.

Design
------
- **Ohmic contact** (SRV form, covers "Dirichlet" as S→∞):
      J_n^out =  q S_n (n - n_eq),       ∂/∂n = q S_n
      J_p^out = -q S_p (p - p_eq),       ∂/∂p = -q S_p
  where (n_eq, p_eq) are **equilibrium** carrier densities consistent with the
  contact Fermi level μ_M and the local band edges (E_C, E_V).

- **Schottky contact** (thermionic-emission-limited Robin BC, unified SRV form):
      J_n^out =  q s_n^TE (n - n_eq^M),   s_n^TE ≈ α_n v_th,n exp(-Φ_Bn/kT)
      J_p^out = -q s_p^TE (p - p_eq^M),   s_p^TE ≈ α_p v_th,p exp(-Φ_Bp/kT)
  Here Φ_Bn (Φ_Bp) are electron (hole) barriers; α_* are tuning factors (≈1).

- **Insulating / symmetry**:
      J_n^out = 0,   J_p^out = 0.

All formulas are **vectorized-safe**, but are typically used at a single boundary node.

Public API
----------
    ContactKind
    OhmicBC, SchottkyBC, InsulatingBC
    Contact
    equilibrium_np_from_mu(...)
    schottky_emission_velocity(...)
    barrier_from_workfunction(...)
    eval_contact_flux(...)

Notes
-----
- Units: SI throughout. q>0. Temperatures in K; energies in J.
- We rely on carriers/statistics for equilibrium n,p (FD/MB) at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from ..physics.carriers.statistics import carriers_3d, Nc_3d, Nv_3d

# ---- constants (SI) ----
Q = 1.602176634e-19       # C
K_B = 1.380649e-23        # J/K
M0 = 9.1093837015e-31     # kg (electron mass)

__all__ = [
    "ContactKind",
    "OhmicBC",
    "SchottkyBC",
    "InsulatingBC",
    "Contact",
    "equilibrium_np_from_mu",
    "schottky_emission_velocity",
    "barrier_from_workfunction",
    "eval_contact_flux",
]


# ---------------------------------------------------------------------
# Contact descriptors
# ---------------------------------------------------------------------

ContactKind = Literal["ohmic", "schottky", "insulating"]


@dataclass(slots=True)
class OhmicBC:
    """
    Ohmic contact (quasi-Fermi pinning) modeled with SRVs.

    S_n, S_p   : surface recombination velocities [m/s].
                 Large values (e.g., 1e7–1e9) enforce near-Dirichlet behavior.
    mu_M_J     : metal Fermi level [J] at the contact (to compute n_eq, p_eq).
                 If None, you must supply (n_eq, p_eq) directly to eval_contact_flux(...).
    stats      : "FD" or "MB" for equilibrium evaluation.
    """
    S_n_m_per_s: float
    S_p_m_per_s: float
    mu_M_J: Optional[float] = None
    stats: Literal["FD", "MB"] = "FD"


@dataclass(slots=True)
class SchottkyBC:
    """
    Schottky (thermionic emission limited) contact.

    Parameters
    ----------
    phi_Bn_J, phi_Bp_J : barrier heights [J] (electron / hole). Use None to disable a carrier.
    alpha_n, alpha_p   : emission prefactors (dimensionless, ~1) multiplying thermal velocity.
    mu_M_J             : metal Fermi level [J] to compute n_eq^M, p_eq^M at the interface.
    stats              : "FD" or "MB" for equilibrium evaluation.

    Effective emission velocities:
        s_n^TE = alpha_n * v_th,n * exp(-phi_Bn / (k T))
        s_p^TE = alpha_p * v_th,p * exp(-phi_Bp / (k T))
    where v_th ≈ sqrt(3 k T / m*).
    """
    phi_Bn_J: Optional[float]
    phi_Bp_J: Optional[float]
    alpha_n: float = 1.0
    alpha_p: float = 1.0
    mu_M_J: Optional[float] = None
    stats: Literal["FD", "MB"] = "FD"


@dataclass(slots=True)
class InsulatingBC:
    """Insulating / symmetry boundary (zero carrier flux)."""
    pass


@dataclass(slots=True)
class Contact:
    """
    Unified contact container.
      side: "left" or "right" (used only by caller to place at the correct boundary).
      kind: "ohmic" | "schottky" | "insulating".
      model: OhmicBC | SchottkyBC | InsulatingBC
    """
    side: Literal["left", "right"]
    kind: ContactKind
    model: object


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _c64(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def equilibrium_np_from_mu(
    E_C_J: float,
    E_V_J: float,
    mu_J: float,
    T_K: float,
    *,
    me_rel: float,
    mh_rel: float,
    gvc: float = 1.0,
    gvv: float = 1.0,
    stats: Literal["FD", "MB"] = "FD",
    exp_clip: float = 60.0,
) -> Tuple[float, float]:
    """
    Equilibrium carrier densities (n_eq, p_eq) at given μ and band edges.

    Uses the same statistics as the rest of the codebase (FD/MB).
    """
    # Use length-1 arrays to reuse the vectorized carriers_3d.
    E_C = np.array([E_C_J], dtype=np.float64)
    E_V = np.array([E_V_J], dtype=np.float64)
    n, p = carriers_3d(
        E_C, E_V, float(mu_J), float(T_K),
        me_rel=float(me_rel), mh_rel=float(mh_rel),
        gvc=float(gvc), gvv=float(gvv),
        stats=stats, exp_clip=exp_clip,
    )
    return float(n[0]), float(p[0])


def schottky_emission_velocity(
    T_K: float,
    m_rel: float,
    phi_B_J: float,
    alpha: float = 1.0,
) -> float:
    """
    Effective thermionic emission velocity:
        s_TE = alpha * v_th * exp(-phi_B / kT),   v_th ≈ sqrt(3 k T / m*).

    Notes
    -----
    - This is a **velocity** [m/s], not a current density. The boundary flux is
      J^out = ± q s_TE (carrier - carrier_eq_M).
    - 'alpha' absorbs Richardson mismatch and detailed velocity prefactors.
    """
    if phi_B_J is None:
        return 0.0
    v_th = np.sqrt(3.0 * K_B * float(T_K) / (float(m_rel) * M0))
    return float(alpha) * float(v_th) * float(np.exp(-phi_B_J / (K_B * float(T_K))))


def barrier_from_workfunction(
    phi_M_eV: float,
    chi_J: float,
    *,
    for_carrier: Literal["electron", "hole"] = "electron",
    Eg_J: Optional[float] = None,
) -> float:
    """
    Schottky barrier from metal work function and semiconductor affinity.

    Electron barrier:
        Φ_Bn = (φ_M [J]) - χ
    Hole barrier (needs band gap Eg):
        Φ_Bp = Eg - Φ_Bn

    Inputs
    ------
    phi_M_eV : metal work function [eV] (converted to J).
    chi_J    : electron affinity [J].
    Eg_J     : band gap [J] (required when for_carrier="hole").

    Returns
    -------
    barrier [J] (non-negative).
    """
    phi_M_J = float(phi_M_eV) * Q
    phi_Bn = max(0.0, phi_M_J - float(chi_J))
    if for_carrier == "electron":
        return phi_Bn
    if Eg_J is None:
        raise ValueError("Eg_J is required to compute Φ_Bp")
    phi_Bp = max(0.0, float(Eg_J) - phi_Bn)
    return phi_Bp


# ---------------------------------------------------------------------
# Boundary evaluation
# ---------------------------------------------------------------------


def eval_contact_flux(
    *,
    kind: ContactKind,
    E_C_J: float,
    E_V_J: float,
    n_m3: float,
    p_m3: float,
    T_K: float,
    me_rel: float,
    mh_rel: float,
    gvc: float = 1.0,
    gvv: float = 1.0,
    # Models (provide the one consistent with 'kind')
    ohmic: Optional[OhmicBC] = None,
    schottky: Optional[SchottkyBC] = None,
    # Optional explicit equilibrium (overrides μ-based computation if provided)
    n_eq_override: Optional[float] = None,
    p_eq_override: Optional[float] = None,
) -> Tuple[float, float, float, float, float, float]:
    """
    Evaluate outward-normal fluxes at a boundary node and their Jacobians.

    Returns
    -------
    (Jn_out, Jp_out, dJn_dn, dJn_dp, dJp_dn, dJp_dp)

    Conventions
    -----------
    - "Outward" means **out of the simulation domain** at that boundary.
    - For the continuity residual assembled as div(J)/q + ..., use J_out
      directly as the boundary face flux (with your face orientation).
    """
    kind = str(kind).lower()  # normalize

    # Defaults: zero flux (insulating)
    Jn_out = 0.0
    Jp_out = 0.0
    dJn_dn = 0.0
    dJn_dp = 0.0
    dJp_dn = 0.0
    dJp_dp = 0.0

    if kind == "insulating":
        return Jn_out, Jp_out, dJn_dn, dJn_dp, dJp_dn, dJp_dp

    if kind == "ohmic":
        if ohmic is None:
            raise ValueError("Ohmic BC requires 'ohmic' model")
        # Equilibrium densities at the contact
        if (n_eq_override is not None) and (p_eq_override is not None):
            n_eq, p_eq = float(n_eq_override), float(p_eq_override)
        elif ohmic.mu_M_J is not None:
            n_eq, p_eq = equilibrium_np_from_mu(
                E_C_J, E_V_J, float(ohmic.mu_M_J), float(T_K),
                me_rel=float(me_rel), mh_rel=float(mh_rel),
                gvc=float(gvc), gvv=float(gvv), stats=ohmic.stats,
            )
        else:
            raise ValueError("Ohmic BC needs either (mu_M_J) or explicit (n_eq, p_eq)")

        # SRV Robin boundary (outward)
        Jn_out = Q * float(ohmic.S_n_m_per_s) * (float(n_m3) - n_eq)
        Jp_out = -Q * float(ohmic.S_p_m_per_s) * (float(p_m3) - p_eq)
        dJn_dn = Q * float(ohmic.S_n_m_per_s)
        dJp_dp = -Q * float(ohmic.S_p_m_per_s)
        return Jn_out, Jp_out, dJn_dn, dJn_dp, dJp_dn, dJp_dp

    if kind == "schottky":
        if schottky is None:
            raise ValueError("Schottky BC requires 'schottky' model")
        if schottky.mu_M_J is None:
            raise ValueError("Schottky BC needs metal μ (mu_M_J) to form n_eq^M, p_eq^M")

        # Contact-side equilibrium (metal-referenced) densities
        n_eq, p_eq = equilibrium_np_from_mu(
            E_C_J, E_V_J, float(schottky.mu_M_J), float(T_K),
            me_rel=float(me_rel), mh_rel=float(mh_rel),
            gvc=float(gvc), gvv=float(gvv), stats=schottky.stats,
        )

        # Emission velocities (Robin coefficients)
        s_n = 0.0
        s_p = 0.0
        if schottky.phi_Bn_J is not None:
            s_n = schottky_emission_velocity(T_K, me_rel, float(schottky.phi_Bn_J), alpha=float(schottky.alpha_n))
        if schottky.phi_Bp_J is not None:
            s_p = schottky_emission_velocity(T_K, mh_rel, float(schottky.phi_Bp_J), alpha=float(schottky.alpha_p))

        # Outward fluxes
        Jn_out = Q * s_n * (float(n_m3) - n_eq)
        Jp_out = -Q * s_p * (float(p_m3) - p_eq)
        dJn_dn = Q * s_n
        dJp_dp = -Q * s_p
        return Jn_out, Jp_out, dJn_dn, dJn_dp, dJp_dn, dJp_dp

    raise ValueError(f"Unknown contact kind: {kind!r}")


# ---------------------------------------------------------------------
# Lightweight builders (convenience)
# ---------------------------------------------------------------------

def make_contact_left_schottky(
    *,
    phi_M_eV: float,
    chi_J: float,
    Eg_J: float,
    mu_M_J: float,
    stats: Literal["FD", "MB"] = "FD",
    alpha_n: float = 1.0,
    alpha_p: float = 1.0,
) -> Contact:
    """Left boundary = Schottky gate (vertical HEMT slice)."""
    phi_Bn = barrier_from_workfunction(phi_M_eV, chi_J, for_carrier="electron")
    phi_Bp = barrier_from_workfunction(phi_M_eV, chi_J, for_carrier="hole", Eg_J=Eg_J)
    model = SchottkyBC(
        phi_Bn_J=float(phi_Bn),
        phi_Bp_J=float(phi_Bp),
        alpha_n=float(alpha_n),
        alpha_p=float(alpha_p),
        mu_M_J=float(mu_M_J),
        stats=stats,
    )
    return Contact(side="left", kind="schottky", model=model)


def make_contact_right_insulating() -> Contact:
    """Right boundary = insulating (deep buffer)."""
    return Contact(side="right", kind="insulating", model=InsulatingBC())


# expose new helpers
__all__.extend(["make_contact_left_schottky", "make_contact_right_insulating"])