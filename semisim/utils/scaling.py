"""
semisim/utils/scaling.py

Lightweight nondimensional scaling helpers for Poisson/continuity problems.
Keep this in utils (not physics) so solvers and assemblers can use it without
pulling in extra physics modules.

Typical usage:
    sc = Scaling.from_problem(z=problem.z, T_K=problem.T_K)
    phi_nd = sc.scale_phi(phi)
    resid_nd = sc.scale_residual(resid)
    a_nd, b_nd, c_nd = sc.scale_tridiagonal(a, b, c)

All functions are safe no-ops if you pass unity scales explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


Q = 1.602176634e-19  # C
K_B = 1.380649e-23   # J/K
EPS0 = 8.8541878128e-12  # F/m


def _to_f64(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return a


@dataclass
class Scaling:
    """Reference scales for nondimensionalization."""
    V_ref: float   # volts
    N_ref: float   # m^-3
    L_ref: float   # meters
    eps_ref: float # F/m

    def __post_init__(self) -> None:
        # Guard against zeros
        self.V_ref = float(self.V_ref or 1.0)
        self.N_ref = float(self.N_ref or 1.0)
        self.L_ref = float(self.L_ref or 1.0)
        self.eps_ref = float(self.eps_ref or EPS0)

    # ---- Scalar helpers -----------------------------------------------------

    @staticmethod
    def thermal_voltage(T_K: float) -> float:
        return float(K_B * float(T_K) / Q)

    # ---- Variable scaling (in-place safe) ----------------------------------

    def scale_phi(self, phi: np.ndarray) -> np.ndarray:
        return _to_f64(phi) / self.V_ref

    def unscale_phi(self, phi_nd: np.ndarray) -> np.ndarray:
        return _to_f64(phi_nd) * self.V_ref

    def scale_density(self, n: np.ndarray) -> np.ndarray:
        return _to_f64(n) / self.N_ref

    def unscale_density(self, n_nd: np.ndarray) -> np.ndarray:
        return _to_f64(n_nd) * self.N_ref

    def scale_residual(self, r: np.ndarray) -> np.ndarray:
        # Poisson residual has units of C/m^3; scale by q*N_ref so order-unity.
        return _to_f64(r) / (Q * self.N_ref)

    def unscale_residual(self, r_nd: np.ndarray) -> np.ndarray:
        return _to_f64(r_nd) * (Q * self.N_ref)

    # ---- Jacobian scaling ---------------------------------------------------

    def scale_tridiagonal(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale Poisson Jacobian so that diagonal magnitudes are O(1):
          J has units (F/m) / m^2 = F/m^3 effectively; scale by eps_ref/L_ref^2.
        """
        s = self.eps_ref / (self.L_ref ** 2)
        return _to_f64(a) / s, _to_f64(b) / s, _to_f64(c) / s

    def unscale_tridiagonal(self, a_nd: np.ndarray, b_nd: np.ndarray, c_nd: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = self.eps_ref / (self.L_ref ** 2)
        return _to_f64(a_nd) * s, _to_f64(b_nd) * s, _to_f64(c_nd) * s

    # ---- Builders -----------------------------------------------------------

    @classmethod
    def from_problem(
        cls,
        *,
        z: np.ndarray,
        T_K: float,
        eps_bg: float | None = None,
        N_ref: float = 1e21,
        V_ref: float | None = None,
        L_ref: float | None = None,
    ) -> "Scaling":
        """
        Construct a sensible default Scaling from problem arrays.

        Parameters
        ----------
        z : node coordinates [m]
        T_K : temperature [K]
        eps_bg : background permittivity for scaling [F/m] (default EPS0)
        N_ref : density scale [m^-3] (default 1e21, tweak per device)
        V_ref : potential scale [V] (default = V_T)
        L_ref : length scale [m] (default = domain length)
        """
        z = _to_f64(z)
        L = float(z[-1] - z[0]) if z.size >= 2 else 1.0
        Vt = cls.thermal_voltage(T_K)
        return cls(
            V_ref=float(V_ref or Vt),
            N_ref=float(N_ref),
            L_ref=float(L_ref or L),
            eps_ref=float(eps_bg or EPS0),
        )
    
# ---------------------------------------------------------------------------
# Lightweight facade used by the assembler/solvers
# ---------------------------------------------------------------------------
@dataclass
class PoissonScales:
    """Minimal scale set the solvers/assembler need."""
    V_scale: float   # [V]   potential scale (use ~V_T)
    R_scale: float   # [C/m^3] residual scale (use ~q*N_ref)

def compute_poisson_scales(
    *,
    z: np.ndarray,
    Vi: np.ndarray,
    eps_face: np.ndarray,
    ND: np.ndarray,
    NA: np.ndarray,
    rho_extra: np.ndarray,
    T_K: float,
) -> PoissonScales:
    """
    Choose robust scales without touching physics kernels:
      - V_scale = thermal voltage ~ kT/q
      - N_ref   = max(|ND|,|NA|,|rho_extra|/q, 1e20 m^-3)  (floor avoids zeros)
      - R_scale = q * N_ref  (Poisson residual is in C/m^3)
    """
    z = _to_f64(z); Vi = _to_f64(Vi)
    ND = _to_f64(ND); NA = _to_f64(NA); rho_extra = _to_f64(rho_extra)

    V_scale = Scaling.thermal_voltage(float(T_K))
    # Build a safe density reference
    q = Q
    dens_from_rho = np.max(np.abs(rho_extra)) / q if rho_extra.size else 0.0
    dens_candidates = [
        float(np.max(np.abs(ND))) if ND.size else 0.0,
        float(np.max(np.abs(NA))) if NA.size else 0.0,
        float(dens_from_rho),
        1e20,  # floor to avoid tiny scales; tweak per device class if needed
    ]
    N_ref = max(dens_candidates)
    R_scale = q * N_ref
    return PoissonScales(V_scale=float(V_scale), R_scale=float(R_scale))

