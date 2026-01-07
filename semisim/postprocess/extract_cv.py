# -*- coding: utf-8 -*-
"""
C–V extractors:
  - quasi_static(): numerical dQ/dV from a DC sweep
  - small_signal_placeholder(): AC linearization hook (wire to solver later)
"""
import numpy as np

def quasi_static(Vg: np.ndarray, Qg: np.ndarray) -> np.ndarray:
    """Central-difference slope -> C(V). Arrays must be monotonically swept."""
    dV = np.diff(Vg)
    dQ = np.diff(Qg)
    C_mid = dQ / np.where(np.abs(dV) < 1e-30, np.sign(dV)*1e-30, dV)
    # return mid-point V and C as same-length arrays (pad ends)
    V_mid = 0.5*(Vg[1:] + Vg[:-1])
    return V_mid, C_mid

def small_signal_placeholder(Yjw: complex, omega: float) -> float:
    """Given terminal admittance at ω, return capacitance = Im(Y)/ω."""
    return float(np.imag(Yjw))/max(omega, 1e-30)
