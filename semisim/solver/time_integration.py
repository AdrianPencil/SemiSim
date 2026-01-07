# -*- coding: utf-8 -*-
"""
Time integration utilities (BDF1/θ-method with adaptivity placeholder).
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ThetaMethod:
    theta: float = 1.0  # 1.0: backward Euler

def step(theta: ThetaMethod, y_n, t_n, dt, residual, jacobian):
    # Newton on R(y_{n+1}) = 0; placeholder
    return y_n, t_n + dt
