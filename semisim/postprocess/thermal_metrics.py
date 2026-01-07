# -*- coding: utf-8 -*-
"""
Thermal KPIs: Rθ and Zθ(ω) placeholders.
"""
from __future__ import annotations
import numpy as np

def rtheta(delta_T: float, power_W: float) -> float:
    return delta_T / max(power_W, 1e-30)
