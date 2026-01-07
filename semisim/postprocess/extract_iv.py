# -*- coding: utf-8 -*-
"""
DC I–V metrics.
"""
from __future__ import annotations
import numpy as np

def ron(v: np.ndarray, i: np.ndarray, v_window: tuple[float,float] | None = None) -> float:
    """
    Extract R_on as the inverse slope around a window (or entire range).
    """
    if v_window:
        mask = (v >= v_window[0]) & (v <= v_window[1])
        v, i = v[mask], i[mask]
    p = np.polyfit(v, i, 1)
    slope = p[0]
    return 1.0 / max(slope, 1e-30)
