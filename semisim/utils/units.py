# -*- coding: utf-8 -*-
"""
Unit helpers (very small; expand if you add nondimensionalization).
"""
from __future__ import annotations

def clamp(x, lo, hi): return max(lo, min(hi, x))
