# -*- coding: utf-8 -*-
r"""
Heat equation with sources.

Strong form:
  ρ c_p ∂T/∂t = ∇·(k ∇T) + H_Joule + H_recomb + H_TE

Here we just expose the standard source decomposition.
"""
from __future__ import annotations

def joule_heat(Jx: float, Jy: float, Jz: float, Ex: float, Ey: float, Ez: float) -> float:
    return Jx*Ex + Jy*Ey + Jz*Ez
