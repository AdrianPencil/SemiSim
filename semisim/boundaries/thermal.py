# -*- coding: utf-8 -*-
"""
Thermal boundary conditions.

Types:
  - Dirichlet: T = T_sink
  - Neumann:   n·(k∇T) = q''
  - Robin:     -k ∂T/∂n = h (T - T_inf)
  - TBR jump:  ΔT = q'' · TBR  (at interface)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal

Kind = Literal["dirichlet","neumann","robin"]

@dataclass
class ThermalBC:
    name: str
    kind: Kind
    value: Callable[[float], float]
    h: float | None = None
