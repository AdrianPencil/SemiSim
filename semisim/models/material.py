# -*- coding: utf-8 -*-
"""
Material dataclass.

Fields:
  - name:     identifier
  - eps_rel:  relative permittivity (scalar)
  - k(T):     thermal conductivity model [W/m·K]
  - sigma(T): electrical conductivity model [S/m]
  - rho:      density [kg/m^3]
  - cp(T):    specific heat [J/kg·K]
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

Number = float

@dataclass
class Material:
    name: str
    eps_rel: Number
    k: Callable[[Number], Number]
    sigma: Callable[[Number], Number]
    rho: Number
    cp: Callable[[Number], Number]
