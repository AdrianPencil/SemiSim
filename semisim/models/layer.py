# -*- coding: utf-8 -*-
"""
Layer dataclass for a stack element.

Fields:
  - material: cross-reference to Material.name
  - thickness_m: layer thickness in meters
  - area_m2: in-plane area in m^2
  - tbr_m2K_per_W: Thermal Boundary Resistance (defaults to 0.0)
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Layer:
    name: str
    material: str
    thickness_m: float
    area_m2: float
    tbr_m2K_per_W: float = 0.0
