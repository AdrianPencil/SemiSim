# -*- coding: utf-8 -*-
"""
Materials CSV ingest.

Expected columns (example):
  name,eps_rel,k0,alpha_k,sigma0,alpha_sigma,rho,cp0,alpha_cp

All temperature-dependent properties are modeled as:
  k(T)     = k0 * (1 + alpha_k * (T - 300))
  sigma(T) = sigma0 * exp(alpha_sigma * (T - 300))
  cp(T)    = cp0 * (1 + alpha_cp * (T - 300))

We compute volumetric heat capacity on demand as:
  Cv(T) = rho * cp(T)

Units:
  eps_rel [–], k [W/m·K], sigma [S/m], rho [kg/m^3], cp [J/kg·K]
"""
from __future__ import annotations
import csv, math
from pathlib import Path
from typing import Dict
from semisim.models.material import Material

def load_materials(csv_path: Path) -> Dict[str, Material]:
    M: Dict[str, Material] = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            k = lambda T, k0=float(r['k0']), a=float(r['alpha_k']): k0*(1.0 + a*(T-300.0))
            sigma = lambda T, s0=float(r['sigma0']), a=float(r['alpha_sigma']): s0*math.exp(a*(T-300.0))
            cp = lambda T, c0=float(r['cp0']), a=float(r['alpha_cp']): c0*(1.0 + a*(T-300.0))
            M[r['name']] = Material(
                name=r['name'],
                eps_rel=float(r['eps_rel']),
                k=k, sigma=sigma,
                rho=float(r['rho']),
                cp=cp
            )
    return M

def volumetric_Cv(materials: Dict[str, Material], T: float = 300.0):
    """Return {material_name: Cv(T)} with Cv = rho * cp(T)."""
    return {m.name: m.rho * m.cp(T) for m in materials.values()}
