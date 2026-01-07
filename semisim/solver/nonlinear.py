# -*- coding: utf-8 -*-
"""
Nonlinear coupling drivers.

Two strategies:
  - Gummel (staggered): solve φ → (n,p) → T → update coefficients → repeat
  - Newton–Krylov: solve all unknowns together (faster if well-preconditioned)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any

@dataclass
class SolverOptions:
    atol: float = 1e-8
    rtol: float = 1e-6
    max_iter: int = 50
    method: str = "newton"  # or "gummel"

def gummel_cycle(cfg) -> Dict[str, Any]:
    # placeholder structure; wire into your assembly & linear solver
    return {"converged": True, "iterations": 3}

def newton_solve(cfg) -> Dict[str, Any]:
    # placeholder Newton; wire residual/Jacobian callbacks + linear solves
    return {"converged": True, "iterations": 5}
