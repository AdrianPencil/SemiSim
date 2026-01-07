# -*- coding: utf-8 -*-
"""
Linear solver backends (sparse direct, Krylov, preconditioners).
Keep the API tiny so you can swap SciPy/PETSc/etc. later.
"""
from __future__ import annotations
from typing import Any

def solve_linear(A: Any, b: Any) -> Any:
    # replace with actual linear algebra; A could be CSR, etc.
    return "x_solution"
