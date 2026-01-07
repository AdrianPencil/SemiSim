# -*- coding: utf-8 -*-
"""
Quasi-static C–V slope basic behavior: monotone in accumulation for ideal MOSCAP.
"""
import numpy as np
from semisim.postprocess.extract_cv import quasi_static

def test_quasi_static_cv():
    Vg = np.linspace(-1, 1, 201)
    # toy monotone Q(V) in accumulation
    Qg = 1e-9 * (Vg + 1.0).clip(min=0.0)
    Vmid, C = quasi_static(Vg, Qg)
    assert (C[Vmid > 0.5] > 0).all()
