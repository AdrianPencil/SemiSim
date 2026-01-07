# C:\Users\A Asheri\PROJECT_ROOT\semisim\tests\test_band_edges_shift.py
# -*- coding: utf-8 -*-
"""
Sanity: MOSCAP band edges move by q*Δφ and align with χ, Eg.
"""
import math
from semisim.physics.bands import BandParams, band_edges

def test_band_edges_shift():
    bp = BandParams(chi_J=4.05*1.602e-19, Eg_J=1.12*1.602e-19)  # ~Si numbers
    Ec0, Ev0 = band_edges(phi_V=0.0, bp=bp)
    Ec1, Ev1 = band_edges(phi_V=+0.1, bp=bp)  # +0.1 V lowers Ec by q*0.1
    assert math.isclose(Ec1, Ec0 - 1.602e-20, rel_tol=0, abs_tol=1e-22)
    assert math.isclose((Ec1-Ev1), bp.Eg_J, rel_tol=1e-12)
