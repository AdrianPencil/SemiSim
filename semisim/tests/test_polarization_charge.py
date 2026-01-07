# -*- coding: utf-8 -*-
"""
Test: interface sheet charge from polarization jump.
σ_pol = (P2 - P1)·n̂
"""
from semisim.physics.polarization import interface_sheet_charge

def test_interface_sheet_charge():
    P1 = (0.0, 0.0, 0.05)  # C/m^2
    P2 = (0.0, 0.0, 0.15)
    n  = (0.0, 0.0, 1.0)
    sigma = interface_sheet_charge(P1, P2, n)
    assert abs(sigma - 0.10) < 1e-12
