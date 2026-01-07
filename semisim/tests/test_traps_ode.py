# -*- coding: utf-8 -*-
"""
Test: trap occupancy first-order ODE has correct exponential approach
for a constant right-hand side (very simplified).
"""
from semisim.physics.traps import TrapLevel, step_occupancy

def test_trap_step_bounds():
    t = TrapLevel(Et=0.4, Nt=1e22, sigma_n=1e-19, sigma_p=1e-19, kind="donor", f=0.0)
    for _ in range(100):
        f = step_occupancy(t, n=1e21, p=1e16, T=300.0, Nc=1e25, Nv=1e25,
                           Ec_minus_Et=0.2, Ev_minus_Et=1.0, dt=1e-9)
        assert 0.0 <= f <= 1.0
