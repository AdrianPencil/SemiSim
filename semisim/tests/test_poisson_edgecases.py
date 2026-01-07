# semisim/tests/test_poisson_edgecases.py
"""Pytest-style edge-case checks for the 1D Poisson solver.
Run with:  pytest -q
"""
from __future__ import annotations

import numpy as np

from physics.poisson import (
    Material,
    Doping,
    Domain,
    PoissonSetup,
    solve_poisson_1d,
    compute_residual_inf,
)

Q = 1.602176634e-19


def _mk_si():
    return Material(eps_rel=11.7, ni=1.0e16, Eg_J=1.12 * Q)


def test_depletion_dirichlet_dirichlet():
    mat = _mk_si()
    dop = Doping(NA=1e23)
    dom = Domain(L=200e-9, N=400)
    setup = PoissonSetup(T=300.0, mat=mat, dop=dop, dom=dom, phi_s=0.4, phi_b=0.0)
    out = solve_poisson_1d(setup)
    res = compute_residual_inf(out.phi, setup)
    assert res < 1e-6
    # monotonic potential between boundaries in this setup
    assert np.all(np.diff(out.phi) < 1e-9)  # non-increasing


def test_accumulation_n_type():
    mat = _mk_si()
    dop = Doping(ND=5e23)
    dom = Domain(L=100e-9, N=300)
    setup = PoissonSetup(T=300.0, mat=mat, dop=dop, dom=dom, phi_s=-0.3)
    out = solve_poisson_1d(setup)
    res = compute_residual_inf(out.phi, setup)
    assert res < 1e-6
    # near surface electrons dominate
    assert out.n[0] > out.p[0]


def test_strong_inversion_with_clipping():
    mat = _mk_si()
    dop = Doping(NA=1e23)
    dom = Domain(L=150e-9, N=500)
    setup = PoissonSetup(T=300.0, mat=mat, dop=dop, dom=dom, phi_s=0.8, exp_clip=60.0)
    out = solve_poisson_1d(setup)
    res = compute_residual_inf(out.phi, setup)
    assert res < 1e-5
    # inversion at the surface: n >> p
    assert out.n[0] > 10.0 * out.p[0]


def test_near_intrinsic():
    mat = Material(eps_rel=11.7, ni=1.0e14, Eg_J=1.12 * Q)
    dop = Doping()
    dom = Domain(L=300e-9, N=600)
    setup = PoissonSetup(T=300.0, mat=mat, dop=dop, dom=dom, phi_s=0.05)
    out = solve_poisson_1d(setup)
    res = compute_residual_inf(out.phi, setup)
    assert res < 1e-6


def test_bulk_neumann_zero_field():
    mat = _mk_si()
    dop = Doping(NA=2e23)
    dom = Domain(L=200e-9, N=400)
    setup = PoissonSetup(T=300.0, mat=mat, dop=dop, dom=dom, phi_s=0.4, bulk_bc="NeumannZero")
    out = solve_poisson_1d(setup)
    res = compute_residual_inf(out.phi, setup)
    assert res < 1e-6
    # zero-field at bulk: discrete derivative ~ 0
    phi = out.phi
    assert abs(phi[-1] - phi[-2]) < 1e-9


def test_minimal_grid_n3():
    mat = _mk_si()
    dop = Doping(NA=1e22)
    dom = Domain(L=50e-9, N=3)
    setup = PoissonSetup(T=300.0, mat=mat, dop=dop, dom=dom, phi_s=0.2)
    out = solve_poisson_1d(setup)
    res = compute_residual_inf(out.phi, setup)
    assert res < 1e-4
