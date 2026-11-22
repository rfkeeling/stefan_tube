"""Validation conditions of Appendix C5: T = 298.15 K, p = 101325 Pa, L = 0.10 m,
z_pivot = L/2, ambient x_H2O = 0.010, N_grid = 61."""
import numpy as np
import pytest

from stefan_tube import StefanTubeBVP, StefanTubeSolver

L = 0.10


@pytest.fixture(scope="module")
def solver():
    return StefanTubeSolver()


@pytest.fixture(scope="module")
def base(solver):
    return solver.solve(0.010, L / 2, N_grid=61, t_factor=20.0)


def test_steady_state_reference(base):
    assert base["F_O2"] == pytest.approx(1.6313179e-03, rel=1e-6)
    assert base["x_CO2_wall"] == pytest.approx(0.241688, rel=1e-5)
    assert base["F_O2"] + base["F_CO2"] == pytest.approx(0.0, abs=1e-12)


def test_nitrogen_flux_vanishes_without_being_imposed(base):
    assert np.max(np.abs(base["F_N2"])) < 1e-15


def test_cross_flux_drag_removal(solver, base):
    warm = (base["x_O2"], base["x_H2O"], base["x_CO2"], base["z"])
    r = solver.solve(0.010, L / 2, no_cfd=True, warm_start=warm, t_factor_relax=8.0)
    assert r["F_O2"] == pytest.approx(1.656515e-03, rel=1e-6)
    assert r["x_CO2_wall"] == pytest.approx(0.24542, abs=5e-6)


def test_no_pivot_matches_bvp(solver):
    r_tm = solver.solve(0.010, L, N_grid=61, t_factor=20.0)
    r_bvp = StefanTubeBVP().solve(0.010, solver.chi_O2d_0, solver.x_sat, 0.0)
    assert r_tm["F_O2"] == pytest.approx(r_bvp["F_O2"], rel=2e-4)
    assert r_tm["x_CO2_wall"] == pytest.approx(r_bvp["x_CO2_wall"], rel=2e-4)


def test_outer_domain_bvp_reproduces_pivot_co2(solver, base):
    z = base["z"]
    i = int(np.argmin(np.abs(z - L / 2)))
    x3_0 = solver.chi_CO2_dry_ambient * (1 - 0.010)
    x1_0 = solver.chi_O2d_0 * (1 - 0.010)
    outer = StefanTubeBVP(L=L / 2).solve_concentration_bc(
        x1_0, 0.010, x3_0, base["x_O2"][i], solver.x_sat,
        F_O2_guess=base["F_O2"], x3_guess_L=base["x_CO2"][i])
    assert outer["F_O2"] == pytest.approx(base["F_O2"], rel=1e-6)
    assert outer["x_CO2_wall"] == pytest.approx(base["x_CO2"][i], rel=1e-5)


def test_humidity_sensitivity_ordering(solver):
    """Moving the pivot toward the wall increases the O2 flux (Fig. 2)."""
    F = [solver.solve(0.0, f * L, N_grid=61, t_factor=20.0)["F_O2"] for f in (0.1, 0.5, 1.0)]
    assert F[0] < F[1] < F[2]
