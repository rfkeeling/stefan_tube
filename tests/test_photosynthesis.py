import numpy as np
import pytest

from stefan_tube import StefanTubeSolverPhotosynthesis, severinghaus_chi_wall

L, CHI_CO2_COMP = 0.10, 280e-6


@pytest.fixture(scope="module")
def solver():
    return StefanTubeSolverPhotosynthesis()


def test_steady_state_reference(solver):
    r = solver.solve(0.010, L / 2, CHI_CO2_COMP, N_grid=61, t_factor=20.0)
    assert r["F_O2"] == pytest.approx(-7.116101e-07, rel=1e-5)
    assert r["F_O2"] + r["F_CO2"] == pytest.approx(0.0, abs=1e-12)
    assert np.max(np.abs(r["F_N2"])) < 1e-12
    assert r["x_CO2_wall"] == pytest.approx(CHI_CO2_COMP, rel=1e-4)
    assert r["x_CO2_wall"] > CHI_CO2_COMP


def test_wall_o2_independent_of_pivot_position(solver):
    x = [solver.solve(0.010, f * L, CHI_CO2_COMP, N_grid=61, t_factor=20.0)["x_O2"][-1]
         for f in (0.25, 0.5, 0.75)]
    assert np.ptp(x) / x[1] < 1e-6


def test_severinghaus_closed_form_limits(solver):
    chi_o = solver.chi_O2d_0
    # no humidity gradient: no fractionation
    assert severinghaus_chi_wall(solver.x_sat, solver.x_sat, chi_o) == pytest.approx(chi_o)
    # humid wall enriches dry-air O2 slightly, by eps chi (1 - chi) / (1 - x) dx
    chi_wall = severinghaus_chi_wall(0.0, solver.x_sat, chi_o)
    assert 0 < chi_wall - chi_o < 1e-3
