"""
Sect. C3.5: the cross-flux-drag variant (term x_O2 F_H2O / (c_t D_O2-H2O)
removed from the O2 Stefan-Maxwell relation) is started from the converged
full-model profile and relaxed for 3, 8 and 15 diffusive timescales; the
results should agree to six significant figures.

Conditions: T = 298.15 K, p = 101325 Pa, L = 0.10 m, z_pivot = L/2, N_grid = 61,
ambient x_H2O = 0.010.
"""
import _common  # noqa: F401
from stefan_tube import StefanTubeSolver

tm = StefanTubeSolver()
full = tm.solve(0.010, tm.L / 2, N_grid=61, t_factor=20.0)
warm = (full["x_O2"], full["x_H2O"], full["x_CO2"], full["z"])
print(f"Full model:             F_O2 = {full['F_O2']:.6e} mol m-2 s-1   x_CO2(wall) = {full['x_CO2_wall']:.5f}")
for t in (3.0, 8.0, 15.0):
    r = tm.solve(0.010, tm.L / 2, no_cfd=True, warm_start=warm, t_factor_relax=t)
    print(f"No cross-flux drag, relaxed {t:>4.0f} t_diff: F_O2 = {r['F_O2']:.6e} mol m-2 s-1   "
          f"x_CO2(wall) = {r['x_CO2_wall']:.5f}")
