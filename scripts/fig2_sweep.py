"""
Data for Fig. 2: O2 flux through the respiratory tube vs. ambient water-vapor
mole fraction, at pivot positions z_pivot/L = 0.1, 0.5 (base case) and 1.0
(pivot disabled), and with the cross-flux-drag term removed at z_pivot/L = 0.5.

Each sweep is warm-started from the converged profile at the previous humidity;
the no-cross-flux-drag run at each humidity is warm-started from the full
solution at that same humidity.

Writes data/fig2_o2_flux_vs_humidity.csv.
"""
import time
import _common
import numpy as np
from stefan_tube import StefanTubeSolver

solver = StefanTubeSolver()  # T = 298.15 K, p = 101325 Pa, L = 0.10 m
x_h2o_0 = np.linspace(0.0, solver.x_sat, 15)
N_GRID, T_FACTOR_COLD, T_FACTOR_WARM, T_FACTOR_NOCFD = 61, 20.0, 8.0, 10.0

curves = {}
t0 = time.time()
for frac in (0.1, 0.5, 1.0):
    F_full = np.full_like(x_h2o_0, np.nan)
    F_nocfd = np.full_like(x_h2o_0, np.nan)
    warm = None
    for i, xh in enumerate(x_h2o_0):
        if warm is None:
            r = solver.solve(xh, frac * solver.L, N_grid=N_GRID, t_factor=T_FACTOR_COLD)
        else:
            r = solver.solve(xh, frac * solver.L, warm_start=warm, t_factor_relax=T_FACTOR_WARM)
        F_full[i] = r["F_O2"]
        warm = (r["x_O2"], r["x_H2O"], r["x_CO2"], r["z"])
        if frac == 0.5:
            r_no = solver.solve(xh, frac * solver.L, no_cfd=True, warm_start=warm,
                                t_factor_relax=T_FACTOR_NOCFD)
            F_nocfd[i] = r_no["F_O2"]
    curves[frac] = (F_full, F_nocfd)
    print(f"z_pivot/L = {frac}: done [{time.time() - t0:.0f} s]")

_common.write_csv(_common.DATA / "fig2_o2_flux_vs_humidity.csv", {
    "x_H2O_ambient": x_h2o_0,
    "F_O2_pivot_0.10": curves[0.1][0],
    "F_O2_pivot_0.50": curves[0.5][0],
    "F_O2_pivot_1.00": curves[1.0][0],
    "F_O2_pivot_0.50_no_cross_flux_drag": curves[0.5][1],
})
print("saved data/fig2_o2_flux_vs_humidity.csv (flux in mol m-2 s-1)")
