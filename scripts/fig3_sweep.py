"""
Data for Fig. 3: photosynthesis variant, O2 mole fraction at the end wall vs.
ambient water-vapor mole fraction, at z_pivot/L = 0.25, 0.5 and 0.75 (Fig. 3
shows 0.5; the other two support the statement that the result is independent
of pivot position). Wall CO2 compensation level 280 ppm, ambient CO2 400 ppm
(dry basis). Each sweep is warm-started from the previous humidity.

Writes data/fig3_photosynthesis_wall_o2.csv.
"""
import time
import _common
import numpy as np
from stefan_tube import StefanTubeSolverPhotosynthesis

CHI_CO2_COMP = 280e-6
solver = StefanTubeSolverPhotosynthesis()  # T = 298.15 K, p = 101325 Pa, L = 0.10 m
x_h2o_0 = np.linspace(0.0, solver.x_sat, 15)

cols = {"x_H2O_ambient": x_h2o_0}
t0 = time.time()
for frac in (0.25, 0.5, 0.75):
    x_O2_wall = np.full_like(x_h2o_0, np.nan)
    x_CO2_wall = np.full_like(x_h2o_0, np.nan)
    F_O2 = np.full_like(x_h2o_0, np.nan)
    warm = None
    for i, xh in enumerate(x_h2o_0):
        if warm is None:
            r = solver.solve(xh, frac * solver.L, CHI_CO2_COMP, N_grid=61, t_factor=20.0)
        else:
            r = solver.solve(xh, frac * solver.L, CHI_CO2_COMP, warm_start=warm, t_factor_relax=10.0)
        x_O2_wall[i], x_CO2_wall[i], F_O2[i] = r["x_O2"][-1], r["x_CO2"][-1], r["F_O2"]
        warm = (r["x_O2"], r["x_H2O"], r["x_CO2"], r["z"])
    cols[f"x_O2_wall_pivot_{frac:.2f}"] = x_O2_wall
    cols[f"x_CO2_wall_pivot_{frac:.2f}"] = x_CO2_wall
    cols[f"F_O2_pivot_{frac:.2f}"] = F_O2
    print(f"z_pivot/L = {frac}: done [{time.time() - t0:.0f} s]")

_common.write_csv(_common.DATA / "fig3_photosynthesis_wall_o2.csv", cols)
print("saved data/fig3_photosynthesis_wall_o2.csv")
