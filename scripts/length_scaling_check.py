"""
Fluxes scale as 1/L at fixed z_pivot/L (Sect. 5.1) and the photosynthesis
result is independent of L (Fig. 3 caption). Runs L = 0.05, 0.10, 0.20, 0.40 m
at ambient x_H2O = 0.010, z_pivot/L = 0.5, N_grid = 61.

Respiratory tube: F x L should be constant. Photosynthesis tube: x_O2(wall)
should be constant.
"""
import _common  # noqa: F401
import numpy as np
from stefan_tube import StefanTubeSolver, StefanTubeSolverPhotosynthesis

X_H2O_0 = 0.010
Ls = (0.05, 0.10, 0.20, 0.40)

print("Respiratory tube: flux x L (should be constant)")
print(f"{'L (m)':>8}{'F_O2 L':>14}{'F_CO2 L':>14}{'F_H2O,outer L':>16}")
rows = []
for L in Ls:
    tm = StefanTubeSolver(L=L)
    r = tm.solve(X_H2O_0, L / 2, N_grid=61, t_factor=20.0)
    outer = 0.5 * (r["z"][:-1] + r["z"][1:]) < L / 2
    rows.append((r["F_O2"] * L, r["F_CO2"] * L, r["F_H2O"][outer].mean() * L))
    print(f"{L:>8.2f}{rows[-1][0]:>14.6e}{rows[-1][1]:>14.6e}{rows[-1][2]:>16.6e}")
rows = np.array(rows)
print("max relative spread across L:", np.ptp(rows, axis=0) / np.abs(rows[0]))

print("\nPhotosynthesis tube: x_O2 at the wall (should be constant)")
x_wall = []
for L in Ls:
    tm = StefanTubeSolverPhotosynthesis(L=L)
    r = tm.solve(X_H2O_0, L / 2, 280e-6, N_grid=61, t_factor=20.0)
    x_wall.append(r["x_O2"][-1])
    print(f"{L:>8.2f}  x_O2(wall) = {x_wall[-1]:.9e}")
print(f"max relative spread across L: {np.ptp(x_wall) / x_wall[0]:.2e}")
