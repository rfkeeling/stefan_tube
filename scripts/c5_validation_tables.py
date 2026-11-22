"""
Appendix C validation, Sects. C5.1-C5.3 (Tables C2 and C3), plus the
related checks quoted in Sects. C2.3 and C3.4.

Validation conditions: T = 298.15 K, p = 101325 Pa, L = 0.10 m,
z_pivot = L/2, ambient x_H2O = 0.010, chi_O2,d = 0.209339, chi_CO2,d = 400 ppm.

  C5.1  F_N2 = 0 recovered at steady state without being imposed along the tube.
  C5.2  Table C2: the time-marching solution split at the pivot into an inner
        and an outer boundary-value problem (adaptive-mesh collocation), each
        solved independently using concentrations read off the pivot.
  C5.3  Table C3: z_pivot = L (pivot disabled), time-marching vs. one BVP over
        the whole tube.
"""
import _common  # noqa: F401
import numpy as np
from stefan_tube import StefanTubeBVP, StefanTubeSolver

T, P, L = 298.15, 101325.0, 0.10
X_H2O_0 = 0.010


def rel(a, b):
    return abs(a - b) / abs(b)


def run(N_grid):
    tm = StefanTubeSolver(T=T, p=P, L=L)
    z_pivot = L / 2
    r = tm.solve(X_H2O_0, z_pivot, N_grid=N_grid, t_factor=20.0)
    z = r["z"]
    dz = z[1] - z[0]
    seg_mid = 0.5 * (z[:-1] + z[1:])
    outer = seg_mid < z_pivot

    print(f"\n{'=' * 78}\nN_grid = {N_grid}   (segments: {N_grid - 1}, "
          f"pivot at grid index {int(np.argmin(np.abs(z - z_pivot)))})\n{'=' * 78}")

    print("\nC5.1 Steady-state internal consistency")
    print(f"  max |F_N2| over all segments = {np.max(np.abs(r['F_N2'])):.2e} mol m-2 s-1")

    print("\nC3.4 Flux uniformity at the end of the run (spread = (max - min)/mean)")
    # F_O2 and F_CO2 are returned as means; recompute the per-segment fluxes directly.
    F1, F2, F3, F4 = tm._solve_all_fluxes(r["x_O2"], r["x_H2O"], r["x_CO2"], dz, r["z"] >= z_pivot)
    for name, F, mask in [("F_O2 (all segments)", F1, slice(None)),
                          ("F_CO2 (all segments)", F4, slice(None)),
                          ("F_H2O (outer segments)", F2, outer)]:
        v = F[mask]
        print(f"  {name:<24} {100 * (v.max() - v.min()) / abs(v.mean()):.2e} %")
    print(f"  F_H2O inner segments: {F2[~outer].min():.3e} to {F2[~outer].max():.3e} "
          f"(not uniform: wetted wall is a distributed source)")

    k_rxn = 200 * tm.ct * tm.D_min / dz**2
    print("\nC2.3 Wall O2")
    print(f"  x_O2(wall) = {r['x_O2'][-1]:.3e}  (< 1e-4: {r['x_O2'][-1] < 1e-4}); "
          f"F_O2/(k_rxn dz) = {r['F_O2'] / (k_rxn * dz):.3e}")

    # ---- Table C2: split at the pivot ----
    i_p = int(np.argmin(np.abs(z - z_pivot)))
    x_O2_p, x_CO2_p = r["x_O2"][i_p], r["x_CO2"][i_p]
    x_sat = tm.x_sat
    inner = StefanTubeBVP(T=T, p=P, L=L - z_pivot).solve_concentration_bc(
        x1_0=x_O2_p, x2_0=x_sat, x3_0=x_CO2_p, x1_L=0.0, x2_L=x_sat,
        F_O2_guess=r["F_O2"], x3_guess_L=r["x_CO2_wall"])
    x3_amb = tm.chi_CO2_dry_ambient * (1 - X_H2O_0)
    x1_amb = tm.chi_O2d_0 * (1 - X_H2O_0)
    outer_bvp = StefanTubeBVP(T=T, p=P, L=z_pivot).solve_concentration_bc(
        x1_0=x1_amb, x2_0=X_H2O_0, x3_0=x3_amb, x1_L=x_O2_p, x2_L=x_sat,
        F_O2_guess=r["F_O2"], x3_guess_L=x_CO2_p)

    print("\nTable C2. Validation against BVP solutions")
    print(f"  {'Quantity':<40}{'Time-marching':>16}{'BVP':>16}{'Rel. diff.':>13}")
    rows = [
        ("F_O2, outer domain (mol m-2 s-1)", r["F_O2"], outer_bvp["F_O2"], "{:.6e}"),
        ("x_CO2 at pivot (outer domain, solved)", x_CO2_p, outer_bvp["x_CO2_wall"], "{:.6f}"),
        ("F_O2, inner domain (mol m-2 s-1)", r["F_O2"], inner["F_O2"], "{:.6e}"),
        ("x_CO2(wall), inner domain", r["x_CO2_wall"], inner["x_CO2_wall"], "{:.6f}"),
    ]
    for label, a, b, fmt in rows:
        print(f"  {label:<40}{fmt.format(a):>16}{fmt.format(b):>16}{rel(b, a):>13.2e}")

    # ---- Table C3: pivot disabled ----
    r_np = tm.solve(X_H2O_0, L, N_grid=N_grid, t_factor=20.0)
    bvp = StefanTubeBVP(T=T, p=P, L=L).solve(X_H2O_0, tm.chi_O2d_0, x_sat, 0.0)
    print("\nTable C3. No-pivot cross check")
    print(f"  {'Quantity':<40}{'Time-marching':>16}{'BVP':>16}{'Rel. diff.':>13}")
    for label, a, b, fmt in [("F_O2 (mol m-2 s-1)", r_np["F_O2"], bvp["F_O2"], "{:.6e}"),
                             ("x_CO2(wall)", r_np["x_CO2_wall"], bvp["x_CO2_wall"], "{:.6f}")]:
        print(f"  {label:<40}{fmt.format(a):>16}{fmt.format(b):>16}{rel(b, a):>13.2e}")


run(61)
