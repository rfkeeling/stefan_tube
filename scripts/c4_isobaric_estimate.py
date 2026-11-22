"""
Sect. C4: pressure difference needed to drive the net mass flow through the
tube against Hagen-Poiseuille drag, to check the isobaric approximation.

Hagen-Poiseuille needs the mass-averaged velocity u = j_mass / rho, not the
molar-flux velocity. With F_N2 = 0 and F_CO2 = -F_O2,

    j_mass = F_O2 (M_O2 - M_CO2) + M_H2O F_H2O,

so the O2 -> CO2 exchange contributes to the mass flux even though it cancels
in the molar flux. F_O2 is uniform along the tube; F_H2O is uniform in the outer
domain and is averaged over the inner domain. Pressure drop per domain:
dp = 8 mu L_domain u / r^2, summed over the two domains. Viscosity of air at
T from Sutherland's formula; density of dry air from c_t and the dry-air
molar mass.

Conditions: T = 298.15 K, p = 101325 Pa, L = 0.10 m, z_pivot = L/2, N_grid = 61,
ambient x_H2O = 0.010.
"""
import _common  # noqa: F401
import numpy as np
from stefan_tube import StefanTubeSolver, MOLAR_MASS

T, P = 298.15, 101325.0
M = {k: v * 1e-3 for k, v in MOLAR_MASS.items()}  # kg mol-1


def sutherland_viscosity(T, mu_ref=1.716e-5, T_ref=273.15, S=110.4):
    return mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)


def mass_fluxes(L, x_h2o_0=0.010, N_grid=61):
    """Mass flux (kg m-2 s-1) in the outer and inner domains, z_pivot = L/2."""
    tm = StefanTubeSolver(T=T, p=P, L=L)
    z_pivot = L / 2
    r = tm.solve(x_h2o_0, z_pivot, N_grid=N_grid, t_factor=20.0)
    z = r["z"]
    outer = 0.5 * (z[:-1] + z[1:]) < z_pivot
    F_O2 = r["F_O2"]
    j_o2_co2 = F_O2 * (M["O2"] - M["CO2"])
    j_outer = j_o2_co2 + M["H2O"] * r["F_H2O"][outer].mean()
    j_inner = j_o2_co2 + M["H2O"] * r["F_H2O"][~outer].mean()
    return tm, F_O2, r["F_H2O"][outer].mean(), r["F_H2O"][~outer].mean(), j_outer, j_inner, j_o2_co2


tm, F_O2, F_H2O_out, F_H2O_in, j_out, j_in, j_o2co2 = mass_fluxes(0.10)
L = tm.L
chi = tm.chi_O2d_0
rho = tm.ct * (chi * M["O2"] + (1 - chi) * M["N2"])
mu = sutherland_viscosity(T)

print(f"F_O2 (uniform)         = {F_O2:.4e} mol m-2 s-1")
print(f"F_H2O, outer / inner   = {F_H2O_out:.4e} / {F_H2O_in:.4e} mol m-2 s-1")
print(f"j_mass from O2/CO2     = {j_o2co2:.4e} kg m-2 s-1")
print(f"j_mass, outer domain   = {j_out:.4e} kg m-2 s-1")
print(f"j_mass, inner domain   = {j_in:.4e} kg m-2 s-1   (inner/outer = {j_in / j_out:.4f})")
print(f"rho = {rho:.4f} kg m-3, mu = {mu:.4e} Pa s")
u_out, u_in = abs(j_out) / rho, abs(j_in) / rho
print(f"u, outer / inner       = {u_out:.3e} / {u_in:.3e} m s-1\n")

print("Pressure difference across the two domains (each of length L/2):")
print(f"{'r':>8}{'dp outer (Pa)':>16}{'dp inner (Pa)':>16}{'dp total (Pa)':>16}{'dp / p':>12}")
for r_tube in (1e-4, 1e-3, 1e-2):
    dp_o = 8 * mu * (L / 2) * u_out / r_tube**2
    dp_i = 8 * mu * (L / 2) * u_in / r_tube**2
    print(f"{r_tube*1e3:>6.1f}mm{dp_o:>16.3e}{dp_i:>16.3e}{dp_o + dp_i:>16.3e}{(dp_o + dp_i) / P:>12.2e}")

print("\nLength dependence (j_mass x L should be constant, so dp is independent of L):")
print(f"{'L (m)':>8}{'F_O2':>13}{'j_outer':>13}{'j_outer*L':>13}{'j_inner':>13}{'j_inner*L':>13}{'inner/outer':>13}")
for L_ in (0.05, 0.10, 0.20, 0.40):
    _, F, _, _, jo, ji, _ = mass_fluxes(L_)
    print(f"{L_:>8.2f}{F:>13.4e}{jo:>13.4e}{jo * L_:>13.4e}{ji:>13.4e}{ji * L_:>13.4e}{ji / jo:>13.4f}")
