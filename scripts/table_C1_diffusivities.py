"""
Table C1: binary diffusivities at T = 298.15 K, p = 101.325 kPa (Eq. C3),
and the O2-H2O diffusivity derived from Severinghaus et al. (1996).
"""
import _common  # noqa: F401
import numpy as np
from stefan_tube import (FULLER_V, MOLAR_MASS, D_RATIO_SEVERINGHAUS,
                         fuller_diffusivity, binary_diffusivities)

T, P = 298.15, 101325.0
p_bar = P / 1e5
D = binary_diffusivities(T, P)

print(f"Table C1. Binary diffusivities at T = {T} K, p = {P/1e3:.3f} kPa\n")
print(f"{'pair':<10}{'Sv_A':>7}{'Sv_B':>7}{'M_AB':>9}{'D (cm2/s)':>12}")
print(f"{'O2-H2O':<10}{'--':>7}{'--':>7}{'--':>9}{D['O2-H2O']*1e4:>12.4f}"
      f"   (= D_H2O-N2 / {D_RATIO_SEVERINGHAUS})")
for a, b in [("O2", "N2"), ("H2O", "N2"), ("O2", "CO2"), ("H2O", "CO2"), ("N2", "CO2")]:
    M_ab = 2.0 / (1.0 / MOLAR_MASS[a] + 1.0 / MOLAR_MASS[b])
    print(f"{a + '-' + b:<10}{FULLER_V[a]:>7.1f}{FULLER_V[b]:>7.1f}{M_ab:>9.2f}"
          f"{fuller_diffusivity(a, b, T, p_bar)*1e4:>12.4f}")
print(f"\nD_min = {min(D.values())*1e4:.4f} cm2/s ({min(D, key=D.get)})")
