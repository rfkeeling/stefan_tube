"""
Physical constants and correlations shared by the Stefan-Maxwell tube models.
"""
import numpy as np

R = 8.314462618  # J mol-1 K-1

# Ambient dry-air mole fractions. Dry air includes its CO2: chi_O2,d = x_O2 / (1 - x_H2O).
# chi_O2,d = 0.209339 is the value of Aoki et al. (2019).
AMBIENT_CHI_O2_DRY = 0.209339
AMBIENT_CHI_CO2_DRY = 0.0004

# Severinghaus et al. (1996) ratio D_N2-H2O / D_O2-H2O.
D_RATIO_SEVERINGHAUS = 0.965

# Fuller-Schettler-Giddings atomic diffusion volumes and molar masses (g mol-1).
FULLER_V = {"O2": 16.6, "N2": 17.9, "H2O": 12.7, "CO2": 26.9}
MOLAR_MASS = {"O2": 32.00, "N2": 28.01, "H2O": 18.02, "CO2": 44.01}


def fuller_diffusivity(species_a, species_b, T, p_bar):
    """
    Binary diffusivity [m2 s-1] from the Fuller-Schettler-Giddings correlation
    (Eq. C3); T in K, p in bar.

    D_AB [cm2 s-1] = 0.00143 T^1.75 / (p M_AB^0.5 (Sv_A^(1/3) + Sv_B^(1/3))^2),
    M_AB = 2 / (1/M_A + 1/M_B).
    """
    Va, Vb = FULLER_V[species_a], FULLER_V[species_b]
    Ma, Mb = MOLAR_MASS[species_a], MOLAR_MASS[species_b]
    M_ab = 2.0 / (1.0 / Ma + 1.0 / Mb)
    D_cm2_s = 0.00143 * T**1.75 / (p_bar * np.sqrt(M_ab) * (Va**(1 / 3) + Vb**(1 / 3))**2)
    return D_cm2_s * 1e-4


def binary_diffusivities(T, p):
    """
    The six pairwise diffusivities [m2 s-1] at temperature T [K] and pressure p [Pa].

    All pairs except O2-H2O come from the Fuller correlation. O2-H2O is set to
    D_N2-H2O / 0.965 (Severinghaus et al., 1996), keeping D_N2-H2O at its
    Fuller value.
    """
    p_bar = p / 1e5
    D = {
        "O2-N2": fuller_diffusivity("O2", "N2", T, p_bar),
        "H2O-N2": fuller_diffusivity("H2O", "N2", T, p_bar),
        "O2-CO2": fuller_diffusivity("O2", "CO2", T, p_bar),
        "H2O-CO2": fuller_diffusivity("H2O", "CO2", T, p_bar),
        "N2-CO2": fuller_diffusivity("N2", "CO2", T, p_bar),
    }
    D["O2-H2O"] = D["H2O-N2"] / D_RATIO_SEVERINGHAUS
    return D


def e_sat_bolton(T_K):
    """Saturation vapor pressure of water [Pa], Bolton (1980)."""
    Tc = T_K - 273.15
    return 6.112 * np.exp(17.67 * Tc / (Tc + 243.5)) * 100.0


def x_sat_bolton(T_K, p):
    """Saturation mole fraction of water vapor at temperature T_K and pressure p [Pa]."""
    return e_sat_bolton(T_K) / p
