"""
Steady-state boundary-value solver for the four-species (O2, H2O, N2, CO2)
Stefan-Maxwell tube, used to cross-check the time-marching solver
(Appendix C, Sect. C5).

F_N2 = 0 and F_CO2 = -F_O2 are imposed throughout the tube (bootstrap
condition). State vector: [x_O2, x_H2O, x_CO2, F_O2, F_H2O]; x_N2 = 1 - sum of
the other three mole fractions.
"""
import numpy as np
from scipy.integrate import solve_bvp

from .constants import R, binary_diffusivities, AMBIENT_CHI_CO2_DRY


class StefanTubeBVP:
    def __init__(self, T=298.15, p=101325.0, L=0.10, chi_CO2_dry_ambient=AMBIENT_CHI_CO2_DRY):
        self.T, self.p, self.L = T, p, L
        D = binary_diffusivities(T, p)
        self.D_O2H2O, self.D_O2N2, self.D_H2ON2 = D["O2-H2O"], D["O2-N2"], D["H2O-N2"]
        self.D_O2CO2, self.D_H2OCO2, self.D_N2CO2 = D["O2-CO2"], D["H2O-CO2"], D["N2-CO2"]
        self.ct = p / (R * T)  # mol m-3
        self.chi_CO2_dry_ambient = chi_CO2_dry_ambient

    def _rhs(self, z, y):
        x1, x2, x3, F1, F2 = y  # O2, H2O, CO2 mole fractions; F1 = F_O2, F2 = F_H2O; F_CO2 = -F1
        xN2 = 1.0 - x1 - x2 - x3
        ct = self.ct
        dx1 = ((x1 * F2 - x2 * F1) / (ct * self.D_O2H2O)
               - xN2 * F1 / (ct * self.D_O2N2)
               - F1 * (x1 + x3) / (ct * self.D_O2CO2))
        dx2 = ((x2 * F1 - x1 * F2) / (ct * self.D_O2H2O)
               - xN2 * F2 / (ct * self.D_H2ON2)
               - F1 * x2 / (ct * self.D_H2OCO2)
               - x3 * F2 / (ct * self.D_H2OCO2))
        dx3 = (F1 * (x3 + x1) / (ct * self.D_O2CO2)
               + (x3 * F2 + x2 * F1) / (ct * self.D_H2OCO2)
               + xN2 * F1 / (ct * self.D_N2CO2))
        return np.vstack([dx1, dx2, dx3, np.zeros_like(F1), np.zeros_like(F2)])

    def solve(self, x_h2o_0, chi_O2d_0, x_h2o_L, chi_O2d_L, n_mesh=80,
              F_O2_guess=1.9e-3, F_H2O_guess=2e-4):
        """
        Whole-tube solve with the ambient dry-air convention of the
        time-marching solver: ambient O2 is chi_O2d_0 (1 - x_H2O) and ambient
        CO2 is chi_CO2_dry_ambient (1 - x_H2O). x_CO2 at z = L is not
        prescribed; it is solved for.
        """
        x3_0 = self.chi_CO2_dry_ambient * (1 - x_h2o_0)
        x1_0 = chi_O2d_0 * (1 - x_h2o_0)
        x1_L = chi_O2d_L * (1 - x_h2o_L)
        return self.solve_concentration_bc(x1_0, x_h2o_0, x3_0, x1_L, x_h2o_L,
                                           n_mesh=n_mesh, F_O2_guess=F_O2_guess,
                                           F_H2O_guess=F_H2O_guess)

    def solve_concentration_bc(self, x1_0, x2_0, x3_0, x1_L, x2_L, n_mesh=80,
                               F_O2_guess=1.9e-3, F_H2O_guess=2e-4, x3_guess_L=0.15):
        """
        Solve with mole fractions prescribed directly: (x_O2, x_H2O, x_CO2) at
        z = 0 and (x_O2, x_H2O) at z = L. Used to solve the inner and outer
        sub-domains of a pivot solution as separate problems (Table C2).
        Uses scipy's adaptive-mesh collocation solver.
        """
        z = np.linspace(0, self.L, n_mesh)
        y_guess = np.vstack([
            np.linspace(x1_0, x1_L, n_mesh),
            np.linspace(x2_0, x2_L, n_mesh),
            np.linspace(x3_0, x3_guess_L, n_mesh),
            np.full(n_mesh, F_O2_guess),
            np.full(n_mesh, F_H2O_guess),
        ])

        def bc(ya, yb):
            return np.array([ya[0] - x1_0, ya[1] - x2_0, ya[2] - x3_0,
                             yb[0] - x1_L, yb[1] - x2_L])

        sol = solve_bvp(self._rhs, bc, z, y_guess, max_nodes=30000, tol=1e-10)
        if not sol.success:
            raise RuntimeError(f"BVP failed: {sol.message}")
        return {"F_O2": sol.y[3, 0], "F_H2O": sol.y[4, 0], "F_CO2": -sol.y[3, 0],
                "x_CO2_wall": sol.y[2, -1], "sol": sol}
