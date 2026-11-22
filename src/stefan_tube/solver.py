"""
Time-marching (method-of-lines) solver for the pivot Stefan-Maxwell tube
(Appendix C, Sect. C3).

z is discretized onto one uniform grid spanning the whole tube, and the
mole-fraction profiles are integrated forward in time (BDF) to steady state.
At every evaluation of the right-hand side, the four species' fluxes on all
N_grid - 1 segments between grid points are obtained from one direct linear
system of 4 (N_grid - 1) equations: three Stefan-Maxwell relations per
segment, sum_i dx_i/dt = 0 at each interior grid point, and F_N2 = 0 at the
wall-adjacent segment only. The wall reaction is a fast finite-rate term at
the last grid point.

For z >= z_pivot, x_H2O is held at saturation (moist wall); for z < z_pivot
it evolves freely. z_pivot >= L disables the pivot, reducing to the plain
single-domain problem solved by bvp.StefanTubeBVP.

The ambient O2 and CO2 mole fractions are specified on a dry-air basis (dry air
includes its CO2) and diluted by ambient humidity: x_O2 = chi_O2,d (1 - x_H2O),
x_CO2 = chi_CO2,d (1 - x_H2O).

Time is physical time. The concentrations c_i = c_t x_i obey dc_i/dt = -dF_i/dz,
so dx_i/dt = -(1/c_t) dF_i/dz, and the wall reaction acts as a source of
k_rxn x_O2 [mol m-3 s-1] in the O2 and CO2 budgets. Integration lengths are in
diffusive timescales L^2 / D_min.
"""
import numpy as np
from scipy.integrate import solve_ivp

from .constants import (R, AMBIENT_CHI_O2_DRY, AMBIENT_CHI_CO2_DRY,
                        binary_diffusivities, x_sat_bolton)


class StefanTubeSolver:
    def __init__(self, T=298.15, p=101325.0, L=0.10, chi_CO2_dry_ambient=AMBIENT_CHI_CO2_DRY):
        self.T, self.p, self.L = T, p, L
        self.ct = p / (R * T)  # mol m-3
        D = binary_diffusivities(T, p)
        self.D_O2H2O, self.D_O2N2, self.D_H2ON2 = D["O2-H2O"], D["O2-N2"], D["H2O-N2"]
        self.D_O2CO2, self.D_H2OCO2, self.D_N2CO2 = D["O2-CO2"], D["H2O-CO2"], D["N2-CO2"]
        self.D_min = min(D.values())
        self.x_sat = x_sat_bolton(T, p)
        self.chi_O2d_0 = AMBIENT_CHI_O2_DRY
        self.chi_CO2_dry_ambient = chi_CO2_dry_ambient

    def _solve_all_fluxes(self, x1, x2, x4, dz, pinned, no_cfd=False):
        """
        Solve the 4 (N_grid - 1) linear system for the fluxes on all segments.

        x1, x2, x4 are the O2, H2O and CO2 mole fractions at the grid points
        (x3 = N2 is the remainder). Returns (F_O2, F_H2O, F_N2, F_CO2), each of
        length N_grid - 1. no_cfd=True removes the term x_O2 F_H2O / (c_t D_O2-H2O)
        from the O2 Stefan-Maxwell relation (Sect. C3.5).
        """
        ct = self.ct
        D_O2H2O, D_O2N2, D_H2ON2 = self.D_O2H2O, self.D_O2N2, self.D_H2ON2
        D_O2CO2, D_H2OCO2, D_N2CO2 = self.D_O2CO2, self.D_H2OCO2, self.D_N2CO2

        N_grid = len(x1)
        Nf = N_grid - 1
        idx1 = np.arange(Nf)  # F_O2 unknowns
        idx2 = Nf + idx1      # F_H2O
        idx3 = 2 * Nf + idx1  # F_N2
        idx4 = 3 * Nf + idx1  # F_CO2

        A = np.zeros((4 * Nf, 4 * Nf))
        b = np.zeros(4 * Nf)

        # Stefan-Maxwell relations for O2, H2O, CO2 on every segment, with
        # midpoint-averaged mole fractions and centered-difference gradients.
        x1f = 0.5 * (x1[:-1] + x1[1:])
        x2f = 0.5 * (x2[:-1] + x2[1:])
        x4f = 0.5 * (x4[:-1] + x4[1:])
        x3f = 1 - x1f - x2f - x4f
        dx1dz = (x1[1:] - x1[:-1]) / dz
        dx2dz = (x2[1:] - x2[:-1]) / dz
        dx4dz = (x4[1:] - x4[:-1]) / dz

        row_O2 = 3 * idx1
        row_H2O = row_O2 + 1
        row_CO2 = row_O2 + 2
        cfd_term = np.zeros(Nf) if no_cfd else x1f / (ct * D_O2H2O)

        A[row_O2, idx1] = -x2f / (ct * D_O2H2O) - x3f / (ct * D_O2N2) - x4f / (ct * D_O2CO2)
        A[row_O2, idx2] = cfd_term
        A[row_O2, idx3] = x1f / (ct * D_O2N2)
        A[row_O2, idx4] = x1f / (ct * D_O2CO2)
        b[row_O2] = dx1dz

        A[row_H2O, idx1] = x2f / (ct * D_O2H2O)
        A[row_H2O, idx2] = -x1f / (ct * D_O2H2O) - x3f / (ct * D_H2ON2) - x4f / (ct * D_H2OCO2)
        A[row_H2O, idx3] = x2f / (ct * D_H2ON2)
        A[row_H2O, idx4] = x2f / (ct * D_H2OCO2)
        b[row_H2O] = dx2dz

        A[row_CO2, idx1] = x4f / (ct * D_O2CO2)
        A[row_CO2, idx2] = x4f / (ct * D_H2OCO2)
        A[row_CO2, idx3] = x4f / (ct * D_N2CO2)
        A[row_CO2, idx4] = -x1f / (ct * D_O2CO2) - x2f / (ct * D_H2OCO2) - x3f / (ct * D_N2CO2)
        b[row_CO2] = dx4dz

        # sum_i dx_i/dt = 0 at each interior grid point j, written in terms of
        # the fluxes on the segments to its left (j-1) and right (j). The H2O
        # term is dropped where x_H2O is pinned.
        j = np.arange(1, N_grid - 1)
        left, right = j - 1, j
        cons_rows = 3 * Nf + np.arange(len(j))
        for idx in (idx1, idx3, idx4):
            A[cons_rows, idx[right]] += 1
            A[cons_rows, idx[left]] += -1
        free = ~pinned[j]
        A[cons_rows[free], idx2[right[free]]] += 1
        A[cons_rows[free], idx2[left[free]]] += -1

        # F_N2 = 0 on the wall-adjacent segment.
        A[4 * Nf - 1, idx3[Nf - 1]] = 1.0

        sol = np.linalg.solve(A, b)
        return sol[0:Nf], sol[Nf:2 * Nf], sol[2 * Nf:3 * Nf], sol[3 * Nf:4 * Nf]

    def _wall_reaction(self, x_O2_wall, x_CO2_wall, k_rxn):
        """
        Respiration, O2 -> CO2 at the last grid point: O2 is consumed at rate
        k_rxn * x_O2(wall) [mol m-3 s-1] and CO2 is produced in equal amount.
        Returns the reaction's contributions (d_O2, d_CO2) to c_t dx/dt there.
        """
        rate = k_rxn * x_O2_wall
        return -rate, +rate

    def solve(self, x_h2o_0, z_pivot, N_grid=101, t_factor=8.0, k_rxn=None, no_cfd=False,
              warm_start=None, t_factor_relax=3.0):
        """
        Integrate to steady state at ambient water-vapor mole fraction x_h2o_0.

        With warm_start=None, start from a linear initial profile and
        integrate for t_factor diffusive timescales (L^2 / D_min). With
        warm_start = (x_O2, x_H2O, x_CO2, z) from an earlier solve at the same
        z_pivot and N_grid, start from that profile and integrate for
        t_factor_relax diffusive timescales (used for humidity sweeps and for
        the no_cfd comparison).
        """
        x4_0 = self.chi_CO2_dry_ambient * (1 - x_h2o_0)
        x1_0 = self.chi_O2d_0 * (1 - x_h2o_0)
        x2_0 = x_h2o_0

        if warm_start is None:
            z = np.linspace(0, self.L, N_grid)
            pinned = z >= z_pivot
            x1_init = np.linspace(x1_0, 0.0, N_grid)
            x2_init = np.linspace(x2_0, self.x_sat, N_grid)
            x2_init[pinned] = self.x_sat
            x4_init = np.full(N_grid, x4_0)
            y0 = np.concatenate([x1_init, x2_init, x4_init])
            t_use = t_factor
        else:
            x1w, x2w, x4w, z = warm_start
            N_grid = len(z)
            pinned = z >= z_pivot
            y0 = np.concatenate([x1w, x2w, x4w])
            t_use = t_factor_relax

        return self._integrate(y0, z, pinned, x1_0, x2_0, x4_0, N_grid, t_use, k_rxn, no_cfd)

    def _integrate(self, y0, z, pinned, x1_0, x2_0, x4_0, N_grid, t_factor, k_rxn, no_cfd):
        dz = z[1] - z[0]
        if k_rxn is None:
            k_rxn = 200 * self.ct * self.D_min / dz ** 2  # Sect. C2.3

        def rhs(t, y):
            x1 = y[0:N_grid].copy()
            x2 = y[N_grid:2 * N_grid].copy()
            x4 = y[2 * N_grid:3 * N_grid].copy()
            x1[0], x2[0], x4[0] = x1_0, x2_0, x4_0
            x2[pinned] = self.x_sat

            F1, F2, _, F4 = self._solve_all_fluxes(x1, x2, x4, dz, pinned, no_cfd=no_cfd)

            dx1dt = np.zeros(N_grid)
            dx2dt = np.zeros(N_grid)
            dx4dt = np.zeros(N_grid)
            dx1dt[1:-1] = -(F1[1:] - F1[:-1]) / dz
            dx2dt[1:-1] = np.where(pinned[1:-1], 0.0, -(F2[1:] - F2[:-1]) / dz)
            dx4dt[1:-1] = -(F4[1:] - F4[:-1]) / dz
            d_rxn_O2, d_rxn_CO2 = self._wall_reaction(x1[-1], x4[-1], k_rxn)
            dx1dt[-1] = F1[-1] / dz + d_rxn_O2
            dx4dt[-1] = F4[-1] / dz + d_rxn_CO2
            return np.concatenate([dx1dt, dx2dt, dx4dt]) / self.ct  # c_t dx/dt = -dF/dz + reaction

        t_diff = self.L ** 2 / self.D_min
        sol = solve_ivp(rhs, [0, t_factor * t_diff], y0, method='BDF', rtol=1e-10, atol=1e-13)

        x1f = sol.y[0:N_grid, -1]
        x2f = sol.y[N_grid:2 * N_grid, -1]
        x4f = sol.y[2 * N_grid:3 * N_grid, -1]
        x1f[0], x2f[0], x4f[0] = x1_0, x2_0, x4_0
        x2f[pinned] = self.x_sat

        F1, F2, F3, F4 = self._solve_all_fluxes(x1f, x2f, x4f, dz, pinned, no_cfd=no_cfd)

        return {
            "F_O2": F1.mean(),
            "F_CO2": F4.mean(),
            "F_H2O": F2,            # per segment
            "F_N2": F3,             # per segment
            "x_CO2_wall": x4f[-1],
            "x_O2": x1f, "x_H2O": x2f, "x_CO2": x4f, "z": z,
        }
