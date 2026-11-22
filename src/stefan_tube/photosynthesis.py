"""
Photosynthesis variant of the pivot tube (Sect. 5.2 of the main text): O2 is
produced at the end wall and CO2 consumed, in equimolar amounts.

CO2 at the last grid point is relaxed toward a compensation level
chi_CO2_comp (mole fraction, moist basis) at rate k_rxn * (x_CO2(wall) -
chi_CO2_comp) [mol m-3 s-1], instead of toward zero, because net photosynthetic uptake
stops at a nonzero CO2 compensation point. As in the respiration model,
x_CO2(wall) settles just above the target by the fast-but-finite-reaction
residual F / (k_rxn dz).
"""
from .solver import StefanTubeSolver


class StefanTubeSolverPhotosynthesis(StefanTubeSolver):
    def solve(self, x_h2o_0, z_pivot, chi_CO2_comp, **kwargs):
        self.chi_CO2_comp = chi_CO2_comp
        return super().solve(x_h2o_0, z_pivot, **kwargs)

    def _wall_reaction(self, x_O2_wall, x_CO2_wall, k_rxn):
        rate = k_rxn * (x_CO2_wall - self.chi_CO2_comp)
        return +rate, -rate
