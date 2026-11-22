"""
stefan_tube: Stefan-Maxwell multicomponent diffusion in a tube containing
O2, H2O, N2 and CO2, with N2 stagnant and equimolar O2-CO2 exchange at the end
wall, and a "pivot" beyond which the side wall is moist (H2O held at
saturation). Code supporting Figs. 2-3 and Appendix C of Keeling et al.
"""
import os

# The flux linear system is small (~240 x 240) and solved tens of thousands of
# times per run; multithreaded BLAS is much slower than one thread here.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import threadpoolctl

threadpoolctl.threadpool_limits(1)

from .constants import (R, AMBIENT_CHI_O2_DRY, AMBIENT_CHI_CO2_DRY, D_RATIO_SEVERINGHAUS,
                        FULLER_V, MOLAR_MASS, fuller_diffusivity, binary_diffusivities,
                        e_sat_bolton, x_sat_bolton)
from .bvp import StefanTubeBVP
from .solver import StefanTubeSolver
from .photosynthesis import StefanTubeSolverPhotosynthesis
from .severinghaus import severinghaus_chi_wall
