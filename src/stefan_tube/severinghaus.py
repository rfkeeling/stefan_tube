"""
Closed-form Severinghaus et al. (1996) prediction used in Fig. 3 (Eq. 5 of the
main text; derivation in Appendix B), for the dry-air O2 fraction at a
water-saturated surface given ambient air and a humidity difference.
"""
from .constants import D_RATIO_SEVERINGHAUS


def severinghaus_chi_wall(x_h2o_0, x_sat, chi_o, d_ratio=D_RATIO_SEVERINGHAUS):
    """
    chi at the water-saturated end wall, to first order in eps = 1 - d_ratio,
    where chi = x_O2 / (1 - x_H2O) is the dry-air O2 fraction and chi_o its
    ambient value (chi_O2,d):

        chi_wall = chi_o + eps chi_o (1 - chi_o) / (1 - x_H2O,0) (x_sat - x_H2O,0)
    """
    eps = 1.0 - d_ratio
    return chi_o + eps * chi_o * (1.0 - chi_o) / (1.0 - x_h2o_0) * (x_sat - x_h2o_0)
