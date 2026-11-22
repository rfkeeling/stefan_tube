import pytest

from stefan_tube import StefanTubeBVP, AMBIENT_CHI_O2_DRY


def test_wet_sink_reference():
    """Ambient dry air at z = 0, saturated wall (x_H2O = 0.03126) with chi_O2,d = 0 at z = L."""
    r = StefanTubeBVP().solve(x_h2o_0=0.010, chi_O2d_0=AMBIENT_CHI_O2_DRY, x_h2o_L=0.03126, chi_O2d_L=0.0)
    assert r["F_O2"] == pytest.approx(1.6378421178e-03, rel=1e-6)
    assert r["x_CO2_wall"] == pytest.approx(0.2414218074, rel=1e-6)
