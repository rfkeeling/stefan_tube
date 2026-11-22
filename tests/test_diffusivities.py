import pytest

from stefan_tube import binary_diffusivities, x_sat_bolton, D_RATIO_SEVERINGHAUS

T, P = 298.15, 101325.0


def test_table_C1_diffusivities():
    """Table C1 (cm2 s-1), with the Severinghaus-corrected O2-H2O value."""
    D = {k: v * 1e4 for k, v in binary_diffusivities(T, P).items()}
    assert D["O2-N2"] == pytest.approx(0.2069, abs=5e-5)
    assert D["H2O-N2"] == pytest.approx(0.2632, abs=5e-5)
    assert D["O2-CO2"] == pytest.approx(0.1612, abs=5e-5)
    assert D["H2O-CO2"] == pytest.approx(0.2102, abs=5e-5)
    assert D["N2-CO2"] == pytest.approx(0.1638, abs=5e-5)
    assert D["O2-H2O"] == pytest.approx(0.2728, abs=5e-5)
    assert D["H2O-N2"] / D["O2-H2O"] == pytest.approx(D_RATIO_SEVERINGHAUS)
    assert min(D.values()) == D["O2-CO2"]


def test_saturation_mole_fraction():
    assert x_sat_bolton(T, P) == pytest.approx(0.03126, abs=1e-6)
