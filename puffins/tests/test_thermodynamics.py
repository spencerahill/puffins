"""Tests for thermodynamics module."""

import numpy as np
import pytest

from puffins.constants import C_P, GRAV_EARTH, L_V, P0, R_D
from puffins.thermodynamics import (
    dsat_entrop_dtemp_approx,
    equiv_pot_temp,
    exner_func,
    mixing_ratio,
    moist_enthalpy,
    moist_entropy,
    moist_static_energy,
    pot_temp,
    pseudoadiabatic_lapse_rate,
    rel_hum_from_temp_dewpoint,
    relative_humidity,
    sat_equiv_pot_temp,
    sat_vap_press_tetens_kelvin,
    saturation_entropy,
    saturation_mixing_ratio,
    saturation_mse,
    saturation_specific_humidity,
    specific_humidity,
    temp_from_equiv_pot_temp,
    vap_press_from_mix_ratio,
    water_vapor_mixing_ratio,
)

# ---------------------------------------------------------------------------
# TestExnerFunc
# ---------------------------------------------------------------------------


class TestExnerFunc:
    """Tests for exner_func."""

    def test_reference_pressure(self) -> None:
        """Exner function equals 1 at p = p0."""
        result = exner_func(1000.0)
        np.testing.assert_allclose(result, 1.0)

    def test_half_pressure(self) -> None:
        """Exner function at half reference pressure."""
        result = exner_func(500.0)
        expected = (500.0 / 1000.0) ** (R_D / C_P)
        np.testing.assert_allclose(result, expected)

    def test_array_input(self) -> None:
        """Works with numpy array input."""
        pressures = np.array([500.0, 750.0, 1000.0])
        result = exner_func(pressures)
        assert result.shape == (3,)
        np.testing.assert_allclose(result[-1], 1.0)

    def test_custom_p0(self) -> None:
        """Exner function with custom reference pressure."""
        result = exner_func(1e5, p0=1e5)
        np.testing.assert_allclose(result, 1.0)


# ---------------------------------------------------------------------------
# TestPotTemp
# ---------------------------------------------------------------------------


class TestPotTemp:
    """Tests for pot_temp."""

    def test_at_reference_pressure(self) -> None:
        """Potential temperature equals temperature at p0."""
        result = pot_temp(300.0, 1000.0)
        np.testing.assert_allclose(result, 300.0)

    def test_lower_pressure(self) -> None:
        """Potential temperature is higher than actual temp at lower pressure."""
        result = pot_temp(250.0, 500.0)
        assert result > 250.0

    def test_array_input(self) -> None:
        """Works with numpy arrays."""
        temps = np.array([250.0, 275.0, 300.0])
        pressures = np.array([500.0, 750.0, 1000.0])
        result = pot_temp(temps, pressures)
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# TestMoistEnthalpy
# ---------------------------------------------------------------------------


class TestMoistEnthalpy:
    """Tests for moist_enthalpy."""

    def test_zero_humidity(self) -> None:
        """Moist enthalpy equals temperature when humidity is zero."""
        result = moist_enthalpy(300.0, 0.0)
        np.testing.assert_allclose(result, 300.0)

    def test_positive_humidity(self) -> None:
        """Moist enthalpy exceeds temperature with nonzero humidity."""
        result = moist_enthalpy(300.0, 0.01)
        assert result > 300.0

    def test_known_value(self) -> None:
        """Check against hand-calculated value."""
        temp, sphum = 300.0, 0.01
        expected = temp + L_V * sphum / C_P
        np.testing.assert_allclose(moist_enthalpy(temp, sphum), expected)


# ---------------------------------------------------------------------------
# TestWaterVaporMixingRatio
# ---------------------------------------------------------------------------


class TestWaterVaporMixingRatio:
    """Tests for water_vapor_mixing_ratio."""

    def test_zero_vapor_pressure(self) -> None:
        """Mixing ratio is zero when vapor pressure is zero."""
        result = water_vapor_mixing_ratio(0.0, 1e5)
        np.testing.assert_allclose(result, 0.0)

    def test_positive_value(self) -> None:
        """Mixing ratio is positive for positive vapor pressure."""
        result = water_vapor_mixing_ratio(1000.0, 1e5)
        assert result > 0.0

    def test_known_value(self) -> None:
        """Check against hand-calculated value."""
        from puffins.constants import EPSILON

        e, p = 2000.0, 1e5
        expected = EPSILON * e / (p - e)
        np.testing.assert_allclose(water_vapor_mixing_ratio(e, p), expected)


# ---------------------------------------------------------------------------
# TestVapPressFromMixRatio
# ---------------------------------------------------------------------------


class TestVapPressFromMixRatio:
    """Tests for vap_press_from_mix_ratio."""

    def test_zero_mixing_ratio(self) -> None:
        """Vapor pressure is zero when mixing ratio is zero."""
        result = vap_press_from_mix_ratio(0.0, 1e5)
        np.testing.assert_allclose(result, 0.0)

    def test_roundtrip(self) -> None:
        """Roundtrip: mixing ratio -> vapor pressure -> mixing ratio."""
        e_orig = 2000.0
        p = 1e5
        w = water_vapor_mixing_ratio(e_orig, p)
        e_recovered = vap_press_from_mix_ratio(w, p)
        np.testing.assert_allclose(e_recovered, e_orig)


# ---------------------------------------------------------------------------
# TestSpecificHumidity / TestMixingRatio
# ---------------------------------------------------------------------------


class TestSpecificHumidity:
    """Tests for specific_humidity."""

    def test_zero(self) -> None:
        result = specific_humidity(0.0)
        np.testing.assert_allclose(result, 0.0)

    def test_small_mixing_ratio(self) -> None:
        """For small w, q ≈ w."""
        w = 0.01
        q = specific_humidity(w)
        np.testing.assert_allclose(q, w, atol=1e-3)

    def test_array(self) -> None:
        w = np.array([0.0, 0.005, 0.01, 0.02])
        q = specific_humidity(w)
        assert q.shape == (4,)
        assert np.all(q <= w)


class TestMixingRatioFunc:
    """Tests for mixing_ratio."""

    def test_zero(self) -> None:
        result = mixing_ratio(0.0)
        np.testing.assert_allclose(result, 0.0)

    def test_roundtrip(self) -> None:
        """specific_humidity and mixing_ratio are inverses."""
        w_orig = 0.015
        q = specific_humidity(w_orig)
        w_recovered = mixing_ratio(q)
        np.testing.assert_allclose(w_recovered, w_orig)

    def test_array(self) -> None:
        q = np.array([0.0, 0.005, 0.01])
        w = mixing_ratio(q)
        assert np.all(w >= q)


# ---------------------------------------------------------------------------
# TestSatVapPressTetensKelvin
# ---------------------------------------------------------------------------


class TestSatVapPressTetensKelvin:
    """Tests for sat_vap_press_tetens_kelvin."""

    def test_positive(self) -> None:
        """Saturation vapor pressure is positive for Earth-like temps."""
        result = sat_vap_press_tetens_kelvin(300.0)
        assert result > 0.0

    def test_increases_with_temp(self) -> None:
        """Clausius-Clapeyron: sat vap press increases with temperature."""
        temps = np.array([270.0, 280.0, 290.0, 300.0])
        result = sat_vap_press_tetens_kelvin(temps)
        assert np.all(np.diff(result) > 0)

    def test_approximate_value_at_20c(self) -> None:
        """Roughly 2337 Pa at 20°C (293.15 K), within 5%."""
        result = sat_vap_press_tetens_kelvin(293.15)
        np.testing.assert_allclose(result, 2337.0, rtol=0.05)

    def test_array_input(self) -> None:
        temps = np.array([273.15, 293.15, 313.15])
        result = sat_vap_press_tetens_kelvin(temps)
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# TestSaturationMixingRatio / TestSaturationSpecificHumidity
# ---------------------------------------------------------------------------


class TestSaturationMixingRatio:
    """Tests for saturation_mixing_ratio."""

    def test_positive(self) -> None:
        result = saturation_mixing_ratio(1e5, 300.0)
        assert result > 0.0

    def test_increases_with_temp(self) -> None:
        temps = np.array([270.0, 280.0, 290.0, 300.0])
        results = [saturation_mixing_ratio(1e5, t) for t in temps]
        assert all(results[i] < results[i + 1] for i in range(len(results) - 1))

    def test_decreases_with_pressure(self) -> None:
        """Saturation mixing ratio decreases with increasing pressure at fixed T."""
        r1 = saturation_mixing_ratio(5e4, 300.0)
        r2 = saturation_mixing_ratio(1e5, 300.0)
        assert r1 > r2


class TestSaturationSpecificHumidity:
    """Tests for saturation_specific_humidity."""

    def test_positive(self) -> None:
        result = saturation_specific_humidity(1e5, 300.0)
        assert result > 0.0

    def test_less_than_sat_mixing_ratio(self) -> None:
        """Sat specific humidity < sat mixing ratio (by definition)."""
        q_sat = saturation_specific_humidity(1e5, 300.0)
        w_sat = saturation_mixing_ratio(1e5, 300.0)
        assert q_sat < w_sat


# ---------------------------------------------------------------------------
# TestRelativeHumidity
# ---------------------------------------------------------------------------


class TestRelativeHumidity:
    """Tests for relative_humidity."""

    def test_saturated(self) -> None:
        """RH = 1 when vapor pressure equals saturation."""
        result = relative_humidity(2000.0, 2000.0)
        np.testing.assert_allclose(result, 1.0)

    def test_half_saturated(self) -> None:
        result = relative_humidity(1000.0, 2000.0)
        np.testing.assert_allclose(result, 0.5)

    def test_array(self) -> None:
        e = np.array([500.0, 1000.0, 2000.0])
        e_sat = np.array([2000.0, 2000.0, 2000.0])
        result = relative_humidity(e, e_sat)
        np.testing.assert_allclose(result, [0.25, 0.5, 1.0])


# ---------------------------------------------------------------------------
# TestRelHumFromTempDewpoint
# ---------------------------------------------------------------------------


class TestRelHumFromTempDewpoint:
    """Tests for rel_hum_from_temp_dewpoint."""

    def test_dewpoint_equals_temp(self) -> None:
        """RH = 1 when dewpoint equals temperature."""
        result = rel_hum_from_temp_dewpoint(300.0, 300.0)
        np.testing.assert_allclose(result, 1.0)

    def test_dewpoint_below_temp(self) -> None:
        """RH < 1 when dewpoint is below temperature."""
        result = rel_hum_from_temp_dewpoint(300.0, 290.0)
        assert 0.0 < result < 1.0

    def test_array(self) -> None:
        temps = np.array([300.0, 300.0])
        dews = np.array([300.0, 290.0])
        result = rel_hum_from_temp_dewpoint(temps, dews)
        np.testing.assert_allclose(result[0], 1.0)
        assert result[1] < 1.0


# ---------------------------------------------------------------------------
# TestMoistStaticEnergy
# ---------------------------------------------------------------------------


class TestMoistStaticEnergy:
    """Tests for moist_static_energy."""

    def test_zero_height_zero_humidity(self) -> None:
        """MSE = c_p * T when height and humidity are zero."""
        result = moist_static_energy(300.0, 0.0, 0.0)
        np.testing.assert_allclose(result, C_P * 300.0)

    def test_known_value(self) -> None:
        temp, height, q = 300.0, 5000.0, 0.015
        expected = C_P * temp + GRAV_EARTH * height + L_V * q
        np.testing.assert_allclose(moist_static_energy(temp, height, q), expected)


# ---------------------------------------------------------------------------
# TestSaturationMSE
# ---------------------------------------------------------------------------


class TestSaturationMSE:
    """Tests for saturation_mse."""

    def test_exceeds_dry_mse(self) -> None:
        """Saturation MSE >= MSE with zero humidity."""
        dry_mse = moist_static_energy(300.0, 0.0, 0.0)
        sat = saturation_mse(300.0, 0.0)
        assert sat >= dry_mse

    def test_increases_with_temp(self) -> None:
        s1 = saturation_mse(280.0, 0.0)
        s2 = saturation_mse(300.0, 0.0)
        assert s2 > s1


# ---------------------------------------------------------------------------
# TestSaturationEntropy
# ---------------------------------------------------------------------------


class TestSaturationEntropy:
    """Tests for saturation_entropy."""

    def test_positive(self) -> None:
        result = saturation_entropy(300.0)
        assert result > 0.0

    def test_increases_with_temp(self) -> None:
        s1 = saturation_entropy(280.0)
        s2 = saturation_entropy(300.0)
        assert s2 > s1

    def test_with_provided_sat_vap_press(self) -> None:
        """Providing sat_vap_press explicitly gives same result as default."""
        svp = sat_vap_press_tetens_kelvin(300.0)
        result_auto = saturation_entropy(300.0)
        result_manual = saturation_entropy(300.0, sat_vap_press=svp)
        np.testing.assert_allclose(result_auto, result_manual)


# ---------------------------------------------------------------------------
# TestDsatEntropDtempApprox
# ---------------------------------------------------------------------------


class TestDsatEntropDtempApprox:
    """Tests for dsat_entrop_dtemp_approx."""

    def test_positive(self) -> None:
        """Derivative is positive (entropy increases with temperature)."""
        result = dsat_entrop_dtemp_approx(300.0)
        assert result > 0.0

    def test_increases_with_temp_at_fixed_pressure(self) -> None:
        """Derivative is larger at warmer temperatures (more moisture)."""
        cold = dsat_entrop_dtemp_approx(260.0)
        warm = dsat_entrop_dtemp_approx(300.0)
        assert warm > cold

    def test_array_input(self) -> None:
        temps = np.array([260.0, 280.0, 300.0])
        result = dsat_entrop_dtemp_approx(temps)
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# TestEquivPotTemp
# ---------------------------------------------------------------------------


class TestEquivPotTemp:
    """Tests for equiv_pot_temp."""

    def test_exceeds_temperature(self) -> None:
        """Equivalent potential temperature >= actual temperature."""
        result = equiv_pot_temp(300.0, 0.7, 1e5)
        assert result >= 300.0

    def test_dry_limit(self) -> None:
        """At RH=0 (approximately), theta_e approaches potential temperature."""
        # Use very small RH to avoid log(0)
        result = equiv_pot_temp(300.0, 1e-10, 1e5)
        theta = pot_temp(300.0, 1e5, p0=P0)
        # Should be within same order of magnitude
        np.testing.assert_allclose(result, theta, rtol=0.1)

    def test_increases_with_humidity(self) -> None:
        t1 = equiv_pot_temp(300.0, 0.3, 1e5)
        t2 = equiv_pot_temp(300.0, 0.9, 1e5)
        assert t2 > t1


# ---------------------------------------------------------------------------
# TestSatEquivPotTemp
# ---------------------------------------------------------------------------


class TestSatEquivPotTemp:
    """Tests for sat_equiv_pot_temp."""

    def test_equals_equiv_pot_temp_at_saturation(self) -> None:
        """sat_equiv_pot_temp is equiv_pot_temp with rel_hum=1."""
        sat = sat_equiv_pot_temp(300.0, 1e5)
        full = equiv_pot_temp(300.0, 1.0, 1e5)
        np.testing.assert_allclose(sat, full)

    def test_exceeds_equiv_pot_temp(self) -> None:
        """Saturation theta_e >= theta_e at sub-saturation."""
        sat = sat_equiv_pot_temp(300.0, 1e5)
        subsaturated = equiv_pot_temp(300.0, 0.5, 1e5)
        assert sat >= subsaturated


# ---------------------------------------------------------------------------
# TestTempFromEquivPotTemp
# ---------------------------------------------------------------------------


class TestTempFromEquivPotTemp:
    """Tests for temp_from_equiv_pot_temp."""

    def test_scalar_roundtrip(self) -> None:
        """Roundtrip: T -> theta_e -> T."""
        temp_orig = 290.0
        rh = 0.7
        theta_e = equiv_pot_temp(temp_orig, rh, P0)
        temp_recovered = temp_from_equiv_pot_temp(theta_e, rel_hum=rh, pressure=P0)
        np.testing.assert_allclose(temp_recovered, temp_orig, atol=1.0)

    def test_result_less_than_theta_e(self) -> None:
        """Temperature < equivalent potential temperature."""
        # Use a value that converges with default parameters
        theta_e = 330.0
        temp = temp_from_equiv_pot_temp(theta_e)
        assert not np.isnan(temp)
        assert temp < theta_e

    def test_array_input(self) -> None:
        """Works with array of theta_e values."""
        theta_es = np.array([330.0, 340.0, 350.0])
        temps = temp_from_equiv_pot_temp(theta_es)
        # Should return array-like with same length
        assert len(np.atleast_1d(temps)) == 3


# ---------------------------------------------------------------------------
# TestMoistEntropy
# ---------------------------------------------------------------------------


class TestMoistEntropy:
    """Tests for moist_entropy."""

    def test_positive(self) -> None:
        result = moist_entropy(300.0, 0.7, 1e5)
        assert result > 0.0

    def test_increases_with_temp(self) -> None:
        s1 = moist_entropy(280.0, 0.7, 1e5)
        s2 = moist_entropy(300.0, 0.7, 1e5)
        assert s2 > s1


# ---------------------------------------------------------------------------
# TestPseudoadiabaticLapseRate
# ---------------------------------------------------------------------------


class TestPseudoadiabaticLapseRate:
    """Tests for pseudoadiabatic_lapse_rate."""

    def test_positive(self) -> None:
        """Lapse rate is positive (temperature decreases with height)."""
        result = pseudoadiabatic_lapse_rate(300.0, 1e5)
        assert result > 0.0

    def test_less_than_dry_adiabatic(self) -> None:
        """Pseudoadiabatic lapse rate < dry adiabatic lapse rate (g/c_p)."""
        dry_lapse = GRAV_EARTH / C_P
        result = pseudoadiabatic_lapse_rate(300.0, 1e5, rel_hum=1.0)
        assert result < dry_lapse

    def test_approaches_dry_at_low_temp(self) -> None:
        """At very cold temperatures, approaches dry adiabatic rate."""
        dry_lapse = GRAV_EARTH / C_P
        cold_result = pseudoadiabatic_lapse_rate(200.0, 1e5)
        np.testing.assert_allclose(cold_result, dry_lapse, rtol=0.05)

    def test_array_input(self) -> None:
        temps = np.array([260.0, 280.0, 300.0])
        pressures = np.array([8e4, 9e4, 1e5])
        result = pseudoadiabatic_lapse_rate(temps, pressures)
        assert result.shape == (3,)
