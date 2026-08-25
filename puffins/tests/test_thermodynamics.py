"""Tests for thermodynamics module."""

import numpy as np
import pytest
import xarray as xr

from puffins.constants import C_P, EPSILON, GRAV_EARTH, L_V, P0, R_D
from puffins.thermodynamics import (
    dry_static_energy,
    dsat_entrop_dtemp_approx,
    equiv_pot_temp,
    exner_func,
    kinetic_energy,
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
    total_energy,
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
        results = saturation_mixing_ratio(1e5, temps)
        assert np.all(np.diff(results) > 0)

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
# TestKineticEnergy
# ---------------------------------------------------------------------------


class TestKineticEnergy:
    """Tests for kinetic_energy."""

    def test_known_value(self) -> None:
        """Reconstruct 0.5 * (u^2 + v^2) from raw numbers."""
        u, v = 3.0, 4.0
        np.testing.assert_allclose(kinetic_energy(u, v), 0.5 * (9.0 + 16.0))

    def test_zero_wind(self) -> None:
        """Zero wind has zero kinetic energy."""
        np.testing.assert_allclose(kinetic_energy(0.0, 0.0), 0.0)

    def test_symmetric_in_components(self) -> None:
        """Swapping u and v leaves the result unchanged."""
        np.testing.assert_allclose(kinetic_energy(2.0, 7.0), kinetic_energy(7.0, 2.0))

    def test_even_in_sign(self) -> None:
        """Reversing the flow direction leaves the result unchanged."""
        np.testing.assert_allclose(kinetic_energy(-2.0, -7.0), kinetic_energy(2.0, 7.0))

    def test_quadratic_scaling(self) -> None:
        """Doubling both components quadruples the kinetic energy."""
        np.testing.assert_allclose(
            kinetic_energy(4.0, 6.0), 4.0 * kinetic_energy(2.0, 3.0)
        )

    def test_dataarray_input(self) -> None:
        """DataArray input returns a DataArray with the same coords."""
        u = xr.DataArray([1.0, 2.0], dims="x", coords={"x": [0.0, 1.0]}, name="u")
        v = xr.DataArray([2.0, 4.0], dims="x", coords={"x": [0.0, 1.0]}, name="v")
        result = kinetic_energy(u, v)
        assert isinstance(result, xr.DataArray)
        np.testing.assert_allclose(result.values, [2.5, 10.0])
        np.testing.assert_allclose(result["x"].values, [0.0, 1.0])


# ---------------------------------------------------------------------------
# TestDryStaticEnergy
# ---------------------------------------------------------------------------


class TestDryStaticEnergy:
    """Tests for dry_static_energy."""

    def test_known_value_defaults(self) -> None:
        """Reconstruct c_p * T + g * z from the module's constants."""
        temp, height = 300.0, 5000.0
        expected = C_P * temp + GRAV_EARTH * height
        np.testing.assert_allclose(dry_static_energy(temp, height), expected)

    def test_known_value_nondefault_coeffs(self) -> None:
        """Both c_p and grav are honored, reconstructed from raw numbers.

        Uses values unlike the defaults in both magnitude and ratio, so a
        swapped or dropped coefficient cannot coincidentally pass.
        """
        temp, height = 250.0, 8000.0
        c_p, grav = 800.0, 3.71  # Roughly Mars.
        expected = 800.0 * 250.0 + 3.71 * 8000.0
        np.testing.assert_allclose(
            dry_static_energy(temp, height, c_p=c_p, grav=grav), expected
        )

    def test_zero_height(self) -> None:
        """At zero height, DSE reduces to c_p * T."""
        np.testing.assert_allclose(dry_static_energy(300.0, 0.0), C_P * 300.0)

    def test_c_p_scales_temp_term_only(self) -> None:
        """Doubling c_p adds exactly one more c_p * T, leaving g * z alone."""
        temp, height = 280.0, 4000.0
        base = dry_static_energy(temp, height, c_p=C_P)
        doubled = dry_static_energy(temp, height, c_p=2 * C_P)
        np.testing.assert_allclose(doubled - base, C_P * temp)

    def test_grav_scales_height_term_only(self) -> None:
        """Doubling grav adds exactly one more g * z, leaving c_p * T alone."""
        temp, height = 280.0, 4000.0
        base = dry_static_energy(temp, height, grav=GRAV_EARTH)
        doubled = dry_static_energy(temp, height, grav=2 * GRAV_EARTH)
        np.testing.assert_allclose(doubled - base, GRAV_EARTH * height)

    def test_is_mse_minus_latent_term(self) -> None:
        """DSE equals MSE with the latent term removed."""
        temp, height, q = 290.0, 3000.0, 0.012
        np.testing.assert_allclose(
            dry_static_energy(temp, height),
            moist_static_energy(temp, height, q) - L_V * q,
        )

    def test_dataarray_input(self) -> None:
        """DataArray input returns a DataArray."""
        temp = xr.DataArray([280.0, 300.0], dims="x")
        height = xr.DataArray([0.0, 5000.0], dims="x")
        result = dry_static_energy(temp, height)
        assert isinstance(result, xr.DataArray)
        np.testing.assert_allclose(
            result.values, [C_P * 280.0, C_P * 300.0 + GRAV_EARTH * 5000.0]
        )


# ---------------------------------------------------------------------------
# TestTotalEnergy
# ---------------------------------------------------------------------------


class TestTotalEnergy:
    """Tests for total_energy."""

    def test_known_value_defaults(self) -> None:
        """Reconstruct c_p*T + g*z + L_v*q + 0.5*(u^2+v^2) from constants."""
        u, v, temp, height, q = 10.0, 5.0, 290.0, 3000.0, 0.012
        expected = C_P * temp + GRAV_EARTH * height + L_V * q + 0.5 * (10.0**2 + 5.0**2)
        np.testing.assert_allclose(total_energy(u, v, temp, height, q), expected)

    def test_known_value_nondefault_coeffs(self) -> None:
        """All three of c_p, grav, and l_v are honored simultaneously.

        Reconstructed entirely from raw numbers, so a coefficient attached to
        the wrong term fails here.
        """
        u, v, temp, height, q = 8.0, 6.0, 250.0, 4000.0, 0.02
        c_p, grav, l_v = 800.0, 3.71, 1.2e6
        expected = (
            800.0 * 250.0 + 3.71 * 4000.0 + 1.2e6 * 0.02 + 0.5 * (8.0**2 + 6.0**2)
        )
        np.testing.assert_allclose(
            total_energy(u, v, temp, height, q, c_p=c_p, grav=grav, l_v=l_v),
            expected,
        )

    def test_l_v_scales_moisture_term_only(self) -> None:
        """Doubling l_v adds exactly one more L_v * q."""
        args = (10.0, 5.0, 290.0, 3000.0, 0.012)
        base = total_energy(*args, l_v=L_V)
        doubled = total_energy(*args, l_v=2 * L_V)
        np.testing.assert_allclose(doubled - base, L_V * 0.012)

    def test_c_p_scales_temp_term_only(self) -> None:
        """Doubling c_p adds exactly one more c_p * T."""
        args = (10.0, 5.0, 290.0, 3000.0, 0.012)
        base = total_energy(*args, c_p=C_P)
        doubled = total_energy(*args, c_p=2 * C_P)
        np.testing.assert_allclose(doubled - base, C_P * 290.0)

    def test_grav_scales_height_term_only(self) -> None:
        """Doubling grav adds exactly one more g * z."""
        args = (10.0, 5.0, 290.0, 3000.0, 0.012)
        base = total_energy(*args, grav=GRAV_EARTH)
        doubled = total_energy(*args, grav=2 * GRAV_EARTH)
        np.testing.assert_allclose(doubled - base, GRAV_EARTH * 3000.0)

    def test_equals_mse_plus_ke(self) -> None:
        """Total energy is the sum of its two documented pieces."""
        u, v, temp, height, q = 12.0, -4.0, 285.0, 2500.0, 0.009
        np.testing.assert_allclose(
            total_energy(u, v, temp, height, q),
            moist_static_energy(temp, height, q) + kinetic_energy(u, v),
        )

    def test_zero_wind_reduces_to_mse(self) -> None:
        """With no wind, total energy is just moist static energy."""
        temp, height, q = 285.0, 2500.0, 0.009
        np.testing.assert_allclose(
            total_energy(0.0, 0.0, temp, height, q),
            moist_static_energy(temp, height, q),
        )

    def test_dataarray_input(self) -> None:
        """DataArray input returns a DataArray."""
        ones = xr.DataArray([1.0, 1.0], dims="x")
        result = total_energy(10.0 * ones, 0.0 * ones, 290.0 * ones, 0.0 * ones, 0.0)
        assert isinstance(result, xr.DataArray)
        np.testing.assert_allclose(result.values, C_P * 290.0 + 50.0)


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

    def test_with_provided_sat_vap_press_matches_default(self) -> None:
        """Providing the Tetens sat_vap_press gives the same result as default."""
        svp = sat_vap_press_tetens_kelvin(300.0)
        result_auto = saturation_entropy(300.0)
        result_manual = saturation_entropy(300.0, sat_vap_press=svp)
        np.testing.assert_allclose(result_auto, result_manual)

    def test_sat_vap_press_parameter_affects_result(self) -> None:
        """Providing a different sat_vap_press changes the result."""
        correct_svp = sat_vap_press_tetens_kelvin(300.0)
        wrong_svp = correct_svp * 0.5
        r1 = saturation_entropy(300.0, sat_vap_press=correct_svp)
        r2 = saturation_entropy(300.0, sat_vap_press=wrong_svp)
        assert not np.allclose(r1, r2)


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


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
class TestTempFromEquivPotTemp:
    """Tests for temp_from_equiv_pot_temp.

    brentq probes temperatures near the top of its bracket where the Tetens
    saturation pressure exceeds ambient, producing transient float overflows
    in np.exp / scalar power. The solver still converges, so the warnings
    are filtered at the class level.
    """

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

    def test_zero_dim_array(self) -> None:
        """Works with 0-dimensional numpy array."""
        theta_e = np.float64(330.0)
        temp = temp_from_equiv_pot_temp(theta_e)
        assert not np.isnan(temp)
        assert temp < theta_e


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

    def test_with_tot_wat_mix_ratio(self) -> None:
        """Non-None tot_wat_mix_ratio changes the result."""
        s_default = moist_entropy(300.0, 0.7, 1e5)
        s_with_water = moist_entropy(300.0, 0.7, 1e5, tot_wat_mix_ratio=0.02)
        assert not np.allclose(s_default, s_with_water)


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


# ---------------------------------------------------------------------------
# DataArray input tests
# ---------------------------------------------------------------------------


class TestDataArrayInputs:
    """Tests that key functions work with xarray DataArray inputs."""

    def test_sat_vap_press_dataarray(self) -> None:
        temps = xr.DataArray([270.0, 280.0, 290.0, 300.0], dims="temp")
        result = sat_vap_press_tetens_kelvin(temps)
        assert isinstance(result, xr.DataArray)
        assert result.dims == ("temp",)
        assert np.all(np.diff(result.values) > 0)

    def test_pot_temp_dataarray(self) -> None:
        temp = xr.DataArray([250.0, 275.0, 300.0], dims="lev")
        pressure = xr.DataArray([500.0, 750.0, 1000.0], dims="lev")
        result = pot_temp(temp, pressure)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (3,)
        # At reference pressure, pot temp = temp
        np.testing.assert_allclose(result.values[-1], 300.0)

    def test_moist_static_energy_dataarray(self) -> None:
        temp = xr.DataArray([280.0, 300.0], dims="x")
        height = xr.DataArray([0.0, 5000.0], dims="x")
        spec_hum = xr.DataArray([0.005, 0.015], dims="x")
        result = moist_static_energy(temp, height, spec_hum)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (2,)

    def test_equiv_pot_temp_dataarray(self) -> None:
        temp = xr.DataArray([280.0, 290.0, 300.0], dims="lat")
        result = equiv_pot_temp(temp, 0.7, 1e5)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (3,)
        # theta_e >= T everywhere
        assert np.all(result.values >= temp.values)
