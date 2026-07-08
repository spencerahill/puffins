"""Tests for therm_inert module."""

import numpy as np
import pytest
import xarray as xr

from puffins._typing import ArrayLike
from puffins.constants import (
    C_VL,
    DENS_LIQ_WAT,
    ORB_FREQ_EARTH,
    STEF_BOLTZ_CONST,
    THETA_REF,
)
from puffins.therm_inert import (
    mixed_layer_heat_cap,
    seas_damp_factor,
    seas_lag,
    seas_therm_inert_ratio,
    temp_rad_eq_eff,
    therm_inert_timescale,
)


class TestMixedLayerHeatCap:
    """Tests for mixed_layer_heat_cap."""

    def test_unit_depth(self) -> None:
        """A 1 m column of water has heat capacity density * spec_heat."""
        result = mixed_layer_heat_cap(1.0)
        np.testing.assert_allclose(result, DENS_LIQ_WAT * C_VL)

    def test_linear_in_depth(self) -> None:
        """Heat capacity scales linearly with depth."""
        assert mixed_layer_heat_cap(2.0) == pytest.approx(2 * mixed_layer_heat_cap(1.0))

    def test_positive(self) -> None:
        """Heat capacity is positive for a positive depth."""
        assert mixed_layer_heat_cap(50.0) > 0


class TestThermInertTimescale:
    """Tests for therm_inert_timescale."""

    def test_positive(self) -> None:
        """Timescale is positive for positive inputs."""
        assert therm_inert_timescale(1e8, 300.0) > 0

    def test_scales_inverse_cube_temp(self) -> None:
        """Timescale scales as T^-3 (radiative restoring ~ T^3)."""
        tau_cold = therm_inert_timescale(1e8, 250.0)
        tau_warm = therm_inert_timescale(1e8, 300.0)
        np.testing.assert_allclose(tau_cold / tau_warm, (300.0 / 250.0) ** 3)

    def test_linear_in_heat_cap(self) -> None:
        """Timescale scales linearly with heat capacity."""
        assert therm_inert_timescale(2e8, 300.0) == pytest.approx(
            2 * therm_inert_timescale(1e8, 300.0)
        )

    def test_known_value(self) -> None:
        """Matches hc / (4 sigma T^3) for a concrete case."""
        heat_cap, temp = 2e8, 300.0
        expected = heat_cap / (4.0 * STEF_BOLTZ_CONST * temp**3)
        np.testing.assert_allclose(therm_inert_timescale(heat_cap, temp), expected)


class TestSeasThermInertRatio:
    """Tests for seas_therm_inert_ratio."""

    def test_linear_in_orb_period(self) -> None:
        """Ratio scales linearly with orbital period."""
        assert seas_therm_inert_ratio(2.0, 1.0) == pytest.approx(
            2 * seas_therm_inert_ratio(1.0, 1.0)
        )

    def test_inverse_in_timescale(self) -> None:
        """Ratio scales inversely with the thermal-inertia timescale."""
        assert seas_therm_inert_ratio(1.0, 2.0) == pytest.approx(
            0.5 * seas_therm_inert_ratio(1.0, 1.0)
        )


class TestSeasDampFactor:
    """Tests for seas_damp_factor."""

    def test_unit_alpha(self) -> None:
        """At alpha = 1 the damping factor is 1/sqrt(2)."""
        np.testing.assert_allclose(seas_damp_factor(1.0), 1.0 / np.sqrt(2.0))

    def test_zero_limit(self) -> None:
        """Damping vanishes as thermal inertia vanishes (alpha -> 0)."""
        np.testing.assert_allclose(seas_damp_factor(0.0), 0.0)

    def test_large_alpha_limit(self) -> None:
        """Damping approaches 1 for large alpha (small thermal inertia)."""
        assert seas_damp_factor(1e6) == pytest.approx(1.0)

    def test_monotonic_and_bounded(self) -> None:
        """Damping increases monotonically with alpha and stays in [0, 1)."""
        alpha = np.linspace(0.0, 10.0, 50)
        damp = seas_damp_factor(alpha)
        assert np.all(np.diff(damp) > 0)
        assert np.all((damp >= 0) & (damp < 1))


class TestSeasLag:
    """Tests for seas_lag."""

    def test_unit_alpha(self) -> None:
        """At alpha = 1 the nondimensional lag is pi/4 = arctan(1)."""
        np.testing.assert_allclose(seas_lag(1.0), np.pi / 4.0)

    def test_small_alpha_limit(self) -> None:
        """Lag approaches pi/2 as thermal inertia dominates (alpha -> 0)."""
        assert seas_lag(1e-8) == pytest.approx(np.pi / 2.0)

    def test_large_alpha_limit(self) -> None:
        """Lag approaches 0 for small thermal inertia (large alpha)."""
        assert seas_lag(1e8) == pytest.approx(0.0, abs=1e-6)

    def test_dimensional_divides_by_orb_freq(self) -> None:
        """Providing an orbital frequency converts the lag to a time."""
        alpha = 2.0
        np.testing.assert_allclose(
            seas_lag(alpha, ORB_FREQ_EARTH), seas_lag(alpha) / ORB_FREQ_EARTH
        )


class TestTempRadEqEff:
    """Tests for temp_rad_eq_eff."""

    def test_equator_time_independent(self) -> None:
        """At the equator the annual cycle term vanishes (proportional to sin(lat))."""
        delta_h = 1 / 6.0
        expected = THETA_REF * (1 + delta_h / 3)
        for time in (0.0, 1e7, 2e7):
            result = temp_rad_eq_eff(0.0, time, alpha=1.0, delta_h=delta_h)
            np.testing.assert_allclose(result, expected)

    def test_preserves_lat_shape(self) -> None:
        """Output broadcasts over an array of latitudes."""
        lat = np.linspace(-90.0, 90.0, 19)
        result = temp_rad_eq_eff(lat, 0.0, alpha=1.0)
        assert np.asarray(result).shape == lat.shape

    def test_lat_symmetry_cancels_annual_cycle(self) -> None:
        """T(lat) + T(-lat) removes the odd (sin-lat) annual-cycle term.

        The sum is therefore time-independent and equal to twice the even
        annual-mean term.
        """
        lat, delta_h = 30.0, 1 / 6.0
        even_part = (
            2 * THETA_REF * (1 + delta_h / 3 * (1 - 3 * np.sin(np.deg2rad(lat)) ** 2))
        )
        sums = [
            temp_rad_eq_eff(lat, t, alpha=1.0, delta_h=delta_h)
            + temp_rad_eq_eff(-lat, t, alpha=1.0, delta_h=delta_h)
            for t in (0.0, 1e7, 2e7)
        ]
        for s in sums:
            np.testing.assert_allclose(s, even_part)

    def test_annual_cycle_term_known_value(self) -> None:
        """Pin the annual-cycle term's coefficient chain and phase.

        The symmetry and equator tests only constrain the annual-cycle term
        through its odd-in-lat symmetry and its vanishing at the equator; they
        leave its magnitude and phase unchecked. This reconstructs the full
        expected temperature at an off-equator latitude and non-trivial time
        from raw numpy, so a bug in the ``2 * delta_h * damping *
        sin(maxlat_ann) * sin(orb_freq * (time - lag))`` chain is caught.
        """
        lat, time, alpha, delta_h, maxlat_ann = 30.0, 5.0e6, 2.0, 1 / 6.0, 44.0
        sinlat = np.sin(np.deg2rad(lat))
        damping = alpha / np.sqrt(1 + alpha**2)
        lag = np.arctan(1.0 / alpha) / ORB_FREQ_EARTH
        ann_mean = 1 + delta_h / 3 * (1 - 3 * sinlat**2)
        ann_cyc = (
            2.0
            * delta_h
            * damping
            * np.sin(np.deg2rad(maxlat_ann))
            * sinlat
            * np.sin(ORB_FREQ_EARTH * (time - lag))
        )
        expected = THETA_REF * (ann_mean + ann_cyc)
        result = temp_rad_eq_eff(
            lat, time, alpha=alpha, maxlat_ann=maxlat_ann, delta_h=delta_h
        )
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "alpha",
    [
        1.0,
        np.array([0.5, 1.0, 2.0]),
        xr.DataArray([0.5, 1.0, 2.0], dims=["case"]),
    ],
    ids=["scalar", "ndarray", "DataArray"],
)
def test_seas_damp_factor_array_like(alpha: ArrayLike) -> None:
    """seas_damp_factor accepts float, np.ndarray, and xr.DataArray."""
    result = seas_damp_factor(alpha)
    assert np.all((np.asarray(result) >= 0) & (np.asarray(result) < 1))


@pytest.mark.parametrize(
    "heat_cap,temp",
    [
        (1e8, 300.0),
        (np.array([1e8, 2e8]), 300.0),
        (
            xr.DataArray([1e8, 2e8], dims=["case"]),
            xr.DataArray([300.0, 280.0], dims=["case"]),
        ),
    ],
    ids=["scalar", "ndarray", "DataArray"],
)
def test_therm_inert_timescale_array_like(heat_cap: ArrayLike, temp: ArrayLike) -> None:
    """therm_inert_timescale accepts float, np.ndarray, and xr.DataArray."""
    result = therm_inert_timescale(heat_cap, temp)
    assert np.all(np.asarray(result) > 0)
