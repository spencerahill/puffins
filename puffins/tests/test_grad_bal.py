"""Tests for the grad_bal module.

Closed-form functions are pinned with raw-numpy known-value reconstructions
(reconstructed independently of grad_bal's own helpers) and exercised with
non-default physical parameters so every parameter has teeth. The
convective-quasi-equilibrium and pressure-coordinate functions, which lean on
``calculus.lat_deriv``, are pinned by reconstructing their prefactors around
that (external) derivative.
"""

import numpy as np
import pytest
import xarray as xr

from puffins.calculus import lat_deriv
from puffins.grad_bal import (
    abs_ang_mom_unif_ro,
    abs_vort_zero_cross_cqe,
    grad_wind_bouss,
    grad_wind_cqe,
    grad_wind_p_coords,
    pot_temp_amc_cqe,
    pot_temp_avg_amc_bouss,
    pot_temp_avg_unif_ro,
    pot_temp_avg_unif_ro_small_ang,
    pot_temp_avg_unif_ro_small_ang_eq_ascent,
    pot_temp_lin_ro_lata0_small_ang,
    thermal_wind_p_coords,
    thermal_wind_shear_p_coords,
    u_ang_mom_cons,
    u_ang_mom_cons_small_ang,
    u_lin_ro,
    u_lin_ro_small_ang,
    u_rce_minus_u_amc_bouss,
    u_rce_minus_u_amc_cqe,
    u_unif_ro,
    u_unif_ro_small_ang,
)
from puffins.names import LAT_STR, LEV_STR

# Non-default (non-Earth) parameters, so every parameter is exercised away
# from its default value.
THETA_REF = 330.0
GRAV = 9.81
RADIUS = 6.371e6
ROT_RATE = 7.292e-5
HEIGHT = 12.0e3


def _lat_da(lats: np.ndarray, values: np.ndarray) -> xr.DataArray:
    """A latitude-indexed DataArray (coordinate is latitude, data is values)."""
    return xr.DataArray(
        np.asarray(values, dtype=float), dims=[LAT_STR], coords={LAT_STR: lats}
    )


# ---------------------------------------------------------------------------
# Angular-momentum-conserving winds.
# ---------------------------------------------------------------------------
class TestUAngMomCons:
    def test_known_values(self) -> None:
        lats = np.array([-40.0, -20.0, 10.0, 30.0])
        lat_a = 5.0
        expected = (
            ROT_RATE
            * RADIUS
            * (np.cos(np.deg2rad(lat_a)) ** 2 - np.cos(np.deg2rad(lats)) ** 2)
            / np.cos(np.deg2rad(lats))
        )
        actual = u_ang_mom_cons(lats, lat_a, rot_rate=ROT_RATE, radius=RADIUS)
        np.testing.assert_allclose(actual, expected, rtol=1e-13)

    def test_zero_at_ascent_latitude(self) -> None:
        """AMC wind vanishes at the ascent latitude."""
        assert u_ang_mom_cons(15.0, 15.0) == pytest.approx(0.0, abs=1e-9)

    def test_easterly_equatorward_of_ascent(self) -> None:
        """Equatorward of the ascent latitude the AMC wind is easterly."""
        assert u_ang_mom_cons(0.0, 20.0) < 0.0

    def test_westerly_poleward_of_ascent(self) -> None:
        """Poleward of the ascent latitude the AMC wind is westerly."""
        assert u_ang_mom_cons(40.0, 20.0) > 0.0


class TestUAngMomConsSmallAng:
    def test_known_values(self) -> None:
        lats = np.array([-30.0, -10.0, 20.0])
        lat_a = 8.0
        expected = ROT_RATE * RADIUS * (np.deg2rad(lats) ** 2 - np.deg2rad(lat_a) ** 2)
        actual = u_ang_mom_cons_small_ang(lats, lat_a, rot_rate=ROT_RATE, radius=RADIUS)
        np.testing.assert_allclose(actual, expected, rtol=1e-13)


class TestUUnifRo:
    def test_scales_amc_by_ross_num(self) -> None:
        lats = np.array([-25.0, 5.0, 35.0])
        lat_a, ross = 10.0, 0.6
        expected = ross * u_ang_mom_cons(lats, lat_a, rot_rate=ROT_RATE, radius=RADIUS)
        actual = u_unif_ro(lats, lat_a, ross, rot_rate=ROT_RATE, radius=RADIUS)
        np.testing.assert_allclose(actual, expected, rtol=1e-13)

    def test_ross_num_has_teeth(self) -> None:
        """Doubling Ro doubles the wind (guards the Ro factor)."""
        a = u_unif_ro(20.0, 5.0, 0.5)
        b = u_unif_ro(20.0, 5.0, 1.0)
        assert b == pytest.approx(2.0 * a, rel=1e-12)


class TestUUnifRoSmallAng:
    def test_scales_small_ang_amc_by_ross_num(self) -> None:
        lats = np.array([-20.0, 15.0])
        lat_a, ross = 5.0, 0.75
        expected = ross * u_ang_mom_cons_small_ang(
            lats, lat_a, rot_rate=ROT_RATE, radius=RADIUS
        )
        actual = u_unif_ro_small_ang(
            lats, lat_a, ross, rot_rate=ROT_RATE, radius=RADIUS
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13)


class TestAbsAngMomUnifRo:
    def test_known_values(self) -> None:
        lats = np.array([-35.0, -5.0, 25.0])
        lat_a, ross = 12.0, 0.4
        expected = (
            ROT_RATE
            * RADIUS**2
            * (
                (1 - ross) * np.cos(np.deg2rad(lats)) ** 2
                + ross * np.cos(np.deg2rad(lat_a)) ** 2
            )
        )
        actual = abs_ang_mom_unif_ro(
            lats, lat_a, ross, rot_rate=ROT_RATE, radius=RADIUS
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13)

    def test_ross_one_is_uniform(self) -> None:
        """At Ro=1 the angular momentum is uniform (= Omega a^2 cos^2 phi_a)."""
        lats = np.array([-30.0, 0.0, 30.0])
        vals = np.asarray(abs_ang_mom_unif_ro(lats, 15.0, 1.0))
        np.testing.assert_allclose(vals, vals[0], rtol=1e-13)


# ---------------------------------------------------------------------------
# Potential-temperature fields balancing the uniform-Ro winds.
# ---------------------------------------------------------------------------
class TestPotTempAvgUnifRo:
    @staticmethod
    def _raw(lats, lat_a, theta_a, ross, height, theta_ref, rot, rad, grav):
        prefactor = ross * rot**2 * rad**2 / (2 * grav * height)
        coslat = np.cos(np.deg2rad(lats))
        cosa = np.cos(np.deg2rad(lat_a))
        ratio = cosa / coslat
        return theta_a - theta_ref * prefactor * (
            (2 - ross) * coslat**2
            + cosa**2 * (4 * (1 - ross) * np.log(ratio) + ross * ratio**2 - 2)
        )

    def test_known_values(self) -> None:
        lats = np.array([-30.0, -10.0, 20.0])
        expected = self._raw(
            lats, 5.0, 300.0, 0.7, HEIGHT, THETA_REF, ROT_RATE, RADIUS, GRAV
        )
        actual = pot_temp_avg_unif_ro(
            lats,
            5.0,
            300.0,
            0.7,
            height=HEIGHT,
            theta_ref=THETA_REF,
            rot_rate=ROT_RATE,
            radius=RADIUS,
            grav=GRAV,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_equals_ascent_value_at_ascent_lat(self) -> None:
        """The bracket vanishes at the ascent latitude for Ro=1."""
        val = pot_temp_avg_unif_ro(15.0, 15.0, 300.0, 1.0)
        assert val == pytest.approx(300.0, rel=1e-12)


class TestPotTempAvgUnifRoSmallAng:
    @staticmethod
    def _raw(lats, lat_a, theta_a, ross, burg):
        prefactor = 0.5 * ross / burg
        lat_sq = lats**2
        lata_sq = lat_a**2
        return theta_a - THETA_REF * prefactor * (
            (2 - ross) * (1 - lat_sq)
            + (1 - lata_sq) * (ross * (1 - lat_sq) / (1 - lata_sq) - 2)
        )

    def test_explicit_burg_num(self) -> None:
        lats = np.array([-0.3, 0.0, 0.2])
        burg = 0.05
        expected = self._raw(lats, 0.1, 305.0, 0.8, burg)
        actual = pot_temp_avg_unif_ro_small_ang(
            lats, 0.1, 305.0, 0.8, burg_num=burg, theta_ref=THETA_REF
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_default_burg_is_planetary(self) -> None:
        lats = np.array([-0.3, 0.2])
        burg = GRAV * HEIGHT / (ROT_RATE * RADIUS) ** 2
        expected = self._raw(lats, 0.1, 305.0, 0.8, burg)
        actual = pot_temp_avg_unif_ro_small_ang(
            lats,
            0.1,
            305.0,
            0.8,
            theta_ref=THETA_REF,
            height=HEIGHT,
            rot_rate=ROT_RATE,
            radius=RADIUS,
            grav=GRAV,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)


class TestPotTempAvgUnifRoSmallAngEqAscent:
    def test_known_values(self) -> None:
        lats = np.array([-20.0, 10.0, 30.0])
        ross = 0.9
        burg = 0.07
        expected = 300.0 - THETA_REF * 0.5 * ross / burg * np.deg2rad(lats) ** 4
        actual = pot_temp_avg_unif_ro_small_ang_eq_ascent(
            lats, 300.0, ross_num=ross, theta_ref=THETA_REF, burg_num=burg
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13)

    def test_default_burg_is_planetary(self) -> None:
        lats = np.array([15.0])
        burg = GRAV * HEIGHT / (ROT_RATE * RADIUS) ** 2
        expected = 300.0 - THETA_REF * 0.5 * 1.0 / burg * np.deg2rad(lats) ** 4
        actual = pot_temp_avg_unif_ro_small_ang_eq_ascent(
            lats,
            300.0,
            theta_ref=THETA_REF,
            height=HEIGHT,
            rot_rate=ROT_RATE,
            radius=RADIUS,
            grav=GRAV,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13)


# ---------------------------------------------------------------------------
# Linear-in-latitude Rossby-number winds.
# ---------------------------------------------------------------------------
class TestULinRo:
    @staticmethod
    def _raw(lats, lat_a, lat_d, ro_a, ro_d, rot, rad):
        u_ro_a = ro_a * (
            rot
            * rad
            * (np.cos(np.deg2rad(lat_a)) ** 2 - np.cos(np.deg2rad(lats)) ** 2)
            / np.cos(np.deg2rad(lats))
        )
        sinlat = np.sin(np.deg2rad(lats))
        return u_ro_a - 2 * rot * rad * (ro_a - ro_d) / (
            3 * np.sin(np.deg2rad(lat_d)) * np.cos(np.deg2rad(lats))
        ) * (sinlat**3 - np.sin(np.deg2rad(lat_a)) ** 3)

    def test_known_values(self) -> None:
        lats = np.array([-25.0, 5.0, 30.0])
        expected = self._raw(lats, 8.0, 25.0, 1.0, 0.3, ROT_RATE, RADIUS)
        actual = u_lin_ro(lats, 8.0, 25.0, 1.0, 0.3, rot_rate=ROT_RATE, radius=RADIUS)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_reduces_to_unif_ro_when_ross_equal(self) -> None:
        """When Ro_a == Ro_d the linear term drops out."""
        lats = np.array([-20.0, 15.0])
        expected = u_unif_ro(lats, 8.0, 0.6)
        actual = u_lin_ro(lats, 8.0, 25.0, 0.6, 0.6)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)


class TestULinRoSmallAng:
    @staticmethod
    def _raw(lats, lat_a, lat_d, ro_a, ro_d, rot, rad):
        u_ro_a = ro_a * rot * rad * (np.deg2rad(lats) ** 2 - np.deg2rad(lat_a) ** 2)
        delta_ro = ro_a - ro_d
        latrad = np.deg2rad(lats)
        return u_ro_a - 2 * delta_ro * rot * rad / (3 * np.deg2rad(lat_d)) * (
            latrad**3 - np.deg2rad(lat_a) ** 3
        )

    def test_known_values(self) -> None:
        lats = np.array([-15.0, 10.0, 20.0])
        expected = self._raw(lats, 5.0, 20.0, 1.0, 0.4, ROT_RATE, RADIUS)
        actual = u_lin_ro_small_ang(
            lats, 5.0, 20.0, 1.0, 0.4, rot_rate=ROT_RATE, radius=RADIUS
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)


class TestPotTempLinRoLata0SmallAng:
    """Retained from the original (Burger-number handling) test suite."""

    @staticmethod
    def _raw_profile(lat_deg, lat_descent_deg, ro_a, ro_d, pot_temp_lat0, burg_num):
        delro = ro_a - ro_d
        lat = np.deg2rad(lat_deg)
        latd = np.deg2rad(lat_descent_deg)
        bracket = (
            ro_a / 2
            - 4 * delro * lat / (15 * latd)
            + ro_a**2 * lat**2 / 6
            - 4 * delro * ro_a * lat**3 / (21 * latd)
            + delro**2 * lat**4 / (18 * latd**2)
        )
        return pot_temp_lat0 - THETA_REF * lat**4 / burg_num * bracket

    def test_explicit_burg_num(self) -> None:
        burg = 0.06
        lats = np.array([0.0, 10.0, 20.0, 30.0])
        expected = self._raw_profile(lats, 33.0, 1.0, 0.5, 365.0, burg)
        actual = pot_temp_lin_ro_lata0_small_ang(
            lats, 33.0, 1.0, 0.5, 365.0, theta_ref=THETA_REF, burg_num=burg
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13)

    def test_default_burg_num_is_planetary(self) -> None:
        burg_planet = GRAV * HEIGHT / (ROT_RATE * RADIUS) ** 2
        lats = np.array([0.0, 10.0, 20.0, 30.0])
        expected = self._raw_profile(lats, 33.0, 1.0, 0.5, 365.0, burg_planet)
        actual = pot_temp_lin_ro_lata0_small_ang(
            lats,
            33.0,
            1.0,
            0.5,
            365.0,
            rot_rate=ROT_RATE,
            radius=RADIUS,
            theta_ref=THETA_REF,
            grav=GRAV,
            height=HEIGHT,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13)


# ---------------------------------------------------------------------------
# Boussinesq atmospheres.
# ---------------------------------------------------------------------------
class TestGradWindBouss:
    def test_known_values(self) -> None:
        lats = np.array([-30.0, -10.0, 20.0])
        dtheta_dlat = -2.0
        coslat = np.cos(np.deg2rad(lats))
        sinlat = np.sin(np.deg2rad(lats))
        sqrt_fac = (
            GRAV
            * HEIGHT
            * dtheta_dlat
            / (THETA_REF * ROT_RATE**2 * RADIUS**2 * coslat * sinlat)
        )
        expected = ROT_RATE * RADIUS * coslat * ((1 - sqrt_fac) ** 0.5 - 1)
        actual = grad_wind_bouss(
            lats,
            HEIGHT,
            THETA_REF,
            dtheta_dlat,
            grav=GRAV,
            rot_rate=ROT_RATE,
            radius=RADIUS,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_zero_gradient_gives_zero_wind(self) -> None:
        assert grad_wind_bouss(20.0, HEIGHT, THETA_REF, 0.0) == pytest.approx(0.0)


class TestPotTempAvgAmcBouss:
    def test_known_values(self) -> None:
        lats = np.array([-35.0, -5.0, 25.0])
        lat_max, theta_max, theta_ref, extra = 15.0, 300.0, THETA_REF, 1.3
        chi = -extra * (ROT_RATE**2 * RADIUS**2) / (2.0 * GRAV * HEIGHT)
        numerator = (
            np.sin(np.deg2rad(lats)) ** 2 - np.sin(np.deg2rad(lat_max)) ** 2
        ) ** 2
        expected = (
            theta_max + theta_ref * chi * numerator / np.cos(np.deg2rad(lats)) ** 2
        )
        actual = pot_temp_avg_amc_bouss(
            lats,
            lat_max,
            theta_max,
            theta_ref,
            HEIGHT,
            rot_rate=ROT_RATE,
            radius=RADIUS,
            grav=GRAV,
            extra_factor=extra,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_extra_factor_has_teeth(self) -> None:
        """The departure from theta_max scales linearly in extra_factor."""
        base = pot_temp_avg_amc_bouss(
            30.0, 10.0, 300.0, 290.0, HEIGHT, extra_factor=1.0
        )
        doubled = pot_temp_avg_amc_bouss(
            30.0, 10.0, 300.0, 290.0, HEIGHT, extra_factor=2.0
        )
        assert (doubled - 300.0) == pytest.approx(2.0 * (base - 300.0), rel=1e-12)


class TestURceMinusUAmcBouss:
    def test_known_values(self) -> None:
        lats = np.array([-30.0, -10.0, 25.0])
        lat_max, dtheta_dlat = 12.0, -1.5
        coslat = np.cos(np.deg2rad(lats))
        sinlat = np.sin(np.deg2rad(lats))
        sqrt_fac = (
            GRAV
            * HEIGHT
            * dtheta_dlat
            / (coslat * sinlat * THETA_REF * ROT_RATE**2 * RADIUS**2)
        )
        expected = (
            ROT_RATE
            * RADIUS
            * coslat
            * ((1 - sqrt_fac) ** 0.5 - np.cos(np.deg2rad(lat_max)) ** 2 / coslat**2)
        )
        actual = u_rce_minus_u_amc_bouss(
            lats,
            lat_max,
            HEIGHT,
            THETA_REF,
            dtheta_dlat,
            grav=GRAV,
            rot_rate=ROT_RATE,
            radius=RADIUS,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Convective quasi-equilibrium atmospheres.
# ---------------------------------------------------------------------------
class TestPotTempAmcCqe:
    def test_known_values(self) -> None:
        lats = np.array([-30.0, -5.0, 20.0])
        lat_max, theta_max, dtemp, extra = 10.0, 340.0, 100.0, 0.8
        chi = extra * (ROT_RATE**2 * RADIUS**2) / (1003.5 * dtemp)
        numerator = (
            np.cos(np.deg2rad(lat_max)) ** 2 - np.cos(np.deg2rad(lats)) ** 2
        ) ** 2
        expected = theta_max * np.exp(
            -0.5 * chi * numerator / np.cos(np.deg2rad(lats)) ** 2
        )
        actual = pot_temp_amc_cqe(
            lats,
            lat_max,
            theta_max,
            dtemp,
            rot_rate=ROT_RATE,
            radius=RADIUS,
            extra_factor=extra,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_max_at_lat_max(self) -> None:
        """The field peaks at lat_max (exp argument is zero there)."""
        assert pot_temp_amc_cqe(10.0, 10.0, 340.0, 100.0) == pytest.approx(340.0)


class TestGradWindCqe:
    """Pin the assembly around calculus.lat_deriv via the const_stab branch."""

    def _theta_b(self) -> xr.DataArray:
        lats = np.array([-30.0, -20.0, -10.0, 10.0, 20.0, 30.0])
        return _lat_da(lats, 320.0 - 0.01 * lats**2)

    def test_known_values_const_stab(self) -> None:
        theta_b = self._theta_b()
        lats = theta_b[LAT_STR]
        const_stab, c_p = 4.0, 1003.5
        numer = c_p * const_stab
        coslat = np.cos(np.deg2rad(lats.values))
        sinlat = np.sin(np.deg2rad(lats.values))
        denom = coslat * sinlat * ROT_RATE**2 * RADIUS**2
        log_theta = theta_b.copy(data=np.log(theta_b.values))
        dlnth = lat_deriv(log_theta, LAT_STR).values
        sqrt_term = (1 - (numer / denom) * dlnth) ** 0.5
        expected = ROT_RATE * RADIUS * coslat * (-1 + sqrt_term)
        actual = grad_wind_cqe(
            theta_b,
            const_stab=const_stab,
            c_p=c_p,
            rot_rate=ROT_RATE,
            radius=RADIUS,
        )
        assert isinstance(actual, xr.DataArray)
        np.testing.assert_allclose(actual.values, expected, rtol=1e-11)

    def test_requires_temp_tropo_without_const_stab(self) -> None:
        with pytest.raises(ValueError, match="temp_tropo"):
            grad_wind_cqe(self._theta_b(), const_stab=False)

    def test_temp_tropo_path_runs(self) -> None:
        out = grad_wind_cqe(self._theta_b(), temp_tropo=200.0)
        assert isinstance(out, xr.DataArray)
        assert out.dims == (LAT_STR,)

    def test_compute_temp_sfc_changes_result(self) -> None:
        """With compute_temp_sfc the surface temperature is derived from the
        equivalent potential temperature rather than taken as theta_b, so the
        balanced wind differs (gives compute_temp_sfc and rel_hum teeth)."""
        theta_b = self._theta_b()
        derived = grad_wind_cqe(
            theta_b, temp_tropo=200.0, compute_temp_sfc=True, rel_hum=0.8
        )
        direct = grad_wind_cqe(theta_b, temp_tropo=200.0, compute_temp_sfc=False)
        assert isinstance(derived, xr.DataArray)
        assert np.isfinite(derived.values).all()
        assert not np.allclose(derived.values, direct.values)


class TestURceMinusUAmcCqe:
    """Regression: this previously always raised because it forwarded an
    unsupported ``plus_solution`` kwarg to ``grad_wind_cqe``."""

    def _theta_b(self) -> xr.DataArray:
        lats = np.array([-30.0, -20.0, -10.0, 10.0, 20.0, 30.0])
        return _lat_da(lats, 320.0 - 0.01 * lats**2)

    def test_returns_dataarray(self) -> None:
        out = u_rce_minus_u_amc_cqe(
            10.0, self._theta_b(), temp_tropo=200.0, rot_rate=ROT_RATE, radius=RADIUS
        )
        assert isinstance(out, xr.DataArray)

    def test_equals_grad_wind_minus_amc(self) -> None:
        theta_b = self._theta_b()
        lats = theta_b[LAT_STR]
        u_rce = grad_wind_cqe(
            theta_b, temp_tropo=200.0, rot_rate=ROT_RATE, radius=RADIUS
        )
        u_amc = u_ang_mom_cons(lats, 10.0, ROT_RATE, RADIUS)
        expected = u_rce - u_amc
        actual = u_rce_minus_u_amc_cqe(
            10.0, theta_b, temp_tropo=200.0, rot_rate=ROT_RATE, radius=RADIUS
        )
        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-12)


class TestAbsVortZeroCrossCqe:
    """Thin composition of grad_wind_cqe -> abs_vort_from_u -> zero_cross_interp,
    restricted to the northern hemisphere. For a smooth, sub-critical RCE
    profile the absolute vorticity stays positive, so the pipeline runs to
    completion and the (separately tested) zero-cross finder reports no
    crossing. An equator-free grid keeps grad_wind_cqe non-singular."""

    def test_no_crossing_for_subcritical_profile(self) -> None:
        lats = np.linspace(1.0, 40.0, 40)
        theta_b = _lat_da(lats, 340.0 - 0.1 * lats**2)
        with pytest.raises(ValueError, match="zero cross"):
            abs_vort_zero_cross_cqe(
                theta_b, temp_tropo=200.0, rot_rate=ROT_RATE, radius=RADIUS
            )


# ---------------------------------------------------------------------------
# Pressure-coordinate thermal / gradient winds.
# ---------------------------------------------------------------------------
def _temp_field() -> xr.DataArray:
    lats = np.array([-40.0, -20.0, -10.0, 10.0, 20.0, 40.0])
    levs = np.array([25000.0, 50000.0, 85000.0])
    # Temperature warm at equator, decreasing with height.
    base = 290.0 - 0.02 * lats[None, :] ** 2
    lapse = np.array([0.7, 0.85, 1.0])[:, None]
    data = base * lapse
    return xr.DataArray(
        data,
        dims=[LEV_STR, LAT_STR],
        coords={LEV_STR: levs, LAT_STR: lats},
    )


class TestThermalWindShearPCoords:
    def test_known_values(self) -> None:
        temp = _temp_field()
        pressure = temp[LEV_STR]
        r_d = 287.06
        expected = (
            -1
            * r_d
            * lat_deriv(temp, LAT_STR)
            / (2 * ROT_RATE * RADIUS * pressure * np.sin(np.deg2rad(temp[LAT_STR])))
        )
        actual = thermal_wind_shear_p_coords(
            temp, pressure, radius=RADIUS, rot_rate=ROT_RATE, r_d=r_d
        )
        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-12)


class TestThermalWindPCoords:
    def test_known_values_from_height(self) -> None:
        lats = np.array([-40.0, -20.0, 20.0, 40.0])
        height = _lat_da(lats, 1.0e4 - 5.0 * lats**2)
        sinlat = np.sin(np.deg2rad(height[LAT_STR]))
        expected = (
            -1 * GRAV * lat_deriv(height, LAT_STR) / (2 * ROT_RATE * RADIUS * sinlat)
        )
        actual = thermal_wind_p_coords(
            height=height, radius=RADIUS, rot_rate=ROT_RATE, grav=GRAV
        )
        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-12)

    def test_requires_height_or_temp(self) -> None:
        with pytest.raises(ValueError, match="height.*temp"):
            thermal_wind_p_coords()


class TestGradWindPCoords:
    def test_known_values_from_height(self) -> None:
        lats = np.array([-40.0, -20.0, 20.0, 40.0])
        height = _lat_da(lats, 1.0e4 - 5.0 * lats**2)
        lat = height[LAT_STR]
        sinlat = np.sin(np.deg2rad(lat))
        coslat = np.cos(np.deg2rad(lat))
        sqrt_arg = 1 - GRAV * lat_deriv(height, LAT_STR) / (
            sinlat * coslat * (ROT_RATE * RADIUS) ** 2
        )
        expected = (ROT_RATE * RADIUS * coslat * (np.sqrt(sqrt_arg) - 1)).transpose(
            *height.dims
        )
        actual = grad_wind_p_coords(
            height=height, radius=RADIUS, rot_rate=ROT_RATE, grav=GRAV
        )
        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-12)

    def test_requires_height_or_temp(self) -> None:
        with pytest.raises(ValueError, match="height.*temp"):
            grad_wind_p_coords()
