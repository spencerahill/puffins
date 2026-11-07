"""Tests for puffins.lindzen_hou_1988.

Expected values are transcribed from Hill, Bordoni and Mitchell (2021,
J. Atmos. Sci. 78, doi:10.1175/JAS-D-20-0341.1), pp. 2003-2004, whose
thermal Rossby number is Ro_th = burg_delh * sin(lat_max) (their Eq. 7).
The planetary parameters are Mars-like, so that no Earth default can stand in
for an argument that is dropped.
"""

import inspect
from typing import cast

import numpy as np
import pytest
import xarray as xr

from puffins import lindzen_hou_1988
from puffins.constants import GRAV_EARTH, HEIGHT_TROPO, RAD_EARTH, ROT_RATE_EARTH
from puffins.dynamics import abs_vort_from_u
from puffins.lindzen_hou_1988 import (
    abs_vort_rce_lh88,
    abs_vort_rce_lh88_zero_approx,
    abs_vort_rce_zero_lh88,
    ang_mom_rce_lh88,
    ang_mom_rce_lh88_zero,
    ang_mom_rce_lh88_zero_large_r,
    ang_mom_rce_lh88_zero_small_r,
    ang_mom_rce_lh88_zero_unity_r,
    crit_vort_metric_rce_lh88,
    u_rce_lh88,
)
from puffins.names import LAT_STR

GRAV = 3.71
ROT_RATE = 7.088e-5
RADIUS = 3.39e6
HEIGHT = 1.1e4
DELTA_H = 0.2
LAT_MAX = 20.0
BURG_DELH = GRAV * HEIGHT * DELTA_H / (ROT_RATE * RADIUS) ** 2
RO_TH = BURG_DELH * np.sin(np.deg2rad(LAT_MAX))
# The northern latitudes start at 10 degrees, where the square-root argument
# of Eq. 6 stays positive even for twice DELTA_H.
LATS = np.concatenate([np.arange(-85.0, 0.0, 5.0), np.arange(10.0, 90.0, 5.0)])


def _sqrt_arg(lat: np.ndarray, lat_max: float, ro_th: float) -> np.ndarray:
    """The square-root argument of Eqs. 6, 8 and 9."""
    sin = np.sin(np.deg2rad(lat))
    return cast(np.ndarray, 1 + 2 * ro_th * (1 / np.sin(np.deg2rad(lat_max)) - 1 / sin))


def _u_eq6(lat: np.ndarray, lat_max: float, ro_th: float) -> np.ndarray:
    cos = np.cos(np.deg2rad(lat))
    return cast(
        np.ndarray,
        ROT_RATE * RADIUS * cos * (np.sqrt(_sqrt_arg(lat, lat_max, ro_th)) - 1),
    )


def _m_eq8(lat: np.ndarray, lat_max: float, ro_th: float) -> np.ndarray:
    cos = np.cos(np.deg2rad(lat))
    return cast(
        np.ndarray,
        ROT_RATE * RADIUS**2 * cos**2 * np.sqrt(_sqrt_arg(lat, lat_max, ro_th)),
    )


def _eta_eq9(
    lat: np.ndarray, lat_max: float, ro_th: float, rot_rate: float = ROT_RATE
) -> np.ndarray:
    sin, cos = np.sin(np.deg2rad(lat)), np.cos(np.deg2rad(lat))
    arg = _sqrt_arg(lat, lat_max, ro_th)
    return cast(
        np.ndarray,
        2 * rot_rate * sin * np.sqrt(arg) * (1 - 0.5 * cos**2 / sin**3 * ro_th / arg),
    )


def _eq10_root(lat_max: float, burg_delh: float) -> float:
    """Root in (0, 90] degrees of Eq. 10, a cubic in sin(lat), by np.roots."""
    sin_m = np.sin(np.deg2rad(lat_max))
    ro_th = burg_delh * sin_m
    roots = np.roots([1 + 2 * ro_th / sin_m, -1.5 * ro_th, 0.0, -0.5 * ro_th])
    real = roots[(np.abs(roots.imag) < 1e-12) & (roots.real > 0) & (roots.real <= 1)]
    assert real.size == 1
    return float(np.rad2deg(np.arcsin(real.real[0])))


# Each closed form is checked under forcing peaked in either hemisphere: sign -1
# mirrors the latitudes and lat_max, so Ro_th changes sign with sin(lat_max).
HEMISPHERES = pytest.mark.parametrize("sign", [1.0, -1.0])


def test_renamed_arguments() -> None:
    """Callers pass Bu * delta_h by the keyword burg_delh, so the name is pinned (#115)."""
    names = [
        "u_rce_lh88",
        "ang_mom_rce_lh88",
        "ang_mom_rce_lh88_zero",
        "ang_mom_rce_lh88_zero_small_r",
        "ang_mom_rce_lh88_zero_unity_r",
        "ang_mom_rce_lh88_zero_large_r",
        "abs_vort_rce_lh88",
        "abs_vort_rce_zero_lh88",
    ]
    for name in names:
        params = inspect.signature(getattr(lindzen_hou_1988, name)).parameters
        assert "burg_delh" in params, name
        assert not {"thermal_ro", "therm_ro"} & set(params), name


class TestURceLh88:
    @HEMISPHERES
    def test_matches_eq6(self, sign: float) -> None:
        result = u_rce_lh88(
            sign * LATS,
            sign * LAT_MAX,
            burg_delh=BURG_DELH,
            rot_rate=ROT_RATE,
            radius=RADIUS,
        )
        # The atol covers lat = lat_max, where u is zero.
        np.testing.assert_allclose(
            result,
            _u_eq6(sign * LATS, sign * LAT_MAX, sign * RO_TH),
            rtol=1e-12,
            atol=1e-10,
        )


class TestAngMomRceLh88:
    @HEMISPHERES
    def test_matches_eq8(self, sign: float) -> None:
        result = ang_mom_rce_lh88(
            sign * LATS,
            sign * LAT_MAX,
            burg_delh=BURG_DELH,
            rot_rate=ROT_RATE,
            radius=RADIUS,
        )
        np.testing.assert_allclose(
            result, _m_eq8(sign * LATS, sign * LAT_MAX, sign * RO_TH), rtol=1e-12
        )

    def test_is_angular_momentum_of_u_rce(self) -> None:
        """M = a cos(lat) (Omega a cos(lat) + u), Eq. 2."""
        kwargs = dict(burg_delh=BURG_DELH, rot_rate=ROT_RATE, radius=RADIUS)
        u = u_rce_lh88(LATS, LAT_MAX, **kwargs)
        cos = np.cos(np.deg2rad(LATS))
        np.testing.assert_allclose(
            ang_mom_rce_lh88(LATS, LAT_MAX, **kwargs),
            RADIUS * cos * (ROT_RATE * RADIUS * cos + u),
            rtol=1e-12,
        )


class TestAngMomRceLh88Zero:
    def test_square_root_argument_vanishes_there(self) -> None:
        lat = ang_mom_rce_lh88_zero(LAT_MAX, burg_delh=BURG_DELH)
        assert 0 < lat < LAT_MAX
        assert _sqrt_arg(np.array(lat), LAT_MAX, RO_TH) == pytest.approx(0, abs=1e-12)

    def test_small_r_approaches_exact(self) -> None:
        small = ang_mom_rce_lh88_zero_small_r(LAT_MAX, burg_delh=1e-4)
        expected = np.rad2deg(np.arcsin(2e-4 * np.sin(np.deg2rad(LAT_MAX))))
        assert small == pytest.approx(expected, rel=1e-12)
        assert small == pytest.approx(ang_mom_rce_lh88_zero(LAT_MAX, 1e-4), rel=1e-3)

    def test_unity_r_equals_exact_at_unity(self) -> None:
        assert ang_mom_rce_lh88_zero_unity_r(LAT_MAX, burg_delh=1.0) == pytest.approx(
            ang_mom_rce_lh88_zero(LAT_MAX, 1.0), rel=1e-12
        )

    def test_large_r_approaches_exact(self) -> None:
        assert ang_mom_rce_lh88_zero_large_r(LAT_MAX, burg_delh=1e8) == LAT_MAX
        assert ang_mom_rce_lh88_zero(LAT_MAX, 1e8) == pytest.approx(LAT_MAX, abs=1e-6)


class TestCritVortMetricRceLh88:
    @HEMISPHERES
    def test_matches_derivative_of_ang_mom_squared(self, sign: float) -> None:
        """a^-2 d(M^2)/d(lat), by centered differences of Eq. 8.

        The difference error measured 4.1e-10 relative with this step.
        """
        lats, lat_max, ro_th = sign * LATS, sign * LAT_MAX, sign * RO_TH
        result = crit_vort_metric_rce_lh88(
            lats, lat_max, DELTA_H, HEIGHT, grav=GRAV, rot_rate=ROT_RATE, radius=RADIUS
        )
        step = 1e-4
        m2_minus = _m_eq8(lats - step, lat_max, ro_th) ** 2
        m2_plus = _m_eq8(lats + step, lat_max, ro_th) ** 2
        expected = (m2_plus - m2_minus) / (2 * np.deg2rad(step)) / RADIUS**2
        np.testing.assert_allclose(result, expected, rtol=1e-8)

    def test_responds_to_delta_h(self) -> None:
        """Before #110 was fixed it returned the zero-forcing value for any delta_h."""
        kwargs = dict(grav=GRAV, rot_rate=ROT_RATE, radius=RADIUS)
        base = crit_vort_metric_rce_lh88(LATS, LAT_MAX, DELTA_H, HEIGHT, **kwargs)
        doubled = crit_vort_metric_rce_lh88(
            LATS, LAT_MAX, 2 * DELTA_H, HEIGHT, **kwargs
        )
        assert not np.allclose(base, doubled, rtol=1e-2)


class TestAbsVortRceLh88:
    @HEMISPHERES
    def test_matches_eq9(self, sign: float) -> None:
        result = abs_vort_rce_lh88(
            sign * LATS, sign * LAT_MAX, rot_rate=ROT_RATE, burg_delh=BURG_DELH
        )
        np.testing.assert_allclose(
            result, _eta_eq9(sign * LATS, sign * LAT_MAX, sign * RO_TH), rtol=1e-12
        )

    @HEMISPHERES
    def test_computed_burg_delh_matches_eq9(self, sign: float) -> None:
        """burg_delh computed from delta_h, height, grav, rot_rate and radius."""
        result = abs_vort_rce_lh88(
            sign * LATS,
            sign * LAT_MAX,
            delta_h=DELTA_H,
            height=HEIGHT,
            grav=GRAV,
            rot_rate=ROT_RATE,
            radius=RADIUS,
        )
        np.testing.assert_allclose(
            result, _eta_eq9(sign * LATS, sign * LAT_MAX, sign * RO_TH), rtol=1e-12
        )

    def test_all_defaults_match_eq9_for_earth(self) -> None:
        """Defaults: delta_h = 1/6, Earth's height, gravity, rotation and radius."""
        burg_delh = GRAV_EARTH * HEIGHT_TROPO / 6 / (ROT_RATE_EARTH * RAD_EARTH) ** 2
        ro_th = burg_delh * np.sin(np.deg2rad(LAT_MAX))
        np.testing.assert_allclose(
            abs_vort_rce_lh88(LATS, LAT_MAX),
            _eta_eq9(LATS, LAT_MAX, ro_th, rot_rate=ROT_RATE_EARTH),
            rtol=1e-12,
        )

    def test_zero_burg_delh_gives_planetary_vorticity(self) -> None:
        """An explicit burg_delh of 0 is used as given, not replaced by the default."""
        result = abs_vort_rce_lh88(LATS, LAT_MAX, rot_rate=ROT_RATE, burg_delh=0.0)
        np.testing.assert_allclose(
            result, 2 * ROT_RATE * np.sin(np.deg2rad(LATS)), rtol=1e-12
        )

    def test_responds_to_delta_h(self) -> None:
        kwargs = dict(height=HEIGHT, grav=GRAV, rot_rate=ROT_RATE, radius=RADIUS)
        base = abs_vort_rce_lh88(LATS, LAT_MAX, delta_h=DELTA_H, **kwargs)
        doubled = abs_vort_rce_lh88(LATS, LAT_MAX, delta_h=2 * DELTA_H, **kwargs)
        assert not np.allclose(base, doubled, rtol=1e-2)

    @pytest.mark.parametrize(("lat_lo", "lat_hi"), [(5.0, 85.0), (-85.0, -5.0)])
    def test_is_vorticity_of_u_rce(self, lat_lo: float, lat_hi: float) -> None:
        """Finite-difference absolute vorticity of u_rce_lh88 on a 0.01-degree grid.

        The difference error measured 3.7e-5 of the largest |eta| in the
        northern band and 3.0e-6 in the southern one; a factor of 2 would
        give order 1.
        """
        lats = np.arange(lat_lo, lat_hi + 1e-9, 0.01)
        lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats}, name="lat")
        u = u_rce_lh88(lat, LAT_MAX, BURG_DELH, rot_rate=ROT_RATE, radius=RADIUS)
        eta_fd = abs_vort_from_u(u, rot_rate=ROT_RATE, radius=RADIUS).isel(
            {LAT_STR: slice(1, -1)}
        )
        eta = abs_vort_rce_lh88(
            eta_fd[LAT_STR], LAT_MAX, rot_rate=ROT_RATE, burg_delh=BURG_DELH
        )
        err = np.abs(eta_fd - eta).max().item()
        assert err < 1e-4 * np.abs(eta).max().item()


class TestAbsVortRceZeroLh88:
    @pytest.mark.parametrize(
        ("lat_max", "burg_delh"),
        [
            (LAT_MAX, BURG_DELH),
            (48.86, 0.853),
            (90.0, 5.0),
            # Root at 83.41 degrees, poleward of the guesses before #114.
            (90.0, 50.0),
            # Root at 89.93 degrees, poleward of any guess short of 90.
            (90.0, 5e5),
        ],
    )
    def test_matches_eq10(self, lat_max: float, burg_delh: float) -> None:
        result = abs_vort_rce_zero_lh88(lat_max, burg_delh)
        assert result.item() == pytest.approx(_eq10_root(lat_max, burg_delh), abs=1e-9)

    def test_abs_vort_vanishes_there(self) -> None:
        lat = abs_vort_rce_zero_lh88(LAT_MAX, BURG_DELH).item()
        eta = abs_vort_rce_lh88(lat, LAT_MAX, rot_rate=ROT_RATE, burg_delh=BURG_DELH)
        assert eta == pytest.approx(0, abs=1e-12 * 2 * ROT_RATE)

    def test_scalar_gives_0d(self) -> None:
        result = abs_vort_rce_zero_lh88(48.86, 0.853)
        assert np.ndim(result) == 0
        assert float(result) == pytest.approx(_eq10_root(48.86, 0.853), abs=1e-9)

    def test_0d_dataarray_input(self) -> None:
        lat_max = xr.DataArray(48.86, name="lat_max")
        result = abs_vort_rce_zero_lh88(lat_max, 0.853)
        assert np.ndim(result) == 0
        assert result.item() == pytest.approx(_eq10_root(48.86, 0.853), abs=1e-9)

    def test_negative_lat_max_mirrors(self) -> None:
        north = abs_vort_rce_zero_lh88(48.86, 0.853).item()
        south = abs_vort_rce_zero_lh88(-48.86, 0.853).item()
        assert north > 0
        assert south == pytest.approx(-north, rel=1e-12)

    def test_array_input(self) -> None:
        lat_maxes = np.array([20.0, 48.86, 90.0])
        result = abs_vort_rce_zero_lh88(lat_maxes, 0.853)
        assert result.shape == (3,)
        expected = [_eq10_root(lm, 0.853) for lm in lat_maxes]
        np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_dataarray_input_keeps_dims(self) -> None:
        lat_maxes = xr.DataArray(
            [20.0, 48.86], dims=["month"], coords={"month": [1, 2]}, name="lat_max"
        )
        result = abs_vort_rce_zero_lh88(lat_maxes, 0.853)
        assert result.dims == ("month",)
        np.testing.assert_array_equal(result["month"], [1, 2])
        expected = [_eq10_root(20.0, 0.853), _eq10_root(48.86, 0.853)]
        np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_init_guess_finds_root_equatorward_of_first_guess(self) -> None:
        """At Ro_th of order 1e-10 the root lies near 0.03 degrees.

        The default search, which starts at 5 degrees and next tries 0.1,
        never brackets it.
        """
        result = abs_vort_rce_zero_lh88(LAT_MAX, 1e-9, init_guess=0.001)
        expected = _eq10_root(LAT_MAX, 1e-9)
        assert expected < 0.1
        assert result.item() == pytest.approx(expected, rel=1e-9)

    def test_init_guess_does_not_move_root(self) -> None:
        """Eq. 10 has one root in (0, 90] degrees, wherever the search starts."""
        default = abs_vort_rce_zero_lh88(48.86, 0.853).item()
        assert abs_vort_rce_zero_lh88(
            48.86, 0.853, init_guess=60
        ).item() == pytest.approx(default, abs=1e-9)


class TestAbsVortRceLh88ZeroApprox:
    @pytest.mark.parametrize("factor", [1.0, 1.5])
    def test_matches_exact_root_within_grid_spacing(self, factor: float) -> None:
        """The two values of delta_h have exact roots more than 0.5 degrees apart.

        The grid starts at 8 degrees north, poleward of where the square-root
        argument goes negative at 1.5 DELTA_H. On a grid reaching that band,
        or the Southern Hemisphere, this deprecated function returns NaN.
        The interpolation error measured 3.4e-5 degrees at most.
        """
        lats = np.arange(8.0, 89.5, 0.05)
        lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats}, name="lat")
        result = abs_vort_rce_lh88_zero_approx(
            lat,
            LAT_MAX,
            delta_h=factor * DELTA_H,
            height=HEIGHT,
            rot_rate=ROT_RATE,
            radius=RADIUS,
            grav=GRAV,
        )
        expected = _eq10_root(LAT_MAX, factor * BURG_DELH)
        assert abs(result.item() - expected) < 1e-3
        assert abs(expected - _eq10_root(LAT_MAX, BURG_DELH * (2.5 - factor))) > 0.5
