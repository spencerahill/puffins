"""Tests for the held_hou_1980 module.

Held & Hou (1980), "Nonlinear Axially Symmetric Circulations in a Nearly
Inviscid Atmosphere", J. Atmos. Sci., 37, 515-533.

Each function with a closed-form expression carries at least one known-value
test that reconstructs the expected output from raw numpy (not from the
module's own helpers), so a perturbed coefficient in the source is caught. The
small-angle functions are additionally checked against their full counterparts
in the limit where the two must agree.
"""

import numpy as np
import pytest
import xarray as xr

from puffins.constants import (
    DELTA_H,
    DELTA_V,
    HEIGHT_TROPO,
    RAD_EARTH,
    ROT_RATE_EARTH,
    THETA_REF,
)
from puffins.held_hou_1980 import (
    _hc_edge_hh80_lhs,
    dpot_temp_rce_hh80_dlat,
    hc_edge_hh80,
    hc_edge_hh80_small_angle,
    pot_temp_rce_hh80,
    pot_temp_rce_hh80_small_ang,
    u_crit_switch_lat_hh80,
    u_crit_switch_lat_hh80_small_angle,
    u_rce_hh80,
)


class TestPotTempRceHH80:
    """RCE potential temperature, HH80 Eq. 2."""

    def test_known_value_raw_numpy(self) -> None:
        # Non-default values for every parameter so a mutation of any one
        # coefficient changes the result. lat=30 -> cos^2 = 0.75; z/height =
        # 0.8 so the vertical term (0.3) and horizontal term are both nonzero
        # and distinct.
        lat, z = 30.0, 8.0e3
        theta_ref, height, delta_h, delta_v = 300.0, 10.0e3, 0.2, 0.1
        expected = theta_ref * (
            1.0
            + delta_h * (np.cos(np.deg2rad(lat)) ** 2 - 2.0 / 3.0)
            + (z / height - 0.5) * delta_v
        )
        result = pot_temp_rce_hh80(lat, z, theta_ref, height, delta_h, delta_v)
        np.testing.assert_allclose(result, expected)

    def test_global_mean_removes_horizontal_term(self) -> None:
        # <cos^2 phi> over the sphere is 2/3, so the (cos^2 - 2/3) term
        # integrates to zero: the area-weighted mean at mid-height equals
        # theta_ref exactly. This pins the -2/3 offset.
        lats = np.linspace(-90.0, 90.0, 1801)
        theta = pot_temp_rce_hh80(
            lats, 0.5 * HEIGHT_TROPO, THETA_REF, HEIGHT_TROPO, DELTA_H, DELTA_V
        )
        weights = np.cos(np.deg2rad(lats))
        mean = np.sum(theta * weights) / np.sum(weights)
        np.testing.assert_allclose(mean, THETA_REF, rtol=1e-4)

    def test_vertical_term_sign(self) -> None:
        # Warmer aloft: increasing z raises the potential temperature.
        low = pot_temp_rce_hh80(0.0, 0.0, THETA_REF, HEIGHT_TROPO, DELTA_H, DELTA_V)
        high = pot_temp_rce_hh80(
            0.0, HEIGHT_TROPO, THETA_REF, HEIGHT_TROPO, DELTA_H, DELTA_V
        )
        assert high > low

    def test_preserves_dataarray(self) -> None:
        lats = xr.DataArray([0.0, 30.0, 60.0], dims=["lat"])
        out = pot_temp_rce_hh80(
            lats, 0.5 * HEIGHT_TROPO, THETA_REF, HEIGHT_TROPO, DELTA_H, DELTA_V
        )
        assert isinstance(out, xr.DataArray)
        assert out.dims == ("lat",)


class TestPotTempRceHH80SmallAng:
    """RCE potential temperature in the small-angle limit."""

    def test_known_value_raw_numpy(self) -> None:
        lat, z = 10.0, 8.0e3
        theta_ref, height, delta_h, delta_v = 300.0, 10.0e3, 0.2, 0.1
        expected = theta_ref * (
            1.0
            + delta_h * (1.0 - np.deg2rad(lat) ** 2 - 2.0 / 3.0)
            + delta_v * (z / height - 0.5)
        )
        result = pot_temp_rce_hh80_small_ang(
            lat,
            z=z,
            theta_ref=theta_ref,
            height=height,
            delta_h=delta_h,
            delta_v=delta_v,
        )
        np.testing.assert_allclose(result, expected)

    def test_defaults(self) -> None:
        # With the default mid-tropospheric height the vertical term vanishes.
        lat = 15.0
        expected = THETA_REF * (
            1.0 + DELTA_H * (1.0 - np.deg2rad(lat) ** 2 - 2.0 / 3.0)
        )
        np.testing.assert_allclose(pot_temp_rce_hh80_small_ang(lat), expected)

    def test_agrees_with_full_at_small_latitude(self) -> None:
        # cos^2(phi) ~ 1 - phi^2 for small phi, so the small-angle profile
        # matches the full one near the equator.
        lat = 2.0
        full = pot_temp_rce_hh80(
            lat, 0.5 * HEIGHT_TROPO, THETA_REF, HEIGHT_TROPO, DELTA_H, DELTA_V
        )
        small = pot_temp_rce_hh80_small_ang(lat)
        np.testing.assert_allclose(small, full, rtol=1e-4)


class TestURceHH80:
    """Zonal wind in gradient balance with the RCE temperatures."""

    def test_known_value_raw_numpy_non_earth(self) -> None:
        # Non-default rotation rate and radius so both are exercised.
        lat, ro = 25.0, 0.3
        rot_rate, radius = 1.5e-4, 3.4e6
        expected = (
            rot_rate
            * radius
            * np.cos(np.deg2rad(lat))
            * ((1.0 + 2.0 * ro) ** 0.5 - 1.0)
        )
        result = u_rce_hh80(lat, ro, rot_rate=rot_rate, radius=radius)
        np.testing.assert_allclose(result, expected)

    def test_vanishes_at_zero_thermal_rossby(self) -> None:
        lats = np.array([0.0, 30.0, 60.0])
        np.testing.assert_allclose(u_rce_hh80(lats, 0.0), 0.0, atol=1e-12)

    def test_zero_at_pole_max_at_equator(self) -> None:
        ro = 0.2
        assert u_rce_hh80(90.0, ro) == pytest.approx(0.0, abs=1e-9)
        eq = u_rce_hh80(0.0, ro)
        mid = u_rce_hh80(45.0, ro)
        assert eq > mid > 0.0


class TestDPotTempRceHH80Dlat:
    """Meridional derivative of the RCE potential temperature."""

    def test_known_value_raw_numpy(self) -> None:
        lat, delta_h = 20.0, 0.2
        # -2 delta_h sin cos = -delta_h sin(2 phi)
        expected = -2.0 * delta_h * np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(lat))
        np.testing.assert_allclose(dpot_temp_rce_hh80_dlat(lat, delta_h), expected)

    def test_matches_finite_difference_of_profile(self) -> None:
        # It is (1/theta_ref) d/dphi of the horizontal part of Eq. 2, with phi
        # in radians. Compare to a central difference of the full profile.
        lat, delta_h = 35.0, DELTA_H
        dphi_deg = 1e-4
        theta_ref = THETA_REF
        plus = pot_temp_rce_hh80(
            lat + dphi_deg,
            0.5 * HEIGHT_TROPO,
            theta_ref,
            HEIGHT_TROPO,
            delta_h,
            DELTA_V,
        )
        minus = pot_temp_rce_hh80(
            lat - dphi_deg,
            0.5 * HEIGHT_TROPO,
            theta_ref,
            HEIGHT_TROPO,
            delta_h,
            DELTA_V,
        )
        fd = (plus - minus) / (2.0 * np.deg2rad(dphi_deg)) / theta_ref
        np.testing.assert_allclose(dpot_temp_rce_hh80_dlat(lat, delta_h), fd, rtol=1e-6)

    def test_antisymmetric_and_zero_at_equator(self) -> None:
        assert dpot_temp_rce_hh80_dlat(0.0, DELTA_H) == pytest.approx(0.0, abs=1e-12)
        north = dpot_temp_rce_hh80_dlat(30.0, DELTA_H)
        south = dpot_temp_rce_hh80_dlat(-30.0, DELTA_H)
        np.testing.assert_allclose(north, -south)
        assert north < 0.0  # theta decreases poleward in the NH


class TestUCritSwitchLatHH80:
    """Supercriticality (RCE == AMC wind) latitude."""

    def test_known_value_raw_numpy(self) -> None:
        ro = 0.4
        expected = np.rad2deg(np.arccos((1.0 + 2.0 * ro) ** -0.25))
        np.testing.assert_allclose(u_crit_switch_lat_hh80(ro), expected)

    def test_zero_at_zero_thermal_rossby(self) -> None:
        assert u_crit_switch_lat_hh80(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_monotonic_increasing(self) -> None:
        ros = np.array([0.05, 0.2, 0.5, 1.0])
        lats = u_crit_switch_lat_hh80(ros)
        assert np.all(np.diff(lats) > 0.0)


class TestUCritSwitchLatHH80SmallAngle:
    """Small-angle supercriticality latitude."""

    def test_known_value_raw_numpy(self) -> None:
        ro = 0.09
        expected = np.rad2deg(ro**0.5)
        np.testing.assert_allclose(u_crit_switch_lat_hh80_small_angle(ro), expected)

    def test_agrees_with_full_at_small_rossby(self) -> None:
        ro = 1e-3
        full = u_crit_switch_lat_hh80(ro)
        small = u_crit_switch_lat_hh80_small_angle(ro)
        np.testing.assert_allclose(small, full, rtol=1e-2)


class TestHcEdgeHH80SmallAngle:
    """Small-angle Hadley cell edge, HH80 Eq. 16."""

    def test_known_value_raw_numpy(self) -> None:
        ro = 0.06
        expected = np.rad2deg((5.0 * ro / 3.0) ** 0.5)
        np.testing.assert_allclose(hc_edge_hh80_small_angle(ro), expected)

    def test_scales_as_sqrt_rossby(self) -> None:
        # Doubling Ro widens the cell by sqrt(2).
        edge1 = hc_edge_hh80_small_angle(0.05)
        edge2 = hc_edge_hh80_small_angle(0.10)
        np.testing.assert_allclose(edge2 / edge1, np.sqrt(2.0))


class TestHcEdgeHH80Lhs:
    """Left-hand side of the transcendental Eq. 17 (root at the cell edge)."""

    def test_known_value_raw_numpy(self) -> None:
        lat, ro = 25.0, 0.3
        y = np.sin(np.deg2rad(lat))
        expected = (
            (1.0 / 3.0) * (4.0 * ro - 1.0) * y**3
            - y**5 / (1.0 - y**2)
            - y
            + 0.5 * np.log((1.0 + y) / (1.0 - y))
        )
        np.testing.assert_allclose(_hc_edge_hh80_lhs(lat, ro), expected)

    def test_returns_python_float(self) -> None:
        # `brentq_solver_sweep_param` declares `Callable[..., float]`, and the
        # bare expression yields np.float64. `type(...) is float` rather than
        # `isinstance`, because np.float64 subclasses float and so would pass
        # an isinstance check even with the conversion removed.
        assert type(_hc_edge_hh80_lhs(20.0, 0.2)) is float


class TestHcEdgeHH80:
    """Numerically-solved (full) Hadley cell edge, HH80 Eq. 17."""

    def test_solution_is_root_of_eq17(self) -> None:
        ro = 0.3
        edge = hc_edge_hh80(ro)
        np.testing.assert_allclose(_hc_edge_hh80_lhs(edge.item(), ro), 0.0, atol=1e-8)

    def test_returns_dataarray_over_param_range(self) -> None:
        ros = np.array([0.1, 0.3, 0.6])
        edges = hc_edge_hh80(ros)
        assert isinstance(edges, xr.DataArray)
        assert edges.size == ros.size
        # Each entry solves its own Eq. 17.
        for ro, edge in zip(ros, edges.values, strict=True):
            np.testing.assert_allclose(
                _hc_edge_hh80_lhs(float(edge), ro), 0.0, atol=1e-8
            )

    def test_approaches_small_angle_for_small_rossby(self) -> None:
        ro = 0.01
        full = hc_edge_hh80(ro).item()
        small = hc_edge_hh80_small_angle(ro)
        np.testing.assert_allclose(full, small, rtol=0.05)

    def test_widens_with_thermal_rossby(self) -> None:
        ros = np.array([0.1, 0.3, 0.6])
        edges = hc_edge_hh80(ros).values
        assert np.all(np.diff(edges) > 0.0)

    def test_custom_bound_guess_range(self) -> None:
        # A custom bracket range that still contains the root reproduces the
        # default-range solution.
        ro = 0.3
        default = hc_edge_hh80(ro).item()
        custom = hc_edge_hh80(ro, bound_guess_range=np.arange(5.0, 60.1, 5.0)).item()
        np.testing.assert_allclose(custom, default, rtol=1e-6)

    def test_init_guess_sets_one_end_of_the_bracket(self) -> None:
        # `init_guess` supplies one end of the sign-change bracket, so it can
        # decide whether a root is found at all. With `bound_guess_range`
        # restricted to latitudes entirely poleward of the root (~34.9 deg for
        # ro=0.3), Eq. 17 is negative at every one of them. An `init_guess`
        # equatorward of the root is positive there, so a sign change is found;
        # one that is also poleward is negative, so none is, and the solver
        # returns nan. Pins the parameter: hardcoding it inside the function
        # would make both calls agree.
        ro = 0.3
        poleward_only = np.array([40.0, 50.0])
        found = hc_edge_hh80(ro, init_guess=0.1, bound_guess_range=poleward_only)
        np.testing.assert_allclose(found.item(), hc_edge_hh80(ro).item(), rtol=1e-6)

        not_found = hc_edge_hh80(ro, init_guess=45.0, bound_guess_range=poleward_only)
        assert np.isnan(not_found.item())
