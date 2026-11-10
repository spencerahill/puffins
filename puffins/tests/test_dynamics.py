"""Tests for dynamics module."""

import numpy as np
import pytest
import xarray as xr

from puffins.constants import (
    GRAV_EARTH,
    HEIGHT_TROPO,
    RAD_EARTH,
    RAD_MARS,
    ROT_RATE_EARTH,
    ROT_RATE_MARS,
)
from puffins.dynamics import (
    abs_ang_mom,
    abs_vort_from_u,
    abs_vort_vert_comp,
    brunt_vaisala_freq,
    bulk_stat_stab,
    coriolis_param,
    plan_burg_num,
    rel_vort_from_u,
    ross_num_from_uwind,
    ross_num_gen,
    rossby_radius,
    therm_ross_num,
    u_bci_2layer_qg,
    z_from_hypso,
    zonal_fric_inferred_steady,
)
from puffins.names import LAT_STR, LEV_STR


def _lat_dataarray(values: list[float] | np.ndarray) -> xr.DataArray:
    """Create a DataArray with a latitude coordinate."""
    vals = np.array(values, dtype=float)
    return xr.DataArray(vals, coords={LAT_STR: vals}, dims=[LAT_STR])


def _uwind_dataarray(
    values: list[float] | np.ndarray,
    lats: list[float] | np.ndarray,
) -> xr.DataArray:
    """Create a zonal wind DataArray with latitude coordinate."""
    return xr.DataArray(
        np.array(values, dtype=float),
        coords={LAT_STR: np.array(lats, dtype=float)},
        dims=[LAT_STR],
    )


def _cell_centered_lats(dlat: float) -> xr.DataArray:
    """Latitudes from pole to pole whose end rows lie dlat/2 from the poles."""
    return _lat_dataarray(np.arange(-90.0 + dlat / 2, 90.0, dlat))


# Zonal winds u(phi) with their analytic derivatives du/dphi, phi in radians.
# Both vanish only as fast as cos(phi) toward the poles, so both have nonzero
# vorticity at the poles, which the cos(phi) form of the relative vorticity
# gets wrong by about 100% at the end rows of a cell-centered grid.
_U_PROFILES = {
    "sin2phi": (lambda phi: 20.0 * np.sin(2 * phi), lambda phi: 40.0 * np.cos(2 * phi)),
    "solid_body": (lambda phi: 30.0 * np.cos(phi), lambda phi: -30.0 * np.sin(phi)),
}

_P0 = 1e5
_CUSTOM_NAMES = {LAT_STR: "latitude", LEV_STR: "pfull"}

# Latitudes for a NumPy lat that are neither _wind_fields(36)'s coordinate
# nor symmetric about the equator, so a lat that is label-aligned, reversed
# or ignored changes the result.
_LAT_OFF_GRID = np.linspace(-80.0, 85.0, 36)


def _wind_fields(
    nlat: int,
    dims: tuple[str, str] = (LAT_STR, LEV_STR),
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """u, v (m/s) and omega (Pa/s) on 17 levels in Pa, with dims ordered as given.

    u is linear in pressure, so its centered differences in pressure are
    exact, and v never vanishes, so omega * du/dp / v is finite everywhere.
    """
    lat = _lat_dataarray(np.linspace(-87.5, 87.5, nlat))
    levs = np.linspace(1e4, 9e4, 17)
    plev = xr.DataArray(levs, coords={LEV_STR: levs}, dims=[LEV_STR])
    phi = np.deg2rad(lat)
    uwind = 20.0 * np.sin(2 * phi) * (1 - plev / _P0)
    vwind = 2.0 * np.sin(3 * phi) * np.cos(np.pi * plev / _P0) + 3.0
    omega = 0.05 * np.cos(3 * phi) * np.sin(np.pi * plev / _P0)
    return (
        uwind.transpose(*dims).rename("uwind"),
        vwind.transpose(*dims).rename("vwind"),
        omega.transpose(*dims).rename("omega"),
    )


# ---------------------------------------------------------------------------
# TestCoriolisParam
# ---------------------------------------------------------------------------


class TestCoriolisParam:
    """Tests for coriolis_param."""

    def test_equator_is_zero(self) -> None:
        """Coriolis parameter is zero at the equator."""
        result = coriolis_param(0.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_north_pole(self) -> None:
        """Coriolis parameter at the North Pole equals 2*Omega."""
        result = coriolis_param(90.0)
        np.testing.assert_allclose(result, 2.0 * ROT_RATE_EARTH)

    def test_south_pole(self) -> None:
        """Coriolis parameter at the South Pole equals -2*Omega."""
        result = coriolis_param(-90.0)
        np.testing.assert_allclose(result, -2.0 * ROT_RATE_EARTH)

    def test_antisymmetric(self) -> None:
        """Coriolis parameter is antisymmetric about the equator."""
        f_30n = coriolis_param(30.0)
        f_30s = coriolis_param(-30.0)
        np.testing.assert_allclose(f_30n, -f_30s)

    def test_30n(self) -> None:
        """Coriolis parameter at 30N is Omega (sin(30)=0.5)."""
        result = coriolis_param(30.0)
        np.testing.assert_allclose(result, ROT_RATE_EARTH)

    def test_custom_rot_rate(self) -> None:
        """Works with a custom rotation rate."""
        result = coriolis_param(90.0, rot_rate=1.0)
        np.testing.assert_allclose(result, 2.0)

    def test_array_input(self) -> None:
        """Works with numpy array input."""
        lats = np.array([-90, -30, 0, 30, 90], dtype=float)
        result = coriolis_param(lats)
        assert result.shape == (5,)
        np.testing.assert_allclose(result[2], 0.0, atol=1e-15)

    def test_dataarray_input(self) -> None:
        """Works with xarray DataArray input."""
        lats = _lat_dataarray([0.0, 30.0, 60.0, 90.0])
        result = coriolis_param(lats)
        assert isinstance(result, xr.DataArray)
        np.testing.assert_allclose(result.sel(lat=0.0).item(), 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# TestPlanBurgNum
# ---------------------------------------------------------------------------


class TestPlanBurgNum:
    """Tests for plan_burg_num."""

    def test_default_earth(self) -> None:
        """Planetary Burger number with default Earth values."""
        result = plan_burg_num()
        expected = HEIGHT_TROPO * GRAV_EARTH / (ROT_RATE_EARTH * RAD_EARTH) ** 2
        np.testing.assert_allclose(result, expected)

    def test_positive(self) -> None:
        """Planetary Burger number is positive."""
        assert plan_burg_num() > 0

    def test_custom_params(self) -> None:
        """Works with custom parameters."""
        result = plan_burg_num(height=1.0, grav=1.0, rot_rate=1.0, radius=1.0)
        np.testing.assert_allclose(result, 1.0)

    def test_scales_with_height(self) -> None:
        """Burger number scales linearly with height."""
        bu1 = plan_burg_num(height=1e4)
        bu2 = plan_burg_num(height=2e4)
        np.testing.assert_allclose(bu2 / bu1, 2.0)


# ---------------------------------------------------------------------------
# TestThermRossNum
# ---------------------------------------------------------------------------


class TestThermRossNum:
    """Tests for therm_ross_num."""

    def test_zero_delta_h(self) -> None:
        """Thermal Rossby number is zero when delta_h is zero."""
        result = therm_ross_num(0.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_positive_for_positive_delta_h(self) -> None:
        """Thermal Rossby number is positive for positive delta_h."""
        result = therm_ross_num(1.0 / 6.0)
        assert result > 0

    def test_default_lat_max_90(self) -> None:
        """Default lat_max of 90 gives sin(90)=1, so result equals delta_h * Bu."""
        delta_h = 0.2
        result = therm_ross_num(delta_h)
        expected = delta_h * plan_burg_num()
        np.testing.assert_allclose(result, expected)

    def test_scales_with_delta_h(self) -> None:
        """Thermal Rossby number scales linearly with delta_h."""
        r1 = therm_ross_num(0.1)
        r2 = therm_ross_num(0.2)
        np.testing.assert_allclose(r2 / r1, 2.0)

    def test_array_delta_h(self) -> None:
        """Works with array-valued delta_h."""
        delta_h = np.array([0.1, 0.2, 0.3])
        result = therm_ross_num(delta_h)
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# TestAbsAngMom
# ---------------------------------------------------------------------------


class TestAbsAngMom:
    """Tests for abs_ang_mom."""

    def test_zero_wind_equator(self) -> None:
        """At the equator with u=0, angular momentum is Omega*a^2."""
        u = _uwind_dataarray([0.0], [0.0])
        result = abs_ang_mom(u)
        expected = ROT_RATE_EARTH * RAD_EARTH**2
        np.testing.assert_allclose(result.item(), expected)

    def test_zero_wind_pole(self) -> None:
        """At the pole with u=0, angular momentum is zero (cos(90)=0)."""
        u = _uwind_dataarray([0.0], [90.0])
        result = abs_ang_mom(u)
        np.testing.assert_allclose(result.item(), 0.0, atol=1e-5)

    def test_nonzero_wind(self) -> None:
        """Angular momentum with nonzero wind at the equator."""
        u_val = 10.0
        u = _uwind_dataarray([u_val], [0.0])
        result = abs_ang_mom(u)
        expected = RAD_EARTH * (ROT_RATE_EARTH * RAD_EARTH + u_val)
        np.testing.assert_allclose(result.item(), expected)

    def test_explicit_lat(self) -> None:
        """Works when lat is explicitly provided."""
        u = _uwind_dataarray([0.0], [45.0])
        result = abs_ang_mom(u, lat=45.0)
        assert isinstance(result, xr.DataArray)

    def test_multiple_latitudes(self) -> None:
        """Works with multiple latitudes."""
        lats = [0.0, 30.0, 60.0, 90.0]
        u = _uwind_dataarray([0.0, 0.0, 0.0, 0.0], lats)
        result = abs_ang_mom(u)
        assert result.shape == (4,)
        # Monotonically decreasing from equator to pole for u=0
        assert result.values[0] > result.values[-1]

    @pytest.mark.parametrize("nlat", [36, 17])
    @pytest.mark.parametrize("dims", [(LAT_STR, LEV_STR), (LEV_STR, LAT_STR)])
    def test_numpy_lat_follows_lat_dim(self, nlat: int, dims: tuple[str, str]) -> None:
        """A NumPy lat is taken along the latitude dimension in either order.

        Issue #124: it was multiplied along the last dimension, which raised
        for 36 latitudes on 17 levels and gave wrong values for 17 on 17.
        """
        uwind, _, _ = _wind_fields(nlat, dims)
        result = abs_ang_mom(uwind, lat=uwind[LAT_STR].values)
        xr.testing.assert_identical(result, abs_ang_mom(uwind))

    @pytest.mark.parametrize("numpy_lat", [True, False])
    def test_custom_dim_names(self, numpy_lat: bool) -> None:
        """lat goes along ``lat_str``, whatever it is named, NumPy or default."""
        uwind = _wind_fields(17)[0]
        renamed = uwind.rename(_CUSTOM_NAMES)
        lat = renamed["latitude"].values if numpy_lat else None
        result = abs_ang_mom(renamed, lat=lat, lat_str="latitude")
        assert result.dims == ("latitude", "pfull")
        assert np.array_equal(result.values, abs_ang_mom(uwind).values)

    def test_numpy_lat_of_wrong_length_raises(self) -> None:
        """A NumPy lat of another length than the latitude dimension raises."""
        uwind = _wind_fields(36)[0]
        with pytest.raises(ValueError, match="conflicting dimension sizes"):
            abs_ang_mom(uwind, lat=np.linspace(-87.5, 87.5, 35))

    def test_numpy_lat_as_long_as_levels_raises(self) -> None:
        """17 latitudes on 36 latitudes by 17 levels raise.

        Master multiplied them along the levels with no error.
        """
        uwind = _wind_fields(36)[0]
        with pytest.raises(ValueError, match="conflicting dimension sizes"):
            abs_ang_mom(uwind, lat=np.linspace(-87.5, 87.5, 17))

    def test_numpy_lat_needs_a_lat_str_dim(self) -> None:
        """A NumPy lat with a field lacking a ``lat_str`` dimension raises.

        Without the check it became a new dimension, giving an outer product
        over two latitude dimensions with no error.
        """
        uwind = _wind_fields(36)[0].rename(_CUSTOM_NAMES)
        with pytest.raises(ValueError, match="not a dimension of the field"):
            abs_ang_mom(uwind, lat=uwind["latitude"].values)

    def test_known_value_with_positional_lat_mars(self) -> None:
        """a cos(lat) (Omega a cos(lat) + u) on a NumPy lat, taken in order.

        The lat is not the field's coordinate, u is nonzero away from the
        equator, and the radius and rotation rate are Mars's.
        """
        uwind = _wind_fields(36, (LEV_STR, LAT_STR))[0]
        cos = np.cos(np.deg2rad(_LAT_OFF_GRID))
        expected = RAD_MARS * cos * (ROT_RATE_MARS * RAD_MARS * cos + uwind.values)
        result = abs_ang_mom(
            uwind, lat=_LAT_OFF_GRID, radius=RAD_MARS, rot_rate=ROT_RATE_MARS
        )
        np.testing.assert_allclose(
            result.transpose(LEV_STR, LAT_STR).values, expected, rtol=1e-13
        )

    def test_scalar_lat(self) -> None:
        """A scalar lat is used at every point: cos(60) = 1/2."""
        uwind = _wind_fields(36)[0]
        result = abs_ang_mom(uwind, lat=60.0)
        expected = RAD_EARTH * 0.5 * (ROT_RATE_EARTH * RAD_EARTH * 0.5 + uwind.values)
        np.testing.assert_allclose(result.values, expected, rtol=1e-12)

    def test_two_dimensional_numpy_lat_used_as_given(self) -> None:
        """A NumPy lat of u's own shape multiplies element by element."""
        uwind = _wind_fields(36)[0]
        lat2d = np.broadcast_to(uwind[LAT_STR].values[:, None], uwind.shape).copy()
        result = abs_ang_mom(uwind, lat=lat2d)
        np.testing.assert_array_equal(result.values, abs_ang_mom(uwind).values)

    @pytest.mark.parametrize("as_dataarray", [False, True])
    def test_scalar_wind_with_lat_array(self, as_dataarray: bool) -> None:
        """abs_ang_mom(0, lat=lats), the planetary angular momentum, as callers use.

        A NumPy lat gives a NumPy array, as on master; a DataArray lat gives
        a DataArray on that lat's coordinate.
        """
        lats = np.linspace(-90.0, 90.0, 19)
        lat_in = _lat_dataarray(lats) if as_dataarray else lats
        result = abs_ang_mom(0, lat=lat_in)  # type: ignore[arg-type]
        assert isinstance(result, xr.DataArray) == as_dataarray
        expected = ROT_RATE_EARTH * RAD_EARTH**2 * np.cos(np.deg2rad(lats)) ** 2
        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-13)


# ---------------------------------------------------------------------------
# TestAbsVortVertComp
# ---------------------------------------------------------------------------


class TestAbsVortVertComp:
    """Tests for abs_vort_vert_comp."""

    def test_returns_dataarray(self) -> None:
        """Returns an xarray DataArray."""
        lats = np.linspace(-80, 80, 41)
        u = _uwind_dataarray(np.zeros_like(lats), lats)
        m = abs_ang_mom(u)
        result = abs_vort_vert_comp(m)
        assert isinstance(result, xr.DataArray)

    def test_shape_preserved(self) -> None:
        """Output shape matches input shape."""
        lats = np.linspace(-80, 80, 41)
        u = _uwind_dataarray(np.zeros_like(lats), lats)
        m = abs_ang_mom(u)
        result = abs_vort_vert_comp(m)
        assert result.shape == m.shape

    def test_solid_body_matches_coriolis(self) -> None:
        """For u=0 (solid-body rotation), abs vorticity equals Coriolis parameter."""
        lats = np.linspace(-80, 80, 41)
        u = _uwind_dataarray(np.zeros_like(lats), lats)
        m = abs_ang_mom(u)
        result = abs_vort_vert_comp(m)
        expected = coriolis_param(_lat_dataarray(lats))
        # Finite-difference derivative introduces error at endpoints; compare interior
        np.testing.assert_allclose(
            result.values[2:-2], expected.values[2:-2], rtol=1e-2
        )


# ---------------------------------------------------------------------------
# TestAbsVortFromU
# ---------------------------------------------------------------------------


class TestAbsVortFromU:
    """Tests for abs_vort_from_u."""

    def test_zero_wind(self) -> None:
        """With u=0, absolute vorticity equals planetary vorticity (2*Omega*sin(lat))."""
        lats = np.linspace(-80, 80, 41)
        u = _uwind_dataarray(np.zeros_like(lats), lats)
        result = abs_vort_from_u(u)
        expected = coriolis_param(_lat_dataarray(lats))
        np.testing.assert_allclose(result.values, expected.values, atol=1e-12)

    def test_returns_dataarray(self) -> None:
        """Returns an xarray DataArray."""
        lats = np.linspace(-60, 60, 31)
        u = _uwind_dataarray(np.ones(31) * 5.0, lats)
        result = abs_vort_from_u(u)
        assert isinstance(result, xr.DataArray)

    def test_positive_nh(self) -> None:
        """Absolute vorticity is positive in NH for zero wind."""
        lats = np.linspace(10, 80, 15)
        u = _uwind_dataarray(np.zeros(15), lats)
        result = abs_vort_from_u(u)
        assert (result.values > 0).all()


# ---------------------------------------------------------------------------
# TestRelVortFromU
# ---------------------------------------------------------------------------


class TestRelVortFromU:
    """Tests for rel_vort_from_u."""

    def test_zero_wind(self) -> None:
        """Relative vorticity is zero for zero wind."""
        lats = np.linspace(-80, 80, 41)
        u = _uwind_dataarray(np.zeros(41), lats)
        result = rel_vort_from_u(u)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-15)

    def test_returns_dataarray(self) -> None:
        """Returns an xarray DataArray."""
        lats = np.linspace(-60, 60, 31)
        u = _uwind_dataarray(np.ones(31) * 10.0, lats)
        result = rel_vort_from_u(u)
        assert isinstance(result, xr.DataArray)

    def test_shape_preserved(self) -> None:
        """Output shape matches input shape."""
        lats = np.linspace(-80, 80, 41)
        u = _uwind_dataarray(np.ones(41) * 5.0, lats)
        result = rel_vort_from_u(u)
        assert result.shape == u.shape

    @pytest.mark.parametrize("profile", sorted(_U_PROFILES))
    def test_matches_closed_form_at_every_row(self, profile: str) -> None:
        """Within 5e-4 of the exact vorticity at every row of a 1-degree grid.

        The exact zeta is (-du/dphi + u tan(phi)) / a, with du/dphi written
        out analytically. The largest error measured was 2.5e-4 of the
        largest |zeta|, for sin2phi; the cos(phi) form this replaced (issue
        #126) is off by about 1.0 of it at the two end rows. Mars's radius
        checks that ``radius`` is used.
        """
        u_fn, du_fn = _U_PROFILES[profile]
        lat = _cell_centered_lats(1.0)
        phi = np.deg2rad(lat)
        uwind = u_fn(phi).rename("uwind")
        exact = (-du_fn(phi) + u_fn(phi) * np.tan(phi)) / RAD_MARS
        result = rel_vort_from_u(uwind, radius=RAD_MARS)
        assert np.array_equal(result[LAT_STR].values, lat.values)
        scale = np.abs(exact.values).max()
        np.testing.assert_allclose(
            result.values, exact.values, rtol=0, atol=5e-4 * scale
        )

    def test_second_order_convergence_at_every_row(self) -> None:
        """The largest error falls 6.25-fold from a 2.5- to a 1-degree grid.

        That is (2.5 / 1)^2, second order, end rows included, on this grid
        from pole to pole. The cos(phi) form's largest error, at its end
        rows, did not fall at all. The maximum is NumPy's, so a NaN row fails.
        """
        u_fn, du_fn = _U_PROFILES["sin2phi"]
        errors = []
        for dlat in (2.5, 1.0):
            phi = np.deg2rad(_cell_centered_lats(dlat))
            exact = (-du_fn(phi) + u_fn(phi) * np.tan(phi)) / RAD_EARTH
            result = rel_vort_from_u(u_fn(phi).rename("uwind"))
            errors.append(np.max(np.abs(result.values - exact.values)))
        assert errors[0] / errors[1] > 5.0

    def test_descending_latitude_matches_closed_form(self) -> None:
        """A north-to-south grid, as in ERA5, matches the exact vorticity too."""
        u_fn, du_fn = _U_PROFILES["sin2phi"]
        lat = _cell_centered_lats(1.0).isel({LAT_STR: slice(None, None, -1)})
        phi = np.deg2rad(lat)
        exact = (-du_fn(phi) + u_fn(phi) * np.tan(phi)) / RAD_EARTH
        result = rel_vort_from_u(u_fn(phi).rename("uwind"))
        assert np.array_equal(result[LAT_STR].values, lat.values)
        scale = np.abs(exact.values).max()
        np.testing.assert_allclose(
            result.values, exact.values, rtol=0, atol=5e-4 * scale
        )


# ---------------------------------------------------------------------------
# TestRossNumFromUwind
# ---------------------------------------------------------------------------


class TestRossNumFromUwind:
    """Tests for ross_num_from_uwind."""

    def test_zero_wind(self) -> None:
        """Rossby number is zero for zero wind (away from equator)."""
        lats = np.linspace(10, 80, 15)
        u = _uwind_dataarray(np.zeros(15), lats)
        result = ross_num_from_uwind(u)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-10)

    def test_returns_dataarray(self) -> None:
        """Returns an xarray DataArray."""
        lats = np.linspace(10, 80, 15)
        u = _uwind_dataarray(np.ones(15) * 20.0, lats)
        result = ross_num_from_uwind(u)
        assert isinstance(result, xr.DataArray)

    def test_explicit_lat(self) -> None:
        """Works when lat is explicitly provided."""
        lats = np.linspace(10, 80, 15)
        u = _uwind_dataarray(np.zeros(15), lats)
        result = ross_num_from_uwind(u, lat=lats)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-10)

    def test_matches_closed_form_with_mars_constants(self) -> None:
        """Within 4e-4 of the exact -zeta/f poleward of 5 degrees.

        Exact zeta for u = 20 sin(2 phi) is (-40 cos(2 phi) + u tan(phi)) / a,
        and f is 2 Omega sin(phi). On this 1-degree grid the largest error
        measured poleward of 5 degrees was 1.7e-4, where |Ro| reaches 0.85.
        """
        lat = _cell_centered_lats(1.0)
        phi = np.deg2rad(lat)
        uwind = (20.0 * np.sin(2 * phi)).rename("uwind")
        zeta = (-40.0 * np.cos(2 * phi) + uwind * np.tan(phi)) / RAD_MARS
        exact = -zeta / (2 * ROT_RATE_MARS * np.sin(phi))
        result = ross_num_from_uwind(uwind, radius=RAD_MARS, rot_rate=ROT_RATE_MARS)
        away = np.abs(lat.values) >= 5
        np.testing.assert_allclose(
            result.values[away], exact.values[away], rtol=0, atol=4e-4
        )

    @pytest.mark.parametrize("nlat", [36, 17])
    @pytest.mark.parametrize("dims", [(LAT_STR, LEV_STR), (LEV_STR, LAT_STR)])
    def test_numpy_lat_follows_lat_dim(self, nlat: int, dims: tuple[str, str]) -> None:
        """A NumPy lat is taken along the latitude dimension in either order.

        Issue #124: it was multiplied along the last dimension, which raised
        for 36 latitudes on 17 levels and gave wrong values for 17 on 17.
        """
        uwind, _, _ = _wind_fields(nlat, dims)
        result = ross_num_from_uwind(uwind, lat=uwind[LAT_STR].values)
        xr.testing.assert_identical(result, ross_num_from_uwind(uwind))

    @pytest.mark.parametrize("numpy_lat", [True, False])
    def test_custom_dim_names(self, numpy_lat: bool) -> None:
        """lat goes along ``lat_str``, whatever it is named, NumPy or default."""
        uwind = _wind_fields(17)[0]
        renamed = uwind.rename(_CUSTOM_NAMES)
        lat = renamed["latitude"].values if numpy_lat else None
        result = ross_num_from_uwind(renamed, lat=lat, lat_str="latitude")
        assert result.dims == ("latitude", "pfull")
        assert np.array_equal(result.values, ross_num_from_uwind(uwind).values)

    def test_numpy_lat_of_wrong_length_raises(self) -> None:
        """A NumPy lat of another length than the latitude dimension raises."""
        uwind = _wind_fields(36)[0]
        with pytest.raises(ValueError, match="conflicting dimension sizes"):
            ross_num_from_uwind(uwind, lat=np.linspace(-87.5, 87.5, 35))

    def test_numpy_lat_as_long_as_levels_raises(self) -> None:
        """17 latitudes on 36 latitudes by 17 levels raise.

        Master multiplied them along the levels with no error.
        """
        uwind = _wind_fields(36)[0]
        with pytest.raises(ValueError, match="conflicting dimension sizes"):
            ross_num_from_uwind(uwind, lat=np.linspace(-87.5, 87.5, 17))

    def test_lat_sets_f_alone(self) -> None:
        """-zeta/f with zeta on u's coordinate and f on a NumPy lat, in order."""
        uwind = _wind_fields(36, (LEV_STR, LAT_STR))[0]
        coriolis = 2 * ROT_RATE_EARTH * np.sin(np.deg2rad(_LAT_OFF_GRID))
        zeta = rel_vort_from_u(uwind).transpose(LEV_STR, LAT_STR).values
        result = ross_num_from_uwind(uwind, lat=_LAT_OFF_GRID)
        np.testing.assert_allclose(
            result.transpose(LEV_STR, LAT_STR).values, -zeta / coriolis, rtol=1e-13
        )

    def test_scalar_lat(self) -> None:
        """A scalar lat sets f at every point: f(30N) = Omega."""
        uwind = _wind_fields(36)[0]
        result = ross_num_from_uwind(uwind, lat=30.0)
        expected = -rel_vort_from_u(uwind).values / ROT_RATE_EARTH
        np.testing.assert_allclose(result.values, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# TestRossNumGen
# ---------------------------------------------------------------------------


class TestRossNumGen:
    """Tests for ross_num_gen."""

    def test_returns_dataarray(self) -> None:
        """Returns an xarray DataArray."""
        lats = np.linspace(10, 80, 15)
        levs = np.array([500.0, 700.0, 850.0])
        uwind = xr.DataArray(
            np.ones((15, 3)),
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        vwind = xr.DataArray(
            np.ones((15, 3)) * 0.5,
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        omega = xr.DataArray(
            np.ones((15, 3)) * 0.01,
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        result = ross_num_gen(uwind, vwind, omega)
        assert isinstance(result, xr.DataArray)

    def test_hpa_to_pa_flag(self) -> None:
        """hpa_to_pa flag scales du_dp by 1e-2."""
        lats = np.linspace(10, 80, 15)
        levs = np.array([500.0, 700.0, 850.0])
        # Use non-uniform uwind so du_dp is nonzero.
        u_vals = np.outer(np.ones(15), np.array([5.0, 10.0, 15.0]))
        uwind = xr.DataArray(
            u_vals,
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        vwind = xr.DataArray(
            np.ones((15, 3)) * 0.5,
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        omega = xr.DataArray(
            np.ones((15, 3)) * 0.01,
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        r_pa = ross_num_gen(uwind, vwind, omega, hpa_to_pa=False)
        r_hpa = ross_num_gen(uwind, vwind, omega, hpa_to_pa=True)
        # They should differ because hpa_to_pa scales the du_dp term
        assert not np.allclose(r_pa.values, r_hpa.values)

    def test_zero_omega_equals_ross_num_from_uwind(self) -> None:
        """With omega zero the two functions agree exactly (issue #126).

        Mars's radius and rotation rate check that both are passed through.
        """
        uwind, vwind, omega = _wind_fields(36)
        result = ross_num_gen(
            uwind, vwind, 0 * omega, radius=RAD_MARS, rot_rate=ROT_RATE_MARS
        )
        expected = ross_num_from_uwind(uwind, radius=RAD_MARS, rot_rate=ROT_RATE_MARS)
        assert np.array_equal(result.values, expected.values)

    def test_vertical_term_known_value(self) -> None:
        """ross_num_gen minus ross_num_from_uwind is omega (du/dp) / (v f).

        u is 20 sin(2 phi) (1 - p/p0), so du/dp is -20 sin(2 phi) / p0,
        which centered and one-sided differences give exactly.
        """
        uwind, vwind, omega = _wind_fields(36)
        phi = np.deg2rad(uwind[LAT_STR].values)[:, np.newaxis]
        du_dp = -20.0 * np.sin(2 * phi) / _P0
        coriolis = 2 * ROT_RATE_EARTH * np.sin(phi)
        expected = omega.values * du_dp / (vwind.values * coriolis)
        result = ross_num_gen(uwind, vwind, omega) - ross_num_from_uwind(uwind)
        scale = np.abs(expected).max()
        assert scale > 0.01
        np.testing.assert_allclose(result.values, expected, rtol=0, atol=1e-12 * scale)

    def test_hpa_levels_with_flag_match_pa_levels(self) -> None:
        """Levels in hPa with omega in Pa/s and hpa_to_pa=True match levels in Pa.

        Without the flag the vertical term is 100 times too large (issue #123).
        """
        uwind, vwind, omega = _wind_fields(36)
        to_hpa = {LEV_STR: uwind[LEV_STR].values / 100.0}
        u_h, v_h, om_h = (x.assign_coords(to_hpa) for x in (uwind, vwind, omega))
        in_pa = ross_num_gen(uwind, vwind, omega)
        in_hpa = ross_num_gen(u_h, v_h, om_h, hpa_to_pa=True)
        np.testing.assert_allclose(in_hpa.values, in_pa.values, rtol=0, atol=1e-12)
        vert_pa = in_pa - ross_num_from_uwind(uwind)
        vert_hpa_unflagged = ross_num_gen(u_h, v_h, om_h) - ross_num_from_uwind(u_h)
        np.testing.assert_allclose(vert_hpa_unflagged.values, 100.0 * vert_pa.values)

    @pytest.mark.parametrize("nlat", [36, 17])
    @pytest.mark.parametrize("dims", [(LAT_STR, LEV_STR), (LEV_STR, LAT_STR)])
    def test_numpy_lat_follows_lat_dim(self, nlat: int, dims: tuple[str, str]) -> None:
        """A NumPy lat is taken along the latitude dimension in either order.

        Issue #124: it was multiplied along the last dimension, which raised
        for 36 latitudes on 17 levels and gave wrong values for 17 on 17.
        """
        uwind, vwind, omega = _wind_fields(nlat, dims)
        result = ross_num_gen(uwind, vwind, omega, lat=uwind[LAT_STR].values)
        xr.testing.assert_identical(result, ross_num_gen(uwind, vwind, omega))

    @pytest.mark.parametrize("numpy_lat", [True, False])
    def test_custom_dim_names(self, numpy_lat: bool) -> None:
        """lat goes along ``lat_str`` and du/dlev along ``lev_str``."""
        uwind, vwind, omega = _wind_fields(17)
        u_c, v_c, om_c = (x.rename(_CUSTOM_NAMES) for x in (uwind, vwind, omega))
        result = ross_num_gen(
            u_c,
            v_c,
            om_c,
            lat=u_c["latitude"].values if numpy_lat else None,
            lat_str="latitude",
            lev_str="pfull",
        )
        assert result.dims == ("latitude", "pfull")
        expected = ross_num_gen(uwind, vwind, omega)
        assert np.array_equal(result.values, expected.values)

    def test_numpy_lat_of_wrong_length_raises(self) -> None:
        """A NumPy lat of another length than the latitude dimension raises."""
        with pytest.raises(ValueError, match="conflicting dimension sizes"):
            ross_num_gen(*_wind_fields(36), lat=np.linspace(-87.5, 87.5, 35))

    def test_lat_sets_f_alone(self) -> None:
        """(-zeta + omega (du/dp) / v) / f, with f alone on a NumPy lat.

        Master also used the passed lat in the u tan(lat) term of zeta.
        """
        uwind, vwind, omega = _wind_fields(36)
        phi = np.deg2rad(uwind[LAT_STR].values)[:, np.newaxis]
        du_dp = -20.0 * np.sin(2 * phi) / _P0
        coriolis = 2 * ROT_RATE_EARTH * np.sin(np.deg2rad(_LAT_OFF_GRID))[:, None]
        zeta = rel_vort_from_u(uwind).values
        expected = (-zeta + omega.values * du_dp / vwind.values) / coriolis
        result = ross_num_gen(uwind, vwind, omega, lat=_LAT_OFF_GRID)
        scale = np.abs(expected).max()
        np.testing.assert_allclose(result.values, expected, rtol=0, atol=1e-12 * scale)


# ---------------------------------------------------------------------------
# TestBruntVaisalaFreq
# ---------------------------------------------------------------------------


class TestBruntVaisalaFreq:
    """Tests for brunt_vaisala_freq."""

    def test_positive_for_stable(self) -> None:
        """Brunt-Vaisala frequency is real and positive for stable stratification."""
        result = brunt_vaisala_freq(3e-3)
        assert result > 0

    def test_zero_for_neutral(self) -> None:
        """Brunt-Vaisala frequency is zero for neutral stratification."""
        result = brunt_vaisala_freq(0.0)
        np.testing.assert_allclose(result, 0.0)

    def test_custom_params(self) -> None:
        """Works with custom theta_ref and grav."""
        result = brunt_vaisala_freq(1.0, theta_ref=1.0, grav=1.0)
        np.testing.assert_allclose(result, 1.0)

    def test_array_input(self) -> None:
        """Works with array inputs."""
        dtheta_dz = np.array([3e-3, 4e-3, 5e-3])
        result = brunt_vaisala_freq(dtheta_dz)
        assert result.shape == (3,)

    def test_scales_with_sqrt(self) -> None:
        """Frequency scales with sqrt of dtheta_dz."""
        n1 = brunt_vaisala_freq(1e-3)
        n4 = brunt_vaisala_freq(4e-3)
        np.testing.assert_allclose(n4 / n1, 2.0)

    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    def test_negative_returns_nan_for_every_input_kind(self) -> None:
        """A statically unstable layer (negative dtheta_dz) has no real
        frequency and must give nan uniformly. A plain ``** 0.5`` returns a
        complex for a Python float while giving nan for a numpy scalar/array;
        the Python-float case (checked first) is what this pins.
        """
        val = -1e-3
        for x in (val, np.float64(val), np.array([val, 2 * val])):
            result = brunt_vaisala_freq(x)
            assert np.isnan(result).all()


# ---------------------------------------------------------------------------
# TestRossbyRadius
# ---------------------------------------------------------------------------


class TestRossbyRadius:
    """Tests for rossby_radius."""

    def test_positive(self) -> None:
        """Rossby radius is positive in the NH."""
        result = rossby_radius(45.0, dtheta_dz=3e-3)
        assert result > 0

    def test_larger_at_lower_lat(self) -> None:
        """Rossby radius is larger at lower latitudes (weaker f)."""
        r_low = rossby_radius(20.0, dtheta_dz=3e-3)
        r_high = rossby_radius(60.0, dtheta_dz=3e-3)
        assert r_low > r_high

    def test_larger_with_stronger_stratification(self) -> None:
        """Rossby radius increases with stronger stratification."""
        r_weak = rossby_radius(45.0, dtheta_dz=1e-3)
        r_strong = rossby_radius(45.0, dtheta_dz=4e-3)
        assert r_strong > r_weak


# ---------------------------------------------------------------------------
# TestZonalFricInferredSteady
# ---------------------------------------------------------------------------


class TestZonalFricInferredSteady:
    """Tests for zonal_fric_inferred_steady."""

    def test_zero_inputs(self) -> None:
        """Zero fluxes and wind give zero friction."""
        lats = np.linspace(-80, 80, 41)
        levs = np.array([500.0, 700.0, 850.0])
        zeros = xr.DataArray(
            np.zeros((41, 3)),
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        result = zonal_fric_inferred_steady(zeros, zeros, zeros)
        assert isinstance(result, xr.DataArray)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# TestZFromHypso
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:divide by zero encountered in log:RuntimeWarning")
class TestZFromHypso:
    """Tests for z_from_hypso.

    The top-of-atmosphere half-level pressure is 0 (dynamics.py:233), so
    np.log(p_half) produces a benign -inf at the top. z_from_hypso masks
    the resulting inf to NaN before returning, so the warning is noise.
    """

    def test_returns_dataarray(self) -> None:
        """Returns an xarray DataArray."""
        lats = np.array([0.0, 30.0, 60.0])
        levs = np.array([100000.0, 85000.0, 50000.0, 20000.0])
        temp = xr.DataArray(
            np.full((3, 4), 250.0),
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        result = z_from_hypso(temp)
        assert isinstance(result, xr.DataArray)

    def test_height_increases_with_lower_pressure(self) -> None:
        """Height increases as pressure decreases (levs ordered high-to-low)."""
        lats = np.array([0.0])
        levs = np.array([100000.0, 85000.0, 50000.0, 20000.0])
        temp = xr.DataArray(
            np.full((1, 4), 250.0),
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        result = z_from_hypso(temp)
        heights = result.isel({LAT_STR: 0}).values
        valid = heights[~np.isnan(heights)]
        # Heights should monotonically increase toward lower pressure (index 0→3)
        assert valid.size > 1
        assert np.all(np.diff(valid) >= 0)

    def test_warmer_atmosphere_gives_larger_heights(self) -> None:
        """A warmer atmosphere yields larger heights at the same pressure."""
        lats = np.array([0.0])
        levs = np.array([100000.0, 50000.0, 20000.0])
        temp_cold = xr.DataArray(
            np.full((1, 3), 200.0),
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        temp_warm = xr.DataArray(
            np.full((1, 3), 300.0),
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        z_cold = z_from_hypso(temp_cold)
        z_warm = z_from_hypso(temp_warm)
        # Compare at lowest-pressure level (highest altitude)
        idx = {LAT_STR: 0, LEV_STR: -1}
        assert z_warm.isel(idx).item() > z_cold.isel(idx).item()


# ---------------------------------------------------------------------------
# TestUBci2layerQg
# ---------------------------------------------------------------------------


class TestUBci2layerQg:
    """Tests for u_bci_2layer_qg."""

    def test_positive_nh(self) -> None:
        """Critical shear is positive in the Northern Hemisphere."""
        result = u_bci_2layer_qg(45.0)
        assert result > 0

    def test_larger_near_equator(self) -> None:
        """Critical shear is larger near the equator (weaker f)."""
        r_low = u_bci_2layer_qg(10.0)
        r_high = u_bci_2layer_qg(60.0)
        assert r_low > r_high

    def test_symmetric(self) -> None:
        """Critical shear magnitude is symmetric about the equator."""
        # cos(lat)/sin^2(lat) is positive for lat > 0, negative for lat < 0
        # but the function uses cos/sin^2, so the sign depends on cos(lat)
        r_n = u_bci_2layer_qg(45.0)
        r_s = u_bci_2layer_qg(-45.0)
        # cos(-45) = cos(45), sin(-45)^2 = sin(45)^2, so should be equal
        np.testing.assert_allclose(r_n, r_s)

    def test_array_input(self) -> None:
        """Works with array input."""
        lats = np.array([10.0, 30.0, 45.0, 60.0])
        result = u_bci_2layer_qg(lats)
        assert result.shape == (4,)

    def test_dataarray_input(self) -> None:
        """Works with DataArray input."""
        lats = _lat_dataarray([10.0, 30.0, 45.0, 60.0])
        result = u_bci_2layer_qg(lats)
        assert isinstance(result, xr.DataArray)


# ---------------------------------------------------------------------------
# TestBulkStatStab
# ---------------------------------------------------------------------------


class TestBulkStatStab:
    """Tests for bulk_stat_stab."""

    def test_zero_for_uniform(self) -> None:
        """Bulk static stability is zero for uniform potential temperature."""
        levs = np.array([200.0, 500.0, 700.0, 850.0, 1000.0])
        pot_temp = xr.DataArray(
            np.full(5, 300.0), coords={LEV_STR: levs}, dims=[LEV_STR]
        )
        result = bulk_stat_stab(pot_temp)
        np.testing.assert_allclose(result.item(), 0.0)

    def test_positive_for_stable(self) -> None:
        """Positive static stability when upper level is warmer in theta."""
        levs = np.array([200.0, 500.0, 700.0, 850.0, 1000.0])
        # Potential temperature increasing with height (decreasing pressure)
        theta_vals = np.array([350.0, 320.0, 305.0, 295.0, 290.0])
        pot_temp = xr.DataArray(theta_vals, coords={LEV_STR: levs}, dims=[LEV_STR])
        result = bulk_stat_stab(pot_temp)
        assert result.item() > 0

    def test_custom_levels(self) -> None:
        """Works with custom upper/lower levels."""
        levs = np.array([200.0, 500.0, 700.0, 850.0, 1000.0])
        theta_vals = np.array([350.0, 320.0, 305.0, 295.0, 290.0])
        pot_temp = xr.DataArray(theta_vals, coords={LEV_STR: levs}, dims=[LEV_STR])
        result = bulk_stat_stab(pot_temp, lev_upper=200, lev_lower=1000)
        expected = (350.0 - 290.0) / 300.0
        np.testing.assert_allclose(result.item(), expected)

    def test_custom_ref(self) -> None:
        """Works with custom reference potential temperature."""
        levs = np.array([500.0, 850.0])
        theta_vals = np.array([320.0, 295.0])
        pot_temp = xr.DataArray(theta_vals, coords={LEV_STR: levs}, dims=[LEV_STR])
        result = bulk_stat_stab(pot_temp, pot_temp_ref=250.0)
        expected = (320.0 - 295.0) / 250.0
        np.testing.assert_allclose(result.item(), expected)

    def test_2d_input(self) -> None:
        """Works with 2D (lat x lev) input."""
        lats = np.array([0.0, 30.0, 60.0])
        levs = np.array([200.0, 500.0, 700.0, 850.0, 1000.0])
        theta = xr.DataArray(
            np.tile([350.0, 320.0, 305.0, 295.0, 290.0], (3, 1)),
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        result = bulk_stat_stab(theta)
        assert result.shape == (3,)
