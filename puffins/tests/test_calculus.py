"""Tests for calculus module."""

import numpy as np
import pytest
import xarray as xr

from puffins.calculus import (
    _bounds_from_array,
    _check_uniform_spacing,
    _diff_bounds,
    _grid_sfc_area,
    add_lat_lon_bounds,
    flux_div,
    global_avg_grid_data,
    infer_bounds,
    lat_circumf,
    lat_circumf_weight,
    lat_deriv,
    merid_avg_grid_data,
    merid_avg_point_data,
    merid_avg_sinlat_data,
    merid_integral_grid_data,
    merid_integral_point_data,
    sfc_area_latlon_box,
    to_radians,
)
from puffins.constants import RAD_EARTH
from puffins.names import (
    BOUNDS_STR,
    LAT_BOUNDS_STR,
    LAT_STR,
    LEV_STR,
    LON_BOUNDS_STR,
    LON_STR,
    SFC_AREA_STR,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_1d_field(n: int = 181, value: float = 1.0) -> xr.DataArray:
    """1D DataArray on a uniform latitude grid."""
    lats = np.linspace(-90, 90, n)
    data = np.full(n, value)
    return xr.DataArray(data, dims=[LAT_STR], coords={LAT_STR: lats})


def _make_2d_latlev(nlat: int = 91, nlev: int = 5) -> xr.DataArray:
    """2D (lat, lev) DataArray on a uniform lat grid."""
    lats = np.linspace(-90, 90, nlat)
    levs = np.linspace(100, 1000, nlev)
    data = np.ones((nlat, nlev))
    return xr.DataArray(
        data,
        dims=[LAT_STR, LEV_STR],
        coords={LAT_STR: lats, LEV_STR: levs},
    )


def _make_latlon_field(nlat: int = 91, nlon: int = 180) -> xr.DataArray:
    """2D (lat, lon) DataArray on a uniform grid."""
    lats = np.linspace(-89, 89, nlat)
    lons = np.linspace(0, 358, nlon)
    data = np.ones((nlat, nlon))
    return xr.DataArray(
        data,
        dims=[LAT_STR, LON_STR],
        coords={LAT_STR: lats, LON_STR: lons},
        name="data",
    )


# ---------------------------------------------------------------------------
# TestCheckUniformSpacing
# ---------------------------------------------------------------------------


class TestCheckUniformSpacing:
    """Tests for _check_uniform_spacing."""

    def test_uniform_passes(self) -> None:
        """Uniform coordinate does not raise."""
        vals = np.linspace(0, 10, 11)
        coord = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        _check_uniform_spacing(coord, "x", "test_coord")

    def test_nonuniform_raises(self) -> None:
        """Non-uniform coordinate raises ValueError with name in message."""
        vals = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
        coord = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        with pytest.raises(ValueError, match="my_coord"):
            _check_uniform_spacing(coord, "x", "my_coord")

    def test_custom_tolerance(self) -> None:
        """Tight tolerance rejects, loose tolerance accepts."""
        vals = np.array([0.0, 1.0, 2.1, 3.1, 4.1])
        coord = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        with pytest.raises(ValueError):
            _check_uniform_spacing(coord, "x", "x", tol=0.01)
        # Loose tolerance should pass
        _check_uniform_spacing(coord, "x", "x", tol=0.15)


# ---------------------------------------------------------------------------
# TestInferBounds (existing)
# ---------------------------------------------------------------------------


class TestInferBounds:
    """Tests for the infer_bounds function."""

    def test_uniform_spacing(self) -> None:
        """Uniform spacing should succeed and return correct bounds."""
        vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        arr = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        result = infer_bounds(arr, "x")

        assert result.dims == ("x", BOUNDS_STR)
        assert result.shape == (5, 2)
        # For uniform spacing of 1.0, bounds should be [-0.5, 0.5], [0.5, 1.5], ...
        np.testing.assert_allclose(result.values[0], [-0.5, 0.5])
        np.testing.assert_allclose(result.values[2], [1.5, 2.5])
        np.testing.assert_allclose(result.values[4], [3.5, 4.5])

    def test_nonuniform_spacing_raises(self) -> None:
        """Non-uniform spacing should raise ValueError."""
        vals = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
        arr = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        with pytest.raises(ValueError, match="Uniform.*spacing required"):
            infer_bounds(arr, "x")

    def test_nearly_uniform_within_tolerance(self) -> None:
        """Nearly uniform spacing within tolerance should succeed."""
        vals = np.array([0.0, 1.0, 2.001, 3.001, 4.001])
        arr = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        # Default tol=0.01 should accept this
        result = infer_bounds(arr, "x")
        assert result.shape == (5, 2)

    def test_custom_tolerance(self) -> None:
        """Custom tolerance should be respected."""
        vals = np.array([0.0, 1.0, 2.1, 3.1, 4.1])
        arr = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        # Should fail with tight tolerance
        with pytest.raises(ValueError, match="Uniform.*spacing required"):
            infer_bounds(arr, "x", spacing_tol=0.01)
        # Should pass with loose tolerance
        result = infer_bounds(arr, "x", spacing_tol=0.15)
        assert result.shape == (5, 2)

    def test_identical_values_raises(self) -> None:
        """All-identical values should raise ValueError."""
        vals = np.array([5.0, 5.0, 5.0])
        arr = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        with pytest.raises(ValueError, match="all identical"):
            infer_bounds(arr, "x")

    def test_custom_dim_bounds_name(self) -> None:
        """Custom dim_bounds name should be used."""
        vals = np.linspace(0, 10, 5)
        arr = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        result = infer_bounds(arr, "x", dim_bounds="my_bounds")
        assert result.name == "my_bounds"

    def test_latitude_like_values(self) -> None:
        """Typical latitude-like values should work."""
        lats = np.arange(-90, 91, 2.0)
        arr = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
        result = infer_bounds(arr, LAT_STR)
        assert result.shape == (len(lats), 2)
        # First lower bound should be -91, last upper bound should be 91
        np.testing.assert_allclose(result.values[0, 0], -91.0)
        np.testing.assert_allclose(result.values[-1, 1], 91.0)

    def test_wrong_arr_type_raises(self) -> None:
        """Passing a non-DataArray for arr should raise."""
        with pytest.raises(AttributeError):
            infer_bounds(np.array([1.0, 2.0, 3.0]), "x")

    def test_wrong_dim_type_raises(self) -> None:
        """Passing a non-string for dim should raise."""
        vals = np.array([0.0, 1.0, 2.0])
        arr = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        with pytest.raises(TypeError):
            infer_bounds(arr, 123)

    def test_wrong_spacing_tol_type_raises(self) -> None:
        """Passing a non-numeric spacing_tol should raise."""
        vals = np.array([0.0, 1.0, 2.0])
        arr = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        with pytest.raises(TypeError):
            infer_bounds(arr, "x", spacing_tol="strict")


# ---------------------------------------------------------------------------
# TestBoundsFromArray (existing)
# ---------------------------------------------------------------------------


class TestBoundsFromArray:
    """Tests for _bounds_from_array."""

    def test_1d_uniform_spacing(self) -> None:
        """Basic 1D case with uniform spacing."""
        arr = xr.DataArray([1.0, 2.0, 3.0], dims=["x"], coords={"x": [1.0, 2.0, 3.0]})
        bounds = _bounds_from_array(arr, "x")
        # Spacing is 1.0 everywhere, so bounds should be center +/- 0.5
        lower = bounds.isel(bounds=0)
        upper = bounds.isel(bounds=1)
        np.testing.assert_allclose(lower.values, [0.5, 1.5, 2.5])
        np.testing.assert_allclose(upper.values, [1.5, 2.5, 3.5])

    def test_1d_nonuniform_spacing(self) -> None:
        """1D case with non-uniform spacing."""
        arr = xr.DataArray([0.0, 1.0, 3.0], dims=["x"], coords={"x": [0.0, 1.0, 3.0]})
        bounds = _bounds_from_array(arr, "x")
        lower = bounds.isel(bounds=0)
        upper = bounds.isel(bounds=1)
        # spacing = [1.0, 2.0]; last element reuses spacing[-1]=2.0
        np.testing.assert_allclose(lower.values, [-0.5, 0.0, 2.0])
        np.testing.assert_allclose(upper.values, [0.5, 2.0, 4.0])

    def test_2d_dim_not_axis0(self) -> None:
        """Ensure it works when target dim is not axis 0.

        This is the bug the TODO was about: the old implementation
        used raw numpy [:-1]/[-1] indexing which assumes axis=0.
        """
        # Create a 2D array where the target dim 'x' is axis=1
        data = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        arr = xr.DataArray(
            data,
            dims=["y", "x"],
            coords={"y": [0, 1], "x": [10.0, 20.0, 30.0]},
        )
        bounds = _bounds_from_array(arr, "x")
        lower = bounds.isel(bounds=0)
        upper = bounds.isel(bounds=1)
        # .T transposes, so result dims are (x, y) with shape (3, 2)
        np.testing.assert_allclose(
            lower.values,
            [[5.0, 35.0], [15.0, 45.0], [25.0, 55.0]],
        )
        np.testing.assert_allclose(
            upper.values,
            [[15.0, 45.0], [25.0, 55.0], [35.0, 65.0]],
        )

    def test_2d_dim_is_axis0(self) -> None:
        """When target dim is axis 0, should also work."""
        data = np.array([[10.0, 40.0], [20.0, 50.0], [30.0, 60.0]])
        arr = xr.DataArray(
            data,
            dims=["x", "y"],
            coords={"x": [10.0, 20.0, 30.0], "y": [0, 1]},
        )
        bounds = _bounds_from_array(arr, "x")
        lower = bounds.isel(bounds=0)
        upper = bounds.isel(bounds=1)
        # .T transposes, so result dims are (y, x) with shape (2, 3)
        np.testing.assert_allclose(
            lower.values,
            [[5.0, 15.0, 25.0], [35.0, 45.0, 55.0]],
        )
        np.testing.assert_allclose(
            upper.values,
            [[15.0, 25.0, 35.0], [45.0, 55.0, 65.0]],
        )


# ---------------------------------------------------------------------------
# TestLatDeriv
# ---------------------------------------------------------------------------


class TestLatDeriv:
    """Tests for lat_deriv."""

    def test_constant_field_zero_deriv(self) -> None:
        """Derivative of a constant field should be zero."""
        arr = _make_1d_field(181, value=5.0)
        result = lat_deriv(arr)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-10)

    def test_linear_field(self) -> None:
        """Derivative of f(lat) = lat should be approximately constant."""
        lats = np.linspace(-90, 90, 181)
        arr = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
        result = lat_deriv(arr)
        # d(lat)/d(lat_radians) = 1 / (d(lat_radians)/d(lat)) = rad2deg(1)
        expected = np.rad2deg(1.0)
        np.testing.assert_allclose(result.values, expected, rtol=1e-10)

    def test_returns_dataarray(self) -> None:
        """Result should be an xr.DataArray with the lat coordinate."""
        arr = _make_1d_field(91)
        result = lat_deriv(arr)
        assert isinstance(result, xr.DataArray)
        assert LAT_STR in result.dims

    def test_custom_lat_str(self) -> None:
        """Works with a non-default latitude dimension name."""
        lats = np.linspace(-90, 90, 91)
        arr = xr.DataArray(lats, dims=["latitude"], coords={"latitude": lats})
        result = lat_deriv(arr, lat_str="latitude")
        assert isinstance(result, xr.DataArray)


# ---------------------------------------------------------------------------
# TestFluxDiv
# ---------------------------------------------------------------------------


class TestFluxDiv:
    """Tests for flux_div."""

    def test_zero_fluxes_zero_div(self) -> None:
        """Zero input fluxes produce zero divergence."""
        arr = _make_2d_latlev()
        zero = arr * 0.0
        result = flux_div(zero, zero)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-10)

    def test_merid_flux_only(self) -> None:
        """When vertical flux is zero, result comes only from meridional."""
        arr = _make_2d_latlev()
        zero = arr * 0.0
        result = flux_div(arr, zero)
        assert isinstance(result, xr.DataArray)
        assert LAT_STR in result.dims

    def test_vert_flux_only(self) -> None:
        """When meridional flux is zero, result comes only from vertical."""
        arr = _make_2d_latlev()
        zero = arr * 0.0
        result = flux_div(zero, arr)
        assert isinstance(result, xr.DataArray)
        assert LEV_STR in result.dims

    def test_returns_dataarray(self) -> None:
        """Output is DataArray with correct dimensions."""
        arr = _make_2d_latlev()
        result = flux_div(arr, arr)
        assert isinstance(result, xr.DataArray)
        assert set(result.dims) == {LAT_STR, LEV_STR}


# ---------------------------------------------------------------------------
# TestMeridIntegralPointData
# ---------------------------------------------------------------------------


class TestMeridIntegralPointData:
    """Tests for merid_integral_point_data."""

    def test_constant_field(self) -> None:
        """Integral of constant over full sphere.

        integral of cos(lat) dlat from -pi/2 to pi/2 is 2,
        so integral of C * cos(lat) dlat = 2C.
        """
        arr = _make_1d_field(181, value=3.0)
        result = merid_integral_point_data(arr)
        np.testing.assert_allclose(float(result), 3.0 * 2.0, rtol=0.01)

    def test_symmetric_odd_function_cancels(self) -> None:
        """Integral of sin(lat) from -90 to 90 should be near zero."""
        lats = np.linspace(-90, 90, 361)
        data = np.sin(np.deg2rad(lats))
        arr = xr.DataArray(data, dims=[LAT_STR], coords={LAT_STR: lats})
        result = merid_integral_point_data(arr)
        np.testing.assert_allclose(float(result), 0.0, atol=1e-10)

    def test_cumsum_preserves_lat_dim(self) -> None:
        """do_cumsum=True returns DataArray with lat dimension."""
        arr = _make_1d_field(91)
        result = merid_integral_point_data(arr, do_cumsum=True)
        assert isinstance(result, xr.DataArray)
        assert LAT_STR in result.dims

    def test_nonuniform_spacing_raises(self) -> None:
        """Non-uniform lat spacing raises ValueError."""
        lats = np.array([-90.0, -30.0, 0.0, 60.0, 90.0])
        data = np.ones(5)
        arr = xr.DataArray(data, dims=[LAT_STR], coords={LAT_STR: lats})
        with pytest.raises(ValueError, match="Uniform latitude spacing"):
            merid_integral_point_data(arr)

    def test_hemisphere_restriction(self) -> None:
        """Restricting to NH gives positive result for positive constant."""
        arr = _make_1d_field(181, value=1.0)
        result_nh = merid_integral_point_data(arr, min_lat=0, max_lat=90)
        assert float(result_nh) > 0

    def test_returns_dataarray(self) -> None:
        """Output is a DataArray."""
        arr = _make_1d_field(181)
        result = merid_integral_point_data(arr)
        assert isinstance(result, xr.DataArray)

    def test_2d_input_preserves_lev(self) -> None:
        """2D (lat, lev) input returns result with lev dim, lat reduced."""
        arr = _make_2d_latlev(nlat=181, nlev=5)
        result = merid_integral_point_data(arr)
        assert LAT_STR not in result.dims
        assert LEV_STR in result.dims


# ---------------------------------------------------------------------------
# TestMeridAvgPointData
# ---------------------------------------------------------------------------


class TestMeridAvgPointData:
    """Tests for merid_avg_point_data."""

    def test_constant_field_returns_constant(self) -> None:
        """Average of a constant should equal that constant."""
        arr = _make_1d_field(181, value=7.0)
        result = merid_avg_point_data(arr)
        np.testing.assert_allclose(float(result), 7.0, rtol=0.01)

    def test_hemisphere_restriction(self) -> None:
        """Restricting to NH changes the result for a non-constant field."""
        lats = np.linspace(-90, 90, 181)
        data = np.cos(np.deg2rad(lats))
        arr = xr.DataArray(data, dims=[LAT_STR], coords={LAT_STR: lats})
        full = merid_avg_point_data(arr)
        nh = merid_avg_point_data(arr, min_lat=0, max_lat=90)
        # Both should give the same result for cos(lat) by symmetry
        np.testing.assert_allclose(float(full), float(nh), rtol=0.01)

    def test_cumsum_preserves_lat_dim(self) -> None:
        """do_cumsum=True returns DataArray with lat dimension."""
        arr = _make_1d_field(91)
        result = merid_avg_point_data(arr, do_cumsum=True)
        assert isinstance(result, xr.DataArray)
        assert LAT_STR in result.dims


# ---------------------------------------------------------------------------
# TestMeridIntegralGridData
# ---------------------------------------------------------------------------


class TestMeridIntegralGridData:
    """Tests for merid_integral_grid_data."""

    def test_constant_field_area(self) -> None:
        """Integral of 1 over the sphere should be total zonal-strip area.

        For a zonal strip (no longitude dimension), the integral of 1
        over all latitudes = 2 * pi * R^2 * integral(sin(lat)) from pole to pole
        = 2 * pi * R^2 * 2 = 4 * pi * R^2.
        """
        arr = _make_1d_field(181, value=1.0)
        result = merid_integral_grid_data(arr)
        expected = 4.0 * np.pi * RAD_EARTH**2
        np.testing.assert_allclose(float(result), expected, rtol=0.01)

    def test_nonuniform_lat_raises(self) -> None:
        """Non-uniform latitude spacing raises ValueError."""
        lats = np.array([-90.0, -30.0, 0.0, 60.0, 90.0])
        data = np.ones(5)
        arr = xr.DataArray(data, dims=[LAT_STR], coords={LAT_STR: lats})
        with pytest.raises(ValueError, match="Uniform latitude spacing"):
            merid_integral_grid_data(arr)

    def test_custom_radius(self) -> None:
        """Different planet radius scales the result as R^2."""
        arr = _make_1d_field(181, value=1.0)
        r1 = 1.0
        r2 = 2.0
        result1 = float(merid_integral_grid_data(arr, radius=r1))
        result2 = float(merid_integral_grid_data(arr, radius=r2))
        np.testing.assert_allclose(result2 / result1, 4.0, rtol=1e-10)

    def test_lat_bounds(self) -> None:
        """Restricting min_lat/max_lat reduces the integrated area."""
        arr = _make_1d_field(181, value=1.0)
        full = float(merid_integral_grid_data(arr))
        nh = float(merid_integral_grid_data(arr, min_lat=0, max_lat=90))
        # NH is approximately half the full integral
        np.testing.assert_allclose(nh / full, 0.5, rtol=0.02)

    def test_returns_dataarray(self) -> None:
        """Output is a DataArray."""
        arr = _make_1d_field(181)
        result = merid_integral_grid_data(arr)
        assert isinstance(result, xr.DataArray)

    def test_descending_lat_raises(self) -> None:
        """Descending latitude triggers AssertionError."""
        lats = np.linspace(90, -90, 181)
        data = np.ones(181)
        arr = xr.DataArray(data, dims=[LAT_STR], coords={LAT_STR: lats})
        with pytest.raises(AssertionError):
            merid_integral_grid_data(arr)


# ---------------------------------------------------------------------------
# TestMeridAvgGridData
# ---------------------------------------------------------------------------


class TestMeridAvgGridData:
    """Tests for merid_avg_grid_data."""

    def test_constant_field_returns_constant(self) -> None:
        """Average of a spatially constant field returns that constant."""
        arr = _make_1d_field(181, value=3.5)
        result = merid_avg_grid_data(arr)
        np.testing.assert_allclose(float(result), 3.5, rtol=0.01)

    def test_preserves_other_dims(self) -> None:
        """A 2D (lat, lev) input returns a 1D (lev) result."""
        arr = _make_2d_latlev(nlat=181, nlev=5)
        result = merid_avg_grid_data(arr)
        assert LAT_STR not in result.dims
        assert LEV_STR in result.dims

    def test_hemisphere_restriction(self) -> None:
        """min_lat/max_lat restrict the averaging domain."""
        arr = _make_1d_field(181, value=1.0)
        result = merid_avg_grid_data(arr, min_lat=0, max_lat=90)
        np.testing.assert_allclose(float(result), 1.0, rtol=0.01)


# ---------------------------------------------------------------------------
# TestGlobalAvgGridData
# ---------------------------------------------------------------------------


class TestGlobalAvgGridData:
    """Tests for global_avg_grid_data."""

    def test_constant_field_without_sfc_area(self) -> None:
        """Average of a constant 2D lat-lon field equals that constant."""
        arr = _make_latlon_field(nlat=91, nlon=180) * 4.0
        result = global_avg_grid_data(arr)
        np.testing.assert_allclose(float(result), 4.0, rtol=0.01)

    def test_constant_field_with_sfc_area(self) -> None:
        """When sfc_area coordinate is present, uses it for weighting."""
        arr = _make_latlon_field(nlat=91, nlon=180) * 2.0
        # Attach a uniform surface area coordinate
        sfc_area = xr.ones_like(arr)
        arr.coords[SFC_AREA_STR] = sfc_area
        result = global_avg_grid_data(arr)
        np.testing.assert_allclose(float(result), 2.0, rtol=0.01)

    def test_returns_scalar_like(self) -> None:
        """Result has no lat/lon dims."""
        arr = _make_latlon_field()
        result = global_avg_grid_data(arr)
        assert LAT_STR not in result.dims
        assert LON_STR not in result.dims


# ---------------------------------------------------------------------------
# TestMeridAvgSinlatData
# ---------------------------------------------------------------------------


class TestMeridAvgSinlatData:
    """Tests for merid_avg_sinlat_data."""

    def test_constant_field(self) -> None:
        """Average of a constant is that constant."""
        # Create data on uniform sin(lat) spacing
        n = 100
        sinlats = np.linspace(-1, 1, n)
        lats = np.rad2deg(np.arcsin(sinlats))
        data = np.full(n, 5.0)
        arr = xr.DataArray(data, dims=[LAT_STR], coords={LAT_STR: lats})
        result = merid_avg_sinlat_data(arr)
        np.testing.assert_allclose(float(result), 5.0, rtol=0.01)

    def test_nonuniform_sinlat_raises(self) -> None:
        """Data not uniform in sin(lat) raises ValueError."""
        # Regular lat spacing is NOT uniform in sin(lat) for enough points
        lats = np.linspace(-90, 90, 181)
        data = np.ones(181)
        arr = xr.DataArray(data, dims=[LAT_STR], coords={LAT_STR: lats})
        with pytest.raises(ValueError, match="Uniform sin"):
            merid_avg_sinlat_data(arr)

    def test_custom_sinlat_param(self) -> None:
        """Passing explicit sinlat coordinate works."""
        n = 100
        sinlats = np.linspace(-1, 1, n)
        lats = np.rad2deg(np.arcsin(sinlats))
        data = np.full(n, 3.0)
        arr = xr.DataArray(data, dims=[LAT_STR], coords={LAT_STR: lats})
        sinlat_da = xr.DataArray(sinlats, dims=[LAT_STR], coords={LAT_STR: lats})
        result = merid_avg_sinlat_data(arr, sinlat=sinlat_da)
        np.testing.assert_allclose(float(result), 3.0, rtol=0.01)

    def test_lat_bounds(self) -> None:
        """Restricting to a sub-range works."""
        n = 100
        sinlats = np.linspace(-1, 1, n)
        lats = np.rad2deg(np.arcsin(sinlats))
        data = np.full(n, 2.0)
        arr = xr.DataArray(data, dims=[LAT_STR], coords={LAT_STR: lats})
        result = merid_avg_sinlat_data(arr, min_lat=0, max_lat=90)
        np.testing.assert_allclose(float(result), 2.0, rtol=0.01)


# ---------------------------------------------------------------------------
# TestAddLatLonBounds
# ---------------------------------------------------------------------------


class TestAddLatLonBounds:
    """Tests for add_lat_lon_bounds."""

    def test_dataarray_input_returns_dataset(self) -> None:
        """Passing a DataArray returns a Dataset with bounds added."""
        arr = _make_latlon_field(nlat=10, nlon=20)
        result = add_lat_lon_bounds(arr)
        assert isinstance(result, xr.Dataset)

    def test_dataset_input(self) -> None:
        """Passing a Dataset also works."""
        arr = _make_latlon_field(nlat=10, nlon=20)
        ds = arr.to_dataset(name="data")
        result = add_lat_lon_bounds(ds)
        assert isinstance(result, xr.Dataset)

    def test_bounds_added_as_coords(self) -> None:
        """The returned Dataset has lat_bounds and lon_bounds."""
        arr = _make_latlon_field(nlat=10, nlon=20)
        result = add_lat_lon_bounds(arr)
        assert LAT_BOUNDS_STR in result.coords
        assert LON_BOUNDS_STR in result.coords

    def test_bounds_shape(self) -> None:
        """Bounds should have shape (n, 2)."""
        arr = _make_latlon_field(nlat=10, nlon=20)
        result = add_lat_lon_bounds(arr)
        assert result.coords[LAT_BOUNDS_STR].shape == (10, 2)
        assert result.coords[LON_BOUNDS_STR].shape == (20, 2)


# ---------------------------------------------------------------------------
# TestToRadians
# ---------------------------------------------------------------------------


class TestToRadians:
    """Tests for to_radians."""

    def test_large_values_assumed_degrees(self) -> None:
        """Values > 4*pi are assumed degrees and converted."""
        arr = np.array([180.0])
        result = to_radians(arr)
        np.testing.assert_allclose(result, np.array([np.pi]))

    def test_small_values_assumed_radians(self) -> None:
        """Values < 4*pi are returned unchanged."""
        arr = np.array([1.0])
        result = to_radians(arr)
        np.testing.assert_allclose(result, np.array([1.0]))

    def test_is_delta_threshold(self) -> None:
        """With is_delta=True, the threshold is 0.1*pi."""
        # A value just above 0.1*pi should be converted when is_delta=True
        val = 0.35  # > 0.1*pi ~ 0.314
        arr = np.array([val])
        result = to_radians(arr, is_delta=True)
        np.testing.assert_allclose(result, np.deg2rad(arr))

    def test_is_delta_below_threshold(self) -> None:
        """With is_delta=True, values below 0.1*pi are unchanged."""
        val = 0.2  # < 0.1*pi ~ 0.314
        arr = np.array([val])
        result = to_radians(arr, is_delta=True)
        np.testing.assert_allclose(result, arr)

    def test_numpy_array(self) -> None:
        """Works on plain numpy arrays."""
        arr = np.array([90.0, 180.0, 270.0])
        result = to_radians(arr)
        np.testing.assert_allclose(result, np.deg2rad(arr))

    def test_dataarray(self) -> None:
        """Works on xr.DataArray."""
        arr = xr.DataArray([90.0, 180.0])
        result = to_radians(arr)
        expected = np.deg2rad(np.array([90.0, 180.0]))
        np.testing.assert_allclose(result.values, expected)

    def test_units_attr_degrees(self) -> None:
        """DataArray with units='degrees_north' gets converted."""
        arr = xr.DataArray([90.0, 180.0], attrs={"units": "degrees_north"})
        result = to_radians(arr)
        np.testing.assert_allclose(result.values, np.deg2rad([90.0, 180.0]))

    def test_units_attr_radians(self) -> None:
        """DataArray with units='radians' is not converted via units path.

        Falls through to the magnitude heuristic; small values pass through.
        """
        arr = xr.DataArray([1.0, 2.0], attrs={"units": "radians"})
        result = to_radians(arr)
        np.testing.assert_allclose(result.values, [1.0, 2.0])

    def test_scalar_float(self) -> None:
        """Bare Python float is converted correctly."""
        result = to_radians(180.0)
        np.testing.assert_allclose(result, np.pi)


# ---------------------------------------------------------------------------
# TestDiffBounds
# ---------------------------------------------------------------------------


class TestDiffBounds:
    """Tests for _diff_bounds."""

    def test_2d_bounds(self) -> None:
        """DataArray bounds with shape (n, 2), primary path."""
        vals = np.array([1.0, 2.0, 3.0])
        coord = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        bounds = xr.DataArray(
            [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]],
            dims=["x", BOUNDS_STR],
            coords={"x": vals},
        )
        result = _diff_bounds(bounds, coord)
        np.testing.assert_allclose(result.values, [1.0, 1.0, 1.0])

    def test_returns_dataarray(self) -> None:
        """Result is a DataArray."""
        vals = np.array([1.0, 2.0])
        coord = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        bounds = xr.DataArray(
            [[0.5, 1.5], [1.5, 2.5]],
            dims=["x", BOUNDS_STR],
            coords={"x": vals},
        )
        result = _diff_bounds(bounds, coord)
        assert isinstance(result, xr.DataArray)

    def test_1d_fallback_path(self) -> None:
        """1D bounds triggers the except IndexError fallback."""
        vals = np.array([1.0, 2.0, 3.0])
        coord = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        # 1D bounds array: upper bounds only, no second axis
        bounds_1d = xr.DataArray([1.5, 2.5, 3.5, 4.5], dims=["x_bounds"])
        result = _diff_bounds(bounds_1d, coord)
        assert isinstance(result, xr.DataArray)
        np.testing.assert_allclose(result.values, [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# TestSfcAreaLatlonBox
# ---------------------------------------------------------------------------


class TestSfcAreaLatlonBox:
    """Tests for sfc_area_latlon_box."""

    def _make_ds_with_bounds(self, nlat: int = 91, nlon: int = 180) -> xr.Dataset:
        """Create a Dataset with lat/lon and their bounds."""
        arr = _make_latlon_field(nlat=nlat, nlon=nlon)
        return add_lat_lon_bounds(arr)

    def test_total_area_sphere(self) -> None:
        """Sum of all grid cell areas should approximate 4*pi*R^2."""
        ds = self._make_ds_with_bounds(nlat=90, nlon=180)
        areas = sfc_area_latlon_box(ds)
        total = float(areas.sum())
        expected = 4.0 * np.pi * RAD_EARTH**2
        np.testing.assert_allclose(total, expected, rtol=0.01)

    def test_cell_areas_positive(self) -> None:
        """All cell areas should be positive."""
        ds = self._make_ds_with_bounds(nlat=45, nlon=90)
        areas = sfc_area_latlon_box(ds)
        assert (areas > 0).all()

    def test_custom_radius(self) -> None:
        """Using a different radius scales areas as R^2."""
        ds = self._make_ds_with_bounds(nlat=45, nlon=90)
        areas1 = sfc_area_latlon_box(ds, radius=1.0)
        areas2 = sfc_area_latlon_box(ds, radius=2.0)
        np.testing.assert_allclose(
            float(areas2.sum()) / float(areas1.sum()), 4.0, rtol=1e-10
        )

    def test_returns_dataarray(self) -> None:
        """Output is a DataArray."""
        ds = self._make_ds_with_bounds(nlat=10, nlon=20)
        result = sfc_area_latlon_box(ds)
        assert isinstance(result, xr.DataArray)


# ---------------------------------------------------------------------------
# TestGridSfcArea
# ---------------------------------------------------------------------------


class TestGridSfcArea:
    """Tests for _grid_sfc_area."""

    def test_without_bounds(self) -> None:
        """When bounds are None, they are inferred."""
        lats = np.linspace(-89, 89, 45)
        lons = np.linspace(1, 359, 90)
        lon = xr.DataArray(lons, dims=[LON_STR], coords={LON_STR: lons})
        lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
        result = _grid_sfc_area(lon, lat)
        assert isinstance(result, xr.DataArray)
        assert (result > 0).all()

    def test_with_bounds(self) -> None:
        """When bounds are provided, they are used directly."""
        lats = np.linspace(-89, 89, 10)
        lons = np.linspace(1, 359, 20)
        lon = xr.DataArray(lons, dims=[LON_STR], coords={LON_STR: lons})
        lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
        lon_bounds = _bounds_from_array(lon, LON_STR, LON_BOUNDS_STR)
        lat_bounds = _bounds_from_array(lat, LAT_STR, LAT_BOUNDS_STR)
        result = _grid_sfc_area(lon, lat, lon_bounds=lon_bounds, lat_bounds=lat_bounds)
        assert isinstance(result, xr.DataArray)
        assert (result > 0).all()


# ---------------------------------------------------------------------------
# TestLatCircumf
# ---------------------------------------------------------------------------


class TestLatCircumf:
    """Tests for lat_circumf."""

    def test_equator(self) -> None:
        """Circumference at equator should be 2*pi*R."""
        result = lat_circumf(0.0)
        np.testing.assert_allclose(float(result), 2 * np.pi * RAD_EARTH, rtol=1e-10)

    def test_poles(self) -> None:
        """Circumference at +/- 90 should be approximately zero."""
        for pole in [90.0, -90.0]:
            result = lat_circumf(pole)
            np.testing.assert_allclose(float(result), 0.0, atol=1e-5)

    def test_scalar_and_array(self) -> None:
        """Works with both scalar and DataArray inputs."""
        scalar_result = lat_circumf(0.0)
        arr = xr.DataArray([0.0], dims=[LAT_STR], coords={LAT_STR: [0.0]})
        array_result = lat_circumf(arr)
        np.testing.assert_allclose(float(scalar_result), float(array_result.item()))

    def test_custom_radius(self) -> None:
        """Custom radius is used."""
        result = lat_circumf(0.0, radius=1.0)
        np.testing.assert_allclose(float(result), 2 * np.pi, rtol=1e-10)


# ---------------------------------------------------------------------------
# TestLatCircumfWeight
# ---------------------------------------------------------------------------


class TestLatCircumfWeight:
    """Tests for lat_circumf_weight."""

    def test_equator_weight(self) -> None:
        """At equator, weight equals full circumference times value."""
        lats = np.array([0.0])
        arr = xr.DataArray([1.0], dims=[LAT_STR], coords={LAT_STR: lats})
        result = lat_circumf_weight(arr)
        expected = 2 * np.pi * RAD_EARTH
        np.testing.assert_allclose(result.item(), expected, rtol=1e-10)

    def test_explicit_lat(self) -> None:
        """Passing lat parameter explicitly works."""
        lats = np.array([0.0, 30.0, 60.0])
        arr = xr.DataArray([1.0, 1.0, 1.0], dims=[LAT_STR], coords={LAT_STR: lats})
        lat_da = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
        result = lat_circumf_weight(arr, lat=lat_da)
        assert isinstance(result, xr.DataArray)
        # Equator value should be largest
        assert float(result.isel({LAT_STR: 0})) > float(result.isel({LAT_STR: 2}))

    def test_returns_dataarray(self) -> None:
        """Output is DataArray with same dims."""
        arr = _make_1d_field(91)
        result = lat_circumf_weight(arr)
        assert isinstance(result, xr.DataArray)
        assert LAT_STR in result.dims
