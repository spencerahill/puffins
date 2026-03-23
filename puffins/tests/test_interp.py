"""Tests for interp module."""

import numpy as np
import pytest
import xarray as xr

from puffins.interp import (
    _maybe_interp_in_p,
    drop_all_nan_slices,
    drop_nans_and_interp,
    interp_arrs_in_p,
    interp_ds_p_to_p,
    interp_ds_sigma_to_p,
    interp_eta_to_plevs,
    interp_nested_to_plevs,
    interp_p,
    interpolate,
    zero_cross_bounds,
    zero_cross_interp,
)
from puffins.names import (
    BK_STR,
    LAT_STR,
    LEV_STR,
    P_SFC_STR,
    PFULL_STR,
    PHALF_STR,
    PK_STR,
    SIGMA_STR,
    TIME_STR,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_1d_pressure_field(nlev: int = 10, value: float = 1.0) -> xr.DataArray:
    """1D DataArray on a pressure (LEV_STR) coordinate."""
    levs = np.linspace(100, 1000, nlev)
    data = np.full(nlev, value)
    return xr.DataArray(data, dims=[LEV_STR], coords={LEV_STR: levs}, name="field")


def _make_2d_latlev(nlat: int = 9, nlev: int = 10) -> xr.DataArray:
    """2D (lat, lev) DataArray on uniform lat and pressure grids."""
    lats = np.linspace(-90, 90, nlat)
    levs = np.linspace(100, 1000, nlev)
    data = np.ones((nlat, nlev))
    return xr.DataArray(
        data,
        dims=[LAT_STR, LEV_STR],
        coords={LAT_STR: lats, LEV_STR: levs},
        name="field",
    )


def _make_sigma_dataset(nlat: int = 5, nsigma: int = 8) -> xr.Dataset:
    """Dataset with sigma coordinates and surface pressure."""
    lats = np.linspace(-60, 60, nlat)
    sigma = np.linspace(0.1, 0.95, nsigma)
    p_sfc_vals = np.linspace(95000, 101000, nlat)

    p_sfc = xr.DataArray(
        p_sfc_vals, dims=[LAT_STR], coords={LAT_STR: lats}, name=P_SFC_STR
    )
    temp_data = 250.0 + 30.0 * np.outer(np.ones(nlat), sigma)
    temp = xr.DataArray(
        temp_data,
        dims=[LAT_STR, SIGMA_STR],
        coords={LAT_STR: lats, SIGMA_STR: sigma},
        name="temp",
    )
    return xr.Dataset({P_SFC_STR: p_sfc, "temp": temp})


def _make_eta_dataset(nlat: int = 5, nlev: int = 6) -> xr.Dataset:
    """Dataset with eta (hybrid sigma-pressure) coordinates.

    Uses a pure-sigma setup: bk linearly spaced from 0 to 1, pk = 0.
    Half-levels have nlev+1 points; full-levels have nlev points.
    """
    lats = np.linspace(-60, 60, nlat)

    # Half-level coefficients (nlev + 1 points).
    nhalf = nlev + 1
    bk_vals = np.linspace(0, 1, nhalf)
    pk_vals = np.zeros(nhalf)
    phalf_coord = np.arange(nhalf)

    bk = xr.DataArray(
        bk_vals, dims=[PHALF_STR], coords={PHALF_STR: phalf_coord}, name=BK_STR
    )
    pk = xr.DataArray(
        pk_vals, dims=[PHALF_STR], coords={PHALF_STR: phalf_coord}, name=PK_STR
    )

    # Reference full-level pressures (midpoints of half-level coord).
    pfull_vals = 0.5 * (phalf_coord[:-1] + phalf_coord[1:])
    pfull_ref = xr.DataArray(
        pfull_vals,
        dims=[PFULL_STR],
        coords={PFULL_STR: pfull_vals},
        name=PFULL_STR,
    )

    # Surface pressure per latitude (Pa).
    ps_vals = np.linspace(95000, 101000, nlat)
    ps = xr.DataArray(ps_vals, dims=[LAT_STR], coords={LAT_STR: lats}, name="ps")

    # Data variable on (lat, pfull).
    temp_data = 250.0 + 20.0 * np.outer(np.ones(nlat), pfull_vals / pfull_vals.max())
    temp = xr.DataArray(
        temp_data,
        dims=[LAT_STR, PFULL_STR],
        coords={LAT_STR: lats, PFULL_STR: pfull_vals},
        name="temp",
    )

    return xr.Dataset(
        {"temp": temp, BK_STR: bk, PK_STR: pk, "ps": ps, PFULL_STR: pfull_ref}
    )


# ---------------------------------------------------------------------------
# TestDropAllNanSlices
# ---------------------------------------------------------------------------


class TestDropAllNanSlices:
    """Tests for drop_all_nan_slices."""

    def test_no_nan_passthrough(self) -> None:
        """Arrays without NaNs pass through unchanged."""
        arr = _make_2d_latlev(nlat=5, nlev=4)
        result = drop_all_nan_slices([arr], dim=LAT_STR)
        assert len(result) == 1
        xr.testing.assert_equal(result[0].unstack(), arr)

    def test_all_nan_slice_dropped(self) -> None:
        """Latitude slice that is all-NaN gets dropped."""
        arr = _make_2d_latlev(nlat=5, nlev=4)
        arr.loc[{LAT_STR: arr[LAT_STR][2]}] = np.nan
        result = drop_all_nan_slices([arr], dim=LAT_STR)
        assert result[0].sizes[LAT_STR] == 4

    def test_partial_nan_kept(self) -> None:
        """Latitude slice with only some NaNs is retained."""
        arr = _make_2d_latlev(nlat=5, nlev=4)
        arr.values[2, 0] = np.nan  # only one value NaN
        result = drop_all_nan_slices([arr], dim=LAT_STR)
        assert result[0].sizes[LAT_STR] == 5

    def test_multiple_arrays(self) -> None:
        """All arrays in the list are processed."""
        arr1 = _make_2d_latlev(nlat=5, nlev=4)
        arr2 = _make_2d_latlev(nlat=5, nlev=4)
        arr1.loc[{LAT_STR: arr1[LAT_STR][0]}] = np.nan
        result = drop_all_nan_slices([arr1, arr2], dim=LAT_STR)
        assert len(result) == 2

    def test_custom_dim(self) -> None:
        """Works with a non-default dimension."""
        arr = _make_2d_latlev(nlat=5, nlev=4)
        arr.loc[{LEV_STR: arr[LEV_STR][1]}] = np.nan
        result = drop_all_nan_slices([arr], dim=LEV_STR)
        assert result[0].sizes[LEV_STR] == 3


# ---------------------------------------------------------------------------
# TestInterpP
# ---------------------------------------------------------------------------


class TestInterpP:
    """Tests for interp_p."""

    def test_same_grid_returns_same_values(self) -> None:
        """Interpolating to same pressure values recovers original data."""
        arr = _make_1d_pressure_field(nlev=10, value=3.0)
        result = interp_p(arr, new_p_vals=arr[LEV_STR].values)
        xr.testing.assert_allclose(result, arr)

    def test_linear_function(self) -> None:
        """Linear function of pressure is recovered exactly."""
        levs = np.linspace(100, 1000, 10)
        data = 2.0 * levs + 5.0
        arr = xr.DataArray(data, dims=[LEV_STR], coords={LEV_STR: levs}, name="field")
        new_levs = np.array([200, 500, 800])
        result = interp_p(arr, new_p_vals=new_levs)
        expected = 2.0 * new_levs + 5.0
        np.testing.assert_allclose(result.values, expected, atol=1e-10)

    def test_returns_dataarray(self) -> None:
        """Result is an xarray DataArray."""
        arr = _make_1d_pressure_field()
        result = interp_p(arr, new_p_vals=np.array([200, 500]))
        assert isinstance(result, xr.DataArray)

    def test_custom_method(self) -> None:
        """Non-default interpolation method does not raise."""
        arr = _make_1d_pressure_field(nlev=10)
        result = interp_p(arr, new_p_vals=np.array([200, 500]), method="nearest")
        assert result.sizes[LEV_STR] == 2


# ---------------------------------------------------------------------------
# TestInterpArrsInP
# ---------------------------------------------------------------------------


class TestInterpArrsInP:
    """Tests for interp_arrs_in_p."""

    def test_returns_list_of_dataarrays(self) -> None:
        """Each element of the result is a DataArray."""
        arr = _make_2d_latlev(nlat=5, nlev=10)
        new_levs = np.linspace(200, 900, 5)
        result = interp_arrs_in_p([arr], new_p_vals=new_levs)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], xr.DataArray)

    def test_multiple_arrays(self) -> None:
        """All input arrays are interpolated."""
        arr1 = _make_2d_latlev(nlat=5, nlev=10)
        arr2 = _make_2d_latlev(nlat=5, nlev=10) * 2
        new_levs = np.linspace(200, 900, 5)
        result = interp_arrs_in_p([arr1, arr2], new_p_vals=new_levs)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# TestMaybeInterpInP
# ---------------------------------------------------------------------------


class TestMaybeInterpInP:
    """Tests for _maybe_interp_in_p."""

    def test_do_interp_true(self) -> None:
        """When do_interp is True, output has the new pressure levels."""
        arr = _make_2d_latlev(nlat=5, nlev=10)
        new_levs = np.linspace(200, 900, 4)
        result = _maybe_interp_in_p([arr], new_p_vals=new_levs, do_interp=True)
        assert result[0].sizes[LEV_STR] == 4

    def test_do_interp_false(self) -> None:
        """When do_interp is False, originals are returned unchanged."""
        arr = _make_2d_latlev(nlat=5, nlev=10)
        result = _maybe_interp_in_p([arr], do_interp=False)
        assert result[0] is arr


# ---------------------------------------------------------------------------
# TestDropNansAndInterp
# ---------------------------------------------------------------------------


class TestDropNansAndInterp:
    """Tests for drop_nans_and_interp."""

    def test_combined_drop_and_interp(self) -> None:
        """NaN slices are dropped and interpolation occurs."""
        arr = _make_2d_latlev(nlat=5, nlev=10)
        arr.loc[{LAT_STR: arr[LAT_STR][0]}] = np.nan
        new_levs = np.linspace(200, 900, 4)
        result = drop_nans_and_interp([arr], new_p_vals=new_levs, do_interp=True)
        assert result[0].sizes[LAT_STR] == 4

    def test_no_interp_mode(self) -> None:
        """With do_interp=False, only NaN slices are dropped."""
        arr = _make_2d_latlev(nlat=5, nlev=10)
        arr.loc[{LAT_STR: arr[LAT_STR][0]}] = np.nan
        result = drop_nans_and_interp([arr], do_interp=False)
        assert result[0].sizes[LAT_STR] == 4


# ---------------------------------------------------------------------------
# TestInterpolate
# ---------------------------------------------------------------------------


class TestInterpolate:
    """Tests for interpolate (linear two-point interpolation)."""

    def test_linear_exactness(self) -> None:
        """Midpoint of [0, 1] -> [0, 10] gives 5.0."""
        dim = "x"
        x = xr.DataArray([0.0, 1.0], dims=[dim], coords={dim: [0.0, 1.0]})
        y = xr.DataArray([0.0, 10.0], dims=[dim], coords={dim: [0.0, 10.0]})
        result = interpolate(x, y, 0.5, dim)
        np.testing.assert_allclose(result.item(), 5.0)

    def test_endpoint_value(self) -> None:
        """Interpolating at the first x value returns the first y value."""
        dim = "x"
        x = xr.DataArray([0.0, 1.0], dims=[dim], coords={dim: [0.0, 1.0]})
        y = xr.DataArray([3.0, 7.0], dims=[dim], coords={dim: [3.0, 7.0]})
        result = interpolate(x, y, 0.0, dim)
        np.testing.assert_allclose(result.item(), 3.0)

    def test_returns_dataarray(self) -> None:
        """Result is an xarray DataArray."""
        dim = "x"
        x = xr.DataArray([0.0, 1.0], dims=[dim], coords={dim: [0.0, 1.0]})
        y = xr.DataArray([0.0, 10.0], dims=[dim], coords={dim: [0.0, 10.0]})
        result = interpolate(x, y, 0.5, dim)
        assert isinstance(result, xr.DataArray)


# ---------------------------------------------------------------------------
# TestInterpDsSigmaToP
# ---------------------------------------------------------------------------


class TestInterpDsSigmaToP:
    """Tests for interp_ds_sigma_to_p."""

    def test_int_plevs_level_count(self) -> None:
        """Integer plevs argument creates correct number of output levels."""
        ds = _make_sigma_dataset(nlat=5, nsigma=8)
        nplevs = 6
        result = interp_ds_sigma_to_p(ds, nplevs, method="linear")
        assert result.sizes[LEV_STR] == nplevs

    def test_array_plevs(self) -> None:
        """Array plevs are used as output levels."""
        ds = _make_sigma_dataset(nlat=5, nsigma=8)
        plevs = np.array([20000, 50000, 80000])
        result = interp_ds_sigma_to_p(ds, plevs, method="linear")
        assert result.sizes[LEV_STR] == 3

    def test_output_dimensions(self) -> None:
        """Output Dataset has (lev, lat) dimensions for data vars."""
        ds = _make_sigma_dataset(nlat=5, nsigma=8)
        result = interp_ds_sigma_to_p(ds, 6, method="linear")
        assert set(result["temp"].dims) == {LEV_STR, LAT_STR}

    def test_returns_dataset(self) -> None:
        """Result is an xarray Dataset."""
        ds = _make_sigma_dataset(nlat=5, nsigma=8)
        result = interp_ds_sigma_to_p(ds, 4, method="linear")
        assert isinstance(result, xr.Dataset)


# ---------------------------------------------------------------------------
# TestInterpDsPToP
# ---------------------------------------------------------------------------


class TestInterpDsPToP:
    """Tests for interp_ds_p_to_p."""

    def test_constant_field_stays_constant(self) -> None:
        """A spatially constant field remains constant after interpolation."""
        nlat, nlev = 5, 8
        lats = np.linspace(-60, 60, nlat)
        # Varying pfull per latitude
        pfull_vals = np.linspace(200, 900, nlev)
        pfull = xr.DataArray(
            pfull_vals,
            dims=[PFULL_STR],
            coords={PFULL_STR: pfull_vals},
        )
        temp = xr.DataArray(
            300.0 * np.ones((nlat, nlev)),
            dims=[LAT_STR, PFULL_STR],
            coords={LAT_STR: lats, PFULL_STR: pfull_vals},
            name="temp",
        )
        ds = xr.Dataset({"temp": temp})
        plevs = np.array([300, 500, 700])
        result = interp_ds_p_to_p(ds, plevs, method="linear")
        np.testing.assert_allclose(result["temp"].values, 300.0, atol=1e-10)

    def test_output_dimensions(self) -> None:
        """Output has (lev, lat) dimensions."""
        nlat, nlev = 5, 8
        lats = np.linspace(-60, 60, nlat)
        pfull_vals = np.linspace(200, 900, nlev)
        temp = xr.DataArray(
            np.ones((nlat, nlev)),
            dims=[LAT_STR, PFULL_STR],
            coords={LAT_STR: lats, PFULL_STR: pfull_vals},
            name="temp",
        )
        ds = xr.Dataset({"temp": temp})
        plevs = np.array([300, 500, 700])
        result = interp_ds_p_to_p(ds, plevs, method="linear")
        assert set(result["temp"].dims) == {LEV_STR, LAT_STR}

    def test_returns_dataset(self) -> None:
        """Result is an xarray Dataset."""
        nlat, nlev = 5, 8
        lats = np.linspace(-60, 60, nlat)
        pfull_vals = np.linspace(200, 900, nlev)
        temp = xr.DataArray(
            np.ones((nlat, nlev)),
            dims=[LAT_STR, PFULL_STR],
            coords={LAT_STR: lats, PFULL_STR: pfull_vals},
            name="temp",
        )
        ds = xr.Dataset({"temp": temp})
        plevs = np.array([300, 500, 700])
        result = interp_ds_p_to_p(ds, plevs, method="linear")
        assert isinstance(result, xr.Dataset)


# ---------------------------------------------------------------------------
# TestZeroCrossBounds
# ---------------------------------------------------------------------------


class TestZeroCrossBounds:
    """Tests for zero_cross_bounds."""

    def test_finds_crossing(self) -> None:
        """Returns the two values bounding the zero crossing."""
        dim = "x"
        vals = np.array([-2.0, -1.0, 1.0, 2.0])
        coords = np.arange(4.0)
        arr = xr.DataArray(vals, dims=[dim], coords={dim: coords})
        result = zero_cross_bounds(arr, dim)
        assert result.sizes[dim] == 2
        assert float(result.values[0]) == -1.0
        assert float(result.values[1]) == 1.0

    def test_no_crossing_raises(self) -> None:
        """ValueError when array has no zero crossing."""
        dim = "x"
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        coords = np.arange(4.0)
        arr = xr.DataArray(vals, dims=[dim], coords={dim: coords})
        with pytest.raises(ValueError, match="zero crossings"):
            zero_cross_bounds(arr, dim)

    def test_num_cross_selects_crossing(self) -> None:
        """num_cross selects the correct crossing when there are multiple."""
        dim = "x"
        vals = np.array([-1.0, 1.0, -1.0, 1.0])
        coords = np.arange(4.0)
        arr = xr.DataArray(vals, dims=[dim], coords={dim: coords})
        # First crossing between index 0 and 1.
        result0 = zero_cross_bounds(arr, dim, num_cross=0)
        assert float(result0.values[0]) == -1.0
        assert float(result0.values[1]) == 1.0
        # Second crossing between index 1 and 2.
        result1 = zero_cross_bounds(arr, dim, num_cross=1)
        assert float(result1.values[0]) == 1.0
        assert float(result1.values[1]) == -1.0

    def test_returns_2_element_dataarray(self) -> None:
        """Result is a DataArray with 2 elements."""
        dim = "x"
        vals = np.array([-1.0, 1.0, 2.0])
        coords = np.arange(3.0)
        arr = xr.DataArray(vals, dims=[dim], coords={dim: coords})
        result = zero_cross_bounds(arr, dim)
        assert isinstance(result, xr.DataArray)
        assert result.sizes[dim] == 2


# ---------------------------------------------------------------------------
# TestZeroCrossInterp
# ---------------------------------------------------------------------------


class TestZeroCrossInterp:
    """Tests for zero_cross_interp."""

    def test_linear_crossing_exact(self) -> None:
        """Linear function crossing zero gives the exact root."""
        dim = "x"
        coords = np.array([0.0, 1.0, 2.0, 3.0])
        # f(x) = x - 1.5 crosses zero at x = 1.5
        vals = coords - 1.5
        arr = xr.DataArray(vals, dims=[dim], coords={dim: coords})
        result = zero_cross_interp(arr, dim)
        np.testing.assert_allclose(result.item(), 1.5, atol=1e-10)

    def test_no_crossing_raises(self) -> None:
        """ValueError when array has no zero crossing."""
        dim = "x"
        vals = np.array([1.0, 2.0, 3.0])
        coords = np.arange(3.0)
        arr = xr.DataArray(vals, dims=[dim], coords={dim: coords})
        with pytest.raises(ValueError, match="zero crossings"):
            zero_cross_interp(arr, dim)


# ---------------------------------------------------------------------------
# TestInterpEtaToPlevs
# ---------------------------------------------------------------------------


class TestInterpEtaToPlevs:
    """Tests for interp_eta_to_plevs."""

    def test_output_dimensions(self) -> None:
        """Output Dataset has the correct dimensions."""
        ds = _make_eta_dataset(nlat=5, nlev=6)
        plevs = np.array([200, 400, 600, 800])
        result = interp_eta_to_plevs(ds, plevs, dim=LAT_STR, method="linear")
        assert PFULL_STR in result["temp"].dims
        assert LAT_STR in result["temp"].dims
        assert result["temp"].sizes[PFULL_STR] == len(plevs)
        assert result["temp"].sizes[LAT_STR] == 5

    def test_returns_dataset(self) -> None:
        """Result is an xarray Dataset."""
        ds = _make_eta_dataset(nlat=5, nlev=6)
        plevs = np.array([200, 400, 600, 800])
        result = interp_eta_to_plevs(ds, plevs, dim=LAT_STR, method="linear")
        assert isinstance(result, xr.Dataset)


# ---------------------------------------------------------------------------
# TestInterpNestedToPlevs
# ---------------------------------------------------------------------------


class TestInterpNestedToPlevs:
    """Tests for interp_nested_to_plevs."""

    def test_output_has_both_extra_dims(self) -> None:
        """Output Dataset has both the inner and outer loop dimensions."""
        nlat, nlev, ntime = 3, 6, 2
        ds_base = _make_eta_dataset(nlat=nlat, nlev=nlev)
        # Add a time dimension by concatenating.
        datasets = []
        for t in range(ntime):
            ds_t = ds_base.copy(deep=True)
            ds_t = ds_t.expand_dims({TIME_STR: [t]})
            datasets.append(ds_t)
        ds = xr.concat(datasets, dim=TIME_STR)

        plevs = np.array([200, 400, 600])
        result = interp_nested_to_plevs(
            ds, plevs, dim1=LAT_STR, dim2=TIME_STR, method="linear"
        )
        assert LAT_STR in result["temp"].dims
        assert TIME_STR in result["temp"].dims
        assert result["temp"].sizes[TIME_STR] == ntime

    def test_returns_dataset(self) -> None:
        """Result is an xarray Dataset."""
        nlat, nlev, ntime = 3, 6, 2
        ds_base = _make_eta_dataset(nlat=nlat, nlev=nlev)
        datasets = []
        for t in range(ntime):
            ds_t = ds_base.copy(deep=True)
            ds_t = ds_t.expand_dims({TIME_STR: [t]})
            datasets.append(ds_t)
        ds = xr.concat(datasets, dim=TIME_STR)

        plevs = np.array([200, 400, 600])
        result = interp_nested_to_plevs(
            ds, plevs, dim1=LAT_STR, dim2=TIME_STR, method="linear"
        )
        assert isinstance(result, xr.Dataset)
