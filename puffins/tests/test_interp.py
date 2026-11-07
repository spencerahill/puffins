"""Tests for interp module."""

from typing import cast

import numpy as np
import pytest
import xarray as xr

from puffins.interp import (
    PfullMethod,
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
from puffins.vert_coords import (
    pfull_from_phalf_avg,
    pfull_from_phalf_log_mean,
    pfull_simm_burr,
    phalf_from_psfc,
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


def _make_eta_dataset(nlat: int = 5, nlev: int = 6, p_top: float = 100.0) -> xr.Dataset:
    """Dataset with eta (hybrid sigma-pressure) coordinates.

    bk is linearly spaced from 0 to 1, and pk is zero except at the top half
    level, where it is ``p_top`` (Pa). The default ``p_top`` keeps the top
    half level above zero pressure, where the default log-mean full level of
    ``interp_eta_to_plevs`` is defined. Half-levels have nlev+1 points;
    full-levels have nlev points.
    """
    lats = np.linspace(-60, 60, nlat)

    # Half-level coefficients (nlev + 1 points).
    nhalf = nlev + 1
    bk_vals = np.linspace(0, 1, nhalf)
    pk_vals = np.zeros(nhalf)
    pk_vals[0] = p_top
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


def _make_hybrid_dataset(p_top: float = 2.0, nlev: int = 12) -> xr.Dataset:
    """Hybrid sigma-pressure Dataset whose reference coordinates are pressures.

    The half levels are ``ps * bk + pk``, with ``bk = s**2`` and ``pk`` equal to
    ``p_top`` (Pa) at the top and zero at the surface, for ``s`` evenly spaced
    from 0 to 1. The ``phalf`` and ``pfull`` coordinates are the half levels at
    1000 hPa and their log-means (their arithmetic means if ``p_top`` is 0), in
    hPa. Temperature varies with pressure and with latitude.
    """
    s = np.linspace(0.0, 1.0, nlev + 1)
    bk = s**2
    pk = p_top + (5e4 - p_top) * s * (1 - s)
    pk[-1] = 0.0
    phalf_ref = (1e5 * bk + pk) / 100
    upper, lower = phalf_ref[:-1], phalf_ref[1:]
    if p_top > 0:
        pfull_ref = (lower - upper) / np.log(lower / upper)
    else:
        pfull_ref = 0.5 * (upper + lower)
    lats = np.array([-30.0, 0.0, 30.0])
    temp = (
        200
        + 30 * np.log(pfull_ref)[np.newaxis, :]
        + 10 * np.sin(np.log(pfull_ref))[np.newaxis, :]
        + np.arange(3)[:, np.newaxis]
    )
    return xr.Dataset(
        {
            "temp": ((LAT_STR, PFULL_STR), temp),
            BK_STR: (PHALF_STR, bk),
            PK_STR: (PHALF_STR, pk),
            "ps": (LAT_STR, [95000.0, 97500.0, 100000.0]),
        },
        coords={LAT_STR: lats, PFULL_STR: pfull_ref, PHALF_STR: phalf_ref},
    )


# Each pfull_method of interp_eta_to_plevs and the vert_coords function it uses.
_PFULL_FUNCS = {
    "log_mean": lambda phalf, ds: pfull_from_phalf_log_mean(phalf, ds[PFULL_STR]),
    "arithmetic_mean": lambda phalf, ds: pfull_from_phalf_avg(phalf, ds[PFULL_STR]),
    "simmons_burridge": lambda phalf, ds: pfull_simm_burr(
        phalf, ds[PHALF_STR], ds[PFULL_STR]
    ),
}


def _own_levels_hpa(ds: xr.Dataset, pfull_method: str) -> xr.DataArray:
    """A dataset's full-level pressures (hPa) as ``pfull_method`` computes them."""
    phalf = phalf_from_psfc(ds[BK_STR], ds[PK_STR], ds["ps"])
    return cast(xr.DataArray, _PFULL_FUNCS[pfull_method](phalf, ds) * 1e-2)


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

    def _make_ds_and_plevs(self, value: float = 1.0) -> tuple[xr.Dataset, np.ndarray]:
        """Create a Dataset with pfull coordinates and target plevs."""
        nlat, nlev = 5, 8
        lats = np.linspace(-60, 60, nlat)
        pfull_vals = np.linspace(200, 900, nlev)
        temp = xr.DataArray(
            value * np.ones((nlat, nlev)),
            dims=[LAT_STR, PFULL_STR],
            coords={LAT_STR: lats, PFULL_STR: pfull_vals},
            name="temp",
        )
        ds = xr.Dataset({"temp": temp})
        plevs = np.array([300, 500, 700])
        return ds, plevs

    def test_constant_field_stays_constant(self) -> None:
        """A spatially constant field remains constant after interpolation."""
        ds, plevs = self._make_ds_and_plevs(value=300.0)
        result = interp_ds_p_to_p(ds, plevs, method="linear")
        np.testing.assert_allclose(result["temp"].values, 300.0, atol=1e-10)

    def test_output_dimensions(self) -> None:
        """Output has (lev, lat) dimensions."""
        ds, plevs = self._make_ds_and_plevs()
        result = interp_ds_p_to_p(ds, plevs, method="linear")
        assert set(result["temp"].dims) == {LEV_STR, LAT_STR}

    def test_returns_dataset(self) -> None:
        """Result is an xarray Dataset."""
        ds, plevs = self._make_ds_and_plevs()
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

    def test_surface_pressure_name(self) -> None:
        """The surface pressure is read from the variable ``p_sfc_str`` names."""
        ds = _make_eta_dataset(nlat=5, nlev=6)
        plevs = np.array([200, 400, 600, 800])
        expected = interp_eta_to_plevs(ds, plevs, dim=LAT_STR, method="linear")
        renamed = ds.rename({"ps": "surf_pres"})
        result = interp_eta_to_plevs(
            renamed, plevs, dim=LAT_STR, method="linear", p_sfc_str="surf_pres"
        )
        xr.testing.assert_identical(result["temp"], expected["temp"])
        nested = interp_nested_to_plevs(
            renamed.expand_dims({TIME_STR: [0]}),
            plevs,
            dim1=LAT_STR,
            dim2=TIME_STR,
            method="linear",
            p_sfc_str="surf_pres",
        )
        xr.testing.assert_allclose(
            nested["temp"].isel({TIME_STR: 0}, drop=True), expected["temp"]
        )


# ---------------------------------------------------------------------------
# TestInterpNestedToPlevs
# ---------------------------------------------------------------------------


class TestInterpNestedToPlevs:
    """Tests for interp_nested_to_plevs."""

    def _make_eta_with_time(self) -> tuple[xr.Dataset, np.ndarray]:
        """Create an eta Dataset with a time dimension and target plevs."""
        ds_base = _make_eta_dataset(nlat=3, nlev=6)
        datasets = []
        for t in range(2):
            ds_t = ds_base.copy(deep=True)
            ds_t = ds_t.expand_dims({TIME_STR: [t]})
            datasets.append(ds_t)
        ds = xr.concat(datasets, dim=TIME_STR)
        plevs = np.array([200, 400, 600])
        return ds, plevs

    def test_output_has_both_extra_dims(self) -> None:
        """Output Dataset has both the inner and outer loop dimensions."""
        ds, plevs = self._make_eta_with_time()
        result = interp_nested_to_plevs(
            ds, plevs, dim1=LAT_STR, dim2=TIME_STR, method="linear"
        )
        assert LAT_STR in result["temp"].dims
        assert TIME_STR in result["temp"].dims
        assert result["temp"].sizes[TIME_STR] == 2

    def test_returns_dataset(self) -> None:
        """Result is an xarray Dataset."""
        ds, plevs = self._make_eta_with_time()
        result = interp_nested_to_plevs(
            ds, plevs, dim1=LAT_STR, dim2=TIME_STR, method="linear"
        )
        assert isinstance(result, xr.Dataset)


# ---------------------------------------------------------------------------
# TestFullLevelPressures
# ---------------------------------------------------------------------------


def _column(arr: xr.DataArray, i: int) -> np.ndarray:
    """Values of one latitude's column, in full-level order."""
    return cast(np.ndarray, arr.isel({LAT_STR: i}).values)


class TestFullLevelPressures:
    """Where interp_eta_to_plevs places the model levels (issue #118)."""

    @pytest.mark.parametrize("pfull_method", list(_PFULL_FUNCS))
    def test_column_unchanged_at_its_own_levels(
        self, pfull_method: PfullMethod
    ) -> None:
        """Interpolating a column to its own full levels returns it unchanged.

        Each latitude has its own surface pressure, so each column is checked
        at its own levels. Had the function used another method's levels, some
        column of this dataset would change by at least 0.77 K.
        """
        ds = _make_hybrid_dataset()
        levels = _own_levels_hpa(ds, pfull_method)
        for i in range(ds.sizes[LAT_STR]):
            result = interp_eta_to_plevs(
                ds, _column(levels, i), dim=LAT_STR, pfull_method=pfull_method
            )
            np.testing.assert_allclose(
                _column(result["temp"], i), _column(ds["temp"], i), rtol=1e-12
            )

    def test_default_is_log_mean(self) -> None:
        """By default each level is at the log-mean of its two half levels."""
        ds = _make_hybrid_dataset()
        levels = _own_levels_hpa(ds, "log_mean")
        result = interp_eta_to_plevs(ds, _column(levels, 0), dim=LAT_STR)
        np.testing.assert_allclose(
            _column(result["temp"], 0), _column(ds["temp"], 0), rtol=1e-12
        )

    def test_levels_against_the_fixture_closed_form(self) -> None:
        """At 1000 hPa surface pressure the levels are the fixture's own log-means.

        The ``pfull`` coordinate of the third column's dataset is computed in
        raw numpy, independently of puffins. Its end levels differ from the
        function's by rounding, which puts them just outside the column, so the
        round trip uses the interior levels and the ends are compared directly.
        """
        ds = _make_hybrid_dataset()
        assert float(ds["ps"].isel({LAT_STR: 2})) == 100000.0
        independent = ds[PFULL_STR].values
        np.testing.assert_allclose(
            _column(_own_levels_hpa(ds, "log_mean"), 2), independent, rtol=1e-14
        )
        result = interp_eta_to_plevs(ds, independent[1:-1], dim=LAT_STR)
        np.testing.assert_allclose(
            _column(result["temp"], 2), _column(ds["temp"], 2)[1:-1], rtol=1e-12
        )

    def test_method_sets_the_interpolation(self) -> None:
        """``method`` sets how values between the model levels are interpolated.

        Between the levels of this column, cubic and linear interpolation
        differ by 0.096 to 5.4 K.
        """
        ds = _make_hybrid_dataset()
        levels = _column(_own_levels_hpa(ds, "log_mean"), 2)
        mids = np.sqrt(levels[:-1] * levels[1:])
        linear = interp_eta_to_plevs(ds, mids, dim=LAT_STR, method="linear")
        np.testing.assert_allclose(
            _column(linear["temp"], 2),
            np.interp(mids, levels, _column(ds["temp"], 2)),
            rtol=1e-12,
        )
        cubic = interp_eta_to_plevs(ds, mids, dim=LAT_STR)
        gap = np.abs(_column(cubic["temp"], 2) - _column(linear["temp"], 2))
        assert float(gap.min()) > 0.05

    def test_custom_variable_names(self) -> None:
        """Both functions read bk, pk, pfull and ps under the names given."""
        ds = _make_hybrid_dataset()
        nested = ds.expand_dims({TIME_STR: [0]})
        plevs = np.array([100.0, 500.0, 800.0])
        renamed = {
            BK_STR: "b_coef",
            PK_STR: "a_coef",
            PFULL_STR: "lev_full",
            "ps": "surf_pres",
        }
        expected = interp_eta_to_plevs(ds, plevs, dim=LAT_STR)
        result = interp_eta_to_plevs(
            ds.rename(renamed),
            plevs,
            dim=LAT_STR,
            bk_str="b_coef",
            pk_str="a_coef",
            pfull_str="lev_full",
            p_sfc_str="surf_pres",
        )
        np.testing.assert_array_equal(
            result["temp"].transpose(LAT_STR, "lev_full").values,
            expected["temp"].transpose(LAT_STR, PFULL_STR).values,
        )
        nested_result = interp_nested_to_plevs(
            nested.rename(renamed),
            plevs,
            dim1=LAT_STR,
            dim2=TIME_STR,
            bk_str="b_coef",
            pk_str="a_coef",
            pfull_str="lev_full",
            p_sfc_str="surf_pres",
        )
        np.testing.assert_array_equal(
            nested_result["temp"]
            .isel({TIME_STR: 0})
            .transpose(LAT_STR, "lev_full")
            .values,
            expected["temp"].transpose(LAT_STR, PFULL_STR).values,
        )

    @pytest.mark.parametrize("top_first", [True, False])
    def test_log_mean_raises_with_a_zero_pressure_top(self, top_first: bool) -> None:
        """A top half level at 0 Pa raises under the log-mean, naming the others.

        With the column stored surface first, the zero is the last half level.
        """
        ds = _make_hybrid_dataset(p_top=0.0)
        if not top_first:
            ds = ds.isel(
                {
                    PHALF_STR: slice(None, None, -1),
                    PFULL_STR: slice(None, None, -1),
                }
            )
        plevs = np.array([100.0, 500.0])
        with pytest.raises(ValueError, match="'simmons_burridge'"):
            interp_eta_to_plevs(ds, plevs, dim=LAT_STR)
        result = interp_eta_to_plevs(
            ds, plevs, dim=LAT_STR, pfull_method="arithmetic_mean"
        )
        assert bool(np.isfinite(result["temp"]).all())

    @pytest.mark.filterwarnings(
        "ignore:divide by zero encountered in divide:RuntimeWarning"
    )
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in multiply:RuntimeWarning"
    )
    def test_simmons_burridge_with_a_zero_pressure_top(self) -> None:
        """Simmons-Burridge levels work with a top half level at 0 Pa.

        pfull_simm_burr computes the top layer from ln(p / 0) before replacing
        it, which raises the two RuntimeWarnings filtered here.
        """
        ds = _make_hybrid_dataset(p_top=0.0)
        levels = _own_levels_hpa(ds, "simmons_burridge")
        assert float(levels.min()) > 0
        result = interp_eta_to_plevs(
            ds, _column(levels, 1), dim=LAT_STR, pfull_method="simmons_burridge"
        )
        np.testing.assert_allclose(
            _column(result["temp"], 1), _column(ds["temp"], 1), rtol=1e-12
        )

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="pfull_method must be"):
            interp_eta_to_plevs(
                _make_hybrid_dataset(),
                np.array([500.0]),
                dim=LAT_STR,
                pfull_method=cast(PfullMethod, "geometric_mean"),
            )

    def test_simmons_burridge_needs_pressure_coordinates(self) -> None:
        """Index coordinates on phalf and pfull raise under Simmons-Burridge.

        They would otherwise set the top level's scale factor to 0 / 1, with no
        error; the other two methods do not use the coordinates.
        """
        ds = _make_hybrid_dataset()
        indexed = ds.assign_coords(
            {
                PHALF_STR: np.arange(ds.sizes[PHALF_STR]),
                PFULL_STR: np.arange(ds.sizes[PFULL_STR]),
            }
        )
        plevs = np.array([100.0, 500.0])
        with pytest.raises(ValueError, match="strictly between"):
            interp_eta_to_plevs(
                indexed, plevs, dim=LAT_STR, pfull_method="simmons_burridge"
            )
        xr.testing.assert_identical(
            interp_eta_to_plevs(indexed, plevs, dim=LAT_STR)["temp"],
            interp_eta_to_plevs(ds, plevs, dim=LAT_STR)["temp"],
        )

    @pytest.mark.parametrize("pfull_method", list(_PFULL_FUNCS))
    def test_custom_full_level_name(self, pfull_method: PfullMethod) -> None:
        """A full-level dimension under another name, given by pfull_str, works."""
        ds = _make_hybrid_dataset()
        plevs = np.array([100.0, 500.0, 800.0])
        expected = interp_eta_to_plevs(
            ds, plevs, dim=LAT_STR, pfull_method=pfull_method
        )
        result = interp_eta_to_plevs(
            ds.rename({PFULL_STR: "lev_full"}),
            plevs,
            dim=LAT_STR,
            pfull_str="lev_full",
            pfull_method=pfull_method,
        )
        np.testing.assert_array_equal(
            result["temp"].transpose(LAT_STR, "lev_full").values,
            expected["temp"].transpose(LAT_STR, PFULL_STR).values,
        )

    def test_nested_passes_the_method(self) -> None:
        """interp_nested_to_plevs uses the pfull_method it is given."""
        ds = _make_hybrid_dataset()
        nested = xr.concat(
            [ds, ds.assign(temp=ds["temp"] + 1.0)], dim=TIME_STR
        ).assign_coords({TIME_STR: [0, 1]})
        arith_levels = _column(_own_levels_hpa(ds, "arithmetic_mean"), 0)
        log_levels = _column(_own_levels_hpa(ds, "log_mean"), 0)
        by_arith = interp_nested_to_plevs(
            nested,
            arith_levels,
            dim1=LAT_STR,
            dim2=TIME_STR,
            pfull_method="arithmetic_mean",
        )
        by_default = interp_nested_to_plevs(
            nested, log_levels, dim1=LAT_STR, dim2=TIME_STR
        )
        for result in (by_arith, by_default):
            for t in (0, 1):
                np.testing.assert_allclose(
                    _column(result["temp"].sel({TIME_STR: t}), 0),
                    _column(ds["temp"], 0) + t,
                    rtol=1e-12,
                )
