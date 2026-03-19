"""Tests for dates module."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from puffins.dates import (
    ann_harm,
    ann_subset_ts,
    ann_subsets,
    ann_ts_djf,
    subset_ann,
    time_to_year_and_day,
)


@pytest.fixture
def monthly_ts() -> xr.DataArray:
    """3-year monthly time series."""
    time = pd.date_range("2000-01-01", periods=36, freq="MS")
    return xr.DataArray(
        np.arange(36, dtype=float), dims=["time"], coords={"time": time}
    )


@pytest.fixture
def monthly_ts_2d() -> xr.DataArray:
    """3-year monthly time series with a spatial dimension."""
    time = pd.date_range("2000-01-01", periods=36, freq="MS")
    data = np.arange(72, dtype=float).reshape(36, 2)
    return xr.DataArray(data, dims=["time", "x"], coords={"time": time, "x": [0, 1]})


class TestAnnSubsets:
    """Tests for the ann_subsets dictionary."""

    def test_contains_months(self) -> None:
        assert "jan" in ann_subsets
        assert "dec" in ann_subsets
        assert ann_subsets["jan"] == 1
        assert ann_subsets["dec"] == 12

    def test_contains_seasons(self) -> None:
        assert "jja" in ann_subsets
        assert ann_subsets["jja"] == [6, 7, 8]

    def test_contains_ann(self) -> None:
        assert "ann" in ann_subsets
        assert list(ann_subsets["ann"]) == list(range(1, 13))


class TestSubsetAnn:
    """Tests for subset_ann."""

    def test_string_month(self, monthly_ts: xr.DataArray) -> None:
        result = subset_ann(monthly_ts, "jan", drop=True)
        assert len(result.time) == 3
        # January values are at indices 0, 12, 24
        np.testing.assert_array_equal(result.values, [0, 12, 24])

    def test_list_of_months(self, monthly_ts: xr.DataArray) -> None:
        result = subset_ann(monthly_ts, [6, 7, 8], drop=True)
        assert len(result.time) == 9

    def test_ann_returns_unchanged(self, monthly_ts: xr.DataArray) -> None:
        result = subset_ann(monthly_ts, "ann")
        xr.testing.assert_identical(result, monthly_ts)

    def test_single_int_month(self, monthly_ts: xr.DataArray) -> None:
        result = subset_ann(monthly_ts, 1, drop=True)
        assert len(result.time) == 3

    def test_drop_false_preserves_size(self, monthly_ts: xr.DataArray) -> None:
        result = subset_ann(monthly_ts, "jan", drop=False)
        assert len(result.time) == len(monthly_ts.time)


class TestAnnSubsetTs:
    """Tests for ann_subset_ts."""

    def test_mean_reduction(self, monthly_ts: xr.DataArray) -> None:
        result = ann_subset_ts(monthly_ts, "jja")
        assert "year" in result.dims

    def test_output_has_year_dim(self, monthly_ts: xr.DataArray) -> None:
        result = ann_subset_ts(monthly_ts, 1)
        assert "year" in result.dims
        assert len(result.year) == 3

    def test_single_month_equals_raw_values(self, monthly_ts: xr.DataArray) -> None:
        result = ann_subset_ts(monthly_ts, 1)
        # January values are 0, 12, 24
        np.testing.assert_array_equal(result.values, [0, 12, 24])


class TestAnnTsDjf:
    """Tests for ann_ts_djf."""

    def test_output_has_year_dim(self, monthly_ts: xr.DataArray) -> None:
        result = ann_ts_djf(monthly_ts)
        assert "year" in result.dims

    def test_first_year_is_jf_only(self, monthly_ts: xr.DataArray) -> None:
        result = ann_ts_djf(monthly_ts)
        # First year (2000) should be Jan-Feb average only: (0 + 1) / 2 = 0.5
        jf_mean = ann_subset_ts(monthly_ts, "jf").isel(year=0).values
        np.testing.assert_allclose(result.isel(year=0).values, jf_mean)

    def test_2d_preserves_spatial_dim(self, monthly_ts_2d: xr.DataArray) -> None:
        result = ann_ts_djf(monthly_ts_2d)
        assert "x" in result.dims
        assert "year" in result.dims


class TestAnnHarm:
    """Tests for ann_harm."""

    def test_identity_when_num_harm_equals_length(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = ann_harm(arr, num_harm=4)
        np.testing.assert_array_equal(result, arr)

    def test_dataarray_in_dataarray_out(self) -> None:
        da = xr.DataArray(np.sin(np.linspace(0, 2 * np.pi, 12)), dims=["month"])
        result = ann_harm(da, num_harm=1)
        assert isinstance(result, xr.DataArray)

    def test_ndarray_in_ndarray_out(self) -> None:
        arr = np.sin(np.linspace(0, 2 * np.pi, 12))
        result = ann_harm(arr, num_harm=1)
        assert isinstance(result, np.ndarray)

    def test_normalize(self) -> None:
        arr = np.sin(np.linspace(0, 2 * np.pi, 12))
        result = ann_harm(arr, num_harm=1, normalize=True)
        np.testing.assert_allclose(np.abs(result).max(), 1.0)

    def test_single_harmonic_captures_dominant_frequency(self) -> None:
        # Pure sine wave should be well-captured by a single harmonic
        arr = np.sin(np.linspace(0, 2 * np.pi, 12, endpoint=False))
        result = ann_harm(arr, num_harm=1)
        np.testing.assert_allclose(result, arr, atol=1e-10)


class TestTimeToYearAndDay:
    """Tests for time_to_year_and_day."""

    def test_produces_year_and_dayofyear(self) -> None:
        time = pd.date_range("2000-01-01", periods=365, freq="D")
        arr = xr.DataArray(
            np.arange(365, dtype=float), dims=["time"], coords={"time": time}
        )
        result = time_to_year_and_day(arr)
        assert "year" in result.dims
        assert "dayofyear" in result.dims
