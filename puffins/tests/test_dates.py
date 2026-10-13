"""Tests for dates module."""

import calendar

import numpy as np
import pytest
import xarray as xr

from puffins.dates import (
    ann_harm,
    ann_subset_ts,
    ann_subsets,
    ann_ts_djf,
    monthly_anom,
    subset_ann,
    time_to_year_and_day,
)


@pytest.fixture
def monthly_ts() -> xr.DataArray:
    """3-year monthly time series."""
    time = xr.date_range("2000-01-01", periods=36, freq="MS")
    return xr.DataArray(
        np.arange(36, dtype=float), dims=["time"], coords={"time": time}
    )


@pytest.fixture
def monthly_ts_2d() -> xr.DataArray:
    """3-year monthly time series with a spatial dimension."""
    time = xr.date_range("2000-01-01", periods=36, freq="MS")
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
        ann_months = ann_subsets["ann"]
        # Narrow away the non-iterable `int` member of the value union; the
        # concrete container type is deliberately not pinned, since storing
        # the same months as a list would be an equivalent representation.
        assert not isinstance(ann_months, int)
        assert list(ann_months) == list(range(1, 13))

    def test_months_are_calendar_months(self) -> None:
        """The single-month entries are the 12 calendar months."""
        singles = {k: v for k, v in ann_subsets.items() if isinstance(v, int)}
        assert singles == {calendar.month_abbr[m].lower(): m for m in range(1, 13)}

    def test_season_keys_spell_consecutive_months(self) -> None:
        """Each multi-month key is the initials of consecutive months.

        The oracle is independent of dates.py: the initials come from
        `calendar.month_abbr`.
        """
        seasons = {
            k: list(v)
            for k, v in ann_subsets.items()
            if not isinstance(v, int) and k != "ann"
        }
        # 2- through 6-month windows that do not wrap the year end.
        assert len(seasons) == 11 + 10 + 9 + 8 + 7
        for key, months_ in seasons.items():
            assert months_ == list(range(months_[0], months_[0] + len(key))), key
            initials = "".join(calendar.month_abbr[m][0].lower() for m in months_)
            assert initials == key, key

    def test_total_size(self) -> None:
        """12 months, the 45 seasons and 'ann', with no key repeated."""
        assert len(ann_subsets) == 12 + 11 + 10 + 9 + 8 + 7 + 1


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

    def test_season_crossing_year_end(self, monthly_ts: xr.DataArray) -> None:
        # Selecting between the smallest and largest month listed would keep
        # all twelve months here.
        result = subset_ann(monthly_ts, [12, 1, 2], drop=True)
        np.testing.assert_array_equal(result.values, [0, 1, 11, 12, 13, 23, 24, 25, 35])

    def test_months_not_adjacent(self, monthly_ts: xr.DataArray) -> None:
        result = subset_ann(monthly_ts, [1, 3], drop=True)
        np.testing.assert_array_equal(result.values, [0, 2, 12, 14, 24, 26])


# The fixture's value is its index, so index i is month i % 12 + 1 of year
# 2000 + i // 12, and each expected mean below is written by index.
VALS = np.arange(36, dtype=float)


class TestAnnSubsetTs:
    """Tests for ann_subset_ts."""

    def test_mean_reduction(self, monthly_ts: xr.DataArray) -> None:
        result = ann_subset_ts(monthly_ts, "jja")
        assert "year" in result.dims

    def test_output_has_year_dim(self, monthly_ts: xr.DataArray) -> None:
        result = ann_subset_ts(monthly_ts, 1)
        assert "year" in result.dims
        assert len(result.year) == 3

    def test_max_reduction(self, monthly_ts: xr.DataArray) -> None:
        result = ann_subset_ts(monthly_ts, "jja", reduction="max")
        assert "year" in result.dims
        # Max of JJA in year 1 (indices 5,6,7 -> values 5,6,7) is 7
        assert result.isel(year=0).values == 7

    def test_single_month_equals_raw_values(self, monthly_ts: xr.DataArray) -> None:
        result = ann_subset_ts(monthly_ts, 1)
        # January values are 0, 12, 24
        np.testing.assert_array_equal(result.values, [0, 12, 24])

    def test_equal_weight_mean(self, monthly_ts: xr.DataArray) -> None:
        result = ann_subset_ts(monthly_ts, "jja")
        np.testing.assert_array_equal(result["year"], [2000, 2001, 2002])
        want = [VALS[5:8].mean(), VALS[17:20].mean(), VALS[29:32].mean()]
        np.testing.assert_allclose(result, want, rtol=1e-14)

    def test_month_order_does_not_matter(self, monthly_ts: xr.DataArray) -> None:
        xr.testing.assert_identical(
            ann_subset_ts(monthly_ts, [8, 6, 7]), ann_subset_ts(monthly_ts, "jja")
        )

    @pytest.mark.parametrize("months", [[12, 1, 2], [1, 2, 12], [2, 12, 1]])
    def test_season_crossing_year_end(
        self, monthly_ts: xr.DataArray, months: list[int]
    ) -> None:
        # The 2001 winter is Dec 2000, Jan 2001 and Feb 2001 (indices 11-13).
        # The 2000 winter lacks its December, so it is NaN.
        result = ann_subset_ts(monthly_ts, months)
        np.testing.assert_array_equal(result["year"], [2000, 2001, 2002])
        want = [np.nan, VALS[11:14].mean(), VALS[23:26].mean()]
        np.testing.assert_allclose(result, want, rtol=1e-14)

    def test_two_months_before_year_end(self, monthly_ts: xr.DataArray) -> None:
        result = ann_subset_ts(monthly_ts, [11, 12, 1])
        want = [np.nan, VALS[10:13].mean(), VALS[22:25].mean()]
        np.testing.assert_allclose(result, want, rtol=1e-14)

    def test_december_alone_stays_in_its_year(self, monthly_ts: xr.DataArray) -> None:
        result = ann_subset_ts(monthly_ts, [12])
        np.testing.assert_array_equal(result["year"], [2000, 2001, 2002])
        np.testing.assert_array_equal(result, VALS[[11, 23, 35]])

    def test_record_starting_mid_season(self, monthly_ts: xr.DataArray) -> None:
        # Starts in July 2000, so the 2000 JJA lacks its June.
        result = ann_subset_ts(monthly_ts.isel(time=slice(6, None)), "jja")
        np.testing.assert_array_equal(result["year"], [2000, 2001, 2002])
        want = [np.nan, VALS[17:20].mean(), VALS[29:32].mean()]
        np.testing.assert_allclose(result, want, rtol=1e-14)

    def test_record_ending_mid_season(self, monthly_ts: xr.DataArray) -> None:
        # Ends in July 2002, so the 2002 JJA lacks its August.
        result = ann_subset_ts(monthly_ts.isel(time=slice(None, 31)), "jja")
        want = [VALS[5:8].mean(), VALS[17:20].mean(), np.nan]
        np.testing.assert_allclose(result, want, rtol=1e-14)

    def test_record_starting_in_december(self) -> None:
        time = xr.date_range("1999-12-01", periods=37, freq="MS")
        da = xr.DataArray(np.arange(37.0), dims=["time"], coords={"time": time})
        result = ann_subset_ts(da, [12, 1, 2])
        # The year coordinate is the record's calendar years, 1999-2002.
        np.testing.assert_array_equal(result["year"], [1999, 2000, 2001, 2002])
        vals = da.to_numpy()
        want = np.array(
            [np.nan, vals[0:3].mean(), vals[12:15].mean(), vals[24:27].mean()]
        )
        np.testing.assert_allclose(result, want, rtol=1e-14)

    def test_nan_month_makes_its_year_nan(self, monthly_ts: xr.DataArray) -> None:
        holed = monthly_ts.where(monthly_ts["time"] != np.datetime64("2001-07-01"))
        result = ann_subset_ts(holed, "jja")
        want = [VALS[5:8].mean(), np.nan, VALS[29:32].mean()]
        np.testing.assert_allclose(result, want, rtol=1e-14)

    @pytest.mark.parametrize("reduction", ["sum", "max", "min"])
    def test_incomplete_season_nan_for_any_reduction(
        self, monthly_ts: xr.DataArray, reduction: str
    ) -> None:
        # Without the mask, the 2000 winter would reduce Jan and Feb alone.
        result = ann_subset_ts(monthly_ts, [12, 1, 2], reduction=reduction)
        want = getattr(np, reduction)(VALS[11:14])
        assert np.isnan(result.sel(year=2000).item())
        np.testing.assert_allclose(result.sel(year=2001).item(), want, rtol=1e-14)

    def test_annual_mean_needs_all_twelve_months(
        self, monthly_ts: xr.DataArray
    ) -> None:
        result = ann_subset_ts(monthly_ts.isel(time=slice(1, None)), "ann")
        want = [np.nan, VALS[12:24].mean(), VALS[24:36].mean()]
        np.testing.assert_allclose(result, want, rtol=1e-14)

    def test_nan_at_one_point_leaves_others(self, monthly_ts_2d: xr.DataArray) -> None:
        holed = monthly_ts_2d.copy()
        holed[18, 0] = np.nan  # July 2001 at x = 0
        result = ann_subset_ts(holed, "jja")
        assert result.dims == ("year", "x")
        vals = monthly_ts_2d.to_numpy()
        assert np.isnan(result.sel(year=2001, x=0).item())
        np.testing.assert_allclose(
            result.sel(x=1),
            [vals[s, 1].mean() for s in (slice(5, 8), slice(17, 20), slice(29, 32))],
        )

    def test_year_takes_the_place_of_time(self, monthly_ts_2d: xr.DataArray) -> None:
        swapped = monthly_ts_2d.transpose("x", "time")
        assert ann_subset_ts(swapped, "jja").dims == ("x", "year")

    def test_custom_time_dim(self, monthly_ts: xr.DataArray) -> None:
        result = ann_subset_ts(monthly_ts.rename(time="t"), [12, 1, 2], dim_time="t")
        np.testing.assert_allclose(
            result, [np.nan, VALS[11:14].mean(), VALS[23:26].mean()], rtol=1e-14
        )

    @pytest.mark.parametrize(
        ("months", "match"),
        [
            ([1, 3], "unbroken run"),
            ([12, 2], "unbroken run"),
            ([6, 6, 7], "distinct"),
            ([0, 1], "distinct"),
            ([13], "distinct"),
        ],
    )
    def test_invalid_months_raise(
        self, monthly_ts: xr.DataArray, months: list[int], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            ann_subset_ts(monthly_ts, months)

    def test_daily_data_raises(self) -> None:
        time = xr.date_range("2000-01-01", periods=60, freq="D")
        daily = xr.DataArray(np.arange(60.0), dims=["time"], coords={"time": time})
        with pytest.raises(ValueError, match="monthly"):
            ann_subset_ts(daily, "jf")

    def test_season_absent_from_record_raises(self, monthly_ts: xr.DataArray) -> None:
        with pytest.raises(ValueError, match="none of the months"):
            ann_subset_ts(monthly_ts.isel(time=slice(0, 3)), "jja")


class TestAnnTsDjf:
    """Tests for ann_ts_djf."""

    def test_output_has_year_dim(self, monthly_ts: xr.DataArray) -> None:
        result = ann_ts_djf(monthly_ts)
        assert "year" in result.dims

    def test_known_value_equal_weights(self, monthly_ts: xr.DataArray) -> None:
        # 2000 has no December 1999, so it is NaN.  Weighting the months by
        # their lengths would give 11.97 for 2001 in place of 12.
        result = ann_ts_djf(monthly_ts)
        np.testing.assert_array_equal(result["year"], [2000, 2001, 2002])
        want = [np.nan, VALS[11:14].mean(), VALS[23:26].mean()]
        np.testing.assert_allclose(result, want, rtol=1e-14)

    def test_2d_preserves_spatial_dim(self, monthly_ts_2d: xr.DataArray) -> None:
        result = ann_ts_djf(monthly_ts_2d)
        assert result.dims == ("year", "x")
        vals = monthly_ts_2d.to_numpy()
        np.testing.assert_allclose(result.sel(year=2001), vals[11:14].mean(axis=0))


class TestMonthlyAnom:
    """Tests for monthly_anom."""

    def test_removes_calendar_month_climatology(self, monthly_ts: xr.DataArray) -> None:
        result = monthly_anom(monthly_ts)
        # January values are 0, 12, 24 (mean 12), so anomalies are -12, 0, 12.
        np.testing.assert_allclose(
            result.isel(time=[0, 12, 24]).values, [-12.0, 0.0, 12.0]
        )

    def test_anomalies_sum_to_zero_per_month(self, monthly_ts: xr.DataArray) -> None:
        result = monthly_anom(monthly_ts)
        clim = result.groupby("time.month").mean("time")
        np.testing.assert_allclose(clim.values, 0.0, atol=1e-12)

    def test_2d_preserves_spatial_dim(self, monthly_ts_2d: xr.DataArray) -> None:
        result = monthly_anom(monthly_ts_2d)
        assert result.dims == monthly_ts_2d.dims
        clim = result.groupby("time.month").mean("time")
        np.testing.assert_allclose(clim.values, 0.0, atol=1e-12)

    def test_base_period_known_value(self, monthly_ts: xr.DataArray) -> None:
        # Base 2000-2001: each month's climatology is the mean of its 2000 and
        # 2001 values, indices m and m + 12.
        result = monthly_anom(monthly_ts, base=(2000, 2001))
        clim = (VALS[:12] + VALS[12:24]) / 2
        np.testing.assert_allclose(result, VALS - np.tile(clim, 3), rtol=0, atol=1e-12)

    def test_base_period_per_point(self, monthly_ts_2d: xr.DataArray) -> None:
        result = monthly_anom(monthly_ts_2d, base=(2001, 2002))
        vals = monthly_ts_2d.to_numpy()
        clim = (vals[12:24] + vals[24:36]) / 2
        np.testing.assert_allclose(
            result, vals - np.tile(clim, (3, 1)), rtol=0, atol=1e-12
        )

    def test_base_period_custom_time_dim(self, monthly_ts: xr.DataArray) -> None:
        result = monthly_anom(monthly_ts.rename(time="t"), "t", base=(2000, 2001))
        clim = (VALS[:12] + VALS[12:24]) / 2
        np.testing.assert_allclose(result, VALS - np.tile(clim, 3), rtol=0, atol=1e-12)

    def test_base_beyond_record_raises(self, monthly_ts: xr.DataArray) -> None:
        with pytest.raises(ValueError, match="base years 1999-2001"):
            monthly_anom(monthly_ts, base=(1999, 2001))

    def test_base_missing_a_month_raises(self, monthly_ts: xr.DataArray) -> None:
        holed = monthly_ts.drop_sel(time=np.datetime64("2001-07-01"))
        with pytest.raises(ValueError, match="base years 2000-2001"):
            monthly_anom(holed, base=(2000, 2001))


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
        time = xr.date_range("2000-01-01", periods=365, freq="D")
        arr = xr.DataArray(
            np.arange(365, dtype=float), dims=["time"], coords={"time": time}
        )
        result = time_to_year_and_day(arr)
        assert "year" in result.dims
        assert "dayofyear" in result.dims
