"""Tests for the enso module.

Expected values are rebuilt from raw numpy on a small synthetic SST field, not
from the module's own helpers, so that each test pins a coefficient, a bound or
an offset rather than restating the code.
"""

from typing import cast

import numpy as np
import pytest
import xarray as xr

from puffins.enso import (
    DJF,
    JJAS,
    box_mean,
    monthly_anomaly,
    nino34,
    relative_nino34,
    relative_nino34_jjas,
    roni,
    roni_ratio,
    season_mean,
    tio,
    tropical_mean,
)

LATS = np.arange(-24.0, 25.0, 2.0)
LONS = np.arange(40.0, 250.0, 10.0)
N_MONTHS = 12 * (2022 - 1946 + 1)
ATOL = 1e-12
"""Absolute tolerance, in K, for anomalies of order 1 K.  Rounding from a
different summation order measured 7e-14 at most; a relative tolerance
magnifies that near zero."""


@pytest.fixture
def sst() -> xr.DataArray:
    """Random monthly SST, 1946-2022, with one land cell inside every box."""
    rng = np.random.default_rng(0)
    time = xr.date_range("1946-01-01", periods=N_MONTHS, freq="MS")
    data = 27.0 + rng.normal(size=(N_MONTHS, LATS.size, LONS.size))
    # A seasonal cycle, so the climatology is not flat.
    data += np.cos(2 * np.pi * (np.arange(N_MONTHS) % 12) / 12)[:, None, None]
    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": LATS, "lon": LONS},
        name="sst",
    )
    # Land: one cell in the NINO3.4 box, one in the TIO box.
    land = ((da.lat == 0.0) & (da.lon == 200.0)) | ((da.lat == 10.0) & (da.lon == 60.0))
    return cast(xr.DataArray, da.where(~land))


def numpy_box_mean(
    da: xr.DataArray, lat_b: tuple[float, float], lon_b: tuple[float, float] | None
) -> np.ndarray:
    lat = da["lat"].to_numpy()
    lon = da["lon"].to_numpy()
    ilat = np.flatnonzero((lat >= lat_b[0]) & (lat <= lat_b[1]))
    ilon = (
        np.arange(lon.size)
        if lon_b is None
        else np.flatnonzero((lon >= lon_b[0]) & (lon <= lon_b[1]))
    )
    vals = da.to_numpy()[:, ilat][:, :, ilon].astype(np.float64)
    w = np.cos(np.deg2rad(lat[ilat]))[None, :, None] * np.ones_like(vals)
    w[np.isnan(vals)] = 0.0
    return cast(np.ndarray, np.nansum(vals * w, axis=(1, 2)) / w.sum(axis=(1, 2)))


def numpy_monthly_anomaly(
    x: np.ndarray, years: np.ndarray, months: np.ndarray, base: tuple[int, int]
) -> np.ndarray:
    out = np.empty_like(x)
    for m in range(1, 13):
        sel = (months == m) & (years >= base[0]) & (years <= base[1])
        out[months == m] = x[months == m] - np.nanmean(x[sel])
    return out


def running3(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan)
    out[1:-1] = (x[:-2] + x[1:-1] + x[2:]) / 3.0
    return out


def numpy_ratio(
    n34a: np.ndarray,
    tropa: np.ndarray,
    years: np.ndarray,
    months: np.ndarray,
    span: tuple[int, int],
) -> np.ndarray:
    oni = running3(n34a)
    diff = oni - running3(tropa)
    ratio = np.empty(12)
    for m in range(1, 13):
        sel = (months == m) & (years >= span[0]) & (years <= span[1]) & np.isfinite(oni)
        ratio[m - 1] = np.std(oni[sel], ddof=1) / np.std(diff[sel], ddof=1)
    return ratio


def time_parts(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    return da["time"].dt.year.to_numpy(), da["time"].dt.month.to_numpy()


class TestBoxMean:
    def test_known_value(self, sst: xr.DataArray) -> None:
        got = box_mean(sst, (-5.0, 5.0), (190.0, 240.0))
        np.testing.assert_allclose(
            got, numpy_box_mean(sst, (-5.0, 5.0), (190.0, 240.0)), rtol=1e-13
        )

    def test_bounds_are_inclusive(self, sst: xr.DataArray) -> None:
        # Gridpoints sit exactly on -4, 4, 190 and 240; exclusive bounds would
        # drop them and change the mean.
        got = box_mean(sst, (-4.0, 4.0), (190.0, 240.0))
        np.testing.assert_allclose(
            got, numpy_box_mean(sst, (-4.0, 4.0), (190.0, 240.0)), rtol=1e-13
        )

    def test_cos_lat_weighting(self) -> None:
        da = xr.DataArray(
            [[[1.0], [3.0]]],
            dims=["time", "lat", "lon"],
            coords={"time": [0], "lat": [0.0, 60.0], "lon": [0.0]},
        )
        # Weights 1 and 0.5: (1 * 1 + 3 * 0.5) / 1.5.
        got = box_mean(da, (-90.0, 90.0))
        np.testing.assert_allclose(got, [2.5 / 1.5], rtol=1e-14)

    def test_land_left_out_of_the_weights(self) -> None:
        da = xr.DataArray(
            [[[1.0, np.nan], [3.0, 5.0]]],
            dims=["time", "lat", "lon"],
            coords={"time": [0], "lat": [0.0, 60.0], "lon": [0.0, 1.0]},
        )
        got = box_mean(da, (-90.0, 90.0))
        np.testing.assert_allclose(
            got, [(1.0 + 0.5 * 3.0 + 0.5 * 5.0) / 2.0], rtol=1e-14
        )

    def test_no_lon_bounds_spans_every_longitude(self, sst: xr.DataArray) -> None:
        got = box_mean(sst, (-20.0, 20.0))
        np.testing.assert_allclose(
            got, numpy_box_mean(sst, (-20.0, 20.0), None), rtol=1e-13
        )

    def test_descending_latitude(self, sst: xr.DataArray) -> None:
        flipped = sst.isel(lat=slice(None, None, -1))
        np.testing.assert_allclose(
            box_mean(flipped, (-5.0, 5.0), (190.0, 240.0)),
            box_mean(sst, (-5.0, 5.0), (190.0, 240.0)),
            rtol=1e-13,
        )

    def test_float32_input_accumulates_in_float64(self, sst: xr.DataArray) -> None:
        got = box_mean(sst.astype(np.float32), (-20.0, 20.0))
        assert got.dtype == np.float64
        want = numpy_box_mean(sst.astype(np.float32), (-20.0, 20.0), None)
        np.testing.assert_allclose(got, want, rtol=1e-13)

    def test_empty_box_raises(self, sst: xr.DataArray) -> None:
        with pytest.raises(ValueError, match="no gridpoint"):
            box_mean(sst, (-5.0, 5.0), (-170.0, -120.0))

    def test_custom_dimension_names(self, sst: xr.DataArray) -> None:
        renamed = sst.rename(lat="latitude", lon="longitude")
        got = box_mean(
            renamed,
            (-5.0, 5.0),
            (190.0, 240.0),
            lat_str="latitude",
            lon_str="longitude",
        )
        np.testing.assert_allclose(
            got, box_mean(sst, (-5.0, 5.0), (190.0, 240.0)), rtol=1e-14
        )


class TestBoxes:
    def test_nino34_box(self, sst: xr.DataArray) -> None:
        np.testing.assert_allclose(
            nino34(sst), numpy_box_mean(sst, (-5.0, 5.0), (190.0, 240.0)), rtol=1e-13
        )

    def test_tropical_mean_box(self, sst: xr.DataArray) -> None:
        np.testing.assert_allclose(
            tropical_mean(sst), numpy_box_mean(sst, (-20.0, 20.0), None), rtol=1e-13
        )

    def test_tio_box(self, sst: xr.DataArray) -> None:
        np.testing.assert_allclose(
            tio(sst), numpy_box_mean(sst, (-20.0, 20.0), (50.0, 110.0)), rtol=1e-13
        )

    def test_coordinate_names_pass_through(self, sst: xr.DataArray) -> None:
        renamed = sst.rename(lat="latitude", lon="longitude")
        np.testing.assert_allclose(
            tio(renamed, lat_str="latitude", lon_str="longitude"), tio(sst), rtol=1e-14
        )


class TestMonthlyAnomaly:
    def test_known_value_nondefault_base(self, sst: xr.DataArray) -> None:
        series = nino34(sst)
        years, months = time_parts(series)
        got = monthly_anomaly(series, base=(1960, 1979))
        want = numpy_monthly_anomaly(series.to_numpy(), years, months, (1960, 1979))
        np.testing.assert_allclose(got, want, rtol=0, atol=ATOL)

    def test_default_base_is_1991_2020(self, sst: xr.DataArray) -> None:
        series = nino34(sst)
        years, months = time_parts(series)
        want = numpy_monthly_anomaly(series.to_numpy(), years, months, (1991, 2020))
        np.testing.assert_allclose(monthly_anomaly(series), want, rtol=0, atol=ATOL)

    def test_base_missing_months_raises(self, sst: xr.DataArray) -> None:
        with pytest.raises(ValueError, match="twelve months"):
            monthly_anomaly(nino34(sst), base=(2030, 2040))

    def test_keeps_the_time_axis_only(self, sst: xr.DataArray) -> None:
        got = monthly_anomaly(nino34(sst))
        assert got.dims == ("time",)
        assert "month" not in got.coords


class TestRoniRatio:
    def test_known_value_nondefault_years(self, sst: xr.DataArray) -> None:
        n34a = monthly_anomaly(nino34(sst))
        trop = monthly_anomaly(tropical_mean(sst))
        years, months = time_parts(n34a)
        got = roni_ratio(n34a, trop, years=(1955, 1990))
        want = numpy_ratio(
            n34a.to_numpy(), trop.to_numpy(), years, months, (1955, 1990)
        )
        np.testing.assert_allclose(got.sel(month=np.arange(1, 13)), want, rtol=1e-12)

    def test_default_years_are_1950_2020(self, sst: xr.DataArray) -> None:
        n34a = monthly_anomaly(nino34(sst))
        trop = monthly_anomaly(tropical_mean(sst))
        years, months = time_parts(n34a)
        want = numpy_ratio(
            n34a.to_numpy(), trop.to_numpy(), years, months, (1950, 2020)
        )
        np.testing.assert_allclose(
            roni_ratio(n34a, trop).sel(month=np.arange(1, 13)), want, rtol=1e-12
        )


class TestRoni:
    def test_known_value(self, sst: xr.DataArray) -> None:
        n34 = numpy_box_mean(sst, (-5.0, 5.0), (190.0, 240.0))
        trop = numpy_box_mean(sst, (-20.0, 20.0), None)
        years, months = time_parts(sst)
        n34a = numpy_monthly_anomaly(n34, years, months, (1980, 2009))
        tropa = numpy_monthly_anomaly(trop, years, months, (1980, 2009))
        ratio = numpy_ratio(n34a, tropa, years, months, (1960, 2005))
        want = (running3(n34a) - running3(tropa)) * ratio[months - 1]
        got = roni(sst, base=(1980, 2009), sigma_years=(1960, 2005))
        np.testing.assert_allclose(got, want, rtol=0, atol=ATOL)

    def test_indexed_by_center_month(self, sst: xr.DataArray) -> None:
        got = roni(sst)
        # The July value is the June-August mean, scaled by July's ratio.
        n34a = monthly_anomaly(nino34(sst))
        trop = monthly_anomaly(tropical_mean(sst))
        ratio = roni_ratio(n34a, trop).sel(month=7).item()
        jja = (n34a - trop).sel(time=slice("2000-06-01", "2000-08-01")).mean().item()
        np.testing.assert_allclose(
            got.sel(time="2000-07-01").item(), jja * ratio, rtol=1e-12
        )

    def test_ends_are_nan(self, sst: xr.DataArray) -> None:
        got = roni(sst)
        assert np.isnan(got.isel(time=0)) and np.isnan(got.isel(time=-1))
        assert np.isfinite(got.isel(time=slice(1, -1))).all()


class TestRelativeNino34:
    def test_unscaled_is_the_difference_of_anomalies(self, sst: xr.DataArray) -> None:
        n34 = numpy_box_mean(sst, (-5.0, 5.0), (190.0, 240.0))
        trop = numpy_box_mean(sst, (-20.0, 20.0), None)
        years, months = time_parts(sst)
        want = numpy_monthly_anomaly(
            n34, years, months, (1991, 2020)
        ) - numpy_monthly_anomaly(trop, years, months, (1991, 2020))
        np.testing.assert_allclose(
            relative_nino34(sst, scaled=False), want, rtol=0, atol=ATOL
        )

    def test_scaled_known_value(self, sst: xr.DataArray) -> None:
        n34 = numpy_box_mean(sst, (-5.0, 5.0), (190.0, 240.0))
        trop = numpy_box_mean(sst, (-20.0, 20.0), None)
        years, months = time_parts(sst)
        n34a = numpy_monthly_anomaly(n34, years, months, (1975, 2004))
        tropa = numpy_monthly_anomaly(trop, years, months, (1975, 2004))
        ratio = numpy_ratio(n34a, tropa, years, months, (1952, 2015))
        want = (n34a - tropa) * ratio[months - 1]
        got = relative_nino34(sst, base=(1975, 2004), sigma_years=(1952, 2015))
        np.testing.assert_allclose(got, want, rtol=0, atol=ATOL)

    def test_jjas_mean(self, sst: xr.DataArray) -> None:
        monthly = relative_nino34(sst).to_numpy()
        years, months = time_parts(sst)
        want = [
            monthly[(years == y) & np.isin(months, (6, 7, 8, 9))].mean()
            for y in range(1946, 2023)
        ]
        got = relative_nino34_jjas(sst)
        np.testing.assert_array_equal(got["year"], np.arange(1946, 2023))
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_jjas_mean_unscaled(self, sst: xr.DataArray) -> None:
        np.testing.assert_allclose(
            relative_nino34_jjas(sst, scaled=False),
            season_mean(relative_nino34(sst, scaled=False), JJAS),
            rtol=1e-14,
        )
        assert not np.allclose(
            relative_nino34_jjas(sst, scaled=False), relative_nino34_jjas(sst)
        )


class TestSeasonMean:
    @pytest.fixture
    def monthly(self) -> xr.DataArray:
        time = xr.date_range("2000-01-01", periods=36, freq="MS")
        return xr.DataArray(
            np.arange(36, dtype=float), dims=["time"], coords={"time": time}
        )

    def test_jjas(self, monthly: xr.DataArray) -> None:
        got = season_mean(monthly, JJAS)
        np.testing.assert_array_equal(got["year"], [2000, 2001, 2002])
        np.testing.assert_allclose(got, [6.5, 18.5, 30.5])

    def test_djf_belongs_to_the_year_of_january(self, monthly: xr.DataArray) -> None:
        got = season_mean(monthly, DJF)
        # 2000 lacks its December (1999), so the first complete winter is
        # Dec 2000 (11) + Jan 2001 (12) + Feb 2001 (13).
        np.testing.assert_array_equal(got["year"], [2001, 2002])
        np.testing.assert_allclose(got, [12.0, 24.0])

    def test_two_months_before_the_wrap(self, monthly: xr.DataArray) -> None:
        got = season_mean(monthly, (11, 12, 1))
        np.testing.assert_array_equal(got["year"], [2001, 2002])
        np.testing.assert_allclose(got, [11.0, 23.0])

    def test_nan_month_drops_the_season(self, monthly: xr.DataArray) -> None:
        holed = monthly.where(monthly["time"] != np.datetime64("2001-07-01"))
        got = season_mean(holed, JJAS)
        np.testing.assert_array_equal(got["year"], [2000, 2002])

    def test_custom_dimension_names(self, monthly: xr.DataArray) -> None:
        renamed = monthly.rename(time="t")
        got = season_mean(renamed, JJAS, time_str="t", year_str="yr")
        assert got.dims == ("yr",)
        np.testing.assert_allclose(got, [6.5, 18.5, 30.5])
