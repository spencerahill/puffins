"""Tests for the enso module.

Expected values are rebuilt from raw numpy on a small synthetic SST field, not
from the module's own helpers, so that each test pins a coefficient, a bound or
an offset rather than restating the code.
"""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import xarray as xr

from puffins.dates import ann_subset_ts, monthly_anom
from puffins.enso import (
    CPC_SEASONS,
    RONI_BASE,
    RONI_SIGMA_YEARS,
    box_mean,
    nino34,
    read_cpc_index,
    relative_nino34,
    relative_nino34_jjas,
    roni,
    roni_ratio,
    tio,
    tropical_mean,
)
from puffins.longitude import lon_to_pm180

# Gridpoints sit on each box bound (5 and 20 degrees; 50, 110, 190 and 240E)
# and just outside it (5.5 and 20.5; 45, 115, 185 and 245E), on a full ring of
# longitudes, so a box drawn one gridpoint too wide or too narrow changes the
# mean.
LATS = np.sort(
    np.concatenate([np.arange(-24.0, 25.0, 2.0), [-20.5, -5.5, -5.0, 5.0, 5.5, 20.5]])
)
LONS = np.sort(
    np.concatenate([np.arange(0.0, 360.0, 10.0), [45.0, 115.0, 185.0, 245.0]])
)
N_MONTHS = 12 * (2022 - 1946 + 1)
NINO34_BOX = ((-5.0, 5.0), (190.0, 240.0))
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


@pytest.fixture
def renamed(sst: xr.DataArray) -> xr.DataArray:
    """The same field with non-default time, latitude and longitude names."""
    return sst.rename(time="t", lat="latitude", lon="longitude")


def numpy_box_mean(
    da: xr.DataArray, lat_b: tuple[float, float], lon_b: tuple[float, float] | None
) -> np.ndarray:
    lat = da["lat"].to_numpy().astype(np.float64)
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


def numpy_relative(
    da: xr.DataArray, base: tuple[int, int], span: tuple[int, int] | None
) -> np.ndarray:
    """Monthly relative NINO3.4 anomaly, scaled by the ratio over ``span``."""
    n34 = numpy_box_mean(da, *NINO34_BOX)
    trop = numpy_box_mean(da, (-20.0, 20.0), None)
    years, months = time_parts(da)
    n34a = numpy_monthly_anomaly(n34, years, months, base)
    tropa = numpy_monthly_anomaly(trop, years, months, base)
    if span is None:
        return cast(np.ndarray, n34a - tropa)
    ratio = numpy_ratio(n34a, tropa, years, months, span)
    return cast(np.ndarray, (n34a - tropa) * ratio[months - 1])


def time_parts(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    return da["time"].dt.year.to_numpy(), da["time"].dt.month.to_numpy()


def anomaly(series: xr.DataArray) -> xr.DataArray:
    """Anomaly from the RONI base climatology, through dates.monthly_anom."""
    return monthly_anom(series, base=RONI_BASE).drop_vars("month")


class TestBoxMean:
    def test_known_value(self, sst: xr.DataArray) -> None:
        got = box_mean(sst, *NINO34_BOX)
        np.testing.assert_allclose(got, numpy_box_mean(sst, *NINO34_BOX), rtol=1e-13)

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
            box_mean(flipped, *NINO34_BOX), box_mean(sst, *NINO34_BOX), rtol=1e-13
        )

    def test_reversed_latitude_bounds(self, sst: xr.DataArray) -> None:
        got = box_mean(sst, (5.0, -5.0), (190.0, 240.0))
        np.testing.assert_allclose(got, numpy_box_mean(sst, *NINO34_BOX), rtol=1e-13)

    def test_field_on_pm180_longitudes(self, sst: xr.DataArray) -> None:
        pm = sst.assign_coords(lon=lon_to_pm180(sst["lon"])).sortby("lon")
        assert float(pm["lon"].min()) < 0
        got = box_mean(pm, *NINO34_BOX)
        np.testing.assert_allclose(got, numpy_box_mean(sst, *NINO34_BOX), rtol=1e-13)

    def test_bounds_on_pm180_longitudes(self, sst: xr.DataArray) -> None:
        got = box_mean(sst, (-5.0, 5.0), (-170.0, -120.0))
        np.testing.assert_allclose(got, numpy_box_mean(sst, *NINO34_BOX), rtol=1e-13)

    @pytest.fixture
    def ring(self) -> xr.DataArray:
        """One equatorial row on 0-330E, each cell's value its index."""
        lons = np.arange(0.0, 360.0, 30.0)
        return xr.DataArray(
            np.arange(lons.size, dtype=float)[None, None, :],
            dims=["time", "lat", "lon"],
            coords={"time": [0], "lat": [0.0], "lon": lons},
        )

    def test_box_wrapping_through_0e(self, ring: xr.DataArray) -> None:
        # 270, 300, 330, 0 and 30E hold 9, 10, 11, 0 and 1.  The unwrapped
        # reading, 30-270E, would give 5, and every longitude 5.5.
        got = box_mean(ring, (-1.0, 1.0), (270.0, 30.0))
        np.testing.assert_allclose(got, [np.mean([9, 10, 11, 0, 1])], rtol=1e-14)

    @pytest.mark.parametrize("bounds", [(0.0, 360.0), (-180.0, 180.0)])
    def test_bounds_360_apart_span_every_longitude(
        self, ring: xr.DataArray, bounds: tuple[float, float]
    ) -> None:
        got = box_mean(ring, (-1.0, 1.0), bounds)
        np.testing.assert_allclose(got, [np.arange(12).mean()], rtol=1e-14)

    def test_float32_input_accumulates_in_float64(self, sst: xr.DataArray) -> None:
        # A float32 latitude, as many netCDF SST files store it, would make the
        # weights float32 without the cast.
        s32 = sst.astype(np.float32).assign_coords(lat=sst["lat"].astype(np.float32))
        got = box_mean(s32, (-20.0, 20.0))
        assert got.dtype == np.float64
        np.testing.assert_allclose(
            got, numpy_box_mean(s32, (-20.0, 20.0), None), rtol=1e-13
        )

    def test_equal_bounds_select_one_meridian(self, ring: xr.DataArray) -> None:
        got = box_mean(ring, (-1.0, 1.0), (60.0, 60.0))
        np.testing.assert_allclose(got, [2.0], rtol=1e-14)

    def test_reversed_bounds_select_the_complementary_arc(
        self, ring: xr.DataArray
    ) -> None:
        # (180, 240) holds 180, 210 and 240E; (240, 180) runs from 240E east
        # through 0E to 180E, so it holds every column except 210E (index 7).
        got = box_mean(ring, (-1.0, 1.0), (240.0, 180.0))
        np.testing.assert_allclose(got, [np.delete(np.arange(12.0), 7).mean()])

    @pytest.mark.parametrize("bounds", [None, (330.0, 30.0)])
    def test_repeated_longitude_raises(
        self, ring: xr.DataArray, bounds: tuple[float, float] | None
    ) -> None:
        # A copy of the 0E column at 360E, as some regridded products carry.
        cyclic = xr.concat([ring, ring.isel(lon=[0]).assign_coords(lon=[360.0])], "lon")
        with pytest.raises(ValueError, match=r"longitudes \[0.0\]"):
            box_mean(cyclic, (-1.0, 1.0), bounds)

    def test_no_longitude_inside_raises(self, ring: xr.DataArray) -> None:
        with pytest.raises(ValueError, match="no gridpoint inside lon"):
            box_mean(ring, (-1.0, 1.0), (5.0, 25.0))

    def test_no_latitude_inside_raises(self, sst: xr.DataArray) -> None:
        with pytest.raises(ValueError, match="no gridpoint inside lat"):
            box_mean(sst, (30.0, 40.0))

    def test_all_nan_box_raises(self, sst: xr.DataArray) -> None:
        with pytest.raises(ValueError, match="is NaN"):
            box_mean(sst.where(sst["lat"] > 10.0), *NINO34_BOX)

    def test_custom_dimension_names(
        self, sst: xr.DataArray, renamed: xr.DataArray
    ) -> None:
        got = box_mean(renamed, *NINO34_BOX, lat_str="latitude", lon_str="longitude")
        np.testing.assert_allclose(got, box_mean(sst, *NINO34_BOX), rtol=1e-14)


class TestBoxes:
    def test_nino34_box(self, sst: xr.DataArray) -> None:
        np.testing.assert_allclose(
            nino34(sst), numpy_box_mean(sst, *NINO34_BOX), rtol=1e-13
        )

    def test_tropical_mean_box(self, sst: xr.DataArray) -> None:
        np.testing.assert_allclose(
            tropical_mean(sst), numpy_box_mean(sst, (-20.0, 20.0), None), rtol=1e-13
        )

    def test_tio_box(self, sst: xr.DataArray) -> None:
        np.testing.assert_allclose(
            tio(sst), numpy_box_mean(sst, (-20.0, 20.0), (50.0, 110.0)), rtol=1e-13
        )

    def test_coordinate_names_pass_through(
        self, sst: xr.DataArray, renamed: xr.DataArray
    ) -> None:
        np.testing.assert_allclose(
            tio(renamed, lat_str="latitude", lon_str="longitude"), tio(sst), rtol=1e-14
        )


class TestOutputNames:
    @pytest.mark.parametrize(
        ("func", "name"),
        [
            (nino34, "nino34"),
            (tropical_mean, "tropical_mean"),
            (tio, "tio"),
            (roni, "roni"),
            (relative_nino34, "relative_nino34"),
            (relative_nino34_jjas, "relative_nino34_jjas"),
        ],
    )
    def test_name_and_no_month_coordinate(
        self, sst: xr.DataArray, func: Callable[..., xr.DataArray], name: str
    ) -> None:
        got = func(sst)
        assert got.name == name
        assert "month" not in got.coords

    def test_roni_ratio_name(self, sst: xr.DataArray) -> None:
        got = roni_ratio(anomaly(nino34(sst)), anomaly(tropical_mean(sst)))
        assert got.name == "roni_ratio"


class TestRoniRatio:
    def test_known_value_nondefault_years(self, sst: xr.DataArray) -> None:
        n34a = anomaly(nino34(sst))
        trop = anomaly(tropical_mean(sst))
        years, months = time_parts(n34a)
        got = roni_ratio(n34a, trop, years=(1955, 1990))
        want = numpy_ratio(
            n34a.to_numpy(), trop.to_numpy(), years, months, (1955, 1990)
        )
        np.testing.assert_allclose(got.sel(month=np.arange(1, 13)), want, rtol=1e-12)

    def test_default_years_are_1950_2020(self, sst: xr.DataArray) -> None:
        n34a = anomaly(nino34(sst))
        trop = anomaly(tropical_mean(sst))
        years, months = time_parts(n34a)
        want = numpy_ratio(
            n34a.to_numpy(), trop.to_numpy(), years, months, (1950, 2020)
        )
        np.testing.assert_allclose(
            roni_ratio(n34a, trop).sel(month=np.arange(1, 13)), want, rtol=1e-12
        )

    def test_custom_time_dim(self, sst: xr.DataArray) -> None:
        n34a = anomaly(nino34(sst))
        trop = anomaly(tropical_mean(sst))
        got = roni_ratio(n34a.rename(time="t"), trop.rename(time="t"), time_str="t")
        np.testing.assert_allclose(got, roni_ratio(n34a, trop), rtol=1e-14)

    def test_gap_in_time_axis_raises(self, sst: xr.DataArray) -> None:
        gap = np.datetime64("2000-07-01")
        n34a = anomaly(nino34(sst)).drop_sel(time=gap)
        trop = anomaly(tropical_mean(sst)).drop_sel(time=gap)
        with pytest.raises(ValueError, match="no gaps"):
            roni_ratio(n34a, trop)

    def test_years_outside_record_raise(self, sst: xr.DataArray) -> None:
        n34a = anomaly(nino34(sst))
        trop = anomaly(tropical_mean(sst))
        with pytest.raises(ValueError, match="sigma_years 1930-1960"):
            roni_ratio(n34a, trop, years=(1930, 1960))


class TestRoni:
    def test_known_value(self, sst: xr.DataArray) -> None:
        n34 = numpy_box_mean(sst, *NINO34_BOX)
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
        n34a = anomaly(nino34(sst))
        trop = anomaly(tropical_mean(sst))
        ratio = roni_ratio(n34a, trop).sel(month=7).item()
        jja = (n34a - trop).sel(time=slice("2000-06-01", "2000-08-01")).mean().item()
        np.testing.assert_allclose(
            got.sel(time="2000-07-01").item(), jja * ratio, rtol=1e-12
        )

    def test_ends_are_nan(self, sst: xr.DataArray) -> None:
        got = roni(sst)
        assert np.isnan(got.isel(time=0)) and np.isnan(got.isel(time=-1))
        assert np.isfinite(got.isel(time=slice(1, -1))).all()

    def test_custom_names(self, sst: xr.DataArray, renamed: xr.DataArray) -> None:
        got = roni(renamed, time_str="t", lat_str="latitude", lon_str="longitude")
        assert got.dims == ("t",)
        np.testing.assert_allclose(got, roni(sst), rtol=1e-14)

    def test_gap_outside_base_and_sigma_years_raises(self, sst: xr.DataArray) -> None:
        # July 2021 lies outside both default periods, so only the running
        # mean's gap check can catch it.
        with pytest.raises(ValueError, match="no gaps"):
            roni(sst.drop_sel(time=np.datetime64("2021-07-01")))

    def test_base_beyond_record_raises(self, sst: xr.DataArray) -> None:
        with pytest.raises(ValueError, match="base years 2010-2030"):
            roni(sst, base=(2010, 2030))


class TestRelativeNino34:
    def test_unscaled_is_the_difference_of_anomalies(self, sst: xr.DataArray) -> None:
        want = numpy_relative(sst, (1991, 2020), None)
        np.testing.assert_allclose(
            relative_nino34(sst, scaled=False), want, rtol=0, atol=ATOL
        )

    def test_scaled_known_value(self, sst: xr.DataArray) -> None:
        want = numpy_relative(sst, (1975, 2004), (1952, 2015))
        got = relative_nino34(sst, base=(1975, 2004), sigma_years=(1952, 2015))
        np.testing.assert_allclose(got, want, rtol=0, atol=ATOL)

    def test_default_years(self, sst: xr.DataArray) -> None:
        want = numpy_relative(sst, RONI_BASE, RONI_SIGMA_YEARS)
        np.testing.assert_allclose(relative_nino34(sst), want, rtol=0, atol=ATOL)

    def test_positional_years_follow_roni(self, sst: xr.DataArray) -> None:
        # Positionally the second and third arguments are base and sigma_years,
        # as in roni; a base tuple used to be read as `scaled`.
        got = relative_nino34(sst, (1975, 2004), (1952, 2015))
        want = numpy_relative(sst, (1975, 2004), (1952, 2015))
        np.testing.assert_allclose(got, want, rtol=0, atol=ATOL)

    def test_scaled_is_keyword_only(self, sst: xr.DataArray) -> None:
        with pytest.raises(TypeError):
            relative_nino34(sst, RONI_BASE, RONI_SIGMA_YEARS, "time", False)  # type: ignore[misc]

    def test_custom_names(self, sst: xr.DataArray, renamed: xr.DataArray) -> None:
        got = relative_nino34(
            renamed, time_str="t", lat_str="latitude", lon_str="longitude"
        )
        np.testing.assert_allclose(got, relative_nino34(sst), rtol=1e-14)

    def test_gap_raises_only_when_scaled(self, sst: xr.DataArray) -> None:
        gappy = sst.drop_sel(time=np.datetime64("2021-07-01"))
        with pytest.raises(ValueError, match="no gaps"):
            relative_nino34(gappy)
        assert relative_nino34(gappy, scaled=False).sizes["time"] == N_MONTHS - 1

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
            ann_subset_ts(relative_nino34(sst, scaled=False), "jjas"),
            rtol=1e-14,
        )
        assert not np.allclose(
            relative_nino34_jjas(sst, scaled=False), relative_nino34_jjas(sst)
        )

    def test_jjas_forwards_its_arguments(
        self, sst: xr.DataArray, renamed: xr.DataArray
    ) -> None:
        got = relative_nino34_jjas(
            renamed,
            (1975, 2004),
            (1952, 2015),
            "t",
            lat_str="latitude",
            lon_str="longitude",
        )
        monthly = numpy_relative(sst, (1975, 2004), (1952, 2015))
        years, months = time_parts(sst)
        want = [
            monthly[(years == y) & np.isin(months, (6, 7, 8, 9))].mean()
            for y in range(1946, 2023)
        ]
        np.testing.assert_allclose(got, want, rtol=0, atol=ATOL)

    def test_jjas_incomplete_first_year_is_nan(self, sst: xr.DataArray) -> None:
        # Starting in July 1946 leaves the 1946 season without its June.
        got = relative_nino34_jjas(sst.isel(time=slice(6, None)))
        np.testing.assert_array_equal(got["year"], np.arange(1946, 2023))
        assert np.isnan(got.sel(year=1946).item())
        assert np.isfinite(got.sel(year=slice(1947, None))).all()


# The opening rows of CPC's oni.ascii.txt as published, with a made-up OND and
# NDJ for 1949 before them so the table crosses a year end.
ONI_TABLE = """\
 SEAS  YR   TOTAL   ANOM
  OND 1949  25.20  -0.95
  NDJ 1949  25.10  -1.05
  DJF 1950  25.01  -1.32
  JFM 1950  25.36  -1.20
  FMA 1950  25.88  -1.12
"""
RONI_TABLE = """\
SEAS   YR  ANOM
DJF  1950 -1.19
JFM  1950 -1.08
FMA  1950 -0.96
"""


class TestReadCpcIndex:
    """CPC's season tables, each value at the first day of its center month."""

    @staticmethod
    def write(tmp_path: Path, text: str) -> Path:
        path = tmp_path / "table.txt"
        path.write_text(text)
        return path

    def test_oni_anomalies(self, tmp_path: Path) -> None:
        got = read_cpc_index(self.write(tmp_path, ONI_TABLE))
        np.testing.assert_array_equal(got, [-0.95, -1.05, -1.32, -1.20, -1.12])
        np.testing.assert_array_equal(
            got["time"],
            np.array(
                ["1949-11-01", "1949-12-01", "1950-01-01", "1950-02-01", "1950-03-01"],
                dtype="datetime64[ns]",
            ),
        )
        assert got.name == "anom"
        assert got.dims == ("time",)

    def test_other_column(self, tmp_path: Path) -> None:
        got = read_cpc_index(self.write(tmp_path, ONI_TABLE), column="TOTAL")
        np.testing.assert_array_equal(got, [25.20, 25.10, 25.01, 25.36, 25.88])
        assert got.name == "total"

    def test_roni_table(self, tmp_path: Path) -> None:
        """A table without TOTAL, given as a string path."""
        got = read_cpc_index(str(self.write(tmp_path, RONI_TABLE)))
        np.testing.assert_array_equal(got, [-1.19, -1.08, -0.96])
        assert str(got["time"].values[0])[:10] == "1950-01-01"

    def test_time_str(self, tmp_path: Path) -> None:
        got = read_cpc_index(self.write(tmp_path, RONI_TABLE), time_str="t")
        assert got.dims == ("t",)

    def test_seasons_centered_like_a_running_mean(self, tmp_path: Path) -> None:
        """Each row lands where a centered 3-month mean of its months does.

        A table is written from the running mean of a random monthly series,
        each season labeled by the initials of its three months and the year
        of the middle one.  It reads back onto the running mean's own time
        axis, the year-end seasons included.

        """
        rng = np.random.default_rng(3)
        time = xr.date_range("1949-01-01", periods=48, freq="MS")
        run = running3(rng.normal(size=48))
        initials = "JFMAMJJASOND"
        lines = ["SEAS YR ANOM"] + [
            "".join(initials[(t.month - 1 + k) % 12] for k in (-1, 0, 1))
            + f" {t.year} {float(v)!r}"
            for t, v in zip(time.to_pydatetime(), run)
            if np.isfinite(v)
        ]
        got = read_cpc_index(self.write(tmp_path, "\n".join(lines)))
        want = xr.DataArray(run, dims=["time"], coords={"time": time}).isel(
            time=slice(1, -1)
        )
        xr.testing.assert_equal(got.rename(None), want)

    def test_season_labels_in_order_of_center_month(self) -> None:
        initials = "JFMAMJJASOND"
        want = tuple(
            "".join(initials[(m + k) % 12] for k in (-1, 0, 1)) for m in range(12)
        )
        assert want == CPC_SEASONS

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        got = read_cpc_index(self.write(tmp_path, RONI_TABLE + "\n\n"))
        assert got.size == 3

    @pytest.mark.parametrize(
        ("text", "column", "match"),
        [
            ("YR SEAS ANOM\n1950 DJF -1.19\n", "ANOM", "must begin 'SEAS YR'"),
            ("SEAS YEAR ANOM\nDJF 1950 -1.19\n", "ANOM", "must begin 'SEAS YR'"),
            ("", "ANOM", "must begin 'SEAS YR'"),
            (RONI_TABLE, "TOTAL", "no column 'TOTAL'"),
            (RONI_TABLE, "YR", "no column 'YR'"),
            ("SEAS YR ANOM\n", "ANOM", "no rows"),
            ("SEAS YR ANOM\nDJF 1950\n", "ANOM", "does not match the header"),
            ("SEAS YR ANOM\nDJF 1950 -1.19 0.5\n", "ANOM", "does not match the header"),
            ("SEAS YR ANOM\nDEJ 1950 -1.19\n", "ANOM", "unknown season 'DEJ'"),
            ("SEAS YR ANOM\nDJF 1950 -1.19\nFMA 1950 -0.96\n", "ANOM", "one season"),
            ("SEAS YR ANOM\nJFM 1950 -1.08\nDJF 1950 -1.19\n", "ANOM", "one season"),
            ("SEAS YR ANOM\nDJF 1950 -1.19\nDJF 1950 -1.19\n", "ANOM", "one season"),
            ("SEAS YR ANOM\nNDJ 1950 -0.8\nDJF 1950 -1.19\n", "ANOM", "one season"),
        ],
    )
    def test_malformed_tables_raise(
        self, tmp_path: Path, text: str, column: str, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            read_cpc_index(self.write(tmp_path, text), column=column)
