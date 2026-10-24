"""Tests for the enso module.

Expected values are rebuilt from raw numpy on a small synthetic SST field, not
from the module's own helpers, so that each test pins a coefficient, a bound or
an offset rather than restating the code.
"""

import csv
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import xarray as xr

from puffins.dates import ann_subset_ts, monthly_anom
from puffins.enso import (
    CPC_SEASONS,
    ENSO_THRESHOLD,
    RONI_BASE,
    RONI_SIGMA_YEARS,
    box_mean,
    enso_phase,
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


class TestEnsoPhase:
    """El Niño at or above the threshold, La Niña at or below its negative."""

    VALUES = [-1.5, -0.5, -0.49, 0.0, 0.49, 0.5, 0.51, 1.5, np.nan]

    @staticmethod
    def index(values: list[float]) -> xr.DataArray:
        time = xr.date_range("2000-01-01", periods=len(values), freq="MS")
        return xr.DataArray(values, dims=["time"], coords={"time": time}, name="roni")

    @staticmethod
    def assert_rounds_to(
        index: xr.DataArray, rounded: list[float], decimals: int = 1
    ) -> None:
        """Each value of ``index`` rounds to the one in ``rounded``.

        The rounded value is read back through ``enso_phase``: a value that
        rounds to r meets a threshold of abs(r), and not one a step larger.
        """
        step = 10.0**-decimals
        for i, r in enumerate(rounded):
            one = index.isel(time=[i])
            above = round(abs(r) + step, decimals)
            for threshold, want in ((abs(r), np.sign(r)), (above, 0.0)):
                if threshold > 0:
                    got = enso_phase(one, threshold, decimals=decimals, min_run=1)
                    assert got.item() == want, (float(one.item()), r, threshold)

    def test_inclusive_by_default(self) -> None:
        got = enso_phase(self.index(self.VALUES), decimals=None, min_run=1)
        np.testing.assert_array_equal(got, [-1, -1, 0, 0, 0, 1, 1, 1, np.nan])

    def test_strict(self) -> None:
        got = enso_phase(
            self.index(self.VALUES), inclusive=False, decimals=None, min_run=1
        )
        np.testing.assert_array_equal(got, [-1, 0, 0, 0, 0, 0, 1, 1, np.nan])

    def test_threshold(self) -> None:
        got = enso_phase(
            self.index(self.VALUES), threshold=1.0, decimals=None, min_run=1
        )
        np.testing.assert_array_equal(got, [-1, 0, 0, 0, 0, 0, 0, 1, np.nan])

    def test_strict_with_threshold(self) -> None:
        got = enso_phase(
            self.index([-1.5, -1.0, -0.7, 0.7, 1.0, 1.5]),
            threshold=1.0,
            inclusive=False,
            min_run=1,
        )
        np.testing.assert_array_equal(got, [-1, 0, 0, 0, 0, 1])

    @pytest.mark.parametrize("threshold", [3 * 0.1, 0.3 - 0.1, 0.6, 0.7])
    @pytest.mark.parametrize("inclusive", [True, False])
    def test_threshold_on_the_grid(self, threshold: float, inclusive: bool) -> None:
        """A threshold within float error of the grid acts as the grid value.

        3 * 0.1 is 0.30000000000000004 and 0.3 - 0.1 is 0.19999999999999998.
        """
        t = round(threshold, 1)
        index = self.index([-t - 0.1, -t, t, t + 0.1])
        got = enso_phase(index, threshold, inclusive=inclusive, min_run=1)
        at = 1 if inclusive else 0
        np.testing.assert_array_equal(got, [-1, -at, at, 1])

    def test_default_threshold_is_cpcs(self) -> None:
        assert ENSO_THRESHOLD == 0.5

    CPC_RULES = pytest.mark.parametrize(("decimals", "min_run"), [(None, 1), (1, 3)])

    @CPC_RULES
    def test_coordinates_kept_and_named(
        self, decimals: int | None, min_run: int
    ) -> None:
        index = self.index(self.VALUES).assign_coords(
            season=("time", list("abcdefghi")), member=3
        )
        got = enso_phase(index, decimals=decimals, min_run=min_run)
        xr.testing.assert_identical(got.coords.to_dataset(), index.coords.to_dataset())
        assert got.name == "enso_phase"

    @CPC_RULES
    def test_attributes_dropped(self, decimals: int | None, min_run: int) -> None:
        """A phase has no units, so the index's are not carried over."""
        index = self.index(self.VALUES)
        index.attrs = {"units": "K", "long_name": "RONI"}
        with xr.set_options(keep_attrs=True):
            got = enso_phase(index, decimals=decimals, min_run=min_run)
        assert got.attrs == {}

    def test_any_shape(self) -> None:
        """A field of indices, such as one per ensemble member, keeps its shape."""
        index = xr.DataArray([[0.7, -0.2], [-0.6, np.nan]], dims=["year", "member"])
        got = enso_phase(index, min_run=1)
        assert got.dims == ("year", "member")
        np.testing.assert_array_equal(got, [[1, 0], [-1, np.nan]])

    @pytest.mark.parametrize("min_run", [2, 5])
    def test_no_time_dimension_raises(self, min_run: int) -> None:
        """A run of two or more needs a time dimension to count along."""
        index = xr.DataArray([0.7, -0.6], dims=["member"])
        with pytest.raises(ValueError, match="no 'time' dimension.*min_run=1"):
            enso_phase(index, min_run=min_run)

    def test_gap_in_time_raises(self) -> None:
        """Dropping a month would join the runs on either side of it."""
        index = self.index([0.6, 0.6, 0.6, np.nan, 0.6, 0.6]).dropna("time")
        message = (
            "monthly with no gaps, since min_run counts runs of consecutive "
            "months; pass min_run=1 to label each value alone"
        )
        with pytest.raises(ValueError, match=message):
            enso_phase(index)
        np.testing.assert_array_equal(enso_phase(index, min_run=1), [1] * 5)

    @pytest.mark.parametrize("order", [[0, 1, 1, 2, 3], [4, 3, 2, 1, 0]])
    def test_repeated_or_reversed_months_raise(self, order: list[int]) -> None:
        index = self.index([0.6] * 5).isel(time=order)
        with pytest.raises(ValueError, match="monthly with no gaps"):
            enso_phase(index)

    def test_gap_along_time_str_raises(self) -> None:
        index = self.index([0.6] * 6).rename(time="season").isel(season=[0, 1, 3, 4])
        with pytest.raises(ValueError, match="monthly with no gaps"):
            enso_phase(index, time_str="season")

    @pytest.mark.parametrize("coord", [np.arange(2000, 2005), np.arange(5.0)])
    def test_time_coordinate_not_dates_raises(self, coord: np.ndarray) -> None:
        """Year numbers, say, cannot be checked to step by one month."""
        index = xr.DataArray([0.6] * 5, dims=["year"], coords={"year": coord})
        with pytest.raises(ValueError, match="'year' coordinate must be dates"):
            enso_phase(index, time_str="year")
        np.testing.assert_array_equal(enso_phase(index, min_run=1), [1] * 5)

    def test_yearly_time_raises(self) -> None:
        """A run of five is CPC's rule for monthly seasons."""
        time = xr.date_range("2000-01-01", periods=5, freq="YS")
        index = xr.DataArray([0.6] * 5, dims=["time"], coords={"time": time})
        with pytest.raises(ValueError, match="monthly with no gaps"):
            enso_phase(index)

    def test_gap_in_cftime_time_raises(self) -> None:
        pytest.importorskip("cftime")
        time = xr.date_range(
            "2000-01-01", periods=6, freq="MS", calendar="noleap", use_cftime=True
        )
        index = xr.DataArray([0.6] * 6, dims=["time"], coords={"time": time})
        np.testing.assert_array_equal(enso_phase(index), [1] * 6)
        with pytest.raises(ValueError, match="monthly with no gaps"):
            enso_phase(index.isel(time=[0, 1, 2, 4, 5]))

    def test_time_axis_shorter_than_min_run(self) -> None:
        """Values beyond the record might complete a run, so a warm or cold
        value is NaN; an empty axis gives an empty result."""
        got = enso_phase(self.index([0.6, 0.0, -0.6, -0.6]))
        np.testing.assert_array_equal(got, [np.nan, 0, np.nan, np.nan])
        assert enso_phase(self.index([])).sizes == {"time": 0}

    def test_time_axis_exactly_min_run_long(self) -> None:
        np.testing.assert_array_equal(enso_phase(self.index([0.6] * 5)), [1] * 5)
        np.testing.assert_array_equal(enso_phase(self.index([-0.6] * 4)), [np.nan] * 4)

    def test_non_integer_decimals_or_min_run_raises(self) -> None:
        index = self.index([0.6] * 6)
        with pytest.raises(TypeError, match="decimals must be an integer"):
            enso_phase(index, decimals=1.5)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="min_run must be an integer"):
            enso_phase(index, min_run=2.5)  # type: ignore[arg-type]

    def test_numpy_integer_min_run(self) -> None:
        got = enso_phase(self.index([0.6] * 3 + [0.0]), min_run=np.int64(3))  # type: ignore[arg-type]
        np.testing.assert_array_equal(got, [1, 1, 1, 0])

    def test_inclusive_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            enso_phase(self.index([0.5]), 0.5, False)  # type: ignore[misc]

    @pytest.mark.parametrize("threshold", [0.0, -0.5, np.nan])
    def test_nonpositive_threshold_raises(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="threshold must be positive"):
            enso_phase(self.index([0.5]), threshold=threshold)

    def test_decimals_round_halves_up(self) -> None:
        """0.45 rounds to 0.5 and -0.45 to -0.4, as CPC's tables round."""
        got = enso_phase(
            self.index([-0.46, -0.45, -0.44, 0.44, 0.45, 0.46]), decimals=1, min_run=1
        )
        np.testing.assert_array_equal(got, [-1, 0, 0, 0, 1, 1])

    def test_decimals_with_strict(self) -> None:
        got = enso_phase(
            self.index([-0.56, -0.55, 0.54, 0.55]),
            decimals=1,
            inclusive=False,
            min_run=1,
        )
        np.testing.assert_array_equal(got, [-1, 0, 0, 1])

    def test_decimals_leave_rounded_values(self) -> None:
        """CPC's one-decimal values come through rounding unchanged."""
        values = [k / 10 for k in range(-15, 16)]
        self.assert_rounds_to(self.index(values), values)

    def test_decimals_round_values_below_a_half_down(self) -> None:
        """Only a value within 1e-4 of a step below a half rounds up.

        -0.45002 and 0.44998 sit 2e-4 of a step below theirs.
        """
        self.assert_rounds_to(
            self.index([-0.4501, -0.45002, 0.44998, 0.4499]), [-0.5, -0.5, 0.4, 0.4]
        )

    @pytest.mark.parametrize("to_float64", [False, True])
    def test_decimals_round_float32_halves_up(self, to_float64: bool) -> None:
        """A half stored in float32, as 0.45 is stored as 0.44999999, rounds up,
        and so does one cast from float32 to float64."""
        hundredths = range(-395, 400, 10)
        index = self.index([k / 100 for k in hundredths]).astype(np.float32)
        if to_float64:
            index = index.astype(np.float64)
        self.assert_rounds_to(index, [(k + 5) / 100 for k in hundredths])

    def test_two_decimals_round_halves_up(self) -> None:
        """In float64, -2.555 times 100 is -255.50000000000003, just below -255.5."""
        self.assert_rounds_to(
            self.index([-2.555, -2.365, -2.345, -0.455, 0.455, 2.345, 2.555]),
            [-2.55, -2.36, -2.34, -0.45, 0.46, 2.35, 2.56],
            decimals=2,
        )

    def test_two_decimals_round_float32_halves_up(self) -> None:
        """In float32, 3.905 to 3.995 sit 2.9e-6 to 1.1e-5 of a step below
        their halves, and -3.995 to -3.905 above theirs."""
        thousandths = [*range(-3995, -3900, 10), *range(3905, 4000, 10)]
        index = self.index([k / 1000 for k in thousandths]).astype(np.float32)
        self.assert_rounds_to(index, [(k + 5) / 1000 for k in thousandths], decimals=2)

    @pytest.mark.parametrize(
        ("decimals", "halves", "rounded"),
        [
            (1, [-255.85, 256.05], [-255.8, 256.0]),
            (2, [-31.915, -32.005], [-31.91, -32.01]),
            (3, [-1.9005, 2.0015], [-1.9, 2.001]),
            (4, [-0.22095, -0.25025], [-0.2209, -0.2503]),
        ],
    )
    def test_float32_halves_round_up_only_below_the_documented_limits(
        self, decimals: int, halves: list[float], rounded: list[float]
    ) -> None:
        """The limits the docstring gives for float32 halves: 256, 32, 2, 0.25 K.

        Each first value is the half below its limit that float32 stores
        furthest below its exact value, and it rounds up.  Each second is the
        smallest half above the limit, of either sign, that rounds down.
        """
        index = self.index(halves).astype(np.float32)
        self.assert_rounds_to(index, rounded, decimals=decimals)

    def test_float32_index_meets_a_float64_threshold(self) -> None:
        """The rounded index is float64, so 0.7 in float32 equals 0.7 in float64."""
        index = self.index([-0.7, 0.7]).astype(np.float32)
        got = enso_phase(index, np.float64(0.7), min_run=1)
        np.testing.assert_array_equal(got, [-1, 1])

    def test_threshold_off_the_rounding_grid_raises(self) -> None:
        """Rounding to one decimal would move a threshold of 0.37."""
        index = self.index([-0.37, -0.36, 0.36, 0.37])
        with pytest.raises(ValueError, match="not a multiple of 10"):
            enso_phase(index, 0.37, min_run=1)
        for decimals in (None, 2):
            got = enso_phase(index, 0.37, decimals=decimals, min_run=1)
            np.testing.assert_array_equal(got, [-1, 0, 0, 1])

    @pytest.mark.parametrize("threshold", [0.51, 0.501])
    def test_threshold_just_off_the_grid_raises(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="not a multiple of 10"):
            enso_phase(self.index([0.5]), threshold, min_run=1)

    def test_other_decimals(self) -> None:
        got = enso_phase(
            self.index([-0.51, -0.5, 0.49, 0.5]), 1.0, decimals=0, min_run=1
        )
        np.testing.assert_array_equal(got, [-1, 0, 0, 1])

    def test_min_run(self) -> None:
        """By default a run shorter than five is neutral where values that miss
        the threshold bound it, and NaN at either end of the record, which
        values beyond might lengthen.

        The runs here are 4 warm, 6 warm, 5 cold right after them, and 4 cold.
        """
        values = [0.6] * 4 + [0.0] + [0.6] * 6 + [-0.6] * 5 + [0.0] + [-0.6] * 4
        got = enso_phase(self.index(values))
        want = [np.nan] * 4 + [0] + [1] * 6 + [-1] * 5 + [0] + [np.nan] * 4
        np.testing.assert_array_equal(got, want)

    def test_min_run_two(self) -> None:
        got = enso_phase(self.index([0.0, 0.6, 0.0, 0.6, 0.6, -0.6, 0.0]), min_run=2)
        np.testing.assert_array_equal(got, [0, 0, 0, 1, 1, 0, 0])

    def test_nan_ends_a_run(self) -> None:
        """A NaN might continue the run of three before it, so the warm value
        after it is NaN: it would be El Niño only if the NaN were warm."""
        got = enso_phase(self.index([0.6, 0.6, 0.6, np.nan, 0.6, 0.0]), min_run=3)
        np.testing.assert_array_equal(got, [1, 1, 1, np.nan, np.nan, 0])

    def test_nan_beside_a_short_run(self) -> None:
        """Values beside a NaN are NaN where the NaN might complete a run.

        The warm values on either side of the first NaN would make a run of
        three with it, and so would the cold values on either side of the
        third, while the warm value before the second would make a run of two.
        """
        values = [0.0, 0.6, np.nan, 0.6, 0.0, 0.6, np.nan, 0.0, -0.6, np.nan, -0.6, 0.0]
        got = enso_phase(self.index(values), min_run=3)
        nan = np.nan
        np.testing.assert_array_equal(
            got, [0, nan, nan, nan, 0, 0, nan, 0, nan, nan, nan, 0]
        )

    def test_min_run_along_time_str(self) -> None:
        """Runs are counted along time_str wherever it sits in the dimensions."""
        index = xr.DataArray(
            [[0.6, 0.6, 0.6, 0.0], [0.6, 0.6, 0.0, 0.6]], dims=["member", "season"]
        )
        got = enso_phase(index, min_run=3, time_str="season")
        assert got.dims == ("member", "season")
        nan = np.nan
        np.testing.assert_array_equal(got, [[1, 1, 1, 0], [nan, nan, 0, nan]])
        xr.testing.assert_identical(
            enso_phase(index.T, min_run=3, time_str="season"), got.T
        )

    @pytest.mark.parametrize("min_run", [0, -1])
    def test_min_run_below_one_raises(self, min_run: int) -> None:
        with pytest.raises(ValueError, match="min_run must be at least 1"):
            enso_phase(self.index([0.5]), min_run=min_run)

    def test_reproduces_cpcs_roni_table(self) -> None:
        """The defaults give CPC's coloring of RONI for 1951 and 1952.

        From ``RONI.ascii.txt`` and CPC's RONI table, downloaded 2026-10-23.
        FMA 1951, 0.48, is shown as 0.5 and colored El Niño; FMA 1952, 0.46, is
        shown as 0.5 but is a single season and uncolored; MJJ 1952, -0.45, is
        shown as -0.4.  Neither rule alone gives CPC's colors.
        """
        roni = [
            -0.28, 0.09, 0.48, 0.63, 0.59, 0.69, 0.79, 0.97, 1.04, 0.98, 0.94, 0.71,
            0.65, 0.33, 0.46, 0.24, -0.03, -0.45, -0.41, -0.14, 0.08, 0.04, -0.10, 0.10,
        ]  # fmt: skip
        cpc = [0, 0] + [1] * 11 + [0] * 11
        index = self.index(roni)
        np.testing.assert_array_equal(enso_phase(index), cpc)
        assert not np.array_equal(enso_phase(index, min_run=1), cpc)
        assert not np.array_equal(enso_phase(index, decimals=None), cpc)


class TestEnsoPhaseCPCFiles:
    """The defaults against CPC's files as of 23 October 2026, DJF 1950 to JAS 2026.

    See ``data/cpc/README.md`` for where each file comes from.
    """

    DATA = Path(__file__).parent / "data" / "cpc"
    FILES = {
        "RONI": ("RONI.ascii.txt", "roni-table.csv"),
        "ONI": ("oni.ascii.txt", "oni-table.csv"),
    }
    NAMES = pytest.mark.parametrize("name", ["RONI", "ONI"])

    @staticmethod
    def time(months: list[str]) -> np.ndarray:
        """Months written as YYYY-MM, as datetime64[ns]."""
        return np.array(months, dtype="datetime64[M]").astype("datetime64[ns]")

    @classmethod
    def load(cls, name: str) -> xr.Dataset:
        """The text file's two-decimal values and the table's one-decimal
        values and colors, by each season's center month."""
        text_file, table_file = cls.FILES[name]
        text = read_cpc_index(cls.DATA / text_file)
        with (cls.DATA / table_file).open() as f:
            table = list(csv.DictReader(f))
        months = [
            f"{t['year']}-{CPC_SEASONS.index(t['season']) + 1:02d}" for t in table
        ]
        np.testing.assert_array_equal(text["time"], cls.time(months))
        return xr.Dataset(
            {
                "text": text,
                "shown": ("time", [float(t["shown"]) for t in table]),
                "color": ("time", [float(t["phase"]) for t in table]),
            }
        )

    @pytest.mark.parametrize(("name", "last"), [("RONI", 1.69), ("ONI", 2.16)])
    def test_read_cpc_index_reads_every_season(self, name: str, last: float) -> None:
        """All 920 seasons, DJF 1950 to JAS 2026, at their center months."""
        got = read_cpc_index(self.DATA / self.FILES[name][0])
        assert got.sizes["time"] == 920
        np.testing.assert_array_equal(
            got["time"][[0, -1]], self.time(["1950-01", "2026-08"])
        )
        assert got[-1].item() == last

    @NAMES
    def test_reproduces_cpcs_tables(self, name: str) -> None:
        """From the values the tables show, every season but RONI's last four.

        AMJ to JAS 2026 begin a warm run, which CPC leaves uncolored and
        enso_phase labels NaN.
        """
        seasons = self.load(name)
        assert seasons.sizes["time"] == 920
        want = seasons.color.copy()
        if name == "RONI":
            want[-4:] = np.nan
        np.testing.assert_array_equal(enso_phase(seasons.shown), want)

    @NAMES
    def test_reproduces_cpcs_tables_from_text_files(self, name: str) -> None:
        """The same from the two-decimal files, but for RONI's FMA to MJJ 1984.

        CPC shows FMA's -0.45 as -0.5, which joins the three seasons after it
        to the La Niña run before it; rounded halves up, -0.45 is -0.4.
        """
        seasons = self.load(name)
        want = seasons.color.copy()
        if name == "RONI":
            want[-4:] = np.nan
            spring = slice("1984-03", "1984-06")
            assert seasons.text.sel(time="1984-03").item() == -0.45
            np.testing.assert_array_equal(want.sel(time=spring), [-1] * 4)
            want.loc[spring] = 0
        np.testing.assert_array_equal(enso_phase(seasons.text), want)

    @pytest.mark.parametrize(
        ("name", "up", "halves"), [("RONI", 80, 87), ("ONI", 87, 90)]
    )
    def test_cpcs_tables_round_most_halves_up(
        self, name: str, up: int, halves: int
    ) -> None:
        """CPC's tables show most text values ending in 5 rounded up, and every
        other value rounded to the nearest."""
        seasons = self.load(name)
        hundredths = np.round(seasons.text.values * 100).astype(int)
        tenths = np.round(seasons.shown.values * 10).astype(int)
        half = hundredths % 10 == 5
        assert half.sum() == halves
        assert (10 * tenths[half] == hundredths[half] + 5).sum() == up
        assert (10 * tenths[half] == hundredths[half] - 5).sum() == halves - up
        np.testing.assert_array_equal(tenths[~half], np.round(hundredths[~half] / 10))

    def test_cpcs_oni_file_looks_truncated(self) -> None:
        """CPC's two-decimal ONI values average 0.005 K below the 3-month means
        of its monthly NINO3.4 anomalies, for either sign, as truncation toward
        minus infinity would make them; rounding to the nearest would make that
        average near zero."""
        oni = self.load("ONI").text
        lines = (self.DATA / "detrend.nino34.ascii.txt").read_text().splitlines()[1:]
        rows = [line.split() for line in lines]
        months = [f"{r[0]}-{int(r[1]):02d}" for r in rows]
        monthly = xr.DataArray(
            [float(r[-1]) for r in rows],
            dims=["time"],
            coords={"time": self.time(months)},
        )
        means = monthly.rolling(time=3, center=True).mean().sel(time=oni.time)
        below = (means - oni).values
        assert np.isfinite(below).all()
        for sign in (oni.values > 0, oni.values < 0):
            assert 0.004 < below[sign].mean() < 0.006
        # So read_cpc_index's center-month year is CPC's: with DJF or NDJ
        # placed a year off, those seasons would miss by about 1.5 K rms.
        assert np.abs(below).max() < 0.02


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
        path.write_text(text, encoding="utf-8")
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
        # The resolution of the times roni returns from a decoded SST field.
        assert got["time"].dtype == np.dtype("datetime64[ns]")
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

    def test_byte_order_mark_accepted(self, tmp_path: Path) -> None:
        got = read_cpc_index(self.write(tmp_path, "﻿" + RONI_TABLE))
        np.testing.assert_array_equal(got, [-1.19, -1.08, -0.96])

    def test_not_utf8_raises_with_path(self, tmp_path: Path) -> None:
        """A Latin-1 degree sign, byte 0xb0, is not UTF-8."""
        path = tmp_path / "table.txt"
        path.write_bytes(RONI_TABLE.encode() + b"\xb0\n")
        with pytest.raises(ValueError, match="not UTF-8") as err:
            read_cpc_index(path)
        assert str(path) in str(err.value)

    def test_value_column_named_like_a_key_column(self, tmp_path: Path) -> None:
        """The column is looked up among the value columns only."""
        got = read_cpc_index(
            self.write(tmp_path, "SEAS YR YR\nDJF 1950 -1.19\n"), column="YR"
        )
        np.testing.assert_array_equal(got, [-1.19])

    @pytest.mark.parametrize(
        ("row", "month"),
        [("SON 1677 0.1", "1677-10-01"), ("MAM 2262 0.1", "2262-04-01")],
    )
    def test_years_at_the_ends_of_datetime64_ns(
        self, tmp_path: Path, row: str, month: str
    ) -> None:
        got = read_cpc_index(self.write(tmp_path, f"SEAS YR ANOM\n{row}\n"))
        np.testing.assert_array_equal(
            got["time"], np.array([month], dtype="datetime64[ns]")
        )

    @pytest.mark.parametrize(
        ("text", "column", "match"),
        [
            ("YR SEAS ANOM\n1950 DJF -1.19\n", "ANOM", "must begin 'SEAS YR'"),
            ("SEAS YEAR ANOM\nDJF 1950 -1.19\n", "ANOM", "must begin 'SEAS YR'"),
            ("", "ANOM", "must begin 'SEAS YR'"),
            ("\n  \n\n", "ANOM", "must begin 'SEAS YR'"),
            (RONI_TABLE, "TOTAL", "'TOTAL' is not a value column"),
            (RONI_TABLE, "YR", "'YR' is not a value column"),
            (
                "SEAS YR ANOM ANOM\nDJF 1950 -1.19 0.5\n",
                "ANOM",
                "more than one 'ANOM'",
            ),
            ("SEAS YR ANOM\n", "ANOM", "no rows"),
            ("SEAS YR ANOM\nDJF 1950\n", "ANOM", "does not match the header"),
            (
                "SEAS YR ANOM\nDJF 1950 -1.19\nJFM\n",
                "ANOM",
                "does not match the header",
            ),
            ("SEAS YR ANOM\nDJF 1950 -1.19 0.5\n", "ANOM", "does not match the header"),
            ("SEAS YR ANOM\nDJF 1950 abc\n", "ANOM", "does not parse"),
            ("SEAS YR ANOM\nDJF 19x0 -1.19\n", "ANOM", "does not parse"),
            ("SEAS YR ANOM\nDJF 19_50 -1.19\n", "ANOM", "does not parse"),
            ("SEAS YR ANOM\nDJF 1950 1_0.5\n", "ANOM", "does not parse"),
            ("SEAS YR ANOM\nDJF 1950 nan\n", "ANOM", "does not parse"),
            ("SEAS YR ANOM\nDJF 1950 inf\n", "ANOM", "does not parse"),
            ("SEAS YR ANOM\nDJF 50 -1.19\n", "ANOM", "October 1677 to April 2262"),
            ("SEAS YR ANOM\nASO 1677 0.1\n", "ANOM", "October 1677 to April 2262"),
            ("SEAS YR ANOM\nAMJ 2262 0.1\n", "ANOM", "October 1677 to April 2262"),
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
