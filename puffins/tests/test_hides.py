"""Tests for hides module."""
import numpy as np
import xarray as xr

from puffins.constants import RAD_EARTH, ROT_RATE_EARTH
from puffins.hides import (
    _flip_dim,
    _flip_lats,
    _maybe_flip_lats,
    hides_above_eq_mom,
    hides_negative,
    hides_vort_zero_cross,
)
from puffins.names import LAT_STR


def _make_lat_array(
    values: list[float],
    lats: list[float] | None = None,
) -> xr.DataArray:
    """Create a 1-D DataArray along the latitude dimension."""
    if lats is None:
        lats = list(np.linspace(-90, 90, len(values)))
    return xr.DataArray(values, dims=[LAT_STR], coords={LAT_STR: lats})


class TestFlipDim:
    """Tests for _flip_dim."""

    def test_reverses_values(self) -> None:
        arr = _make_lat_array([1.0, 2.0, 3.0])
        flipped = _flip_dim(arr, LAT_STR)
        np.testing.assert_array_equal(flipped.values, [3.0, 2.0, 1.0])

    def test_reverses_coords(self) -> None:
        arr = _make_lat_array([1.0, 2.0, 3.0], lats=[-90.0, 0.0, 90.0])
        flipped = _flip_dim(arr, LAT_STR)
        np.testing.assert_array_equal(flipped[LAT_STR].values, [90.0, 0.0, -90.0])

    def test_double_flip_is_identity(self) -> None:
        arr = _make_lat_array([1.0, 2.0, 3.0, 4.0])
        double_flipped = _flip_dim(_flip_dim(arr, LAT_STR), LAT_STR)
        np.testing.assert_array_equal(double_flipped.values, arr.values)


class TestFlipLats:
    """Tests for _flip_lats."""

    def test_reverses_lat_dim(self) -> None:
        arr = _make_lat_array([10.0, 20.0, 30.0])
        flipped = _flip_lats(arr)
        np.testing.assert_array_equal(flipped.values, [30.0, 20.0, 10.0])


class TestMaybeFlipLats:
    """Tests for _maybe_flip_lats."""

    def test_flips_when_true(self) -> None:
        arr = _make_lat_array([1.0, 2.0, 3.0])
        result = _maybe_flip_lats(arr, do_flip=True)
        np.testing.assert_array_equal(result.values, [3.0, 2.0, 1.0])

    def test_no_flip_when_false(self) -> None:
        arr = _make_lat_array([1.0, 2.0, 3.0])
        result = _maybe_flip_lats(arr, do_flip=False)
        np.testing.assert_array_equal(result.values, [1.0, 2.0, 3.0])


class TestHidesAboveEqMom:
    """Tests for hides_above_eq_mom."""

    def test_finds_poleward_latitude(self) -> None:
        eq_mom = ROT_RATE_EARTH * RAD_EARTH**2
        lats = [-30.0, -15.0, 0.0, 15.0, 30.0]
        # Values above eq_mom at lats -15 and 0, below elsewhere.
        values = [eq_mom * 0.5, eq_mom * 1.5, eq_mom * 1.2, eq_mom * 0.8, eq_mom * 0.3]
        arr = _make_lat_array(values, lats=lats)
        result = hides_above_eq_mom(arr)
        assert float(result) == 0.0  # Last lat where mom > eq_mom

    def test_returns_dataarray(self) -> None:
        eq_mom = ROT_RATE_EARTH * RAD_EARTH**2
        lats = [-10.0, 0.0, 10.0]
        values = [eq_mom * 1.5, eq_mom * 1.2, eq_mom * 0.5]
        arr = _make_lat_array(values, lats=lats)
        result = hides_above_eq_mom(arr)
        assert isinstance(result, xr.DataArray)


class TestHidesNegative:
    """Tests for hides_negative."""

    def test_finds_nan_latitude(self) -> None:
        lats = [-30.0, -15.0, 0.0, 15.0, 30.0]
        values = [1.0, np.nan, np.nan, 2.0, 3.0]
        arr = _make_lat_array(values, lats=lats)
        result = hides_negative(arr)
        assert float(result) == 0.0  # Last NaN latitude

    def test_returns_dataarray(self) -> None:
        lats = [-10.0, 0.0, 10.0]
        values = [np.nan, 1.0, 2.0]
        arr = _make_lat_array(values, lats=lats)
        result = hides_negative(arr)
        assert isinstance(result, xr.DataArray)


class TestHidesVortZeroCross:
    """Tests for hides_vort_zero_cross."""

    def test_finds_sign_change(self) -> None:
        lats = [-30.0, -15.0, 0.0, 15.0, 30.0]
        # Sign changes between lat -15 (positive) and 0 (negative),
        # and between 15 (negative) and 30 (positive).
        values = [1.0, 2.0, -1.0, -2.0, 3.0]
        arr = _make_lat_array(values, lats=lats)
        result = hides_vort_zero_cross(arr)
        # The last latitude at a sign-change boundary is 30.0
        # (diff detects the change at the latter element of each pair).
        assert float(result) == 30.0

    def test_returns_dataarray(self) -> None:
        lats = [-10.0, 0.0, 10.0]
        values = [1.0, -1.0, 1.0]
        arr = _make_lat_array(values, lats=lats)
        result = hides_vort_zero_cross(arr)
        assert isinstance(result, xr.DataArray)
