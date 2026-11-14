"""Tests for longitude module."""

import math
import operator
import warnings
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
import xarray as xr

from puffins.longitude import Longitude, lon_to_0360, lon_to_pm180

# The float64 just below 180.
_BELOW_180 = float(np.nextafter(180.0, 0.0))

# Each of the six comparison operators, paired with what it returns for two
# floats when either is NaN.
_COMPARISONS: list[tuple[Callable[[Any, Any], Any], bool]] = [
    (operator.lt, False),
    (operator.le, False),
    (operator.gt, False),
    (operator.ge, False),
    (operator.eq, False),
    (operator.ne, True),
]

# Integer dtypes whose every value is tested, and those tested at their
# extremes, with the dtype each gives back: int64 for unsigned and 8-bit
# input (issue #148), the input's own dtype otherwise.
_INT_DTYPES_EVERY_VALUE = [
    (np.int8, np.int64),
    (np.uint8, np.int64),
    (np.int16, np.int16),
    (np.uint16, np.int64),
]
_INT_DTYPES_EXTREMES = [
    (np.int32, np.int32),
    (np.uint32, np.int64),
    (np.int64, np.int64),
    (np.uint64, np.int64),
]


def _int_extremes(dtype: Any) -> list[int]:
    """The ends of ``dtype``'s range, the meridians, and 2**31, 2**32, 2**63."""
    info = np.iinfo(dtype)
    near = [-361, -360, -181, -180, -179, -1, 0, 1, 179, 180, 181, 359, 360, 361]
    near += [2**31 - 1, 2**31, 2**32 - 1, 2**32, 2**63 - 1, 2**63, 2**63 + 1]
    ends = [int(info.min) + k for k in range(3)] + [int(info.max) - k for k in range(3)]
    return sorted({v for v in near + ends if info.min <= v <= info.max})


def _exact_pm180(lons: np.ndarray) -> np.ndarray:
    """``lons`` reduced into [-180, 180) by exact rational arithmetic.

    The reduced value is representable in ``lons``'s dtype, so it is returned
    in that dtype with no rounding, which the function checks.  Its remainder
    is taken on exact fractions, so the reference shares no floating-point
    remainder or floor division with the code under test.
    """
    exact = [(Fraction(float(x)) + 180) % 360 - 180 for x in lons]
    ref = np.array([float(r) for r in exact]).astype(lons.dtype)
    assert all(Fraction(float(r)) == e for r, e in zip(ref, exact, strict=True))
    return ref


def _wrap_points_and_neighbors(dtype: Any) -> np.ndarray:
    """Each multiple of 180 in [-1080, 1080] and its two nearest floats each side.

    The two below 0 are negative values for which ``lon + 360`` rounds to
    exactly 360 (issue #83).
    """
    mults = np.arange(-1080, 1081, 180).astype(dtype)
    down1 = np.nextafter(mults, dtype(-np.inf))
    down2 = np.nextafter(down1, dtype(-np.inf))
    up1 = np.nextafter(mults, dtype(np.inf))
    up2 = np.nextafter(up1, dtype(np.inf))
    return np.concatenate([down2, down1, mults, up1, up2])


def _exact_0360(lons: np.ndarray) -> np.ndarray:
    """``lons`` modulo 360 from exact rational arithmetic, in ``lons``'s dtype.

    Each remainder is computed exactly, rounded to float64 and then to the
    dtype, and a remainder that rounds to 360 is sent to 0.  It shares no
    floor division with the code under test.
    """
    ref = np.array([float(Fraction(float(x)) % 360) for x in lons]).astype(lons.dtype)
    ref[ref == 360] = 0
    return ref


class TestLonTo0360:
    """Tests for lon_to_0360."""

    def test_zero(self) -> None:
        assert lon_to_0360(0) == 0

    def test_positive_in_range(self) -> None:
        assert lon_to_0360(90) == 90

    def test_360_wraps_to_zero(self) -> None:
        assert lon_to_0360(360) == 0

    def test_negative(self) -> None:
        assert lon_to_0360(-90) == 270

    def test_int_stays_int(self) -> None:
        result = lon_to_0360(-90)
        assert type(result) is int
        assert result == 270

    @pytest.mark.parametrize("dtype", [np.uint16, np.uint32, np.uint64])
    def test_unsigned_array(self, dtype: Any) -> None:
        """Unsigned input comes back as int64, as lon_to_pm180's does (#148)."""
        result = lon_to_0360(np.array([0, 180, 359, 360, 720], dtype=dtype))
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64
        np.testing.assert_array_equal(result, [0, 180, 359, 0, 0])

    @pytest.mark.parametrize(("dtype", "out_dtype"), _INT_DTYPES_EVERY_VALUE)
    def test_every_value_of_narrow_int_dtypes(self, dtype: Any, out_dtype: Any) -> None:
        """Every int8, uint8, int16 and uint16 value gives its exact remainder."""
        info = np.iinfo(dtype)
        lons = np.arange(int(info.min), int(info.max) + 1).astype(dtype)
        result = lon_to_0360(lons)
        assert isinstance(result, np.ndarray)
        assert result.dtype == out_dtype
        expected = [v % 360 for v in range(int(info.min), int(info.max) + 1)]
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(("dtype", "out_dtype"), _INT_DTYPES_EXTREMES)
    def test_extremes_of_wide_int_dtypes(self, dtype: Any, out_dtype: Any) -> None:
        """The ends of the 32- and 64-bit ranges, uint64 above 2**63 included."""
        values = _int_extremes(dtype)
        result = lon_to_0360(np.array(values, dtype=dtype))
        assert isinstance(result, np.ndarray)
        assert result.dtype == out_dtype
        np.testing.assert_array_equal(result, [v % 360 for v in values])

    @pytest.mark.parametrize(
        ("dtype", "out_dtype"), _INT_DTYPES_EVERY_VALUE + _INT_DTYPES_EXTREMES
    )
    def test_int_numpy_scalars(self, dtype: Any, out_dtype: Any) -> None:
        """A NumPy integer scalar gives a scalar, int64 for unsigned and 8-bit."""
        for value in (np.iinfo(dtype).min, np.iinfo(dtype).max, 90):
            result = lon_to_0360(dtype(value))
            assert type(result) is out_dtype
            assert result == int(value) % 360

    @pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.uint16, np.uint64])
    def test_int_dataarray_keeps_coords_name_and_attrs(self, dtype: Any) -> None:
        info = np.iinfo(dtype)
        values = [0, 90, int(info.max), int(info.min)]
        arr = xr.DataArray(
            np.array(values, dtype=dtype),
            dims=["x"],
            coords={"x": [5.0, 6.0, 7.0, 8.0]},
            name="lons",
            attrs={"units": "degrees_east"},
        )
        result = lon_to_0360(arr)
        assert isinstance(result, xr.DataArray)
        assert result.dtype == np.int64
        assert result.name == "lons"
        assert result.attrs == {"units": "degrees_east"}
        np.testing.assert_array_equal(result["x"].values, [5.0, 6.0, 7.0, 8.0])
        np.testing.assert_array_equal(result.values, [v % 360 for v in values])

    def test_nan_and_infinity_give_nan(self) -> None:
        """NaN passes through, and an infinity gives NaN, with no warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = lon_to_0360(np.array([np.nan, np.inf, -np.inf, 10.0]))
            assert isinstance(result, np.ndarray)
            np.testing.assert_array_equal(result, [np.nan, np.nan, np.nan, 10.0])
            for value in (math.nan, math.inf, np.float32(np.nan)):
                scalar = lon_to_0360(value)
                assert type(scalar) is type(value)
                assert math.isnan(scalar)

    def test_large_positive(self) -> None:
        assert lon_to_0360(720) == 0

    def test_180(self) -> None:
        assert lon_to_0360(180) == 180

    def test_ndarray(self) -> None:
        result = lon_to_0360(np.array([0, 90, 360, -90]))
        np.testing.assert_array_equal(result, [0, 90, 0, 270])

    def test_dataarray(self) -> None:
        da = xr.DataArray([0, 90, 360, -90], dims=["lon"])
        result = lon_to_0360(da)
        xr.testing.assert_equal(result, xr.DataArray([0, 90, 0, 270], dims=["lon"]))

    # Negative longitudes for which ``lon + 360`` rounds to exactly 360 in
    # float64: -2**-45 is the largest in magnitude and -5e-324 the smallest
    # (issue #83).
    @pytest.mark.parametrize("lon", [-(2.0**-45), -1e-14, -5e-324])
    def test_tiny_negative_wraps_to_zero(self, lon: float) -> None:
        """A negative lon that rounds onto 360 comes back as 0, a float."""
        result = lon_to_0360(lon)
        assert type(result) is float
        assert result == 0

    def test_past_tiny_negative_band_stays_below_360(self) -> None:
        """The next float beyond -2**-45 gives the float just below 360."""
        lon = -np.nextafter(2.0**-45, 1.0)
        assert lon_to_0360(lon) == np.nextafter(360.0, 0.0)

    def test_tiny_negative_float32_array(self) -> None:
        """In float32 the band reaches -2**-16, so -1e-6 also comes back as 0."""
        f32 = np.float32
        past_band = -np.nextafter(f32(2.0**-16), f32(1))
        lons = np.array([-(2.0**-16), -1e-6, past_band], dtype=f32)
        result = lon_to_0360(lons)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [0, 0, np.nextafter(f32(360), f32(0))])

    @pytest.mark.parametrize(
        "dtype", [np.float16, np.float32, np.float64, np.longdouble]
    )
    def test_band_edge_in_each_float_dtype(self, dtype: Any) -> None:
        """The band's edge, minus half the float spacing at 360, gives 0.

        One float beyond the edge gives the float just below 360.  The edge
        is -0.125 in float16, and in longdouble it depends on the platform's
        precision.
        """
        edge = -np.spacing(dtype(360)) / 2
        beyond = np.nextafter(edge, dtype(-np.inf))
        result = lon_to_0360(np.array([edge, beyond], dtype=dtype))
        assert isinstance(result, np.ndarray)
        assert result.dtype == dtype
        expected = np.array([0, np.nextafter(dtype(360), dtype(0))], dtype=dtype)
        np.testing.assert_array_equal(result, expected)

    def test_tiny_negative_numpy_scalar_keeps_type(self) -> None:
        result = lon_to_0360(np.float32(-1e-6))
        assert type(result) is np.float32
        assert result == 0

    def test_tiny_negative_dataarray_keeps_coords_name_and_attrs(self) -> None:
        arr = xr.DataArray(
            [-1e-14, -90.0],
            dims=["x"],
            coords={"x": [5.0, 6.0]},
            name="lons",
            attrs={"units": "degrees_east"},
        )
        result = lon_to_0360(arr)
        assert isinstance(result, xr.DataArray)
        assert result.name == "lons"
        assert result.attrs == {"units": "degrees_east"}
        np.testing.assert_array_equal(result["x"].values, [5.0, 6.0])
        np.testing.assert_array_equal(result.values, [0.0, 270.0])

    @pytest.mark.parametrize(
        ("dtype", "uint"), [(np.float32, np.uint32), (np.float64, np.uint64)]
    )
    def test_values_in_range_returned_exactly(self, dtype: Any, uint: Any) -> None:
        """Values already in [0, 360) come back bit for bit."""
        lons = np.array([0.0, 0.1, 179.9, 359.9], dtype=dtype)
        lons = np.append(lons, np.nextafter(dtype(360), dtype(0)))
        result = lon_to_0360(lons)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result.view(uint), lons.view(uint))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_wrap_points_in_range_and_exact(self, dtype: Any) -> None:
        lons = _wrap_points_and_neighbors(dtype)
        result = lon_to_0360(lons)
        assert isinstance(result, np.ndarray)
        assert result.dtype == dtype
        assert np.all((result >= 0) & (result < 360))
        np.testing.assert_array_equal(result, _exact_0360(lons))

    @pytest.mark.parametrize(
        "lon",
        [
            np.float32(134218448.0),
            np.float32(1.01331035e17),
            np.float64(2.0**60),
            np.float64(-3.818876831979843e34),
        ],
    )
    def test_huge_magnitude_gives_exact_remainder(self, lon: Any) -> None:
        """Far past any physical longitude, the result is the exact remainder.

        ``lon - (lon // 360) * 360`` gives 16 for the first value, whose
        remainder is 8, and values outside [0, 360) for the second and fourth.
        """
        result = lon_to_0360(lon)
        assert type(result) is type(lon)
        assert result == _exact_0360(np.array([lon]))[0]


class TestLonToPm180:
    """Tests for lon_to_pm180."""

    def test_zero(self) -> None:
        assert lon_to_pm180(0) == 0

    def test_positive_east(self) -> None:
        assert lon_to_pm180(90) == 90

    def test_180_becomes_negative(self) -> None:
        assert lon_to_pm180(180) == -180

    def test_270_becomes_negative_90(self) -> None:
        assert lon_to_pm180(270) == -90

    def test_360_wraps_to_zero(self) -> None:
        assert lon_to_pm180(360) == 0

    def test_negative_90(self) -> None:
        assert lon_to_pm180(-90) == -90

    # Inputs spanning several wraps in both directions, with 180 itself and
    # its equivalents (-540, -180, 540), which map to -180, and points either
    # side of each hemisphere boundary.
    _LONS = np.array([-540, -190, -180, -10, 0, 10, 179.5, 180, 190, 359.5, 360, 540])
    _EXPECTED = np.array(
        [-180, 170, -180, -10, 0, 10, 179.5, -180, -170, -0.5, 0, -180]
    )

    def test_numpy_array(self) -> None:
        """Element-wise on a numpy array (issue #17)."""
        np.testing.assert_array_equal(lon_to_pm180(self._LONS), self._EXPECTED)

    def test_dataarray_keeps_coords_and_name(self) -> None:
        """A DataArray comes back as a DataArray with its coordinates."""
        coord = np.arange(self._LONS.size) * 10.0
        arr = xr.DataArray(self._LONS, dims=["x"], coords={"x": coord}, name="lons")
        result = lon_to_pm180(arr)
        assert isinstance(result, xr.DataArray)
        assert result.name == "lons"
        np.testing.assert_array_equal(result["x"].values, coord)
        np.testing.assert_array_equal(result.values, self._EXPECTED)

    def test_scalar_stays_scalar_of_same_type(self) -> None:
        """Python ints stay ints and floats stay floats, not 0-d arrays."""
        result_int = lon_to_pm180(190)
        result_float = lon_to_pm180(359.5)
        assert type(result_int) is int
        assert result_int == -170
        assert type(result_float) is float
        assert result_float == -0.5

    def test_eastern_values_returned_exactly(self) -> None:
        """Values already in [0, 180) come back bit for bit, with no round-off.

        A form such as ``(lon + 180) % 360 - 180`` would return 0.1 as
        0.09999999999999432, which breaks exact coordinate matching.
        """
        lons = np.array([0.1, 45.3, 123.456, 179.9])
        np.testing.assert_array_equal(lon_to_pm180(lons), lons)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    def test_array_dtype_preserved(self, dtype: Any) -> None:
        """The output dtype matches the input's, as lon_to_0360's does."""
        lons = np.array([-190, 10, 190, 350], dtype=dtype)
        result = lon_to_pm180(lons)
        assert isinstance(result, np.ndarray)
        assert result.dtype == lons.dtype
        np.testing.assert_array_equal(result, [170, 10, -170, -10])

    @pytest.mark.parametrize(
        "value", [np.float32(190.0), np.float64(190.0), np.int32(190), np.int64(190)]
    )
    def test_numpy_scalar_type_preserved(self, value: Any) -> None:
        """A numpy scalar comes back as a scalar of the same numpy type."""
        result = lon_to_pm180(value)
        assert type(result) is type(value)
        assert result == -170

    @pytest.mark.parametrize("lon", [-1e-14, -(2.0**-45), -5e-324, -3e-14])
    def test_tiny_negative_returned_exactly(self, lon: float) -> None:
        """A tiny negative lon comes back unchanged, never 0 or 360 (#83, #149)."""
        result = lon_to_pm180(lon)
        assert type(result) is float
        assert result == lon

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_wrap_points_in_range_and_exact(self, dtype: Any) -> None:
        """At the wrap points, the exact value in [-180, 180), unrounded."""
        lons = _wrap_points_and_neighbors(dtype)
        result = lon_to_pm180(lons)
        assert isinstance(result, np.ndarray)
        assert result.dtype == dtype
        assert np.all((result >= -180) & (result < 180))
        np.testing.assert_array_equal(result, _exact_pm180(lons))

    @pytest.mark.parametrize(
        ("dtype", "uint"), [(np.float32, np.uint32), (np.float64, np.uint64)]
    )
    def test_western_tenths_returned_exactly(self, dtype: Any, uint: Any) -> None:
        """Every one of -0.1, -0.2, ..., -180.0 comes back bit for bit (#149).

        Through ``lon % 360`` and back, 928 of the 1,800 float64 values came
        back changed, by up to 2**-45.
        """
        lons = (-np.arange(1, 1801) / 10.0).astype(dtype)
        result = lon_to_pm180(lons)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result.view(uint), lons.view(uint))

    def test_python_float_negative_returned_exactly(self) -> None:
        result = lon_to_pm180(-0.1)
        assert type(result) is float
        assert result == -0.1

    def test_sel_on_converted_coordinate(self) -> None:
        """An exact .sel finds -0.1 on a coordinate passed through (#149)."""
        west = -np.arange(1, 1801) / 10.0
        arr = xr.DataArray(
            np.arange(west.size), dims="lon", coords={"lon": west}, name="vals"
        )
        converted = arr.assign_coords(lon=lon_to_pm180(arr["lon"]))
        assert converted.sel(lon=-0.1).item() == 0
        assert converted.sel(lon=-179.9).item() == 1798

    @pytest.mark.parametrize(
        "lon",
        [
            np.float32(-134218448.0),
            np.float32(1.01331035e17),
            np.float64(2.0**60),
            np.float64(-3.818876831979843e34),
            np.float64(-1e300),
        ],
    )
    def test_huge_magnitude_gives_exact_value(self, lon: Any) -> None:
        """Far past any physical longitude, still the exact value."""
        result = lon_to_pm180(lon)
        assert type(result) is type(lon)
        assert result == _exact_pm180(np.array([lon]))[0]

    def test_signed_zeros_give_positive_zero(self) -> None:
        """-0.0 and the negative multiples of 360 give 0.0, never -0.0."""
        lons = np.array([-0.0, -360.0, -720.0, 0.0, 360.0])
        for dtype in (np.float32, np.float64):
            result = lon_to_pm180(lons.astype(dtype))
            assert isinstance(result, np.ndarray)
            np.testing.assert_array_equal(np.signbit(result), False)
        for value in (-0.0, -360.0, np.float64(-0.0), np.float32(-360.0)):
            scalar = lon_to_pm180(value)
            assert type(scalar) is type(value)
            assert scalar == 0
            assert not np.signbit(scalar)

    def test_nan_and_infinity_give_nan(self) -> None:
        """NaN passes through, and an infinity gives NaN, with no warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = lon_to_pm180(np.array([np.nan, np.inf, -np.inf, 190.0]))
            assert isinstance(result, np.ndarray)
            np.testing.assert_array_equal(result, [np.nan, np.nan, np.nan, -170.0])
            for value in (math.nan, math.inf, -math.inf, np.float32(np.inf)):
                scalar = lon_to_pm180(value)
                assert type(scalar) is type(value)
                assert math.isnan(scalar)

    @pytest.mark.parametrize(("dtype", "out_dtype"), _INT_DTYPES_EVERY_VALUE)
    def test_every_value_of_narrow_int_dtypes(self, dtype: Any, out_dtype: Any) -> None:
        """Every int8, uint8, int16 and uint16 value gives its exact value (#148).

        uint16 180 and 270 gave 65356 and 65446 before, and int8 and uint8
        raised OverflowError.
        """
        info = np.iinfo(dtype)
        lons = np.arange(int(info.min), int(info.max) + 1).astype(dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = lon_to_pm180(lons)
        assert isinstance(result, np.ndarray)
        assert result.dtype == out_dtype
        expected = [(v + 180) % 360 - 180 for v in range(info.min, info.max + 1)]
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(("dtype", "out_dtype"), _INT_DTYPES_EXTREMES)
    def test_extremes_of_wide_int_dtypes(self, dtype: Any, out_dtype: Any) -> None:
        """The ends of the 32- and 64-bit ranges, uint64 above 2**63 included."""
        values = _int_extremes(dtype)
        result = lon_to_pm180(np.array(values, dtype=dtype))
        assert isinstance(result, np.ndarray)
        assert result.dtype == out_dtype
        np.testing.assert_array_equal(result, [(v + 180) % 360 - 180 for v in values])

    @pytest.mark.parametrize(
        ("dtype", "out_dtype"), _INT_DTYPES_EVERY_VALUE + _INT_DTYPES_EXTREMES
    )
    def test_int_numpy_scalars(self, dtype: Any, out_dtype: Any) -> None:
        """A NumPy integer scalar gives a scalar, int64 for unsigned and 8-bit."""
        for value in (np.iinfo(dtype).min, np.iinfo(dtype).max, 90):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = lon_to_pm180(dtype(value))
            assert type(result) is out_dtype
            assert result == (int(value) + 180) % 360 - 180

    def test_uint16_180_gives_minus_180(self) -> None:
        """The issue's case: 65356 with an overflow warning before (#148)."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = lon_to_pm180(np.uint16(180))
        assert type(result) is np.int64
        assert result == -180

    @pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.uint16, np.uint64])
    def test_int_dataarray_keeps_coords_name_and_attrs(self, dtype: Any) -> None:
        info = np.iinfo(dtype)
        values = [0, 90, int(info.max), int(info.min)]
        arr = xr.DataArray(
            np.array(values, dtype=dtype),
            dims=["x"],
            coords={"x": [5.0, 6.0, 7.0, 8.0]},
            name="lons",
            attrs={"units": "degrees_east"},
        )
        result = lon_to_pm180(arr)
        assert isinstance(result, xr.DataArray)
        assert result.dtype == np.int64
        assert result.name == "lons"
        assert result.attrs == {"units": "degrees_east"}
        np.testing.assert_array_equal(result["x"].values, [5.0, 6.0, 7.0, 8.0])
        expected = [(v + 180) % 360 - 180 for v in values]
        np.testing.assert_array_equal(result.values, expected)


class TestLongitudeInit:
    """Tests for Longitude.__init__."""

    def test_from_float_east(self) -> None:
        lon = Longitude(90)
        assert lon.longitude == 90
        assert lon.hemisphere == "E"

    def test_from_float_west(self) -> None:
        lon = Longitude(270)
        assert lon.longitude == 90
        assert lon.hemisphere == "W"

    def test_from_zero(self) -> None:
        lon = Longitude(0)
        assert lon.longitude == 0
        assert lon.hemisphere == "E"

    @pytest.mark.parametrize(
        ("value", "lon", "hem"),
        [(-90, 90.0, "W"), (-270, 90.0, "E"), (450, 90.0, "E"), (630, 90.0, "W")],
    )
    def test_numeric_outside_0_360(self, value: float, lon: float, hem: str) -> None:
        """Numbers below 0 or from 360 up wrap to the right hemisphere."""
        obj = Longitude(value)
        assert obj.longitude == lon
        assert obj.hemisphere == hem

    @pytest.mark.parametrize("value", [-1e-14, -5e-324, -3e-14, -0.1, -179.9])
    def test_negative_is_west_with_its_own_magnitude(self, value: float) -> None:
        """A negative number is that many degrees west, however small (#149).

        -1e-14 gave 0E before, and -3e-14 gave 2**-44 degrees west.
        """
        obj = Longitude(value)
        assert obj.hemisphere == "W"
        assert obj.longitude == -value

    @pytest.mark.parametrize("value", [-0.0, -360.0, np.float32(-0.0)])
    def test_negative_zero_is_zero_east(self, value: float) -> None:
        assert repr(Longitude(value)) == "Longitude('0.0E')"

    def test_negative_tenths_equal_west_strings(self) -> None:
        """Longitude(-k/10) equals Longitude('{k/10}w') for k = 1 to 1800."""
        unequal = [
            k for k in range(1, 1801) if Longitude(-k / 10) != Longitude(f"{k / 10}w")
        ]
        assert unequal == []

    @pytest.mark.parametrize(
        "value",
        [
            math.nan,
            math.inf,
            -math.inf,
            np.float32(np.nan),
            np.float64(np.inf),
            "nan",
            "inf",
            "-inf",
            "nanw",
            "nane",
            "NaNW",
            "-nanw",
            "infe",
            "infw",
            "-infw",
        ],
    )
    def test_non_finite_raises(self, value: float | str) -> None:
        """NaN and infinity raise, as a number or a string (issue #150)."""
        with pytest.raises(ValueError, match="must be finite"):
            Longitude(value)

    @pytest.mark.parametrize("value", ["0w", "0W", "-0w", "-0e", "0e", "0.0w"])
    def test_string_prime_meridian_is_zero_east(self, value: str) -> None:
        """Every string for 0 gives 0.0E, as Longitude(0) does."""
        obj = Longitude(value)
        assert repr(obj) == "Longitude('0.0E')"
        assert obj == Longitude(0)

    @pytest.mark.parametrize("value", ["180e", "180E", "180w", "180.0e"])
    def test_string_antimeridian_is_180_west(self, value: str) -> None:
        """Every string for 180 gives 180.0W, as Longitude(180) does."""
        obj = Longitude(value)
        assert repr(obj) == "Longitude('180.0W')"
        assert obj == Longitude(180)
        assert obj == Longitude(-180)

    @pytest.mark.parametrize(
        ("value", "lon", "hem"),
        [
            ("179.9e", 179.9, "E"),
            (f"{_BELOW_180!r}e", _BELOW_180, "E"),
            ("0.1w", 0.1, "W"),
            ("1e-14w", 1e-14, "W"),
            ("5e-324w", 5e-324, "W"),
        ],
    )
    def test_string_beside_meridians_keeps_hemisphere(
        self, value: str, lon: float, hem: str
    ) -> None:
        """Only exactly 0 and exactly 180 move hemisphere."""
        obj = Longitude(value)
        assert obj.longitude == lon
        assert obj.hemisphere == hem

    def test_from_string_east(self) -> None:
        lon = Longitude("45e")
        assert lon.longitude == 45
        assert lon.hemisphere == "E"

    def test_from_string_west(self) -> None:
        lon = Longitude("10W")
        assert lon.longitude == 10
        assert lon.hemisphere == "W"

    def test_from_string_uppercase(self) -> None:
        lon = Longitude("30E")
        assert lon.longitude == 30
        assert lon.hemisphere == "E"

    def test_invalid_string_no_hemisphere(self) -> None:
        with pytest.raises(ValueError, match="end in 'e' or 'w'"):
            Longitude("45n")

    def test_invalid_string_format(self) -> None:
        with pytest.raises(ValueError, match="improperly formatted"):
            Longitude("abcw")

    def test_string_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="within 0 and"):
            Longitude("200e")

    @pytest.mark.parametrize("value", ["-1e-14w", "-10e", "-180w"])
    def test_negative_string_rejected(self, value: str) -> None:
        """Only a negative zero is accepted with a hemisphere letter."""
        with pytest.raises(ValueError, match="within 0 and"):
            Longitude(value)

    def test_invalid_type(self) -> None:
        with pytest.raises(ValueError, match="scalar or a string"):
            Longitude([1, 2])  # type: ignore[arg-type]


class TestLongitudeProperties:
    """Tests for Longitude properties."""

    def test_longitude_read(self) -> None:
        assert Longitude(45).longitude == 45

    def test_hemisphere_read(self) -> None:
        assert Longitude(45).hemisphere == "E"

    def test_longitude_setter_raises(self) -> None:
        lon = Longitude(45)
        with pytest.raises(ValueError, match="cannot be modified"):
            lon.longitude = 100

    def test_hemisphere_setter_raises(self) -> None:
        lon = Longitude(45)
        with pytest.raises(ValueError, match="cannot be modified"):
            lon.hemisphere = "W"


class TestLongitudeComparisons:
    """Tests for Longitude comparison operators."""

    def test_eq_same(self) -> None:
        assert Longitude(90) == Longitude(90)

    def test_eq_different(self) -> None:
        assert Longitude(90) != Longitude(180)

    def test_eq_with_scalar(self) -> None:
        assert Longitude(90) == 90

    def test_lt_west_less_than_east(self) -> None:
        assert Longitude(270) < Longitude(90)

    def test_lt_within_east(self) -> None:
        assert Longitude(45) < Longitude(90)

    def test_lt_within_west(self) -> None:
        # 100W is more west than 90W
        assert Longitude("100w") < Longitude("90w")

    def test_gt(self) -> None:
        assert Longitude(90) > Longitude(270)

    def test_le_equal(self) -> None:
        assert Longitude(90) <= Longitude(90)

    def test_le_less(self) -> None:
        assert Longitude(45) <= Longitude(90)

    def test_ge_equal(self) -> None:
        assert Longitude(90) >= Longitude(90)

    def test_ge_greater(self) -> None:
        assert Longitude(90) >= Longitude(45)

    def test_same_meridian_equal_whichever_hemisphere_letter(self) -> None:
        assert Longitude("0w") == Longitude("0e")
        assert Longitude("180e") == Longitude("180w")

    def test_antimeridian_string_compares_as_west(self) -> None:
        """180E is stored as 180W, so it compares as Longitude(180) does."""
        assert Longitude(180) < Longitude(170)
        assert Longitude("180e") < Longitude("170e")
        assert not Longitude("180e") > Longitude("170e")
        assert not Longitude("170e") < Longitude("180e")
        assert Longitude("170w") > Longitude("180e")
        assert Longitude("180e") <= Longitude("170e")
        assert not Longitude("180e") >= Longitude("170e")

    def test_dataarray_with_nan_issue_output(self) -> None:
        """The issue's comparisons give what they gave before the raise (#150)."""
        arr = xr.DataArray([np.nan, 10.0, 30.0], dims="x", name="lons")
        np.testing.assert_array_equal((arr < Longitude("20e")).values, [0, 1, 0])
        np.testing.assert_array_equal((arr == Longitude("20e")).values, [0, 0, 0])

    @pytest.mark.parametrize("lon", ["20e", "20w"])
    @pytest.mark.parametrize(("op", "nan_result"), _COMPARISONS)
    def test_dataarray_with_non_finite_elements(
        self, op: Callable[[Any, Any], Any], nan_result: bool, lon: str
    ) -> None:
        """A non-finite element compares as NaN does, either side, with no warning.

        The finite elements, which include 20E and 20W themselves, compare as
        Longitudes.  Longitude(lon) != arr is left out: Python takes it as
        ``not (Longitude(lon) == arr)``, which raises for an array of more than
        one element.
        """
        values = [np.nan, np.inf, -np.inf, 10.0, 20.0, 30.0, 340.0]
        arr = xr.DataArray(values, dims="x", name="lons")
        expected = [nan_result] * 3 + [
            op(Longitude(v), Longitude(lon)) for v in values[3:]
        ]
        expected_flip = [nan_result] * 3 + [
            op(Longitude(lon), Longitude(v)) for v in values[3:]
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = op(arr, Longitude(lon))
            assert isinstance(result, xr.DataArray)
            np.testing.assert_array_equal(result.values, expected)
            if op is not operator.ne:
                flipped = op(Longitude(lon), arr)
                assert isinstance(flipped, xr.DataArray)
                np.testing.assert_array_equal(flipped.values, expected_flip)

    @pytest.mark.parametrize("lon", ["20e", "20w"])
    @pytest.mark.parametrize(
        "other", [math.nan, math.inf, -math.inf, np.float32(np.nan)]
    )
    @pytest.mark.parametrize(("op", "nan_result"), _COMPARISONS)
    def test_scalar_non_finite(
        self, op: Callable[[Any, Any], Any], nan_result: bool, other: Any, lon: str
    ) -> None:
        """Against a non-finite number, False, or True for !=, either side.

        Before, a western Longitude compared as west of NaN, so for example
        Longitude('20w') < math.nan was True.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert bool(op(Longitude(lon), other)) is nan_result
            assert bool(op(other, Longitude(lon))) is nan_result

    @pytest.mark.parametrize("other", ["abc", None])
    def test_non_numeric_scalar(self, other: Any) -> None:
        """== is False and != True; an ordering raises TypeError (#150).

        All of these raised RecursionError before.
        """
        lon = Longitude("20e")
        assert (lon == other) is False
        assert (other == lon) is False
        assert (lon != other) is True
        for op in (operator.lt, operator.le, operator.gt, operator.ge):
            with pytest.raises(TypeError):
                op(lon, other)
            with pytest.raises(TypeError):
                op(other, lon)

    def test_list_with_nan(self) -> None:
        """A list is compared element by element, NaN giving False."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = Longitude("20e") < [np.nan, 10.0, 30.0]
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [False, False, True])

    def test_dataset_with_nan(self) -> None:
        """A Dataset is compared variable by variable, either side."""
        ds = xr.Dataset({"lons": ("x", [np.nan, 10.0, 30.0])})
        expected = xr.Dataset({"lons": ("x", [False, True, False])})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            xr.testing.assert_identical(ds < Longitude("20e"), expected)
            xr.testing.assert_identical(Longitude("20e") > ds, expected)

    @pytest.mark.parametrize("zero_dim", [xr.DataArray(np.nan), np.array(np.nan)])
    def test_zero_dim_nan_gives_false(self, zero_dim: Any) -> None:
        """A 0-d array of NaN gives False, a bool, as a 0-d array of 10 gives True."""
        assert (Longitude("20e") > zero_dim) is False
        assert (Longitude("20e") > np.array(10.0)) is True


class TestLongitudeArithmetic:
    """Tests for Longitude arithmetic operators."""

    def test_add_longitudes(self) -> None:
        result = Longitude(90) + Longitude(45)
        assert result == Longitude(135)

    def test_add_with_scalar(self) -> None:
        result = Longitude(90) + 45
        assert result == Longitude(135)

    def test_sub_longitudes(self) -> None:
        result = Longitude(90) - Longitude(45)
        assert result == Longitude(45)

    def test_sub_with_scalar(self) -> None:
        result = Longitude(90) - 45
        assert result == Longitude(45)


class TestLongitudeRepr:
    """Tests for Longitude.__repr__."""

    def test_east(self) -> None:
        assert repr(Longitude(45)) == "Longitude('45.0E')"

    def test_west(self) -> None:
        assert repr(Longitude("10w")) == "Longitude('10.0W')"


class TestLongitudeTo0360:
    """Tests for Longitude.to_0360."""

    def test_east(self) -> None:
        assert Longitude(90).to_0360() == 90

    def test_west(self) -> None:
        assert Longitude(270).to_0360() == 270

    def test_zero(self) -> None:
        assert Longitude(0).to_0360() == 0

    @pytest.mark.parametrize("value", ["0w", "1e-14w"])
    def test_west_at_or_near_zero_gives_zero(self, value: str) -> None:
        """0 comes back, never 360, which is outside [0, 360)."""
        result = Longitude(value).to_0360()
        assert type(result) is float
        assert result == 0

    def test_west_strings(self) -> None:
        assert Longitude("10w").to_0360() == 350
        assert Longitude("180w").to_0360() == 180
        assert Longitude("0.1w").to_0360() == 360 - 0.1

    def test_west_just_past_band(self) -> None:
        """Past the band, a western longitude gives a value just below 360."""
        assert Longitude("3e-14w").to_0360() == np.nextafter(360.0, 0.0)
        assert Longitude("1e-11w").to_0360() == 360 - 1e-11


class TestLongitudeToPm180:
    """Tests for Longitude.to_pm180."""

    def test_east(self) -> None:
        assert Longitude(90).to_pm180() == 90

    def test_west(self) -> None:
        assert Longitude(270).to_pm180() == -90

    def test_zero(self) -> None:
        assert Longitude(0).to_pm180() == 0

    def test_antimeridian_string(self) -> None:
        """Longitude('180e') gives -180, inside [-180, 180)."""
        assert Longitude("180e").to_pm180() == -180


@pytest.mark.parametrize(
    "value",
    [f"{v!r}{hem}" for hem in ("e", "w") for v in (0.0, 1e-14, 0.1, 90.0, 180.0)]
    + [-3e-14, -1e-14, 0.0, 180.0, -180.0, 360.0, 540.0, -0.1],
)
def test_longitude_conversions_in_range(value: float | str) -> None:
    """to_0360 is in [0, 360) and to_pm180 in [-180, 180), from string or number."""
    obj = Longitude(value)
    assert 0 <= obj.to_0360() < 360
    assert -180 <= obj.to_pm180() < 180
    if obj.hemisphere == "E":
        assert 0 <= obj.longitude < 180
    else:
        assert 0 < obj.longitude <= 180
