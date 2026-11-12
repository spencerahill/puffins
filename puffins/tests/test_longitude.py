"""Tests for longitude module."""

from fractions import Fraction
from typing import Any

import numpy as np
import pytest
import xarray as xr

from puffins.longitude import Longitude, lon_to_0360, lon_to_pm180

# The float64 just below 180.
_BELOW_180 = float(np.nextafter(180.0, 0.0))


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
        result = lon_to_0360(np.array([0, 180, 359, 360, 720], dtype=dtype))
        assert isinstance(result, np.ndarray)
        assert result.dtype == dtype
        np.testing.assert_array_equal(result, [0, 180, 359, 0, 0])

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

    def test_tiny_negative_gives_zero(self) -> None:
        """A negative lon that rounds onto 360 gives 0, not 360 (issue #83)."""
        result = lon_to_pm180(-1e-14)
        assert type(result) is float
        assert result == 0

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_wrap_points_in_range_and_exact(self, dtype: Any) -> None:
        """At the wrap points, the exact remainder in [0, 360) minus 360 if 180 up."""
        lons = _wrap_points_and_neighbors(dtype)
        result = lon_to_pm180(lons)
        assert isinstance(result, np.ndarray)
        assert result.dtype == dtype
        assert np.all((result >= -180) & (result < 180))
        ref = _exact_0360(lons)
        expected = np.where(ref >= 180, ref - 360, ref).astype(dtype)
        np.testing.assert_array_equal(result, expected)


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

    def test_tiny_negative_is_zero_east(self) -> None:
        """-1e-14 gives 0E, not 360W, outside 0 to 180 (issue #83)."""
        obj = Longitude(-1e-14)
        assert obj.longitude == 0
        assert obj.hemisphere == "E"

    def test_negative_just_past_band_stays_west(self) -> None:
        """-3e-14 rounds to the float below 360, so 2**-44 degrees west."""
        obj = Longitude(-3e-14)
        assert obj.hemisphere == "W"
        assert obj.longitude == 2.0**-44

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
