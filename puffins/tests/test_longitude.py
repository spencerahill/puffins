"""Tests for longitude module."""

import numpy as np
import pytest
import xarray as xr

from puffins.longitude import (
    Longitude,
    _lon_in_west_hem,
    _maybe_cast_to_lon,
    lon_to_0360,
    lon_to_pm180,
)


class TestLonTo0360:
    """Tests for lon_to_0360."""

    def test_already_in_range(self) -> None:
        assert lon_to_0360(90.0) == 90.0

    def test_negative_lon(self) -> None:
        assert lon_to_0360(-90.0) == 270.0

    def test_exactly_360(self) -> None:
        assert lon_to_0360(360.0) == 0.0

    def test_large_positive(self) -> None:
        assert lon_to_0360(450.0) == 90.0

    def test_large_negative(self) -> None:
        assert lon_to_0360(-450.0) == 270.0

    def test_zero(self) -> None:
        assert lon_to_0360(0.0) == 0.0

    def test_180(self) -> None:
        assert lon_to_0360(180.0) == 180.0

    def test_numpy_array(self) -> None:
        result = lon_to_0360(np.array([-90.0, 0.0, 90.0, 360.0]))
        np.testing.assert_array_equal(result, [270.0, 0.0, 90.0, 0.0])

    def test_xarray_dataarray(self) -> None:
        da = xr.DataArray([-90.0, 0.0, 90.0], dims=["x"])
        result = lon_to_0360(da)
        np.testing.assert_array_equal(result.values, [270.0, 0.0, 90.0])
        assert isinstance(result, xr.DataArray)


class TestLonInWestHem:
    """Tests for _lon_in_west_hem."""

    def test_eastern_hemisphere(self) -> None:
        assert not _lon_in_west_hem(90.0)

    def test_western_hemisphere(self) -> None:
        assert _lon_in_west_hem(200.0)

    def test_exactly_180_is_western(self) -> None:
        assert _lon_in_west_hem(180.0)

    def test_zero_is_eastern(self) -> None:
        assert not _lon_in_west_hem(0.0)

    def test_negative_western(self) -> None:
        assert _lon_in_west_hem(-10.0)


class TestLonToPm180:
    """Tests for lon_to_pm180."""

    def test_eastern_stays_positive(self) -> None:
        assert lon_to_pm180(90.0) == 90.0

    def test_western_becomes_negative(self) -> None:
        assert lon_to_pm180(270.0) == -90.0

    def test_zero(self) -> None:
        assert lon_to_pm180(0.0) == 0.0

    def test_negative_input(self) -> None:
        assert lon_to_pm180(-90.0) == -90.0

    def test_large_positive(self) -> None:
        assert lon_to_pm180(450.0) == 90.0


class TestMaybeCastToLon:
    """Tests for _maybe_cast_to_lon."""

    def test_longitude_passes_through(self) -> None:
        lon = Longitude(10.0)
        assert _maybe_cast_to_lon(lon) is lon

    def test_float_becomes_longitude(self) -> None:
        result = _maybe_cast_to_lon(90.0)
        assert isinstance(result, Longitude)
        assert result == Longitude(90.0)

    def test_string_becomes_longitude(self) -> None:
        result = _maybe_cast_to_lon("10W")
        assert isinstance(result, Longitude)
        assert result == Longitude("10W")

    def test_invalid_returns_original_non_strict(self) -> None:
        obj = [1, 2, 3]
        assert _maybe_cast_to_lon(obj) is obj

    def test_invalid_raises_in_strict_mode(self) -> None:
        with pytest.raises(ValueError):
            _maybe_cast_to_lon([1, 2, 3], strict=True)


class TestLongitudeInit:
    """Tests for Longitude.__init__."""

    def test_from_positive_float(self) -> None:
        lon = Longitude(90.0)
        assert lon.longitude == 90.0
        assert lon.hemisphere == "E"

    def test_from_negative_float(self) -> None:
        lon = Longitude(-90.0)
        assert lon.longitude == 90.0
        assert lon.hemisphere == "W"

    def test_from_western_float(self) -> None:
        lon = Longitude(200.0)
        assert lon.hemisphere == "W"
        assert lon.longitude == 160.0

    def test_from_string_east(self) -> None:
        lon = Longitude("45E")
        assert lon.longitude == 45.0
        assert lon.hemisphere == "E"

    def test_from_string_west(self) -> None:
        lon = Longitude("45w")
        assert lon.longitude == 45.0
        assert lon.hemisphere == "W"

    def test_from_string_case_insensitive(self) -> None:
        assert Longitude("30e") == Longitude("30E")

    def test_from_zero(self) -> None:
        lon = Longitude(0.0)
        assert lon.longitude == 0.0
        assert lon.hemisphere == "E"

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError, match="string inputs must end"):
            Longitude("45N")

    def test_bad_format_string_raises(self) -> None:
        with pytest.raises(ValueError, match="improperly formatted"):
            Longitude("abcE")

    def test_out_of_range_string_raises(self) -> None:
        with pytest.raises(ValueError, match="within 0 and"):
            Longitude("200E")

    def test_non_scalar_non_string_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a scalar or a string"):
            Longitude([1, 2])


class TestLongitudeProperties:
    """Tests for Longitude properties."""

    def test_longitude_immutable(self) -> None:
        lon = Longitude(90.0)
        with pytest.raises(ValueError):
            lon.longitude = 45.0

    def test_hemisphere_immutable(self) -> None:
        lon = Longitude(90.0)
        with pytest.raises(ValueError):
            lon.hemisphere = "W"

    def test_repr(self) -> None:
        lon = Longitude("45E")
        assert repr(lon) == "Longitude('45.0E')"

    def test_repr_west(self) -> None:
        lon = Longitude("120W")
        assert repr(lon) == "Longitude('120.0W')"


class TestLongitudeComparisons:
    """Tests for Longitude comparison operators."""

    def test_eq_same(self) -> None:
        assert Longitude(90.0) == Longitude(90.0)

    def test_eq_different_convention(self) -> None:
        assert Longitude(270.0) == Longitude(-90.0)

    def test_eq_string_and_float(self) -> None:
        assert Longitude("45E") == Longitude(45.0)

    def test_neq(self) -> None:
        assert not (Longitude(90.0) == Longitude(91.0))

    def test_lt_both_eastern(self) -> None:
        assert Longitude(10.0) < Longitude(20.0)

    def test_lt_both_western(self) -> None:
        # More western = further west = "less than"
        assert Longitude("20W") < Longitude("10W")

    def test_lt_west_less_than_east(self) -> None:
        assert Longitude("10W") < Longitude("10E")

    def test_gt_both_eastern(self) -> None:
        assert Longitude(20.0) > Longitude(10.0)

    def test_gt_east_greater_than_west(self) -> None:
        assert Longitude("10E") > Longitude("10W")

    def test_le(self) -> None:
        assert Longitude(10.0) <= Longitude(20.0)
        assert Longitude(10.0) <= Longitude(10.0)

    def test_ge(self) -> None:
        assert Longitude(20.0) >= Longitude(10.0)
        assert Longitude(10.0) >= Longitude(10.0)

    def test_eq_with_float(self) -> None:
        assert Longitude(90.0) == 90.0

    def test_lt_with_float(self) -> None:
        assert Longitude(10.0) < 20.0


class TestLongitudeConversions:
    """Tests for Longitude.to_0360 and to_pm180."""

    def test_to_0360_eastern(self) -> None:
        assert Longitude(90.0).to_0360() == 90.0

    def test_to_0360_western(self) -> None:
        assert Longitude("90W").to_0360() == 270.0

    def test_to_pm180_eastern(self) -> None:
        assert Longitude(90.0).to_pm180() == 90.0

    def test_to_pm180_western(self) -> None:
        assert Longitude("90W").to_pm180() == -90.0

    def test_to_0360_zero(self) -> None:
        assert Longitude(0.0).to_0360() == 0.0


class TestLongitudeArithmetic:
    """Tests for Longitude addition and subtraction."""

    def test_add_longitudes(self) -> None:
        result = Longitude(10.0) + Longitude(20.0)
        assert isinstance(result, Longitude)
        assert result == Longitude(30.0)

    def test_add_float(self) -> None:
        result = Longitude(10.0) + 20.0
        assert isinstance(result, Longitude)
        assert result == Longitude(30.0)

    def test_sub_longitudes(self) -> None:
        result = Longitude(30.0) - Longitude(10.0)
        assert isinstance(result, Longitude)
        assert result == Longitude(20.0)

    def test_add_wraps_around(self) -> None:
        result = Longitude(350.0) + Longitude(20.0)
        assert result == Longitude(10.0)
