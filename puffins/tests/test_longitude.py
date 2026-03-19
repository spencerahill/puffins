"""Tests for longitude module."""

import numpy as np
import pytest
import xarray as xr

from puffins.longitude import Longitude, lon_to_0360, lon_to_pm180


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


class TestLongitudeToPm180:
    """Tests for Longitude.to_pm180."""

    def test_east(self) -> None:
        assert Longitude(90).to_pm180() == 90

    def test_west(self) -> None:
        assert Longitude(270).to_pm180() == -90

    def test_zero(self) -> None:
        assert Longitude(0).to_pm180() == 0
