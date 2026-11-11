#!/usr/bin/env python
"""Functionality relating to parsing and comparing longitudes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import xarray as xr

from ._typing import ArrayLike


def lon_to_0360(lon: ArrayLike) -> ArrayLike:
    """Convert longitude(s) to be within [0, 360).

    The Eastern hemisphere corresponds to 0 <= lon + (n*360) < 180, and the
    Western Hemisphere corresponds to 180 <= lon + (n*360) < 360, where 'n' is
    any integer (positive, negative, or zero).

    Parameters
    ----------
    lon : scalar, numpy.ndarray, or xarray.DataArray
        One or more longitude values to be converted to lie in the [0, 360)
        range

    Returns
    -------
        If ``lon`` is a scalar, then a scalar of the same type in the range [0,
        360).  If ``lon`` is array-like, then an array-like of the same type
        with each element a scalar in the range [0, 360).

    """
    # A single ``lon % 360`` returns exactly 360 for a negative ``lon`` of
    # magnitude at most half the floating-point spacing at 360 (2**-45, about
    # 2.8e-14, in float64 and 2**-16, about 1.5e-5, in float32), because
    # ``lon + 360`` rounds to 360 there.  The second ``% 360`` maps that 360 to
    # 0 and returns every value already in [0, 360) unchanged and in its own
    # type (issue #83).  ``%`` also stays exact at magnitudes, from about 1.3e8
    # in float32 and 7.2e16 in float64, where ``lon - (lon // 360) * 360`` does
    # not.
    return lon % 360 % 360


def _lon_in_west_hem(lon: ArrayLike) -> Any:
    return lon_to_0360(lon) >= 180


def lon_to_pm180(lon: ArrayLike) -> ArrayLike:
    """Convert longitude(s) to be within [-180, 180).

    The Eastern hemisphere corresponds to 0 <= lon + (n*360) < 180, and the
    Western Hemisphere corresponds to 180 <= lon + (n*360) < 360, where 'n' is
    any integer (positive, negative, or zero).

    Parameters
    ----------
    lon : scalar, numpy.ndarray, or xarray.DataArray
        One or more longitude values to be converted to lie in the [-180, 180)
        range

    Returns
    -------
        If ``lon`` is a scalar, then a scalar of the same type in the range
        [-180, 180).  If ``lon`` is array-like, then an array-like of the same
        type with each element a scalar in the range [-180, 180).

    """
    lon0360 = lon_to_0360(lon)
    # `lon0360 // 180 % 2` is 1 in the western hemisphere and 0 in the
    # eastern, in lon0360's own type, so scalars stay scalars, float32 stays
    # float32, DataArrays keep their coordinates, and the values match the
    # former scalar-only `lon0360 - 360` branch exactly.
    return cast(ArrayLike, lon0360 - 360 * (lon0360 // 180 % 2))


def _maybe_cast_to_lon(obj: Any, strict: bool = False) -> Any:
    if isinstance(obj, Longitude):
        return obj
    try:
        return Longitude(obj)
    except (ValueError, TypeError) as e:
        if strict:
            raise type(e)(str(e))
        else:
            return obj


def _other_to_lon(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrapper for casting Longitude operator arguments to Longitude"""

    def func_other_to_lon(obj: Longitude, other: Any) -> Any:
        return func(obj, _maybe_cast_to_lon(other))

    return func_other_to_lon


class Longitude:
    """Geographic longitude.

    Enables unambiguous comparison of longitudes using the standard comparison
    operators, regardless of they were initially represented with a 0 to 360
    convention, -180 to 180 convention, or anything else, and even if the
    original convention differs between them.

    Specifically, the ``<`` operator assesses if the first object is to the
    west of the second object, with the standard convention that longitudes in
    the Western Hemisphere are always to the west of longitudes in the Eastern
    Hemisphere.  The ``>`` operator is defined analogously.  ``==``, ``>=``,
    and ``<=`` are also all defined.

    The prime meridian counts as eastern and the antimeridian as western,
    however they are given: ``Longitude('0w')`` equals ``Longitude(0)``, 0
    degrees east, and ``Longitude('180e')`` equals ``Longitude(180)``, 180
    degrees west, so ``Longitude('180e') < Longitude('170e')``.

    In addition to other Longitude objects, the operators can be used to
    compare a Longitude object to anything that can be casted to a Longitude
    object, or to any sequence (e.g. a list or xarray.DataArray) whose elements
    can be casted to Longitude objects.

    """

    _longitude: float
    _hemisphere: str

    def __init__(self, value: float | str) -> None:
        """
        Parameters
        ----------
        value : {scalar, str}
            Scalars get converted to longitudes using the convention that 0-180
            corresponds to the Eastern Hemisphere, 180-360 corresponds to the
            Western Hemisphere, 360-540 the Eastern Hemisphere, and so on,
            including for negative numbers.

            Strings must be castable to a float or be a positive number in the
            range 0-180 followed by a single letter 'e' or 'w' (case
            insensitive).  For example, ``Longitude('10w')`` would yield a
            ``Longitude`` object corresponding to 10 degrees west longitude.
            The prime meridian and the antimeridian are stored in the
            representation a scalar gives them, so that each meridian has one:
            ``Longitude('0w')`` is 0 degrees east and ``Longitude('180e')`` is
            180 degrees west.
        """
        try:
            val_as_float = float(value)
        except (ValueError, TypeError):
            if not isinstance(value, str):
                raise ValueError("value must be a scalar or a string")
            if value[-1].lower() not in ("w", "e"):
                raise ValueError("string inputs must end in 'e' or 'w'")
            try:
                lon_value = float(value[:-1])
            except ValueError:
                raise ValueError("improperly formatted string")
            if (lon_value < 0) or (lon_value > 180):
                raise ValueError(
                    "Value given as strings with hemisphere "
                    "identifier must have numerical values "
                    "within 0 and +180.  Value given: "
                    f"{lon_value}"
                )
            hemisphere = value[-1].upper()
            # The scalar path puts 0 in the east and 180 in the west, matching
            # lon_to_pm180's range of [-180, 180).  Store strings the same way,
            # and a negative zero as 0.0.
            if lon_value == 0:
                lon_value, hemisphere = 0.0, "E"
            elif lon_value == 180:
                hemisphere = "W"
            self._longitude = lon_value
            self._hemisphere = hemisphere
        else:
            lon_pm180 = float(lon_to_pm180(val_as_float))
            if _lon_in_west_hem(val_as_float):
                self._longitude = abs(lon_pm180)
                self._hemisphere = "W"
            else:
                self._longitude = lon_pm180
                self._hemisphere = "E"

    @property
    def longitude(self) -> float:
        """(scalar) The unsigned numerical value of the longitude.

        In the range [0, 180) in the Eastern Hemisphere and (0, 180] in the
        Western, so 0 is always east and 180 always west.  Must be combined with
        the ``hemisphere`` attribute to specify the exact longitude.

        """
        return self._longitude

    @longitude.setter
    def longitude(self, value: float) -> None:
        raise ValueError(
            "'longitude' property cannot be modified after "
            "Longitude object has been created."
        )

    @property
    def hemisphere(self) -> str:
        """{'W', 'E'} The longitude's hemisphere, either western or eastern."""
        return self._hemisphere

    @hemisphere.setter
    def hemisphere(self, value: str) -> None:
        raise ValueError(
            "'hemisphere' property cannot be modified after "
            "Longitude object has been created."
        )

    def __repr__(self) -> str:
        return f"Longitude('{self.longitude}{self.hemisphere}')"

    @_other_to_lon
    def __eq__(self, other: Any) -> Any:
        if isinstance(other, Longitude):
            return (
                self.hemisphere == other.hemisphere
                and self.longitude == other.longitude
            )
        else:
            return xr.apply_ufunc(np.equal, other, self)

    @_other_to_lon
    def __lt__(self, other: Any) -> Any:
        if isinstance(other, Longitude):
            if self.hemisphere == "W":
                if other.hemisphere == "E":
                    return True
                else:
                    return self.longitude > other.longitude
            else:
                if other.hemisphere == "W":
                    return False
                else:
                    return self.longitude < other.longitude
        else:
            return xr.apply_ufunc(np.greater, other, self)

    @_other_to_lon
    def __gt__(self, other: Any) -> Any:
        if isinstance(other, Longitude):
            if self.hemisphere == "W":
                if other.hemisphere == "E":
                    return False
                else:
                    return self.longitude < other.longitude
            else:
                if other.hemisphere == "W":
                    return True
                else:
                    return self.longitude > other.longitude
        else:
            return xr.apply_ufunc(np.less, other, self)

    @_other_to_lon
    def __le__(self, other: Any) -> Any:
        if isinstance(other, Longitude):
            return self < other or self == other
        else:
            return xr.apply_ufunc(np.greater_equal, other, self)

    @_other_to_lon
    def __ge__(self, other: Any) -> Any:
        if isinstance(other, Longitude):
            return self > other or self == other
        else:
            return xr.apply_ufunc(np.less_equal, other, self)

    def to_0360(self) -> float:
        """Convert longitude to its numerical value within [0, 360)."""
        if self.hemisphere == "W":
            # Through lon_to_0360, so that a western longitude too small for
            # 360 minus it to differ from 360 gives 0, never 360.
            return cast(float, lon_to_0360(-self.longitude))
        else:
            return self.longitude

    def to_pm180(self) -> float:
        """Convert longitude to its numerical value within [-180, 180)."""
        if self.hemisphere == "W":
            return -1 * self.longitude
        else:
            return self.longitude

    @_other_to_lon
    def __add__(self, other: Any) -> Longitude:
        return Longitude(self.to_0360() + other.to_0360())

    @_other_to_lon
    def __sub__(self, other: Any) -> Longitude:
        return Longitude(self.to_0360() - other.to_0360())


if __name__ == "__main__":
    pass
