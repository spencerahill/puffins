#!/usr/bin/env python
"""Functionality relating to parsing and comparing longitudes."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import xarray as xr

from ._typing import ArrayLike


def _int64_if_unsigned_or_8bit(lon: ArrayLike) -> ArrayLike:
    """Unsigned and 8-bit integer input as int64, otherwise ``lon`` unchanged.

    NumPy 2 will not convert the constant 360 to int8 or uint8, which cannot
    represent it, and an unsigned type cannot represent the negative values
    ``lon_to_pm180`` returns (issue #148).  Unsigned values are reduced modulo
    360 in uint64, which represents 360 and every unsigned value exactly,
    before the cast, since casting a uint64 above the int64 maximum wraps it
    to a negative value.
    """
    dtype = getattr(lon, "dtype", None)
    if dtype is None:
        return lon
    values = cast(Any, lon)
    if dtype.kind == "u":
        return cast(ArrayLike, (values.astype(np.uint64) % 360).astype(np.int64))
    if dtype.kind == "i" and dtype.itemsize == 1:
        return cast(ArrayLike, values.astype(np.int64))
    return lon


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
        with each element a scalar in the range [0, 360).  The one exception
        is unsigned or 8-bit integer input, scalar or array, which comes back
        as int64, the type ``lon_to_pm180`` returns for it.  NaN comes back
        as NaN.

    """
    lon = _int64_if_unsigned_or_8bit(lon)
    # A single ``lon % 360`` returns exactly 360 for a negative ``lon`` of
    # magnitude at most half the floating-point spacing at 360 (2**-45, about
    # 2.8e-14, in float64 and 2**-16, about 1.5e-5, in float32), because
    # ``lon + 360`` rounds to 360 there.  The second ``% 360`` maps that 360 to
    # 0 and returns every value already in [0, 360) unchanged and in its own
    # type (issue #83).  ``%`` also stays exact at magnitudes, from about 1.3e8
    # in float32 and 7.2e16 in float64, where ``lon - (lon // 360) * 360`` does
    # not.
    return lon % 360 % 360


def lon_to_pm180(lon: ArrayLike) -> ArrayLike:
    """Convert longitude(s) to be within [-180, 180).

    The Eastern hemisphere corresponds to 0 <= lon + (n*360) < 180, and the
    Western Hemisphere corresponds to 180 <= lon + (n*360) < 360, where 'n' is
    any integer (positive, negative, or zero).

    The result is the exact value of ``lon`` minus a multiple of 360, with no
    rounding, so every value already in [-180, 180) comes back unchanged:
    ``lon_to_pm180(-0.1)`` is -0.1.  A negative zero comes back as 0.

    Parameters
    ----------
    lon : scalar, numpy.ndarray, or xarray.DataArray
        One or more longitude values to be converted to lie in the [-180, 180)
        range

    Returns
    -------
        If ``lon`` is a scalar, then a scalar of the same type in the range
        [-180, 180).  If ``lon`` is array-like, then an array-like of the same
        type with each element a scalar in the range [-180, 180).  The one
        exception is unsigned or 8-bit integer input, scalar or array, which
        comes back as int64, since an unsigned type cannot represent a negative
        result.  NaN comes back as NaN.

    """
    lon = _int64_if_unsigned_or_8bit(lon)
    rem: Any
    if isinstance(lon, int):
        # A Python int, whose % is exact at any magnitude.  A NumPy integer is
        # not an int subclass and takes the last branch.
        rem = lon % 360
    elif isinstance(lon, float) and not isinstance(lon, np.generic):
        # A Python float stays one.  np.fmod returns NaN for an infinity, as
        # % does, where math.fmod raises.
        rem = float(np.fmod(lon, 360))
    else:
        rem = np.fmod(lon, 360)
    # fmod is exact and takes the sign of lon, so it returns -0.1 for -0.1,
    # where lon % 360 gives 360 - 0.1 rounded, from which subtracting 360 does
    # not recover -0.1 (issue #149).  Adding 0 turns the -0.0 that fmod gives
    # for -0.0 and for negative multiples of 360 into 0.0, as % gives, and
    # changes no other value.
    rem = rem + 0
    # rem // 180 is -2 on (-360, -180), -1 on [-180, 0), 0 on [0, 180) and 1
    # on [180, 360), so (rem // 180 + 1) // 2 is -1, 0, 0 and 1 there: the
    # multiple of 360 to subtract, in rem's own type, so scalars stay
    # scalars, float32 stays float32 and DataArrays keep their coordinates.
    # The subtraction is exact, since rem and 360 are within a factor of 2
    # of each other whenever the multiple is not 0 (Sterbenz's lemma).
    return cast(ArrayLike, rem - 360 * ((rem // 180 + 1) // 2))


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


def _compare_unconverted(ufunc: np.ufunc, other: Any, lon: Longitude) -> Any:
    """Compare ``other``, which could not be cast to a Longitude, with ``lon``.

    ``ufunc`` is the comparison with ``other`` on the left.  A Dataset, or an
    array or list of one or more dimensions, is compared element by element,
    each element coming back through the Longitude comparison method.  A 0-d
    array counts as a scalar, as ``Longitude`` treats it, so a NaN in one gives
    the bool False, as a finite value in one gives a bool.  A NaN or infinite
    number compares as a float compares with NaN: False, which makes ``!=``
    True, since Python takes it as the negation of ``==``.  Passing such an
    element back to ``xr.apply_ufunc`` would call the same comparison method on
    it again, without end (issue #150).  For anything else, such as None or a
    string, this returns NotImplemented, so Python raises TypeError for ``<``,
    ``<=``, ``>`` and ``>=``, and ``==`` falls back to identity.  A string that
    spells NaN or infinity, such as 'nan' or 'inf', also gets NotImplemented,
    though float() accepts it, so that it compares as 'nanw' and 'abc' do.
    """
    # np.ndim raises TypeError for a Dataset.
    if isinstance(other, xr.Dataset) or np.ndim(other) > 0:
        return xr.apply_ufunc(ufunc, other, lon)
    if isinstance(other, str):
        return NotImplemented
    try:
        float(other)
    except (TypeError, ValueError):
        return NotImplemented
    return False


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
    whether given as a number or as a string: ``Longitude('0w')`` equals
    ``Longitude(0)``, 0 degrees east, and ``Longitude('180e')`` equals
    ``Longitude(180)``, 180 degrees west, so
    ``Longitude('180e') < Longitude('170e')``.

    In addition to other Longitude objects, the operators can be used to
    compare a Longitude object to anything that can be casted to a Longitude
    object, or to any sequence (e.g. a list or xarray.DataArray) whose elements
    can be casted to Longitude objects.  A NaN or infinite number, alone or as
    an element, compares as a float compares with NaN: ``==``, ``<``, ``<=``,
    ``>`` and ``>=`` give False and ``!=`` gives True.  Anything else that
    cannot be cast, such as None or a string that is not a longitude ('abc',
    and also 'nan', 'inf' and 'nanw'), gives False for ``==`` and raises
    TypeError for ``<``, ``<=``, ``>`` and ``>=``.

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
            including for negative numbers.  The hemisphere and the magnitude
            both come from ``lon_to_pm180``, which is exact, so a negative
            number is western however small: ``Longitude(-1e-14)`` is 1e-14
            degrees west, and ``Longitude(-0.1)`` equals ``Longitude('0.1w')``.

            Strings must be castable to a float or be a positive number in the
            range 0-180 followed by a single letter 'e' or 'w' (case
            insensitive).  For example, ``Longitude('10w')`` would yield a
            ``Longitude`` object corresponding to 10 degrees west longitude.
            The prime meridian and the antimeridian are stored in the
            representation a scalar gives them, so that each meridian has one:
            ``Longitude('0w')`` is 0 degrees east and ``Longitude('180e')`` is
            180 degrees west.

            A NaN or infinite value, as a number or a string such as
            ``'nanw'``, raises ValueError.
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
            if not math.isfinite(lon_value):
                raise ValueError(f"value must be finite; value given: {value!r}")
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
            if not math.isfinite(val_as_float):
                raise ValueError(f"value must be finite; value given: {value!r}")
            lon_pm180 = float(lon_to_pm180(val_as_float))
            # The hemisphere comes from the sign of the exact value in
            # [-180, 180).  Taking it from lon_to_0360 would put a negative
            # value too small for value + 360 to differ from 360 in the east,
            # since lon_to_0360 rounds that value onto 0 (issue #149).
            # lon_to_pm180 gives 0.0 for -0.0, so 0 is eastern.
            self._longitude = abs(lon_pm180)
            self._hemisphere = "W" if lon_pm180 < 0 else "E"

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
            return _compare_unconverted(np.equal, other, self)

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
            return _compare_unconverted(np.greater, other, self)

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
            return _compare_unconverted(np.less, other, self)

    @_other_to_lon
    def __le__(self, other: Any) -> Any:
        if isinstance(other, Longitude):
            return self < other or self == other
        else:
            return _compare_unconverted(np.greater_equal, other, self)

    @_other_to_lon
    def __ge__(self, other: Any) -> Any:
        if isinstance(other, Longitude):
            return self > other or self == other
        else:
            return _compare_unconverted(np.less_equal, other, self)

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
