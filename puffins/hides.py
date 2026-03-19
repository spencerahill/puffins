#! /usr/bin/env python
"""Hide's theorem."""

from typing import cast

import numpy as np
import xarray as xr

from .constants import RAD_EARTH, ROT_RATE_EARTH
from .names import LAT_STR


def _flip_dim(arr: xr.DataArray, dim: str) -> xr.DataArray:
    return cast(xr.DataArray, arr.isel({dim: slice(None, None, -1)}))


def _flip_lats(arr: xr.DataArray, lat_str: str = LAT_STR) -> xr.DataArray:
    return _flip_dim(arr, lat_str)


def _maybe_flip_lats(
    arr: xr.DataArray, do_flip: bool, lat_str: str = LAT_STR
) -> xr.DataArray:
    if do_flip:
        return _flip_lats(arr, lat_str)
    return arr


def hides_above_eq_mom(
    ang_mom: xr.DataArray,
    radius: float = RAD_EARTH,
    flip_lats: bool = False,
    rot_rate: float = ROT_RATE_EARTH,
    lat_str: str = LAT_STR,
) -> xr.DataArray:
    """Poleward-most latitude where angular momentum exceeds planetary
    equatorial value.

    """
    arr = _maybe_flip_lats(ang_mom, flip_lats)
    return cast(
        xr.DataArray, arr.where(arr > rot_rate * radius**2, drop=True)[-1][lat_str]
    )


def hides_negative(
    ang_mom: xr.DataArray, flip_lats: bool = False, lat_str: str = LAT_STR
) -> xr.DataArray:
    """Poleward-most latitude where gradient wind has no real solution."""
    arr = _maybe_flip_lats(ang_mom, flip_lats)
    return cast(xr.DataArray, arr.where(np.isnan(arr), drop=True)[-1][lat_str])


def hides_vort_zero_cross(
    abs_vort: xr.DataArray, flip_lats: bool = False, lat_str: str = LAT_STR
) -> xr.DataArray:
    """Poleward-most latitude where absolute vorticity changes sign."""
    arr = _maybe_flip_lats(abs_vort, flip_lats)
    sign_change = cast(xr.DataArray, np.sign(arr)).diff(lat_str)
    return cast(
        xr.DataArray, arr.where(sign_change, drop=True).dropna(lat_str)[-1][lat_str]
    )


if __name__ == "__main__":
    pass
