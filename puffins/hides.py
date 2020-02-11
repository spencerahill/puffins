#! /usr/bin/env python
"""Hide's theorem."""

import numpy as np

from .constants import RAD_EARTH, ROT_RATE_EARTH
from .names import LAT_STR


def _flip_dim(arr, dim):
    return arr.isel(**{dim: slice(None, None, -1)})


def _flip_lats(arr, lat_str=LAT_STR):
    return _flip_dim(arr, lat_str)


def _maybe_flip_lats(arr, do_flip, lat_str=LAT_STR):
    if do_flip:
        return _flip_lats(arr, lat_str)
    return arr


def hides_above_eq_mom(ang_mom, radius=RAD_EARTH, flip_lats=False,
                       rot_rate=ROT_RATE_EARTH, lat_str=LAT_STR):
    """Poleward-most latitude where angular momentum exceeds planetary
    equatorial value.

    """
    arr = _maybe_flip_lats(ang_mom, flip_lats)
    return arr.where(arr > rot_rate*radius**2, drop=True)[-1][lat_str]


def hides_negative(ang_mom, flip_lats=False, lat_str=LAT_STR):
    """Poleward-most latitude where gradient wind has no real solution."""
    arr = _maybe_flip_lats(ang_mom, flip_lats)
    return arr.where(np.isnan(arr), drop=True)[-1][lat_str]


def hides_vort_zero_cross(abs_vort, flip_lats=False, lat_str=LAT_STR):
    """Poleward-most latitude where absolute vorticity changes sign."""
    arr = _maybe_flip_lats(abs_vort, flip_lats)
    return arr.where(np.sign(arr).diff(
        lat_str), drop=True).dropna(lat_str)[-1][lat_str]


if __name__ == '__main__':
    pass
