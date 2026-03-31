#! /usr/bin/env python
"""Hide's theorem diagnostics.

Functions for diagnosing violations of Hide's theorem, which constrains
the angular momentum distribution in axisymmetric atmospheres. Provides
diagnostics for the poleward-most latitude where angular momentum exceeds
the planetary equatorial value, where the gradient wind has no real
solution, and where absolute vorticity changes sign.

References
----------
.. [1] Hide, R. (1969). "Dynamics of the Atmospheres of the Major Planets
   with an Appendix on the Viscous Boundary Layer at the Rigid Bounding
   Surface of an Electrically-Conducting Rotating Fluid in the Presence
   of a Magnetic Field." J. Atmos. Sci., 26, 841-853.
"""

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
    """Poleward-most latitude where angular momentum exceeds planetary equatorial value.

    Identifies the outermost latitude at which the absolute angular momentum
    exceeds the planetary angular momentum at the equator
    (:math:`\\Omega a^2`). This is one diagnostic of Hide's theorem
    violation, indicating supercritical forcing.

    Parameters
    ----------
    ang_mom : xarray.DataArray
        Absolute angular momentum field along latitude.
    radius : float, optional
        Planetary radius (m). Default: Earth.
    flip_lats : bool, optional
        If True, reverse the latitude dimension before searching.
        Default: False.
    rot_rate : float, optional
        Planetary rotation rate (rad/s). Default: Earth.
    lat_str : str, optional
        Name of the latitude dimension. Default: 'lat'.

    Returns
    -------
    xarray.DataArray
        Latitude (in degrees) of the poleward-most point where angular
        momentum exceeds the equatorial planetary value.

    See Also
    --------
    hides_negative : Latitude where gradient wind has no real solution.
    hides_vort_zero_cross : Latitude where absolute vorticity changes sign.
    """
    arr = _maybe_flip_lats(ang_mom, flip_lats)
    return cast(
        xr.DataArray, arr.where(arr > rot_rate * radius**2, drop=True)[-1][lat_str]
    )


def hides_negative(
    ang_mom: xr.DataArray, flip_lats: bool = False, lat_str: str = LAT_STR
) -> xr.DataArray:
    """Poleward-most latitude where gradient wind has no real solution.

    Where angular momentum is sufficiently anomalous, the gradient wind
    equation yields no real solution (indicated by NaN values). This
    function finds the poleward-most such latitude.

    Parameters
    ----------
    ang_mom : xarray.DataArray
        Absolute angular momentum field along latitude, with NaN values
        where gradient wind has no real solution.
    flip_lats : bool, optional
        If True, reverse the latitude dimension before searching.
        Default: False.
    lat_str : str, optional
        Name of the latitude dimension. Default: 'lat'.

    Returns
    -------
    xarray.DataArray
        Latitude (in degrees) of the poleward-most point where the
        gradient wind has no real solution.

    See Also
    --------
    hides_above_eq_mom : Latitude where angular momentum exceeds equatorial value.
    hides_vort_zero_cross : Latitude where absolute vorticity changes sign.
    """
    arr = _maybe_flip_lats(ang_mom, flip_lats)
    return cast(xr.DataArray, arr.where(np.isnan(arr), drop=True)[-1][lat_str])


def hides_vort_zero_cross(
    abs_vort: xr.DataArray, flip_lats: bool = False, lat_str: str = LAT_STR
) -> xr.DataArray:
    """Poleward-most latitude where absolute vorticity changes sign.

    A sign change in the absolute vorticity indicates a violation of the
    necessary condition for inertial stability, another diagnostic of
    Hide's theorem violation.

    Parameters
    ----------
    abs_vort : xarray.DataArray
        Absolute vorticity field along latitude.
    flip_lats : bool, optional
        If True, reverse the latitude dimension before searching.
        Default: False.
    lat_str : str, optional
        Name of the latitude dimension. Default: 'lat'.

    Returns
    -------
    xarray.DataArray
        Latitude (in degrees) of the poleward-most point where the
        absolute vorticity changes sign.

    See Also
    --------
    hides_above_eq_mom : Latitude where angular momentum exceeds equatorial value.
    hides_negative : Latitude where gradient wind has no real solution.
    """
    arr = _maybe_flip_lats(abs_vort, flip_lats)
    sign_change = cast(xr.DataArray, np.sign(arr)).diff(lat_str)
    return cast(
        xr.DataArray, arr.where(sign_change, drop=True).dropna(lat_str)[-1][lat_str]
    )


if __name__ == "__main__":
    pass
