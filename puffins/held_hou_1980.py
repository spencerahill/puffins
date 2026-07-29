#! /usr/bin/env python
"""Held and Hou (1980) axisymmetric Hadley cell model.

Implements the radiative-convective equilibrium (RCE) temperature profiles,
angular-momentum-conserving (AMC) wind fields, and Hadley cell edge
diagnostics from the foundational Held & Hou (1980) theory of axisymmetric
Hadley circulations.

References
----------
.. [1] Held, I. M. & Hou, A. Y. (1980). "Nonlinear Axially Symmetric
   Circulations in a Nearly Inviscid Atmosphere." J. Atmos. Sci., 37,
   515-533.
"""

from __future__ import annotations

from typing import cast, overload

import numpy as np
import xarray as xr

from ._typing import (
    ArrayLike,
    Scalar,
    SolverGuessRange,
    SolverParamRange,
)
from .constants import (
    DELTA_H,
    DELTA_V,
    HEIGHT_TROPO,
    RAD_EARTH,
    ROT_RATE_EARTH,
    THETA_REF,
)
from .nb_utils import cosdeg, sindeg
from .num_solver import brentq_solver_sweep_param


@overload
def pot_temp_rce_hh80(
    lats: xr.DataArray,
    z: ArrayLike,
    theta_ref: float,
    height: float,
    delta_h: float,
    delta_v: float,
) -> xr.DataArray: ...
@overload
def pot_temp_rce_hh80(
    lats: ArrayLike,
    z: xr.DataArray,
    theta_ref: float,
    height: float,
    delta_h: float,
    delta_v: float,
) -> xr.DataArray: ...
@overload
def pot_temp_rce_hh80(
    lats: np.ndarray,
    z: np.ndarray | Scalar,
    theta_ref: float,
    height: float,
    delta_h: float,
    delta_v: float,
) -> np.ndarray: ...
@overload
def pot_temp_rce_hh80(
    lats: Scalar,
    z: np.ndarray,
    theta_ref: float,
    height: float,
    delta_h: float,
    delta_v: float,
) -> np.ndarray: ...
@overload
def pot_temp_rce_hh80(
    lats: Scalar,
    z: Scalar,
    theta_ref: float,
    height: float,
    delta_h: float,
    delta_v: float,
) -> Scalar: ...
def pot_temp_rce_hh80(
    lats: ArrayLike,
    z: ArrayLike,
    theta_ref: float,
    height: float,
    delta_h: float,
    delta_v: float,
) -> ArrayLike:
    """Radiative-convective equilibrium potential temperature (Eq. 2 of HH80).

    Parameters
    ----------
    lats : array-like
        Latitude (degrees).
    z : array-like
        Height (m).
    theta_ref : float
        Reference potential temperature (K).
    height : float
        Tropopause height (m).
    delta_h : float
        Fractional horizontal temperature difference.
    delta_v : float
        Fractional vertical temperature difference.

    Returns
    -------
    array-like
        RCE potential temperature (K).
    """
    return cast(
        ArrayLike,
        theta_ref
        * (1 + delta_h * (cosdeg(lats) ** 2 - 2 / 3) + (z / height - 0.5) * delta_v),
    )


@overload
def pot_temp_rce_hh80_small_ang(
    lats: xr.DataArray,
    z: ArrayLike = ...,
    theta_ref: float = ...,
    height: float = ...,
    delta_h: float = ...,
    delta_v: float = ...,
) -> xr.DataArray: ...
# `z` is required in the two overloads where it, rather than `lats`, drives
# the return type. Defaulting it there would let an all-defaults call such as
# `f(scalar_lats)` match them, yielding DataArray for a scalar call and an
# ambiguous (hence `Any`) resolution for an ndarray one.
@overload
def pot_temp_rce_hh80_small_ang(
    lats: ArrayLike,
    z: xr.DataArray,
    theta_ref: float = ...,
    height: float = ...,
    delta_h: float = ...,
    delta_v: float = ...,
) -> xr.DataArray: ...
@overload
def pot_temp_rce_hh80_small_ang(
    lats: np.ndarray,
    z: np.ndarray | Scalar = ...,
    theta_ref: float = ...,
    height: float = ...,
    delta_h: float = ...,
    delta_v: float = ...,
) -> np.ndarray: ...
@overload
def pot_temp_rce_hh80_small_ang(
    lats: Scalar,
    z: np.ndarray,
    theta_ref: float = ...,
    height: float = ...,
    delta_h: float = ...,
    delta_v: float = ...,
) -> np.ndarray: ...
@overload
def pot_temp_rce_hh80_small_ang(
    lats: Scalar,
    z: Scalar = ...,
    theta_ref: float = ...,
    height: float = ...,
    delta_h: float = ...,
    delta_v: float = ...,
) -> Scalar: ...
def pot_temp_rce_hh80_small_ang(
    lats: ArrayLike,
    z: ArrayLike = 0.5 * HEIGHT_TROPO,
    theta_ref: float = THETA_REF,
    height: float = HEIGHT_TROPO,
    delta_h: float = DELTA_H,
    delta_v: float = DELTA_V,
) -> ArrayLike:
    """RCE potential temperature in the small-angle limit (Eq. 2 of HH80).

    Parameters
    ----------
    lats : array-like
        Latitude (degrees).
    z : array-like, optional
        Height (m). Default: mid-troposphere.
    theta_ref : float, optional
        Reference potential temperature (K). Default: THETA_REF.
    height : float, optional
        Tropopause height (m). Default: HEIGHT_TROPO.
    delta_h : float, optional
        Fractional horizontal temperature difference. Default: DELTA_H.
    delta_v : float, optional
        Fractional vertical temperature difference. Default: DELTA_V.

    Returns
    -------
    array-like
        RCE potential temperature in the small-angle limit (K).
    """
    return cast(
        ArrayLike,
        theta_ref
        * (
            1
            + delta_h * (1 - np.deg2rad(lats) ** 2 - 2 / 3)
            + delta_v * (z / height - 0.5)
        ),
    )


@overload
def u_rce_hh80(
    lats: xr.DataArray,
    therm_ross_num: ArrayLike,
    rot_rate: float = ...,
    radius: float = ...,
) -> xr.DataArray: ...
@overload
def u_rce_hh80(
    lats: ArrayLike,
    therm_ross_num: xr.DataArray,
    rot_rate: float = ...,
    radius: float = ...,
) -> xr.DataArray: ...
@overload
def u_rce_hh80(
    lats: np.ndarray,
    therm_ross_num: np.ndarray | Scalar,
    rot_rate: float = ...,
    radius: float = ...,
) -> np.ndarray: ...
@overload
def u_rce_hh80(
    lats: Scalar,
    therm_ross_num: np.ndarray,
    rot_rate: float = ...,
    radius: float = ...,
) -> np.ndarray: ...
@overload
def u_rce_hh80(
    lats: Scalar,
    therm_ross_num: Scalar,
    rot_rate: float = ...,
    radius: float = ...,
) -> Scalar: ...
def u_rce_hh80(
    lats: ArrayLike,
    therm_ross_num: ArrayLike,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
) -> ArrayLike:
    """Zonal wind in gradient balance with RCE temperatures.

    Parameters
    ----------
    lats : array-like
        Latitude (degrees).
    therm_ross_num : float or array-like
        Thermal Rossby number.
    rot_rate : float, optional
        Planetary rotation rate (rad/s). Default: Earth.
    radius : float, optional
        Planetary radius (m). Default: Earth.

    Returns
    -------
    array-like
        Zonal wind (m/s).
    """
    return cast(
        ArrayLike,
        rot_rate * radius * cosdeg(lats) * ((1 + 2 * therm_ross_num) ** 0.5 - 1),
    )


@overload
def dpot_temp_rce_hh80_dlat(lats: xr.DataArray, delta_h: float) -> xr.DataArray: ...
@overload
def dpot_temp_rce_hh80_dlat(lats: np.ndarray, delta_h: float) -> np.ndarray: ...
@overload
def dpot_temp_rce_hh80_dlat(lats: Scalar, delta_h: float) -> Scalar: ...
def dpot_temp_rce_hh80_dlat(lats: ArrayLike, delta_h: float) -> ArrayLike:
    """Meridional derivative of RCE potential temperature with respect to latitude.

    Parameters
    ----------
    lats : array-like
        Latitude (degrees).
    delta_h : float
        Fractional horizontal temperature difference.

    Returns
    -------
    array-like
        d(theta_RCE)/d(lat), normalized by theta_ref.
    """
    return cast(ArrayLike, -2 * delta_h * sindeg(lats) * cosdeg(lats))


@overload
def u_crit_switch_lat_hh80(therm_ross_num: xr.DataArray) -> xr.DataArray: ...
@overload
def u_crit_switch_lat_hh80(therm_ross_num: np.ndarray) -> np.ndarray: ...
@overload
def u_crit_switch_lat_hh80(therm_ross_num: Scalar) -> Scalar: ...
def u_crit_switch_lat_hh80(therm_ross_num: ArrayLike) -> ArrayLike:
    """Latitude where RCE and AMC winds are equal in the Held-Hou 1980 model.

    This is the supercriticality boundary: equatorward of this latitude,
    the RCE state violates Hide's theorem and the Hadley cell must exist.

    Parameters
    ----------
    therm_ross_num : float or array-like
        Thermal Rossby number.

    Returns
    -------
    float or array-like
        Critical latitude (degrees).
    """
    return cast(ArrayLike, np.rad2deg(np.arccos((1 + 2 * therm_ross_num) ** -0.25)))


@overload
def u_crit_switch_lat_hh80_small_angle(
    therm_ross_num: xr.DataArray,
) -> xr.DataArray: ...
@overload
def u_crit_switch_lat_hh80_small_angle(therm_ross_num: np.ndarray) -> np.ndarray: ...
@overload
def u_crit_switch_lat_hh80_small_angle(therm_ross_num: Scalar) -> Scalar: ...
def u_crit_switch_lat_hh80_small_angle(therm_ross_num: ArrayLike) -> ArrayLike:
    """Critical latitude in the small-angle limit of the HH80 model.

    Parameters
    ----------
    therm_ross_num : float or array-like
        Thermal Rossby number.

    Returns
    -------
    float or array-like
        Critical latitude (degrees).

    See Also
    --------
    u_crit_switch_lat_hh80 : Full (non-small-angle) version.
    """
    return cast(ArrayLike, np.rad2deg(therm_ross_num**0.5))


@overload
def hc_edge_hh80_small_angle(therm_ross_num: xr.DataArray) -> xr.DataArray: ...
@overload
def hc_edge_hh80_small_angle(therm_ross_num: np.ndarray) -> np.ndarray: ...
@overload
def hc_edge_hh80_small_angle(therm_ross_num: Scalar) -> Scalar: ...
def hc_edge_hh80_small_angle(therm_ross_num: ArrayLike) -> ArrayLike:
    """Hadley cell edge in the small-angle limit (Eq. 16 of HH80).

    Parameters
    ----------
    therm_ross_num : float or array-like
        Thermal Rossby number.

    Returns
    -------
    float or array-like
        Cell edge latitude (degrees).

    See Also
    --------
    hc_edge_hh80 : Numerical solution of the full (non-small-angle) Eq. 17.
    """
    return cast(ArrayLike, np.rad2deg((5 * therm_ross_num / 3) ** 0.5))


_DEFAULT_BOUND_GUESS_RANGE = np.arange(0.1, 90.1, 10)


def _hc_edge_hh80_lhs(lat: float, therm_ross_num: float) -> float:
    """Left hand side of Eq. 17 of Held Hou 1980 (right hand side is zero)."""
    y = sindeg(lat)
    return float(
        (1 / 3) * (4 * therm_ross_num - 1) * y**3
        - y**5 / (1 - y**2)
        - y
        + 0.5 * np.log((1 + y) / (1 - y))
    )


def hc_edge_hh80(
    therm_ross_num: SolverParamRange,
    init_guess: float = 0.1,
    bound_guess_range: SolverGuessRange | None = None,
) -> xr.DataArray:
    """Hadley cell edge according to Held and Hou 1980, Eq. 17.

    Solved numerically using the Brent (1973) root finding algorithm, as
    implemented in scipy's ``scipy.optimize.brentq`` function.

    The solver needs an interval in which ``_hc_edge_hh80_lhs`` changes sign.
    It finds one by marching through ``bound_guess_range`` until some guess
    gives the opposite sign to that at ``init_guess``, then brackets the root
    between the two.

    Parameters
    ----------

    therm_ross_num : scalar or array-like
        Thermal rossby number value(s) for which to solve.
    init_guess : float, optional
        Latitude (degrees) marking one end of the initial bracket. Default:
        0.1, just off the equator, where Eq. 17 is positive for any physical
        thermal Rossby number.
    bound_guess_range : sequence of float or numpy.ndarray, optional
        Latitudes (degrees) to march through in search of the sign change.
        Default: ``numpy.arange(0.1, 90.1, 10)``.

    Returns
    -------

    hc_edge : xarray.DataArray
        Array of the numerical solution for each thermal Rossby number value in
        `therm_ross_num`. Entries for which no sign change was found anywhere
        in `bound_guess_range` are ``nan`` rather than an error, so a sweep
        over many thermal Rossby numbers is not derailed by one that fails to
        bracket. Check for ``nan`` if the bracket arguments are non-default.

    See Also
    --------
    hc_edge_hh80_small_angle : Closed-form Eq. 16, the small-angle limit.

    """
    if bound_guess_range is None:
        bound_guess_range = _DEFAULT_BOUND_GUESS_RANGE
    return brentq_solver_sweep_param(
        _hc_edge_hh80_lhs, therm_ross_num, init_guess, bound_guess_range
    )


if __name__ == "__main__":
    pass
