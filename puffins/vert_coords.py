#! /usr/bin/env python
"""Vertical coordinate transformations and pressure-level utilities.

Functions for converting between vertical coordinate systems, computing
pressure thicknesses, mass-weighted integrals and averages, and
Simmons-Burridge vertical level spacing. Supports both full-level and
half-level pressure grids, as well as log-pressure weighting.
"""

import functools
import operator
import warnings

import numpy as np
import xarray as xr

from .constants import GRAV_EARTH, MEAN_SLP_EARTH
from .names import (
    LEV_STR,
    PFULL_STR,
    PHALF_STR,
)
from .nb_utils import coord_arr_1d


def to_pascal(arr, is_dp=False, warn=False):
    """Force data with units either hPa or Pa to be in Pa.

    Heuristically detects whether values are in hPa or Pa based on their
    magnitude and converts to Pa if needed.

    Parameters
    ----------
    arr : array-like
        Pressure data in either hPa or Pa.
    is_dp : bool, optional
        If True, use a lower threshold (400) for detection, appropriate
        for pressure differences. Default: False.
    warn : bool, optional
        If True, emit a warning when conversion is applied.
        Default: False.

    Returns
    -------
    array-like
        Pressure values in Pa.
    """
    threshold = 400 if is_dp else 1200
    if np.max(np.abs(arr)) < threshold:
        if warn:
            warn_msg = f"Conversion applied: hPa -> Pa to array: {arr}"
            warnings.warn(warn_msg, stacklevel=2)
        return arr * 100.0
    return arr


def int_dp_g(arr, dp, dim=LEV_STR, grav=GRAV_EARTH):
    """Mass-weighted vertical integral: sum(arr * dp) / g.

    Parameters
    ----------
    arr : xarray.DataArray
        Field to integrate.
    dp : xarray.DataArray
        Pressure thickness of each level (Pa).
    dim : str, optional
        Name of the vertical dimension. Default: 'level'.
    grav : float, optional
        Gravitational acceleration (m/s^2). Default: Earth.

    Returns
    -------
    xarray.DataArray
        Vertically integrated field.
    """
    return (arr * dp).sum(dim=dim) / grav


def int_dlogp(
    arr, p_top=0.0, p_bot=MEAN_SLP_EARTH, pfull_str=LEV_STR, phalf_str=PHALF_STR
):
    """Log-pressure-weighted vertical integral.

    Parameters
    ----------
    arr : xarray.DataArray
        Field to integrate, with a pressure coordinate.
    p_top : float, optional
        Top pressure boundary (Pa). Default: 0.0.
    p_bot : float, optional
        Bottom pressure boundary (Pa). Default: MEAN_SLP_EARTH.
    pfull_str : str, optional
        Name of the full-level pressure dimension. Default: 'level'.
    phalf_str : str, optional
        Name of the half-level pressure dimension. Default: 'phalf'.

    Returns
    -------
    xarray.DataArray
        Log-pressure-weighted integral.
    """
    dlogp = dlogp_from_pfull(
        arr[pfull_str], p_top=p_top, p_bot=p_bot, phalf_str=phalf_str
    )
    return (arr * dlogp).sum(pfull_str)


def col_avg(arr, dp, dim=LEV_STR):
    """Pressure-weighted column average.

    Parameters
    ----------
    arr : xarray.DataArray
        Field to average.
    dp : xarray.DataArray
        Pressure thickness of each level (Pa).
    dim : str, optional
        Name of the vertical dimension. Default: 'level'.

    Returns
    -------
    xarray.DataArray
        Pressure-weighted column-mean value.
    """
    return int_dp_g(arr, dp, dim) / int_dp_g(1.0, dp, dim)


def subtract_col_avg(arr, dp, dim=LEV_STR):
    """Impose zero column integral by subtracting column average at each level.

    Used e.g. for computing the zonally integrated mass flux. In the time-mean
    and neglecting tendencies in column mass, the column integrated meridional
    mass transport should be zero at each latitude; otherwise there would be a
    build up of mass on one side.

    Parameters
    ----------
    arr : xarray.DataArray
        Field to adjust.
    dp : xarray.DataArray
        Pressure thickness of each level (Pa).
    dim : str, optional
        Name of the vertical dimension. Default: 'level'.

    Returns
    -------
    xarray.DataArray
        Field with column average subtracted at each level.
    """
    return arr - col_avg(arr, dp, dim=dim)


def phalf_from_pfull(pfull, p_top=0.0, p_bot=MEAN_SLP_EARTH, phalf_str=PHALF_STR):
    """Pressure at half levels given pressures at level centers.

    Computes interface pressures by averaging adjacent full-level
    pressures, with top and bottom boundaries set to ``p_top`` and
    ``p_bot``.

    Parameters
    ----------
    pfull : xarray.DataArray
        Pressure at level centers.
    p_top : float, optional
        Pressure at the model top (Pa). Default: 0.0.
    p_bot : float, optional
        Pressure at the surface (Pa). Default: MEAN_SLP_EARTH.
    phalf_str : str, optional
        Name of the half-level dimension. Default: 'phalf'.

    Returns
    -------
    xarray.DataArray
        Pressure at half levels (Pa).
    """
    if pfull[0] < pfull[1]:
        p_first = p_top
        p_last = p_bot
    else:
        p_first = p_bot
        p_last = p_top
    phalf_inner_vals = 0.5 * (pfull.values[1:] + pfull.values[:-1])
    phalf_vals = np.concatenate([[p_first], phalf_inner_vals, [p_last]])
    return coord_arr_1d(values=phalf_vals, dim=phalf_str)


def dp_from_pfull(pfull, p_top=0.0, p_bot=MEAN_SLP_EARTH):
    """Pressure thickness of levels given pressures at level centers.

    Parameters
    ----------
    pfull : xarray.DataArray
        Pressure at level centers.
    p_top : float, optional
        Pressure at the model top (Pa). Default: 0.0.
    p_bot : float, optional
        Pressure at the surface (Pa). Default: MEAN_SLP_EARTH.

    Returns
    -------
    xarray.DataArray
        Pressure thickness dp for each level (Pa).
    """
    phalf = phalf_from_pfull(pfull, p_top=p_top, p_bot=p_bot)
    return np.abs(xr.ones_like(pfull) * np.diff(phalf.values))


def dp_from_phalf(phalf, pfull_ref, phalf_str=PHALF_STR, pfull_str=PFULL_STR):
    """Pressure thickness of vertical levels given interface (half-level) pressures.

    Parameters
    ----------
    phalf : xarray.DataArray
        Pressure at half levels (interfaces).
    pfull_ref : xarray.DataArray
        Reference pressure at full levels, used for output coordinates.
    phalf_str : str, optional
        Name of the half-level dimension. Default: 'phalf'.
    pfull_str : str, optional
        Name of the full-level dimension. Default: 'pfull'.

    Returns
    -------
    xarray.DataArray
        Pressure thickness dp for each full level (Pa), named 'dp'.
    """
    dp_vals = np.abs(
        phalf.isel(**{phalf_str: slice(None, -1)}).values
        - phalf.isel(**{phalf_str: slice(1, None)}).values
    )
    dims_out = []
    for dim in phalf.dims:
        if dim == "phalf":
            dims_out.append(pfull_str)
        else:
            dims_out.append(dim)

    vals_template = [
        xr.ones_like(phalf[dim]) for dim in phalf.dims if dim != phalf_str
    ] + [pfull_ref]
    # Use reduce(mul) instead of np.prod: numpy >=2 can't handle a list of
    # DataArrays with different shapes, but pairwise mul triggers xarray
    # broadcasting correctly.
    arr_template = xr.ones_like(
        functools.reduce(operator.mul, vals_template)
    ).transpose(*dims_out)
    return (arr_template * dp_vals).rename("dp").astype("float")


def dlogp_from_phalf(phalf, pressure):
    """Log-pressure thickness from half-level pressures.

    Computes d(ln p) = ln(p_{k+1/2} / p_{k-1/2}) for each full level.
    If the top or bottom interface pressure is zero, it is replaced with
    half the adjacent interface value to avoid division by zero.

    Parameters
    ----------
    phalf : xarray.DataArray
        Pressure at half levels (interfaces).
    pressure : xarray.DataArray
        Pressure at full levels, used as template for output shape.

    Returns
    -------
    xarray.DataArray
        Log-pressure thickness for each full level.
    """
    # Avoid divide-by-zero error by overwriting if top pressure is zero.
    phalf_vals = phalf.copy().values
    if phalf_vals[0] == 0:
        phalf_vals[0] = 0.5 * phalf_vals[1]
    elif phalf_vals[-1] == 0:
        phalf_vals[-1] = 0.5 * phalf_vals[-2]
    dlogp_vals = np.log(phalf_vals[1:] / phalf_vals[:-1])
    return xr.ones_like(pressure) * dlogp_vals


def dlogp_from_pfull(pfull, p_top=0.0, p_bot=MEAN_SLP_EARTH, phalf_str=PHALF_STR):
    """Log-pressure thickness from full-level pressures.

    Convenience wrapper: computes half-level pressures from full-level
    pressures, then calls :func:`dlogp_from_phalf`.

    Parameters
    ----------
    pfull : xarray.DataArray
        Pressure at level centers.
    p_top : float, optional
        Pressure at the model top (Pa). Default: 0.0.
    p_bot : float, optional
        Pressure at the surface (Pa). Default: MEAN_SLP_EARTH.
    phalf_str : str, optional
        Name of the half-level dimension. Default: 'phalf'.

    Returns
    -------
    xarray.DataArray
        Log-pressure thickness for each full level.
    """
    phalf = phalf_from_pfull(pfull, p_top=p_top, p_bot=p_bot, phalf_str=phalf_str)
    return dlogp_from_phalf(phalf, pfull)


def phalf_from_psfc(bk, pk, p_sfc):
    """Compute pressure at half levels for hybrid sigma-pressure coordinates.

    Parameters
    ----------
    bk : array-like
        Sigma coefficients at half levels.
    pk : array-like
        Pressure coefficients at half levels (Pa).
    p_sfc : xarray.DataArray or float
        Surface pressure (Pa).

    Returns
    -------
    xarray.DataArray or array-like
        Pressure at half levels (Pa).
    """
    return p_sfc * bk + pk


def pfull_from_phalf_avg(phalf, pfull_ref, phalf_str=PHALF_STR, pfull_str=PFULL_STR):
    """Full-level pressures as simple averages of bounding half levels.

    Parameters
    ----------
    phalf : xarray.DataArray
        Pressure at half levels (interfaces).
    pfull_ref : xarray.DataArray
        Reference full-level pressures for output coordinates.
    phalf_str : str, optional
        Name of the half-level dimension. Default: 'phalf'.
    pfull_str : str, optional
        Name of the full-level dimension. Default: 'pfull'.

    Returns
    -------
    xarray.DataArray
        Pressure at full levels (Pa).
    """
    dp = dp_from_phalf(phalf, pfull_ref, phalf_str=phalf_str, pfull_str=pfull_str)
    return (phalf.isel(**{phalf_str: slice(None, -1)}).values + 0.5 * dp).rename(
        pfull_str
    )


def pfull_vals_simm_burr(phalf, phalf_ref, pfull_ref, phalf_str=PHALF_STR):
    """Compute full-level pressure values using Simmons-Burridge spacing.

    Returns raw numpy values (not an xarray DataArray). Use
    :func:`pfull_simm_burr` for the full xarray-aware version.

    Parameters
    ----------
    phalf : xarray.DataArray
        Pressure at half levels (interfaces).
    phalf_ref : xarray.DataArray
        Reference half-level pressures (for top-level factor).
    pfull_ref : xarray.DataArray
        Reference full-level pressures (for top-level factor).
    phalf_str : str, optional
        Name of the half-level dimension. Default: 'phalf'.

    Returns
    -------
    numpy.ndarray
        Full-level pressure values (Pa).

    References
    ----------
    .. [1] Simmons, A. J. & Burridge, D. M. (1981). "An Energy and
       Angular-Momentum Conserving Vertical Finite-Difference Scheme and
       Hybrid Vertical Coordinates." Mon. Wea. Rev., 109, 758-766.

    See Also
    --------
    pfull_simm_burr : xarray-aware wrapper.
    """
    dp_vals = phalf.diff(phalf_str).values
    # Above means vertically above (i.e. lower pressure).
    phalf_above = phalf.isel(phalf=slice(None, -1))
    phalf_below = phalf.isel(phalf=slice(1, None))

    dlog_phalf_vals = np.log(phalf_below.values / phalf_above.values)
    phalf_over_dp_vals = phalf_above.values / dp_vals

    alpha_vals = 1.0 - phalf_over_dp_vals * dlog_phalf_vals

    ln_pfull_vals = np.log(phalf_below.values) - alpha_vals
    pfull_vals = np.exp(ln_pfull_vals)
    top_lev_factor = float(pfull_ref[0] / phalf_ref[1])
    pfull_vals[0] = phalf.isel(phalf=1) * top_lev_factor
    return pfull_vals


def pfull_simm_burr(
    phalf, phalf_ref, pfull_ref, phalf_str=PHALF_STR, pfull_str=PFULL_STR
):
    """Compute full-level pressures using Simmons-Burridge spacing.

    Handles both increasing and decreasing pressure ordering.

    Parameters
    ----------
    phalf : xarray.DataArray
        Pressure at half levels (interfaces).
    phalf_ref : xarray.DataArray
        Reference half-level pressures.
    pfull_ref : xarray.DataArray
        Reference full-level pressures.
    phalf_str : str, optional
        Name of the half-level dimension. Default: 'phalf'.
    pfull_str : str, optional
        Name of the full-level dimension. Default: 'pfull'.

    Returns
    -------
    xarray.DataArray
        Pressure at full levels (Pa).

    References
    ----------
    .. [1] Simmons, A. J. & Burridge, D. M. (1981). "An Energy and
       Angular-Momentum Conserving Vertical Finite-Difference Scheme and
       Hybrid Vertical Coordinates." Mon. Wea. Rev., 109, 758-766.

    See Also
    --------
    pfull_vals_simm_burr : Returns raw numpy values.
    """
    # Above means vertically above (i.e. lower pressure).
    if phalf_ref[0] < phalf_ref[1]:
        p_is_increasing = True
        phalf_above = phalf.isel(phalf=slice(None, -1))
        phalf_below = phalf.isel(phalf=slice(1, None))
        ind_phalf_next_to_top = 1
    else:
        p_is_increasing = False
        phalf_above = phalf.isel(phalf=slice(1, None))
        phalf_below = phalf.isel(phalf=slice(None, -1))
        ind_phalf_next_to_top = -2

    dlog_phalf_vals = np.log(phalf_below.values / phalf_above.values)

    dp = dp_from_phalf(phalf, pfull_ref, phalf_str=phalf_str, pfull_str=pfull_str)
    phalf_over_dp_vals = phalf_above.values / dp.values
    alpha_vals = 1.0 - phalf_over_dp_vals * dlog_phalf_vals

    ln_pfull_vals = np.log(phalf_below.values) - alpha_vals
    pfull_vals = np.exp(ln_pfull_vals)
    pfull = xr.ones_like(dp) * pfull_vals

    # Top level has its own procedure.
    if p_is_increasing:
        ind_top = 0
        pfull_not_top = pfull.isel(**{pfull_str: slice(1, None)})
    else:
        ind_top = -1
        pfull_not_top = pfull.isel(**{pfull_str: slice(None, -1)})

    top_lev_factor = float(pfull_ref[ind_top] / phalf_ref[ind_phalf_next_to_top])
    pfull_top = top_lev_factor * phalf.isel(**{phalf_str: ind_phalf_next_to_top})
    pfull_top.coords["pfull"] = pfull_ref.isel(**{pfull_str: ind_top})

    if p_is_increasing:
        return xr.concat([pfull_top, pfull_not_top], pfull_str)
    return xr.concat([pfull_not_top, pfull_top], pfull_str)


def _flip_dim(arr, dim):
    return arr.isel(**{dim: slice(None, None, -1)})


def avg_p_weighted(arr, phalf, pressure, p_str=LEV_STR):
    """Cumulative pressure-weighted vertical average.

    Computes the running pressure-weighted mean from the top of the
    atmosphere downward (or upward, depending on pressure ordering).

    Parameters
    ----------
    arr : xarray.DataArray
        Field to average.
    phalf : xarray.DataArray
        Pressure at half levels (interfaces).
    pressure : xarray.DataArray
        Pressure at full levels.
    p_str : str, optional
        Name of the vertical dimension. Default: 'level'.

    Returns
    -------
    xarray.DataArray
        Cumulative pressure-weighted average.
    """
    dp = np.abs(dp_from_phalf(phalf, pressure))
    if phalf[0] > phalf[1]:
        arr_out = _flip_dim(arr, p_str)
        dp_out = _flip_dim(dp, p_str)
    else:
        arr_out = arr
        dp_out = dp
    return (arr_out * dp_out).cumsum(p_str) / dp_out.cumsum(p_str)


def avg_logp_weighted(arr, phalf, pressure, p_str=LEV_STR):
    """Cumulative log-pressure-weighted vertical average.

    Parameters
    ----------
    arr : xarray.DataArray
        Field to average.
    phalf : xarray.DataArray
        Pressure at half levels (interfaces).
    pressure : xarray.DataArray
        Pressure at full levels.
    p_str : str, optional
        Name of the vertical dimension. Default: 'level'.

    Returns
    -------
    xarray.DataArray
        Cumulative log-pressure-weighted average.
    """
    dlogp = dlogp_from_phalf(phalf, pressure)
    return (arr * dlogp).cumsum(p_str) / dlogp.cumsum(p_str)


def col_extrema(arr, p_str=LEV_STR):
    """Locations and values of local extrema within each column.

    Identifies levels where the vertical derivative changes sign.

    Parameters
    ----------
    arr : xarray.DataArray
        Field to search for extrema.
    p_str : str, optional
        Name of the vertical dimension. Default: 'level'.

    Returns
    -------
    xarray.DataArray
        Values of ``arr`` at extrema locations; NaN elsewhere.
    """
    darr_dp = arr.differentiate(p_str)
    sign_change = np.sign(darr_dp).diff(p_str)
    return arr.where(sign_change)


if __name__ == "__main__":
    pass
