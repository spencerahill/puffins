#! /usr/bin/env python
"""Functionality involving vertical coordinates."""
import numpy as np
import xarray as xr

from .calculus import integrate
from .constants import GRAV_EARTH, MEAN_SLP_EARTH
from .names import (
    LEV_STR,
    PFULL_STR,
    PHALF_STR,
)
from .nb_utils import coord_arr_1d


def to_pascal(arr, is_dp=False, warn=False):
    """Force data with units either hPa or Pa to be in Pa."""
    threshold = 400 if is_dp else 1200
    if np.max(np.abs(arr)) < threshold:
        if warn:
            warn_msg = "Conversion applied: hPa -> Pa to array: {}".format(arr)
            warnings.warn(warn_msg)
        return arr * 100.0
    return arr


def int_dp_g(arr, dp, dim=LEV_STR, grav=GRAV_EARTH):
    """Mass weighted integral.  Assumes `dp` is in Pa (not hPa)."""
    return integrate(arr, dp, dim=dim) / grav


def int_dlogp(arr, p_top=0., p_bot=MEAN_SLP_EARTH, pfull_str=LEV_STR,
              phalf_str=PHALF_STR):
    """Integral of array on pressure levels but weighted by log(pressure)."""
    dlogp = dlogp_from_pfull(arr[pfull_str], p_top=p_top, p_bot=p_bot,
                             phalf_str=phalf_str)
    return integrate(arr, dlogp, dim=pfull_str)


def phalf_from_pfull(pfull, p_top=0., p_bot=MEAN_SLP_EARTH,
                     phalf_str=PHALF_STR):
    """Pressure at half levels given pressures at level centers."""
    if pfull[0] < pfull[1]:
        p_first = p_top
        p_last = p_bot
    else:
        p_first = p_bot
        p_last = p_top
    phalf_inner_vals = 0.5*(pfull.values[1:] + pfull.values[:-1])
    phalf_vals = np.concatenate([[p_first], phalf_inner_vals, [p_last]])
    return coord_arr_1d(values=phalf_vals, dim=phalf_str)


def dp_from_pfull(pfull, p_top=0., p_bot=MEAN_SLP_EARTH):
    """Pressure thickness of levels given pressures at level centers."""
    phalf = phalf_from_pfull(pfull, p_top=p_top, p_bot=p_bot)
    return np.abs(xr.ones_like(pfull) * np.diff(phalf.values))


def dp_from_phalf(phalf, pfull_ref, phalf_str=PHALF_STR, pfull_str=PFULL_STR):
    """Pressure thickness of vertical levels given interface pressures."""
    dp_vals = np.abs(phalf.isel(**{phalf_str: slice(None, -1)}).values -
                     phalf.isel(**{phalf_str: slice(1, None)}).values)
    dims_out = []
    for dim in phalf.dims:
        if dim == "phalf":
            dims_out.append(pfull_str)
        else:
            dims_out.append(dim)

    vals_template = ([xr.ones_like(phalf[dim]) for dim in phalf.dims
                      if dim != phalf_str] + [pfull_ref])
    arr_template = xr.ones_like(np.product(vals_template)).transpose(*dims_out)
    return (arr_template * dp_vals).rename("dp").astype("float")


def dlogp_from_phalf(phalf, pressure):
    """Pressure thickness of vertical levels given interface pressures."""
    # Avoid divide-by-zero error by overwriting if top pressure is zero.
    phalf_vals = phalf.copy().values
    if phalf_vals[0] == 0:
        phalf_vals[0] = 0.5 * phalf_vals[1]
    elif phalf_vals[-1] == 0:
        phalf_vals[-1] = 0.5 * phalf_vals[-2]
    dlogp_vals = np.log(phalf_vals[1:] / phalf_vals[:-1])
    return xr.ones_like(pressure) * dlogp_vals


def dlogp_from_pfull(pfull, p_top=0., p_bot=MEAN_SLP_EARTH,
                     phalf_str=PHALF_STR):
    """Thickness in log(p) of vertical levels given level-center pressures."""
    phalf = phalf_from_pfull(pfull, p_top=p_top, p_bot=p_bot,
                             phalf_str=phalf_str)
    return dlogp_from_phalf(phalf, pfull)


def phalf_from_psfc(bk, pk, p_sfc):
    """Compute pressure of half levels of hybrid sigma-pressure coordinates."""
    return p_sfc * bk + pk


def pfull_from_phalf_avg(phalf, pfull_ref, phalf_str=PHALF_STR,
                         pfull_str=PFULL_STR):
    """Compute pressure of half levels of hybrid sigma-pressure coordinates."""
    dp = dp_from_phalf(phalf, pfull_ref, phalf_str=phalf_str,
                       pfull_str=pfull_str)
    return (phalf.isel(**{phalf_str: slice(None, -1)}).values +
            0.5 * dp).rename(pfull_str)


def pfull_vals_simm_burr(phalf, phalf_ref, pfull_ref, phalf_str=PHALF_STR):
    """Compute pressure at full levels using Simmons-Burridge spacing.

    See Simmons and Burridge, 1981, "An Energy and Angular-Momentum Conserving
    Vertical Finite-Difference Scheme and Hybrid Vertical Coordinates."
    Monthly Weather Review, 109(4), 758-766.

    """
    dp_vals = phalf.diff(phalf_str).values
    # Above means vertically above (i.e. lower pressure).
    phalf_above = phalf.isel(phalf=slice(None, -1))
    phalf_below = phalf.isel(phalf=slice(1, None))

    dlog_phalf_vals = np.log(phalf_below.values / phalf_above.values)
    phalf_over_dp_vals = phalf_above.values / dp_vals

    alpha_vals = 1. - phalf_over_dp_vals*dlog_phalf_vals

    ln_pfull_vals = np.log(phalf_below.values) - alpha_vals
    pfull_vals = np.exp(ln_pfull_vals)
    top_lev_factor = float(pfull_ref[0] / phalf_ref[1])
    pfull_vals[0] = phalf.isel(phalf=1) * top_lev_factor
    return pfull_vals


def pfull_simm_burr(phalf, phalf_ref, pfull_ref, phalf_str=PHALF_STR,
                    pfull_str=PFULL_STR):
    """Compute pressure at full levels using Simmons-Burridge spacing.

    See Simmons and Burridge, 1981, "An Energy and Angular-Momentum Conserving
    Vertical Finite-Difference Scheme and Hybrid Vertical Coordinates."
    Monthly Weather Review, 109(4), 758-766.

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

    dp = dp_from_phalf(phalf, pfull_ref, phalf_str=phalf_str,
                       pfull_str=pfull_str)
    phalf_over_dp_vals = phalf_above.values / dp.values
    alpha_vals = 1. - phalf_over_dp_vals * dlog_phalf_vals

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

    top_lev_factor = float(pfull_ref[ind_top] /
                           phalf_ref[ind_phalf_next_to_top])
    pfull_top = top_lev_factor * phalf.isel(
        **{phalf_str: ind_phalf_next_to_top})
    pfull_top.coords["pfull"] = pfull_ref.isel(**{pfull_str: ind_top})

    if p_is_increasing:
        return xr.concat([pfull_top, pfull_not_top], pfull_str)
    return xr.concat([pfull_not_top, pfull_top], pfull_str)


def _flip_dim(arr, dim):
    return arr.isel(**{dim: slice(None, None, -1)})


def avg_p_weighted(arr, phalf, pressure, p_str=LEV_STR):
    """Pressure-weighted vertical average."""
    dp = np.abs(dp_from_phalf(phalf, pressure))
    if phalf[0] > phalf[1]:
        arr_out = _flip_dim(arr, p_str)
        dp_out = _flip_dim(dp, p_str)
    else:
        arr_out = arr
        dp_out = dp
    return (arr_out*dp_out).cumsum(p_str) / dp_out.cumsum(p_str)


def avg_logp_weighted(arr, phalf, pressure, p_str=LEV_STR):
    """Log-pressure-weighted vertical average."""
    dlogp = dlogp_from_phalf(phalf, pressure)
    return (arr * dlogp).cumsum(p_str) / dlogp.cumsum(p_str)


def col_extrema(arr, p_str=LEV_STR):
    """Locations and values of local extrema within each column."""
    darr_dp = arr.differentiate(p_str)
    sign_change = np.sign(darr_dp).diff(p_str)
    return arr.where(sign_change)


if __name__ == "__main__":
    pass
