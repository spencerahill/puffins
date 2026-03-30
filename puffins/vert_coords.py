#! /usr/bin/env python
"""Functionality involving vertical coordinates."""

from __future__ import annotations

import functools
import operator
import warnings
from typing import cast

import numpy as np
import xarray as xr

from ._typing import ArrayLike
from .constants import GRAV_EARTH, MEAN_SLP_EARTH
from .names import (
    LEV_STR,
    PFULL_STR,
    PHALF_STR,
)
from .nb_utils import coord_arr_1d


def to_pascal(arr: ArrayLike, is_dp: bool = False, warn: bool = False) -> ArrayLike:
    """Force data with units either hPa or Pa to be in Pa."""
    threshold = 400 if is_dp else 1200
    if np.max(np.abs(arr)) < threshold:
        if warn:
            warn_msg = f"Conversion applied: hPa -> Pa to array: {arr}"
            warnings.warn(warn_msg, stacklevel=2)
        return cast(ArrayLike, arr * 100.0)
    return arr


def int_dp_g(
    arr: ArrayLike,
    dp: xr.DataArray,
    dim: str = LEV_STR,
    grav: float = GRAV_EARTH,
) -> xr.DataArray:
    """Mass weighted integral.  Assumes `dp` is in Pa (not hPa)."""
    weighted = cast(xr.DataArray, arr * dp)
    return cast(xr.DataArray, weighted.sum(dim=dim) / grav)


def int_dlogp(
    arr: xr.DataArray,
    p_top: float = 0.0,
    p_bot: float = MEAN_SLP_EARTH,
    pfull_str: str = LEV_STR,
    phalf_str: str = PHALF_STR,
) -> xr.DataArray:
    """Integral of array on pressure levels but weighted by log(pressure)."""
    dlogp = dlogp_from_pfull(
        arr[pfull_str], p_top=p_top, p_bot=p_bot, phalf_str=phalf_str
    )
    return cast(xr.DataArray, (arr * dlogp).sum(pfull_str))


def col_avg(arr: xr.DataArray, dp: xr.DataArray, dim: str = LEV_STR) -> xr.DataArray:
    """Pressure-weighted column average."""
    return cast(xr.DataArray, int_dp_g(arr, dp, dim) / int_dp_g(1.0, dp, dim))


def subtract_col_avg(
    arr: xr.DataArray, dp: xr.DataArray, dim: str = LEV_STR
) -> xr.DataArray:
    """Impoze zero column integral by subtracting column average at each level.

    Used e.g. for computing the zonally integrated mass flux.  In the time-mean
    and neglecting tendencies in column mass, the column integrated meridional
    mass transport should be zero at each latitude; otherwise there would be a
    build up of mass on one side.

    """
    return cast(xr.DataArray, arr - col_avg(arr, dp, dim=dim))


def phalf_from_pfull(
    pfull: xr.DataArray,
    p_top: float = 0.0,
    p_bot: float = MEAN_SLP_EARTH,
    phalf_str: str = PHALF_STR,
) -> xr.DataArray:
    """Pressure at half levels given pressures at level centers."""
    if pfull[0] < pfull[1]:
        p_first = p_top
        p_last = p_bot
    else:
        p_first = p_bot
        p_last = p_top
    phalf_inner_vals = 0.5 * (pfull.values[1:] + pfull.values[:-1])
    phalf_vals = np.concatenate([[p_first], phalf_inner_vals, [p_last]])
    return cast(xr.DataArray, coord_arr_1d(values=phalf_vals, dim=phalf_str))


def dp_from_pfull(
    pfull: xr.DataArray, p_top: float = 0.0, p_bot: float = MEAN_SLP_EARTH
) -> xr.DataArray:
    """Pressure thickness of levels given pressures at level centers."""
    phalf = phalf_from_pfull(pfull, p_top=p_top, p_bot=p_bot)
    return cast(xr.DataArray, np.abs(xr.ones_like(pfull) * np.diff(phalf.values)))


def dp_from_phalf(
    phalf: xr.DataArray,
    pfull_ref: xr.DataArray,
    phalf_str: str = PHALF_STR,
    pfull_str: str = PFULL_STR,
) -> xr.DataArray:
    """Pressure thickness of vertical levels given interface pressures."""
    dp_vals = np.abs(
        phalf.isel({phalf_str: slice(None, -1)}).values
        - phalf.isel({phalf_str: slice(1, None)}).values
    )
    dims_out: list[str] = []
    for dim in phalf.dims:
        if dim == "phalf":
            dims_out.append(pfull_str)
        else:
            dims_out.append(str(dim))

    vals_template = [
        xr.ones_like(phalf[dim]) for dim in phalf.dims if dim != phalf_str
    ] + [pfull_ref]
    # Use reduce(mul) instead of np.prod: numpy >=2 can't handle a list of
    # DataArrays with different shapes, but pairwise mul triggers xarray
    # broadcasting correctly.
    arr_template = xr.ones_like(
        functools.reduce(operator.mul, vals_template)
    ).transpose(*dims_out)
    return cast(xr.DataArray, (arr_template * dp_vals).rename("dp").astype("float"))


def dlogp_from_phalf(phalf: xr.DataArray, pressure: xr.DataArray) -> xr.DataArray:
    """Pressure thickness of vertical levels given interface pressures."""
    # Avoid divide-by-zero error by overwriting if top pressure is zero.
    phalf_vals = phalf.copy().values
    if phalf_vals[0] == 0:
        phalf_vals[0] = 0.5 * phalf_vals[1]
    elif phalf_vals[-1] == 0:
        phalf_vals[-1] = 0.5 * phalf_vals[-2]
    dlogp_vals = np.log(phalf_vals[1:] / phalf_vals[:-1])
    return cast(xr.DataArray, xr.ones_like(pressure) * dlogp_vals)


def dlogp_from_pfull(
    pfull: xr.DataArray,
    p_top: float = 0.0,
    p_bot: float = MEAN_SLP_EARTH,
    phalf_str: str = PHALF_STR,
) -> xr.DataArray:
    """Thickness in log(p) of vertical levels given level-center pressures."""
    phalf = phalf_from_pfull(pfull, p_top=p_top, p_bot=p_bot, phalf_str=phalf_str)
    return dlogp_from_phalf(phalf, pfull)


def phalf_from_psfc(bk: ArrayLike, pk: ArrayLike, p_sfc: ArrayLike) -> ArrayLike:
    """Compute pressure of half levels of hybrid sigma-pressure coordinates."""
    return cast(ArrayLike, p_sfc * bk + pk)


def pfull_from_phalf_avg(
    phalf: xr.DataArray,
    pfull_ref: xr.DataArray,
    phalf_str: str = PHALF_STR,
    pfull_str: str = PFULL_STR,
) -> xr.DataArray:
    """Compute pressure at full levels as average of bounding half levels."""
    dp = dp_from_phalf(phalf, pfull_ref, phalf_str=phalf_str, pfull_str=pfull_str)
    return cast(
        xr.DataArray,
        (phalf.isel({phalf_str: slice(None, -1)}).values + 0.5 * dp).rename(pfull_str),
    )


def pfull_vals_simm_burr(
    phalf: xr.DataArray,
    phalf_ref: xr.DataArray,
    pfull_ref: xr.DataArray,
    phalf_str: str = PHALF_STR,
) -> np.ndarray:
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

    alpha_vals = 1.0 - phalf_over_dp_vals * dlog_phalf_vals

    ln_pfull_vals = np.log(phalf_below.values) - alpha_vals
    pfull_vals: np.ndarray = np.exp(ln_pfull_vals)
    top_lev_factor = float(pfull_ref[0] / phalf_ref[1])
    pfull_vals[0] = phalf.isel(phalf=1) * top_lev_factor
    return pfull_vals


def pfull_simm_burr(
    phalf: xr.DataArray,
    phalf_ref: xr.DataArray,
    pfull_ref: xr.DataArray,
    phalf_str: str = PHALF_STR,
    pfull_str: str = PFULL_STR,
) -> xr.DataArray:
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

    dp = dp_from_phalf(phalf, pfull_ref, phalf_str=phalf_str, pfull_str=pfull_str)
    phalf_over_dp_vals = phalf_above.values / dp.values
    alpha_vals = 1.0 - phalf_over_dp_vals * dlog_phalf_vals

    ln_pfull_vals = np.log(phalf_below.values) - alpha_vals
    pfull_vals = np.exp(ln_pfull_vals)
    pfull = cast(xr.DataArray, xr.ones_like(dp) * pfull_vals)

    # Top level has its own procedure.
    if p_is_increasing:
        ind_top = 0
        pfull_not_top = pfull.isel({pfull_str: slice(1, None)})
    else:
        ind_top = -1
        pfull_not_top = pfull.isel({pfull_str: slice(None, -1)})

    top_lev_factor = float(pfull_ref[ind_top] / phalf_ref[ind_phalf_next_to_top])
    pfull_top = cast(
        xr.DataArray,
        top_lev_factor * phalf.isel({phalf_str: ind_phalf_next_to_top}),
    )
    pfull_top.coords["pfull"] = pfull_ref.isel({pfull_str: ind_top})

    if p_is_increasing:
        return cast(xr.DataArray, xr.concat([pfull_top, pfull_not_top], pfull_str))
    return cast(xr.DataArray, xr.concat([pfull_not_top, pfull_top], pfull_str))


def _flip_dim(arr: xr.DataArray, dim: str) -> xr.DataArray:
    return cast(xr.DataArray, arr.isel({dim: slice(None, None, -1)}))


def avg_p_weighted(
    arr: xr.DataArray,
    phalf: xr.DataArray,
    pressure: xr.DataArray,
    p_str: str = LEV_STR,
) -> xr.DataArray:
    """Pressure-weighted vertical average."""
    dp = cast(xr.DataArray, np.abs(dp_from_phalf(phalf, pressure)))
    if phalf[0] > phalf[1]:
        arr_out = _flip_dim(arr, p_str)
        dp_out = _flip_dim(dp, p_str)
    else:
        arr_out = arr
        dp_out = dp
    return cast(xr.DataArray, (arr_out * dp_out).cumsum(p_str) / dp_out.cumsum(p_str))


def avg_logp_weighted(
    arr: xr.DataArray,
    phalf: xr.DataArray,
    pressure: xr.DataArray,
    p_str: str = LEV_STR,
) -> xr.DataArray:
    """Log-pressure-weighted vertical average."""
    dlogp = dlogp_from_phalf(phalf, pressure)
    return cast(xr.DataArray, (arr * dlogp).cumsum(p_str) / dlogp.cumsum(p_str))


def col_extrema(arr: xr.DataArray, p_str: str = LEV_STR) -> xr.DataArray:
    """Locations and values of local extrema within each column."""
    darr_dp = arr.differentiate(p_str)
    sign_change = np.sign(darr_dp).diff(p_str)
    return cast(xr.DataArray, arr.where(sign_change))


if __name__ == "__main__":
    pass
