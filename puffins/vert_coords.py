#! /usr/bin/env python
"""Vertical coordinate transformations and pressure-level utilities.

Functions for converting between vertical coordinate systems, computing
pressure thicknesses, mass-weighted integrals and averages, and
Simmons-Burridge vertical level spacing. Supports both full-level and
half-level pressure grids, as well as log-pressure weighting.
"""

from __future__ import annotations

import functools
import operator
import warnings
from typing import cast, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from ._typing import ArrayLike, Scalar
from .constants import GRAV_EARTH, MEAN_SLP_EARTH
from .names import (
    LEV_STR,
    PFULL_STR,
    PHALF_STR,
)
from .nb_utils import coord_arr_1d


@overload
def to_pascal(
    arr: xr.DataArray, is_dp: bool = ..., warn: bool = ...
) -> xr.DataArray: ...
@overload
def to_pascal(arr: np.ndarray, is_dp: bool = ..., warn: bool = ...) -> np.ndarray: ...
@overload
def to_pascal(arr: Scalar, is_dp: bool = ..., warn: bool = ...) -> Scalar: ...
def to_pascal(arr: ArrayLike, is_dp: bool = False, warn: bool = False) -> ArrayLike:
    """Force data with units either hPa or Pa to be in Pa.

    Note: unit detection is heuristic (max |value| < threshold). Values that
    are genuinely small Pa (e.g. 600 Pa in the upper atmosphere) will be
    incorrectly treated as hPa and multiplied by 100.

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
        return cast(ArrayLike, arr * 100.0)
    return arr


def int_dp_g(
    arr: ArrayLike,
    dp: xr.DataArray,
    dim: str = LEV_STR,
    grav: float = GRAV_EARTH,
) -> xr.DataArray:
    """Mass-weighted vertical integral: sum(arr * dp) / g.

    Parameters
    ----------
    arr : xarray.DataArray
        Field to integrate.
    dp : xarray.DataArray
        Pressure thickness of each level (Pa).
    dim : str, optional
        Name of the vertical dimension. Default: 'plev'.
    grav : float, optional
        Gravitational acceleration (m/s^2). Default: Earth.

    Returns
    -------
    xarray.DataArray
        Vertically integrated field.

    Notes
    -----
    Levels where ``arr * dp`` is NaN are skipped, so a column with missing
    levels is integrated over the levels that remain, as is correct for levels
    below ground.  A column with no non-NaN level returns NaN, where xarray's
    default sum would give 0.  That includes a layer lying wholly below
    ground, which contains no mass; when integrating over a layer above the
    surface, set to 0 the columns whose surface pressure is lower than the
    layer's top pressure.
    """
    weighted = cast(xr.DataArray, arr * dp)
    return cast(xr.DataArray, weighted.sum(dim=dim, min_count=1) / grav)


def int_dlogp(
    arr: xr.DataArray,
    p_top: float = 0.0,
    p_bot: float = MEAN_SLP_EARTH,
    pfull_str: str = LEV_STR,
    phalf_str: str = PHALF_STR,
) -> xr.DataArray:
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
        Name of the full-level pressure dimension. Default: 'plev'.
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
    return cast(xr.DataArray, (arr * dlogp).sum(pfull_str))


def col_avg(arr: xr.DataArray, dp: xr.DataArray, dim: str = LEV_STR) -> xr.DataArray:
    """Pressure-weighted column average.

    Parameters
    ----------
    arr : xarray.DataArray
        Field to average.
    dp : xarray.DataArray
        Pressure thickness of each level (Pa).
    dim : str, optional
        Name of the vertical dimension. Default: 'plev'.

    Returns
    -------
    xarray.DataArray
        Pressure-weighted column-mean value.
    """
    return cast(xr.DataArray, int_dp_g(arr, dp, dim) / int_dp_g(1.0, dp, dim))


def subtract_col_avg(
    arr: xr.DataArray, dp: xr.DataArray, dim: str = LEV_STR
) -> xr.DataArray:
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
        Name of the vertical dimension. Default: 'plev'.

    Returns
    -------
    xarray.DataArray
        Field with column average subtracted at each level.
    """
    return cast(xr.DataArray, arr - col_avg(arr, dp, dim=dim))


def phalf_from_pfull(
    pfull: xr.DataArray,
    p_top: float = 0.0,
    p_bot: float = MEAN_SLP_EARTH,
    phalf_str: str = PHALF_STR,
) -> xr.DataArray:
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
    return cast(xr.DataArray, coord_arr_1d(values=phalf_vals, dim=phalf_str))


def dp_from_pfull(
    pfull: xr.DataArray, p_top: float = 0.0, p_bot: float = MEAN_SLP_EARTH
) -> xr.DataArray:
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
    return cast(xr.DataArray, np.abs(xr.ones_like(pfull) * np.diff(phalf.values)))


def dp_from_phalf(
    phalf: xr.DataArray,
    pfull_ref: xr.DataArray,
    phalf_str: str = PHALF_STR,
    pfull_str: str = PFULL_STR,
) -> xr.DataArray:
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
        phalf.isel({phalf_str: slice(None, -1)}).values
        - phalf.isel({phalf_str: slice(1, None)}).values
    )
    dims_out: list[str] = []
    for dim in phalf.dims:
        if dim == phalf_str:
            dims_out.append(pfull_str)
        else:
            dims_out.append(str(dim))  # str() narrows Hashable for mypy

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
    return cast(xr.DataArray, xr.ones_like(pressure) * dlogp_vals)


def dlogp_from_pfull(
    pfull: xr.DataArray,
    p_top: float = 0.0,
    p_bot: float = MEAN_SLP_EARTH,
    phalf_str: str = PHALF_STR,
) -> xr.DataArray:
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


@overload
def phalf_from_psfc(
    bk: xr.DataArray, pk: xr.DataArray, p_sfc: xr.DataArray
) -> xr.DataArray: ...
@overload
def phalf_from_psfc(bk: ArrayLike, pk: ArrayLike, p_sfc: ArrayLike) -> ArrayLike: ...
def phalf_from_psfc(bk: ArrayLike, pk: ArrayLike, p_sfc: ArrayLike) -> ArrayLike:
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
    return cast(ArrayLike, p_sfc * bk + pk)


def pfull_from_phalf_avg(
    phalf: xr.DataArray,
    pfull_ref: xr.DataArray,
    phalf_str: str = PHALF_STR,
    pfull_str: str = PFULL_STR,
) -> xr.DataArray:
    """Full-level pressures as simple averages of bounding half levels.

    Works with pressure increasing or decreasing along ``phalf_str``.

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
    # Half the (positive) thickness above the lower-pressure interface of each
    # layer, which is the first of each pair only where pressure increases.
    phalf_top = np.minimum(
        phalf.isel({phalf_str: slice(None, -1)}).values,
        phalf.isel({phalf_str: slice(1, None)}).values,
    )
    return cast(xr.DataArray, (0.5 * dp + phalf_top).rename(pfull_str))


def pfull_from_phalf_log_mean(
    phalf: xr.DataArray,
    pfull_ref: xr.DataArray,
    phalf_str: str = PHALF_STR,
    pfull_str: str = PFULL_STR,
) -> xr.DataArray:
    """Full-level pressures as log-means of bounding half levels.

    The pressure of the layer between half levels p1 and p2 is
    (p2 - p1) / ln(p2 / p1), which lies between p1 and p2 and below their
    arithmetic mean. It is the definition of the Ensemble Mars Atmosphere
    Reanalysis System (EMARS) [1]_. Works with pressure increasing or
    decreasing along ``phalf_str``.

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
        Pressure at full levels, in the units of ``phalf``.

    Raises
    ------
    ValueError
        If any half-level pressure is negative, or zero. The log-mean of a
        layer whose top is at zero pressure is zero, so for a model whose top
        half level is 0 Pa use :func:`pfull_from_phalf_avg` or
        :func:`pfull_simm_burr`.

    References
    ----------
    .. [1] Greybush, S. J., et al. (2019). "The Ensemble Mars Atmosphere
       Reanalysis System (EMARS) Version 1.0." Geosci. Data J., 6, 137-150,
       doi:10.1002/gdj3.77, p. 142.
    """
    if bool((phalf < 0).any()):
        raise ValueError(
            "A half-level pressure is negative; check the surface pressure for"
            " fill values."
        )
    if bool((phalf == 0).any()):
        raise ValueError(
            "The log-mean full-level pressure is zero for a layer whose top half"
            " level is at zero pressure. For a model whose top half level is 0 Pa,"
            " use pfull_from_phalf_avg or pfull_simm_burr"
            " (pfull_method='arithmetic_mean' or 'simmons_burridge' in"
            " interp.interp_eta_to_plevs)."
        )
    dp = dp_from_phalf(phalf, pfull_ref, phalf_str=phalf_str, pfull_str=pfull_str)
    dlogp = np.abs(
        np.log(
            phalf.isel({phalf_str: slice(1, None)}).values
            / phalf.isel({phalf_str: slice(None, -1)}).values
        )
    )
    return cast(xr.DataArray, (dp / dlogp).rename(pfull_str))


def _check_simm_burr_refs(
    phalf: xr.DataArray,
    phalf_ref: xr.DataArray,
    pfull_ref: xr.DataArray,
    phalf_str: str,
) -> None:
    """Raise unless the references can set the Simmons-Burridge top level.

    The top level is scaled by a ratio of the reference pressures, and the
    vertical order is read from the reference half levels. So the references
    must be pressures in one unit, each full level strictly between its two
    half levels, running in the same vertical order as ``phalf``.
    """
    half = np.asarray(phalf_ref, dtype=float)
    full = np.asarray(pfull_ref, dtype=float)
    if full.shape != half[1:].shape or not bool(
        (
            (full > np.minimum(half[:-1], half[1:]))
            & (full < np.maximum(half[:-1], half[1:]))
        ).all()
    ):
        raise ValueError(
            "Each reference full-level pressure must lie strictly between its two"
            " reference half-level pressures, in the same units, since they set"
            " the Simmons-Burridge top level. At least one does not, as when the"
            " references are indices or in different units."
        )
    step = (phalf.isel({phalf_str: 1}) - phalf.isel({phalf_str: 0})).values
    if bool(((step > 0) != (half[1] > half[0]))[np.isfinite(step)].any()):
        raise ValueError(
            "The reference half levels and phalf run in opposite vertical orders."
        )


def pfull_vals_simm_burr(
    phalf: xr.DataArray,
    phalf_ref: xr.DataArray,
    pfull_ref: xr.DataArray,
    phalf_str: str = PHALF_STR,
) -> npt.NDArray[np.float64]:
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
    _check_simm_burr_refs(phalf, phalf_ref, pfull_ref, phalf_str)
    # ``above`` means vertically above (i.e. lower pressure). Handle both
    # pressure orderings the same way as ``pfull_simm_burr`` (issue #26): the
    # top level is index 0 when pressure increases with index and index -1 when
    # it decreases. Without this branch the decreasing case mislabeled both the
    # top and the surface level.
    if float(phalf_ref[0]) < float(phalf_ref[1]):
        phalf_above = phalf.isel({phalf_str: slice(None, -1)})
        phalf_below = phalf.isel({phalf_str: slice(1, None)})
        ind_top = 0
        ind_phalf_next_to_top = 1
    else:
        phalf_above = phalf.isel({phalf_str: slice(1, None)})
        phalf_below = phalf.isel({phalf_str: slice(None, -1)})
        ind_top = -1
        ind_phalf_next_to_top = -2

    # Layer thickness with ``below`` (higher pressure) minus ``above`` (lower
    # pressure), so it is positive regardless of ordering.
    dp_vals = phalf_below.values - phalf_above.values
    dlog_phalf_vals = np.log(phalf_below.values / phalf_above.values)
    phalf_over_dp_vals = phalf_above.values / dp_vals

    alpha_vals = 1.0 - phalf_over_dp_vals * dlog_phalf_vals

    ln_pfull_vals = np.log(phalf_below.values) - alpha_vals
    pfull_vals: npt.NDArray[np.float64] = np.exp(ln_pfull_vals)
    top_lev_factor = float(pfull_ref[ind_top] / phalf_ref[ind_phalf_next_to_top])
    pfull_vals[ind_top] = (
        phalf.isel({phalf_str: ind_phalf_next_to_top}).values * top_lev_factor
    )
    return pfull_vals


def pfull_simm_burr(
    phalf: xr.DataArray,
    phalf_ref: xr.DataArray,
    pfull_ref: xr.DataArray,
    phalf_str: str = PHALF_STR,
    pfull_str: str = PFULL_STR,
) -> xr.DataArray:
    """Compute full-level pressures using Simmons-Burridge spacing.

    Handles both increasing and decreasing pressure ordering. The top level is
    the next half level scaled by the ratio of the top reference full level to
    the same reference half level, and the vertical order is read from
    ``phalf_ref``. A ValueError is raised unless each reference full level
    lies strictly between its two reference half levels, in the same units,
    and the references run in the same vertical order as ``phalf``.

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
    _check_simm_burr_refs(phalf, phalf_ref, pfull_ref, phalf_str)
    # Above means vertically above (i.e. lower pressure).
    if phalf_ref[0] < phalf_ref[1]:
        p_is_increasing = True
        phalf_above = phalf.isel({phalf_str: slice(None, -1)})
        phalf_below = phalf.isel({phalf_str: slice(1, None)})
        ind_phalf_next_to_top = 1
    else:
        p_is_increasing = False
        phalf_above = phalf.isel({phalf_str: slice(1, None)})
        phalf_below = phalf.isel({phalf_str: slice(None, -1)})
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
    # Selecting a single half level drops the vertical dimension, leaving a
    # scalar-coordinate ``pfull``; ``expand_dims`` restores it as a length-1
    # dimension so it concatenates cleanly with ``pfull_not_top`` (issue #26).
    # ``drop=True`` keeps a labeled ``phalf`` from leaving that half level's
    # coordinate value on the result.
    phalf_next_to_top = phalf.isel({phalf_str: ind_phalf_next_to_top}, drop=True)
    pfull_top = cast(
        xr.DataArray, (top_lev_factor * phalf_next_to_top).expand_dims(pfull_str)
    )
    # If the working array carries a ``pfull`` dimension-coordinate, label the
    # expanded top level to match ``pfull_not_top`` so ``xr.concat`` keeps a
    # single consistent index instead of raising "Dimension pfull already
    # exists" (issue #26 for coordinate-labeled inputs).
    if pfull_str in pfull_not_top.coords:
        top_coord = pfull_ref[pfull_str].isel({pfull_str: [ind_top]})
        pfull_top = pfull_top.assign_coords({pfull_str: top_coord.values})

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
        Name of the vertical dimension. Default: 'plev'.

    Returns
    -------
    xarray.DataArray
        Cumulative pressure-weighted average.
    """
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
        Name of the vertical dimension. Default: 'plev'.

    Returns
    -------
    xarray.DataArray
        Cumulative log-pressure-weighted average.
    """
    dlogp = dlogp_from_phalf(phalf, pressure)
    return cast(xr.DataArray, (arr * dlogp).cumsum(p_str) / dlogp.cumsum(p_str))


def col_extrema(arr: xr.DataArray, p_str: str = LEV_STR) -> xr.DataArray:
    """Locations and values of local extrema within each column.

    Identifies levels where the vertical derivative changes sign.

    Parameters
    ----------
    arr : xarray.DataArray
        Field to search for extrema.
    p_str : str, optional
        Name of the vertical dimension. Default: 'plev'.

    Returns
    -------
    xarray.DataArray
        Values of ``arr`` at extrema locations; NaN elsewhere.
    """
    darr_dp = arr.differentiate(p_str)
    sign_change = cast(xr.DataArray, np.sign(darr_dp)).diff(p_str)
    return cast(xr.DataArray, arr.where(sign_change))


if __name__ == "__main__":
    pass
