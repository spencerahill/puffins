#! /usr/bin/env python
"""Functions for interpolation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from xarray.core.types import InterpOptions

from .names import (
    BK_STR,
    LAT_STR,
    LEV_STR,
    P_SFC_STR,
    PFULL_STR,
    PK_STR,
    SIGMA_STR,
)
from .vert_coords import pfull_from_phalf_avg, phalf_from_psfc

P_INTERP_VALS: np.ndarray = np.arange(0, 1000.5, 1.0)


def drop_all_nan_slices(
    arrs: Iterable[xr.DataArray], dim: str = LAT_STR
) -> list[xr.DataArray]:
    """Drop values of the given dimension that are all NaN."""
    out = []
    stack_dim = "stacked_dim"
    for arr in arrs:
        other_dims = [od for od in arr.dims if od != dim]
        assert stack_dim not in other_dims
        stacked = arr.stack({stack_dim: other_dims})
        out.append(stacked.dropna(dim, how="all"))
    return out


def interp_p(
    arr: xr.DataArray,
    new_p_vals: np.ndarray = P_INTERP_VALS,
    method: InterpOptions = "linear",
    p_str: str = LEV_STR,
) -> xr.DataArray:
    """Interpolate an array to new pressure values."""
    return cast(xr.DataArray, arr.interp({p_str: new_p_vals}, method=method))


def interp_arrs_in_p(
    arrs: Iterable[xr.DataArray],
    new_p_vals: np.ndarray = P_INTERP_VALS,
    method: InterpOptions = "cubic",
    lat_str: str = LAT_STR,
    p_str: str = LEV_STR,
) -> list[xr.DataArray]:
    """Interpolate all of the given arrays in pressure."""
    return [
        cast(
            xr.DataArray,
            interp_p(
                arr.where(np.isfinite(arr), drop=True),
                new_p_vals=new_p_vals,
                p_str=p_str,
                method=method,
            ).interp({lat_str: arr[lat_str]}),
        )
        for arr in arrs
    ]


def _maybe_interp_in_p(
    arrs: Iterable[xr.DataArray],
    new_p_vals: np.ndarray = P_INTERP_VALS,
    do_interp: bool = True,
    p_str: str = LEV_STR,
) -> list[xr.DataArray]:
    """Either interpolate arrays in pressure or return originals."""
    if do_interp:
        return interp_arrs_in_p(arrs, new_p_vals=new_p_vals, p_str=p_str)
    return list(arrs)


def drop_nans_and_interp(
    arrs: Iterable[xr.DataArray],
    new_p_vals: np.ndarray = P_INTERP_VALS,
    do_interp: bool = True,
    nan_drop_dim: str = LAT_STR,
    p_str: str = LEV_STR,
) -> list[xr.DataArray]:
    """Drop all-NaN latitudes and maybe interpolate."""
    interped = _maybe_interp_in_p(
        arrs, new_p_vals=new_p_vals, do_interp=do_interp, p_str=p_str
    )
    return drop_all_nan_slices(interped, nan_drop_dim)


def interpolate(
    x: xr.DataArray, y: xr.DataArray, x_target: float, dim: str
) -> xr.DataArray:
    """Linearly interpolate between two points.

    Parameters
    ----------
    x, y : xarray.DataArray
        'x' is the "coordinate", 'y' is the array whose value at the
        specified value of 'x' (namely 'x_target') is to be interpolated
    x_target : scalar
        Value of the 'x' coordinate at which to interpolate the 'y' array.
    dim : str
        Name of dimension that holds the 'x' coordinate

    """
    x_vals = x.values
    y_vals = y.values
    y_target_vals = y_vals[0] + (y_vals[1] - y_vals[0]) / (x_vals[1] - x_vals[0]) * (
        x_target - x_vals[0]
    )
    y_target = cast(xr.DataArray, xr.ones_like(x[0]) * y_target_vals)
    y_target[dim].values = y_target_vals
    return y_target


def interp_ds_sigma_to_p(
    ds: xr.Dataset,
    plevs: int | np.ndarray,
    method: InterpOptions = "cubic",
    p_sfc_str: str = P_SFC_STR,
    sigma_str: str = SIGMA_STR,
    lev_str: str = LEV_STR,
    lat_str: str = LAT_STR,
) -> xr.Dataset:
    """Interpolate Dataset with sigma coordinates to uniform pressures."""
    p_sfc = ds[p_sfc_str]
    if isinstance(plevs, int):
        p_sfc_min = float(ds[sigma_str].max() * p_sfc.min())
        p_top_max = float(ds[sigma_str].min() * p_sfc.max())
        p_fixed_vals = np.linspace(p_top_max, p_sfc_min, plevs)
    else:
        p_fixed_vals = plevs
    p_fixed = xr.DataArray(
        p_fixed_vals, dims=[sigma_str], coords={sigma_str: p_fixed_vals}, name="p"
    )
    sigma_fixed = p_fixed / p_sfc

    interped = []
    lats = ds[lat_str]
    for j, _lat in enumerate(lats):
        ds_tmp = ds.isel({lat_str: j}, drop=True).interp(
            {sigma_str: sigma_fixed.isel(lat=j, drop=True)}, method=method
        )
        interped.append(
            ds_tmp.assign_coords({sigma_str: p_fixed}).rename({sigma_str: lev_str})
        )

    return cast(
        xr.Dataset, xr.concat(interped, dim=lats).transpose(lev_str, lat_str, ...)
    )


def interp_ds_p_to_p(
    ds: xr.Dataset,
    plevs: np.ndarray,
    method: InterpOptions = "cubic",
    pfull_str: str = PFULL_STR,
    lev_str: str = LEV_STR,
    lat_str: str = LAT_STR,
) -> xr.Dataset:
    """Interpolate Dataset with varying-p to uniform-p coordinates."""
    p_fixed = xr.DataArray(plevs, dims=[pfull_str], coords={pfull_str: plevs}, name="p")
    interped = []
    lats = ds[lat_str]
    for lat in lats:
        ds_tmp = ds.sel({lat_str: lat}, drop=True).interp(
            {pfull_str: p_fixed}, method=method
        )
        interped.append(
            ds_tmp.assign_coords({pfull_str: p_fixed}).rename({pfull_str: lev_str})
        )

    return cast(
        xr.Dataset, xr.concat(interped, dim=lats).transpose(lev_str, lat_str, ...)
    )


# Array zero crossings.
def zero_cross_bounds(arr: xr.DataArray, dim: str, num_cross: int = 0) -> xr.DataArray:
    """Find the values bounding an array's zero crossing."""
    sign_switch = cast(xr.DataArray, np.sign(arr)).diff(dim)
    switch_arr = arr[dim].where(sign_switch, drop=True)
    if len(switch_arr) == 0:
        raise ValueError("Didn't find any zero crossings")
    switch_val = switch_arr[num_cross]
    return cast(xr.DataArray, arr.sel({dim: slice(None, switch_val)})[-2:])


def zero_cross_interp(arr: xr.DataArray, dim: str, num_cross: int = 0) -> xr.DataArray:
    """Find an array's zero crossing, with interpolation."""
    bounds = zero_cross_bounds(arr, dim, num_cross)
    return cast(xr.DataArray, interpolate(bounds, bounds[dim], 0, dim).rename(dim))


# Interpolation to fixed pressure levels.
def interp_eta_to_plevs(
    ds: xr.Dataset,
    plevs_target: np.ndarray,
    dim: str,
    method: InterpOptions = "cubic",
    pfull_str: str = PFULL_STR,
    bk_str: str = BK_STR,
    pk_str: str = PK_STR,
    p_sfc_str: str = P_SFC_STR,
    lev_str: str = LEV_STR,
    lat_str: str = LAT_STR,
) -> xr.Dataset:
    """Interpolate Dataset from eta coordinates to fixed pressure levels.

    Note: the logic only works correctly if the dataset is only defined in
    pressure and one other dimension (e.g. latitude).  It will silently give
    errant results if there are other dimensions.

    """
    phalf = phalf_from_psfc(ds[bk_str], ds[pk_str], ds["ps"])
    pfull = pfull_from_phalf_avg(phalf, ds[pfull_str])
    pfull *= 1e-2  # Pa to hPa

    interped = []
    for val in ds[dim]:
        pfull_this_val = pfull.sel({dim: val}, drop=True)
        ds_this_val = ds.sel({dim: val}, drop=True).assign_coords(
            {pfull_str: pfull_this_val}
        )
        interped_this_val = ds_this_val.interp({pfull_str: plevs_target}, method=method)
        interped.append(interped_this_val)
    return cast(xr.Dataset, xr.concat(interped, dim=ds[dim]))


def interp_nested_to_plevs(
    ds: xr.Dataset,
    plevs_target: np.ndarray,
    dim1: str,
    dim2: str,
    method: InterpOptions = "cubic",
    pfull_str: str = PFULL_STR,
    bk_str: str = BK_STR,
    pk_str: str = PK_STR,
    p_sfc_str: str = P_SFC_STR,
    lev_str: str = LEV_STR,
    lat_str: str = LAT_STR,
) -> xr.Dataset:
    """Interpolate Dataset from eta coordinates to fixed pressure levels.

    Note: the logic only works correctly if the dataset is only defined in
    pressure and *two* other dimension (e.g. latitude and time).  It will
    silently give errant results if there are other dimensions.

    """
    coord2 = ds[dim2]
    interped = [
        interp_eta_to_plevs(
            ds.sel({dim2: val2}),
            plevs_target,
            dim1,
            method=method,
            pfull_str=pfull_str,
            bk_str=bk_str,
            pk_str=pk_str,
            p_sfc_str=p_sfc_str,
            lev_str=lev_str,
            lat_str=lat_str,
        )
        for val2 in coord2
    ]
    return cast(xr.Dataset, xr.concat(interped, dim=coord2))


if __name__ == "__main__":
    pass
