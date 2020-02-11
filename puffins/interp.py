#! /usr/bin/env python
"""Functions for interpolation."""

import numpy as np
import xarray as xr

from .names import LAT_STR, LEV_STR


P_INTERP_VALS = np.arange(0, 1000.5, 1.)


def drop_all_nan_slices(arrs, dim=LAT_STR):
    """Drop values of the given dimension that are all NaN."""
    out = []
    stack_dim = 'stacked_dim'
    for arr in arrs:
        other_dims = [od for od in arr.dims if od != dim]
        assert stack_dim not in other_dims
        stacked = arr.stack(**{stack_dim: other_dims})
        out.append(stacked.dropna(dim, how='all'))
    return out


def interp_p(arr, new_p_vals=P_INTERP_VALS,
             method='linear', p_str=LEV_STR):
    """Interpolate an array to new pressure values."""
    return arr.interp(**{p_str: new_p_vals}, method=method)


def interp_arrs_in_p(arrs, new_p_vals=P_INTERP_VALS,
                     method='cubic', lat_str=LAT_STR,
                     p_str=LEV_STR):
    """Interpolate all of the given arrays in pressure."""
    return [interp_p(arr.where(np.isfinite(arr), drop=True),
                     new_p_vals=new_p_vals, p_str=p_str,
                     method=method).interp(
        **{lat_str: arr[lat_str]}) for arr in arrs]


def _maybe_interp_in_p(arrs, new_p_vals=P_INTERP_VALS,
                       do_interp=True, p_str=LEV_STR):
    """Either interpolate arrays in pressure or return originals."""
    if do_interp:
        return interp_arrs_in_p(arrs, new_p_vals=new_p_vals,
                                p_str=p_str)
    else:
        return arrs


def drop_nans_and_interp(arrs, new_p_vals=P_INTERP_VALS, do_interp=True,
                         nan_drop_dim=LAT_STR, p_str=LEV_STR):
    """Drop all-NaN latitudes and maybe interpolate."""
    interped = _maybe_interp_in_p(arrs, new_p_vals=new_p_vals,
                                  do_interp=do_interp, p_str=p_str)
    return drop_all_nan_slices(interped, nan_drop_dim)


def interpolate(x, y, x_target, dim):
    """Interpolate between two points.

    Parameters
    ----------
    x, y : xarray.DataArray or xarray.Dataset
        'x' is the "coordinate", 'y' is the array whose value at the
        specified value of 'x' (namely 'x_target') is to be interpolated
    x_target : scalar
        Value of the 'x' coordinate at which to interpolate the 'y' array.
    dim : str
        Name of dimension that holds the 'x' coordinate

    """
    x_vals = x.values
    y_vals = y.values
    y_target_vals = (y_vals[0] + (y_vals[1] - y_vals[0]) /
                     (x_vals[1] - x_vals[0]) * (x_target - x_vals[0]))
    y_target = xr.ones_like(x[0])*y_target_vals
    y_target[dim].values = y_target_vals
    return y_target


if __name__ == '__main__':
    pass
