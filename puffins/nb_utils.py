#! /usr/bin/env python
"""Utility functions useful for interactive work."""

from subprocess import PIPE, Popen
import warnings.warn

import numpy as np
import xarray as xr

from .names import DAY_OF_YEAR_STR, LAT_STR, TIME_STR


# Quality control for production notebooks
def check_nb_unused_imports(nb_path):
    """List all instances of unused imports in the given Jupyter notebook

    Adapted from:
    - https://stackoverflow.com/a/56591258
    - https://stackoverflow.com/a/15100663

    """
    p1 = Popen(f"jupyter nbconvert {nb_path} --stdout --to python".split(),
               stdout=PIPE)
    p2 = Popen("flake8 - --select=F401".split(), stdin=p1.stdout,
               stdout=PIPE)
    p1.stdout.close()
    return p2.communicate()[0].decode("utf-8")


def warn_if_unused_imports(nb_path):
    """Issue warning if unused imports are found in this notebook."""
    warnlog = check_nb_unused_imports(nb_path)
    if warnlog:
        warnings.warn("This notebook has the following unused imports: "
                      f"\n\n{warnlog}")


# Coordinate arrays.
def coord_arr_1d(start, stop, spacing, dim):
    """Create xr.DataArray of an evenly spaced 1D coordinate ."""
    arr_np = np.arange(start, stop + 0.1*spacing, spacing)
    return xr.DataArray(arr_np, name=dim, dims=[dim],
                        coords={dim: arr_np})


def lat_arr(start=-90, stop=90, spacing=1., dim=LAT_STR):
    """Convenience function to create an array of latitudes."""
    if start is None and stop is None:
        start = -90 + 0.5*spacing
        stop = 90 - 0.5*spacing
    return coord_arr_1d(start, stop, spacing, dim)


def time_arr(start=0, stop=100, spacing=1., dim=TIME_STR):
    """Convenience function to create an array of times."""
    return coord_arr_1d(start, stop, spacing, dim)


def days_of_year_arr(start=1, stop=365, spacing=1., dim=DAY_OF_YEAR_STR):
    """Convenience function to create an array of times."""
    return coord_arr_1d(start, stop, spacing, dim)


# Trigonometric functions with arguments in degrees.
def sindeg(lats_deg):
    """Sine, where the argument is in degrees, not radians."""
    return np.sin(np.deg2rad(lats_deg))


def cosdeg(lats_deg):
    """Cosine, where the argument is in degrees, not radians."""
    return np.cos(np.deg2rad(lats_deg))


def tandeg(lats_deg):
    """Tangent, where the argument is in degrees, not radians."""
    return np.tan(np.deg2rad(lats_deg))


def sin2deg(sinlat):
    """Convert from `sin(lat)` to `lat` in degrees."""
    return np.rad2deg(np.arcsin(sinlat))


# Logic for converting functions that operate on arrays of only one dimension
# to work on multi-dimensional arrays, repeating the calculation over all other
# dimensions.
def _arrs_to_ds(arrs, names=None):
    """Combine DataArrays into a single Dataset."""
    if names is None:
        names = [str(n) for n in range(len(arrs))]
    return xr.Dataset(data_vars=dict(zip(names, arrs)))


def _func_arrs_to_ds(func, args=None, kwargs=None):
    """Create new function that calls the original on a Dataset."""
    def func_on_ds(ds, args=args, kwargs=kwargs):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        return func(*ds.data_vars.values(), *args, **kwargs)
    return func_on_ds


def groupby_apply_func(func, arrs, groupby_dims,
                       func_args=None, func_kwargs=None):
    """Apply a function via 'groupby' over the given dimensions."""
    ds = _arrs_to_ds(arrs)
    func_on_ds = _func_arrs_to_ds(func, args=func_args,
                                  kwargs=func_kwargs)
    if isinstance(groupby_dims, str):
        groupby_dims = (groupby_dims,)

    # Stack/unstack to iterate over all dims that are to be
    # grouped/applied over.
    stacked_dim = 'dummy_stacked_dim'
    assert stacked_dim not in ds.dims
    return ds.stack(**{stacked_dim: groupby_dims}).groupby(
        stacked_dim).apply(func_on_ds).unstack(stacked_dim)


def apply_maybe_groupby(func, non_groupby_dims, arrs, args=None, kwargs=None):
    """Apply the function, grouping over the given dimension if it exists."""
    # Make sure all arrs have same dimensions.
    assert np.all([arr.dims == arrs[0].dims for arr in arrs])
    groupby_dims = [dim for dim in arrs[0].dims if dim not in non_groupby_dims]
    if len(groupby_dims) == 0:
        # No extra dimensions, so just apply the function.
        func_args = arrs
        if args is not None:
            func_args += args
        return func(*func_args, **kwargs)
    else:
        return groupby_apply_func(func, arrs, groupby_dims,
                                  func_args=args, func_kwargs=kwargs)


# Find locations of array extrema.
def max_and_argmax(arr, do_min=False):
    """Get extremum value and associated coordinate values."""
    method = arr.argmin if do_min else arr.argmax
    indexes = np.unravel_index(method(), arr.shape)
    return arr[indexes]


def max_and_argmax_along_dim(dataset, dim, do_min=False):
    """Extremum and its coords for each value of a dimension."""
    grouped = dataset.groupby(dim, squeeze=False)
    return grouped.apply(max_and_argmax, do_min=do_min)


# Array zero crossings.
def zero_cross_nh(arr, lat_str=LAT_STR):
    lats = arr[lat_str]
    return lats[np.abs(arr.where(lats > 0)).argmin()]


if __name__ == '__main__':
    pass
