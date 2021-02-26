#! /usr/bin/env python
"""Utility functions useful for interactive work in Jupyter notebooks."""

import importlib
import os.path
from subprocess import PIPE, Popen
import warnings

from IPython.display import display, Javascript
import git
import numpy as np
import xarray as xr

from .constants import RAD_EARTH
from .names import DAY_OF_YEAR_STR, LAT_STR, TIME_STR


# Quality control for production notebooks
def _checkout_if_clean(repo, branch_name):
    """Checkout the given branch of the given repo."""
    clean_and_tracked = not (repo.is_dirty() or repo.untracked_files)
    if clean_and_tracked:
        repo.git.checkout(branch_name)
    else:
        raise Exception(f"The repo in the directory '{repo.working_dir}' "
                        "has an untracked file or uncommitted changes.  These "
                        "must be handled before puffins gets switched to "
                        "this project's branch.")


def _package_rootdir(name):
    """Get the root directory of the installed package with the given name."""
    initfile = importlib.util.find_spec(name).origin
    return os.path.split(os.path.split(initfile)[0])[0]


def setup_puffins(branch_name="master"):
    """Switch puffins to the specified branch, if current branch is clean."""
    puffins_repo = git.Repo(_package_rootdir("puffins"))
    _checkout_if_clean(puffins_repo, branch_name)


def save_jupyter_nb():
    """From within a Jupyter notebook, save that notebook.

    https://stackoverflow.com/a/57814673/

    """
    display(Javascript('Jupyter.notebook.save_checkpoint();'))


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
def coord_arr_1d(start=None, stop=None, spacing=None, dim=None, values=None,
                 dtype=None):
    """Create xr.DataArray of an evenly spaced 1D coordinate ."""
    if values is None:
        arr_np = np.arange(start, stop + 0.1*spacing, spacing)
    else:
        arr_np = np.asarray(values)
    if dtype is not None:
        arr_np = arr_np.astype(dtype)
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
                       func_args=None, func_kwargs=None,
                       stacked_dim="dummy_stacked_dim"):
    """Apply a function via 'groupby' over the given dimensions."""
    ds = _arrs_to_ds(arrs)
    func_on_ds = _func_arrs_to_ds(func, args=func_args,
                                  kwargs=func_kwargs)
    if isinstance(groupby_dims, str):
        groupby_dims = (groupby_dims,)

    # Stack/unstack to iterate over all dims that are to be grouped/applied
    # over.
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
def zero_cross_bounds(arr, dim, num_cross):
    """Find the values bounding an array's zero crossing."""
    sign_switch = np.sign(arr).diff(dim)
    switch_val = arr[dim].where(sign_switch, drop=True)[num_cross]
    lower_bound = max(0.999*switch_val, np.min(arr[dim]))
    upper_bound = min(1.001*switch_val, np.max(arr[dim]))
    return arr.sel(**{dim: [lower_bound, upper_bound], "method": "backfill"})


def first_zero_cross_bounds(arr, dim):
    """Find the values bounding an array's first zero crossing."""
    return zero_cross_bounds(arr, dim, 0)


def last_zero_cross_bounds(arr, dim):
    """Find the values bounding an array's last zero crossing."""
    return zero_cross_bounds(arr, dim, -1)


def zero_cross_nh(arr, lat_str=LAT_STR):
    lats = arr[lat_str]
    return lats[np.abs(arr.where(lats > 0)).argmin()]


# Data processing and cleaning.
def symmetrize_hemispheres(ds, vars_to_flip_sign=None, lat_str=LAT_STR):
    """Symmetrize data about the equator.

    Parameters
    ----------

    ds : xr.Dataset
        Dataset to be symmetrized

    vars_to_flip_sign: None or sequence of strings, default None
        List of variables, (typically those involving meridional wind), which
        are mirror-symmetric about the equator, rather than just symmetric.  So
        we have to multiply their SH values by -1 before symmetrizing,
        otherwise the results end up being nearly zero.
    lat_str : str, default ``names.LAT_STR``

    Returns
    -------

    ds_symm : xr.Dataset
        Dataset with the variables from the original dataset symmetrized about
        the equator

    """
    lats = ds[lat_str]
    north_hem = ds.where(lats > 0, drop=True)
    south_hem = ds.where(lats < 0, drop=True).isel(lat=slice(-1, None, -1))

    if vars_to_flip_sign is None:
        vars_to_flip_sign = []
    for varname in vars_to_flip_sign:
        south_hem[varname] = -1*south_hem[varname]

    south_hem[lat_str] = north_hem[lat_str]
    ds_hem_avg = 0.5*(south_hem + north_hem)

    ds_opp = ds_hem_avg.copy(deep=True)
    ds_opp = ds_opp.isel(lat=slice(-1, None, -1))

    # Note: because of an xarray bug, can't use `ds_opp[lat_str] *= -1` here,
    # because in that case it also multiplies `ds_avg[lat_str]` by -1.
    ds_opp[lat_str] = ds_opp[lat_str]*-1
    ds_symm = xr.concat([ds_opp, ds_hem_avg], dim=lat_str)

    for varname in vars_to_flip_sign:
        ds_symm[varname] = ds_symm[varname]*np.sign(ds_symm[lat_str])
    return ds_symm


# Misc.
def lat_area_weight(lat, radius=RAD_EARTH):
    """Geometric factor corresponding to surface area at each latitude."""
    return 2.*np.pi*radius*cosdeg(lat)


def to_pascal(arr, is_dp=False, warn=False):
    """Force data with units either hPa or Pa to be in Pa."""
    threshold = 400 if is_dp else 1200
    if np.max(np.abs(arr)) < threshold:
        if warn:
            warn_msg = "Conversion applied: hPa -> Pa to array: {}".format(arr)
            warnings.warn(warn_msg)
        return arr * 100.0
    return arr


def drop_dupes(sequence):
    """Drop duplicate elements from a list or other sequence.

    C.f. https://stackoverflow.com/a/7961390

    """
    orig_type = type(sequence)
    return orig_type(list(dict.fromkeys(sequence)))


if __name__ == '__main__':
    pass
