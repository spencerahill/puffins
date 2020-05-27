"""Functionality relating to statistics, timeseries, etc."""
from eofs.xarray import Eof
import numpy as np
import scipy.stats
import xarray as xr

from .names import LAT_STR, YEAR_STR
from .nb_utils import cosdeg


# Trends: computing trends, detrending, etc.
def trend(arr, dim, order=1, ret_slope_y0=False):
    """Compute linear or higher-order polynomial fit."""
    coord = arr.coords[dim]
    slope, y0 = np.polyfit(coord, arr, order)
    if ret_slope_y0:
        return slope, y0
    return xr.ones_like(arr)*np.polyval([slope, y0], coord)


def detrend(arr, dim, order=1):
    """Subtract off the linear or higher order polynomial fit."""
    return arr - trend(arr, dim, order)


# Common timeseries transforms: anomalies, standardized anomalies, etc.
def anomaly(arr, dim="time"):
    """Deviation at each point from the mean along a dimension."""
    return arr - arr.mean(dim)


def standardize(arr, dim="time"):
    """Anomaly normalized by the standard deviation along a dimension."""
    return anomaly(arr, dim) / arr.std(dim)


# Filtering (time or otherwise)
def run_mean(arr, n=10, dim="time", center=True, **kwargs):
    """Simple running average along a dimension."""
    return arr.rolling(**{dim: n}, center=center, **kwargs).mean().dropna(dim)


def run_mean_anom(arr, n=10, dim="time", center=True, **kwargs):
    """Anomaly at each point from a running average along a dimension."""
    return arr - run_mean(arr, n, dim, center=center, **kwargs)


# Correlations.
def sel_shared_vals(arr1, arr2, dim):
    """Restrict two arrays to their shared values along a dimension.

    Helpful for computing things like correlations which require the
    two arrays to be equal length.

    """
    min_shared_val = max(arr1[dim].min(), arr2[dim].min())
    max_shared_val = min(arr1[dim].max(), arr2[dim].max())
    shared_vals = (arr1[dim] >= min_shared_val) & (arr1[dim] <= max_shared_val)
    arr1_trunc = arr1.where(shared_vals, drop=True)
    arr2_trunc = arr2.where(shared_vals, drop=True)
    return arr1_trunc, arr2_trunc


def corr_where_overlap(arr1, arr2, dim):
    """Compute corr. coeff. for overlapping portion of the two arrays."""
    arr1_shared, arr2_shared = sel_shared_vals(arr1, arr2, dim)
    return scipy.stats.pearsonr(arr1_shared, arr2_shared)[0]


# Empirical orthogonal functions (EOFs)
def eof_solver(arr, lat_str=LAT_STR, time_str=YEAR_STR):
    """Generate an EOF solver for gridded lat-lon data."""
    weights = np.sqrt(cosdeg(arr[lat_str])).values[..., np.newaxis]
    # The eofs package requires that the time dimension be named "time".
    return Eof(arr.rename({time_str: "time"}), weights=weights)
