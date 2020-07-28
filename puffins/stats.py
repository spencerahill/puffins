"""Functionality relating to statistics, timeseries, etc."""
from eofs.xarray import Eof
import numpy as np
import scipy.stats
import xarray as xr

from .names import LAT_STR, LON_STR, YEAR_STR
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


def pointwise_corr_latlon_sweep(arr, arr_sweep, dim_lat=LAT_STR,
                                dim_lon=LON_STR, dim_time=YEAR_STR):
    """Correlation of 1D-arr w/ a (time, lat, lon)-arr at each (lat, lon)."""
    corrs = []
    for lat in arr_sweep[dim_lat]:
        corrs.append([scipy.stats.pearsonr(
            arr_sweep.fillna(0.).sel(lat=lat).sel(lon=lon),
            arr)[0] for lon in arr_sweep[dim_lon]
        ])
    arr_sweep_0 = arr_sweep.isel(**{dim_time: 0}, drop=True)
    return (xr.ones_like(arr_sweep_0) *
            np.array(corrs).reshape(arr_sweep_0.shape))


# Empirical orthogonal functions (EOFs)
def eof_solver(arr, lat_str=LAT_STR, time_str=YEAR_STR):
    """Generate an EOF solver for gridded lat-lon data."""
    weights = np.sqrt(cosdeg(arr[lat_str])).values[..., np.newaxis]
    # The eofs package requires that the time dimension be named "time".
    return Eof(arr.rename({time_str: "time"}), weights=weights)
