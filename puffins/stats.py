"""Functionality relating to statistics, timeseries, etc."""
from eofs.xarray import Eof
import numpy as np
import ruptures as rpt
import sklearn.metrics
import scipy.stats
import xarray as xr

from .names import LAT_STR, LON_STR, YEAR_STR
from .nb_utils import cosdeg


# Trends: computing trends, detrending, etc.
def trend(arr, dim="year", order=1, return_coeffs=False):
    """Compute linear or higher-order polynomial fit.

    If return_coeffs is True, then coeffs.degree(sel=0) is the y-intercept,
    coeffs.degree(sel=1) is the slope, etc.

    """
    coeffs = arr.polyfit(dim, order)["polyfit_coefficients"]
    if return_coeffs:
        return coeffs
    return xr.polyval(arr[dim], coeffs)


def detrend(arr, dim="year", order=1):
    """Subtract off the linear or higher order polynomial fit."""
    return arr - trend(arr, dim, order)


# Common timeseries transforms: anomalies, standardized anomalies, etc.
def anomaly(arr, dim="time"):
    """Deviation at each point from the mean along a dimension."""
    return arr - arr.mean(dim)


def standardize(arr, dim="time"):
    """Anomaly normalized by the standard deviation along a dimension."""
    return anomaly(arr, dim) / arr.std(dim)


def dt_std_anom(arr, dim="year", order=1):
    """Detrended standardized anomaly timeseries."""
    return detrend(standardize(arr, dim), dim=dim, order=order)


# Filtering (time or otherwise)
def run_mean(arr, n=10, dim="time", center=True, **kwargs):
    """Simple running average along a dimension."""
    return arr.rolling(**{dim: n}, center=center, **kwargs).mean().dropna(dim)


def run_mean_anom(arr, n=10, dim="time", center=True, **kwargs):
    """Anomaly at each point from a running average along a dimension."""
    return arr - run_mean(arr, n, dim, center=center, **kwargs)


# Correlations and linear regression.
def corr_detrended(arr1, arr2, dim, order=1):
    """Correlation coefficient of two arrays after they are detrended."""
    return xr.corr(detrend(arr1, dim, order), detrend(arr2, dim, order), dim)


def corr_where_overlap(arr1, arr2, dim):
    """Compute corr. coeff. for overlapping portion of the two arrays."""
    return float(xr.corr(arr1, arr2, dim))


def pointwise_corr_latlon_sweep(arr, arr_sweep, dim_lat=LAT_STR,
                                dim_lon=LON_STR, dim_time=YEAR_STR):
    """Correlation of 1D-arr w/ a (time, lat, lon)-arr at each (lat, lon).

    TODO: This can all be replaced w/ xr.corr right?

    """
    corrs = []
    for lat in arr_sweep[dim_lat]:
        corrs.append([scipy.stats.pearsonr(
            arr_sweep.fillna(0.).sel(lat=lat).sel(lon=lon),
            arr)[0] for lon in arr_sweep[dim_lon]
        ])
    arr_sweep_0 = arr_sweep.isel(**{dim_time: 0}, drop=True)
    return (xr.ones_like(arr_sweep_0) *
            np.array(corrs).reshape(arr_sweep_0.shape))


def lin_regress(arr1, arr2, dim):
    """Use xr.apply_ufunc to broadcast scipy.stats.linregress.

    For example, over latitude and longitude.

    Adapated from
    https://github.com/pydata/xarray/issues/1815#issuecomment-614216243.

    """
    def _linregress(x, y):
        """Wrapper around scipy.stats.linregress to use in apply_ufunc."""
        slope, intercept, r_val, p_val, std_err = scipy.stats.linregress(x, y)
        return np.array([slope, intercept, r_val, p_val, std_err])

    arr = xr.apply_ufunc(
        _linregress,
        arr1_trunc,
        arr2_trunc,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[["parameter"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=['float64'],
        dask_gufunc_kwargs=dict(output_sizes={"parameter": 5}),
    )
    arr.coords["parameter"] = xr.DataArray(
        ["slope", "intercept", "r_value", "p_value", "std_err"],
        dims=["parameter"],
    )
    return arr


def rmse(arr1, arr2, dim):
    """Root mean square error using xr.apply_ufunc to broadcast.

    Adapated from
    https://github.com/pydata/xarray/issues/1815#issuecomment-614216243.

    """
    def _rmse(x, y):
        """Wrapper around scipy.stats.linregress to use in apply_ufunc."""
        return sklearn.metrics.mean_squared_error(x, y, squared=False)

    return xr.apply_ufunc(_rmse, arr1, arr2, input_core_dims=[[dim], [dim]],
                          vectorize=True, dask="parallelized")


# Breakpoint detection
def detect_breakpoint(arr, dim, rpt_class=rpt.Binseg, model="l2",
                      n_bkps=1):
    """Use xr.apply_ufunc to broadcast breakpoint detections from ruptures"""

    rpt_instance = rpt_class(model=model)

    def _detect_bp(arr):
        """Wrapper to use in apply_ufunc."""
        return rpt_instance.fit(arr).predict(n_bkps=n_bkps)[0]

    inds_bp = xr.apply_ufunc(
        _detect_bp,
        arr,
        input_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
    )
    return arr[dim][inds_bp]


# Empirical orthogonal functions (EOFs)
def eof_solver(arr, lat_str=LAT_STR, time_str=YEAR_STR):
    """Generate an EOF solver for gridded lat-lon data."""
    weights = np.sqrt(cosdeg(arr[lat_str])).values[..., np.newaxis]
    # The eofs package requires that the time dimension be named "time".
    return Eof(arr.rename({time_str: "time"}), weights=weights)
