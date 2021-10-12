"""Functionality relating to statistics, timeseries, etc."""
from eofs.xarray import Eof
import numpy as np
import ruptures as rpt
from sklearn import linear_model
import sklearn.metrics
import scipy.stats
import xarray as xr

from .names import LAT_STR, LON_STR, YEAR_STR
from .nb_utils import cosdeg


# TODO: make this a decorator.
def _infer_dim_if_1d(arr, dim):
    """Helper function to get dim name of 1D arrays."""
    if dim is None:
        if arr.ndim == 1:
            dim = arr.dims[0]
        else:
            raise ValueError("Dimension must be specified if array isn't 1D.")
    return dim


# Trends: computing trends, detrending, etc.
def trend(arr, dim=None, order=1, return_coeffs=False):
    """Compute linear or higher-order polynomial fit.

    If return_coeffs is True, then coeffs.degree(sel=0) is the y-intercept,
    coeffs.degree(sel=1) is the slope, etc.

    """
    dim = _infer_dim_if_1d(arr, dim)
    coeffs = arr.polyfit(dim, order)["polyfit_coefficients"]
    if return_coeffs:
        return coeffs
    return xr.polyval(arr[dim], coeffs)


def detrend(arr, dim=None, order=1):
    """Subtract off the linear or higher order polynomial fit."""
    dim = _infer_dim_if_1d(arr, dim)
    return (arr - trend(arr, dim, order) + arr.mean(dim)).transpose(*arr.dims)


# Common timeseries transforms: anomalies, standardized anomalies, etc.
def anomaly(arr, dim=None):
    """Deviation at each point from the mean along a dimension."""
    dim = _infer_dim_if_1d(arr, dim)
    return arr - arr.mean(dim)


def standardize(arr, dim=None):
    """Anomaly normalized by the standard deviation along a dimension."""
    dim = _infer_dim_if_1d(arr, dim)
    return anomaly(arr, dim) / arr.std(dim)


def dt_std_anom(arr, dim=None, order=1):
    """Detrended standardized anomaly timeseries."""
    dim = _infer_dim_if_1d(arr, dim)
    return detrend(standardize(arr, dim), dim=dim, order=order)


# Filtering (time or otherwise)
def run_mean(arr, n=10, dim="time", center=True, **kwargs):
    """Simple running average along a dimension."""
    return arr.rolling(**{dim: n}, center=center, **kwargs).mean().dropna(dim)


def run_mean_anom(arr, n=10, dim="time", center=True, **kwargs):
    """Anomaly at each point from a running average along a dimension."""
    return arr - run_mean(arr, n, dim, center=center, **kwargs)


def xwelch(arr, **kwargs):
    """Wrapper for scipy.signal.welch for xr.DataArrays"""
    freqs, psd = scipy.signal.welch(arr, **kwargs)
    return xr.DataArray(psd, dims=["frequency"], coords={"frequency": freqs},
                        name="psd")


def welch(arr, dim="time", **welch_kwargs):
    """Use xr.apply_ufunc to broadcast scipy.signal.welch.

    For example, over latitude and longitude.

    """
    def _welch(x):
        """Wrapper around scipy.signal.welch to use in apply_ufunc."""
        freqs, densities = scipy.signal.welch(x, **welch_kwargs)
        return np.array([freqs, densities])

    arr = xr.apply_ufunc(
        _welch,
        arr,
        input_core_dims=[[dim]],
        output_core_dims=[["parameter", "frequency"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"],
        dask_gufunc_kwargs=dict(output_sizes={"parameter": 2}),
    )
    freqs = arr.isel(parameter=0, lat=0, lon=0).rename("frequency").reset_coords(drop=True)
    psd = arr.isel(parameter=1)
    ds = psd.to_dataset(name="psd")
    ds.coords["frequency"] = freqs
    return ds["psd"]


def butterworth(arr, n, windows, filttype="bandpass", dim="time"):
    """Apply the Butterworth filter to an xr.DataArray."""
    if filttype == "bandpass" and windows[0] > windows[1]:
        raise ValueError("windows need to be in (high, low) "
                         "frequency order; got (low, high)")
    b, a = scipy.signal.butter(n, windows, filttype)
    axis_num = arr.get_axis_num(dim)
    return scipy.signal.filtfilt(b, a, arr, axis=axis_num)


# Correlations and linear regression.
def corr_detrended(arr1, arr2, dim=None, order=1):
    """Correlation coefficient of two arrays after they are detrended."""
    corr = xr.corr(detrend(arr1, dim, order), detrend(arr2, dim, order), dim)
    if corr.shape == tuple():
        return float(corr)
    return corr


def autocorr(arr, lag=None, dim="time", do_detrend=False):
    """Autocorrelation on xarray.DataArrays computed for specified lag(s).

    Adapted from https://stackoverflow.com/a/21863139/1706640

    """
    if do_detrend:
        arr = detrend(arr, dim=dim)

    if np.isscalar(lag):
        if lag == 0:
            return 1.
        return np.corrcoef(np.array([arr[:-lag], arr[lag:]]))[0, 1]
    else:
        if lag is None:
            lag = range(len(arr) - 2)
        values = [autocorr(arr, l) for l in lag]
        return xr.DataArray(values, dims=["lag"], coords={"lag": lag},
                            name="autocorrelation")


def lag_corr(arr1, arr2, lag=None, dim="time", do_align=True,
             do_detrend=False):
    """Lag correlation on xarray.DataArrays computed for specified lag(s).

    Lags can be negative, zero, or positive.  Positive lag corresponds to arr1
    leading arr2, negative to arr2 leading arr1.  Lag is the desired time
    index, not the time itself.

    Adapted from https://stackoverflow.com/a/21863139/1706640

    """
    if do_align:
        arr1, arr2 = xr.align(arr1, arr2)

    if do_detrend:
        arr1 = detrend(arr1, dim=dim)
        arr2 = detrend(arr2, dim=dim)

    if np.isscalar(lag):
        if lag == 0:
            arrs = [arr1.values, arr2.values]
        elif lag > 0:
            arrs = [arr1.values[lag:], arr2.values[:-lag]]
        elif lag < 0:
            arrs = [arr1.values[:lag], arr2.values[-lag:]]
        return np.corrcoef(np.array(arrs))[0, 1]
    else:
        if lag is None:
            lag = range(min(len(arr1), len(arr2)) - 2)
        values = [lag_corr(arr1, arr2, l) for l in lag]
        return xr.DataArray(values, dims=["lag"], coords={"lag": lag},
                            name="lagged-correlation")


def lin_regress(arr1, arr2, dim):
    """Use xr.apply_ufunc to broadcast scipy.stats.linregress.

    For example, over latitude and longitude.

    Adapted from
    https://github.com/pydata/xarray/issues/1815#issuecomment-614216243.

    """
    def _linregress(x, y):
        """Wrapper around scipy.stats.linregress to use in apply_ufunc."""
        slope, intercept, r_val, p_val, std_err = scipy.stats.linregress(x, y)
        return np.array([slope, intercept, r_val, p_val, std_err])

    arr1_trunc, arr2_trunc = xr.align(arr1, arr2)

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


def multi_regress(target, predictors):
    """Multiple linear regression with xarray."""
    clf_X = np.array([arr.values for arr in predictors]).transpose()
    clf = linear_model.LinearRegression()
    clf.fit(clf_X, target.values)
    return clf, xr.ones_like(target) * clf.predict(clf_X)


def rmse(arr1, arr2, dim):
    """Root mean square error using xr.apply_ufunc to broadcast.

    Adapted from
    https://github.com/pydata/xarray/issues/1815#issuecomment-614216243.

    """
    def _rmse(x, y):
        """Wrapper around scipy.stats.linregress to use in apply_ufunc."""
        return sklearn.metrics.mean_squared_error(x, y, squared=False)

    return xr.apply_ufunc(_rmse, arr1, arr2, input_core_dims=[[dim], [dim]],
                          vectorize=True, dask="parallelized")


# Breakpoint detection
def detect_breakpoint(arr, dim, rpt_class=rpt.Binseg, model="l2", n_bkps=1):
    """Use xr.apply_ufunc to broadcast breakpoint detections using ruptures"""

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
    return Eof(arr.rename({time_str: "time"}).transpose("time", ...),
               weights=weights)