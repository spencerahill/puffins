"""Functionality relating to statistics, timeseries, etc."""
from eofs.xarray import Eof
import numpy as np
import ruptures as rpt
from sklearn import linear_model
import pymannkendall as mk
import sklearn.metrics
import scipy.stats
from statsmodels.distributions.empirical_distribution import ECDF
import xarray as xr

from .interp import zero_cross_interp
from .names import LAT_STR, YEAR_STR
from .nb_utils import coord_arr_1d, cosdeg


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


def xmannken(arr, dim):
    """Use xr.apply_ufunc to broadcast the Mann-Kendall test."""
    dims_bcast = [dim_ for dim_ in arr.dims if dim_ != dim]
    stacked = arr.stack(location=dims_bcast)
    stacked_masked = stacked.where(~np.isnan(stacked), drop=True)

    def _xmk(arr):
        _, _, p, z, tau, s, var_s, slope, intercept = mk.original_test(arr)
        return np.array([p, z, tau, s, var_s, slope, intercept])

    arr = xr.apply_ufunc(
        _xmk,
        stacked_masked,
        input_core_dims=[[dim]],
        output_core_dims=[["parameter"]],
        vectorize=True,
        dask="parallelized",
    ).unstack("location")
    arr.coords["parameter"] = xr.DataArray(
        ["p", "z", "tau", "s", "var_s", "slope", "intercept"], dims=["parameter"])
    # The "sortby" call is because in testing an array had its first value moved to the end.
    return arr.sortby(dims_bcast)


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


# Centroids.
def centroid(arr, dim, weights=None, centroid_thresh=0.5):
    """Compute centroid of a given field."""
    if weights is None:
        weights = 1.0
    arr_int = (weights * arr).cumsum(dim)
    arr_int_norm = arr_int / arr_int.max(dim)
    return zero_cross_interp(arr_int_norm - centroid_thresh, dim)


def merid_centroid(arr, lat_str=LAT_STR, centroid_thresh=0.5, do_cos_weight=True):
    """Compute centroid of a field in latitude.

    By default, includes area weighting by cos(lat).

    """
    if do_cos_weight:
        weights = np.abs(cosdeg(arr[lat_str]))
    else:
        weights = None
    return centroid(arr, lat_str, weights=weights, centroid_thresh=centroid_thresh)


# Filtering (time or otherwise)
def run_mean(arr, n=10, dim="time", center=True, **kwargs):
    """Simple running average along a dimension."""
    return arr.rolling(**{dim: n}, center=center, **kwargs).mean().dropna(dim)


def run_mean_anom(arr, n=10, dim="time", center=True, **kwargs):
    """Anomaly at each point from a running average along a dimension."""
    return arr - run_mean(arr, n, dim, center=center, **kwargs)


def avg_monthly(arr, dim="time"):
    """Average across months weighting by number of days in each month."""
    return arr.weighted(arr[dim].dt.days_in_month).mean(dim)


def rolling_avg(arr, weight, **rolling_kwargs):
    """Rolling weighted average."""
    return (arr * weight).rolling(**rolling_kwargs).sum() / weight.rolling(
        **rolling_kwargs
    ).sum()


def xwelch(arr, **kwargs):
    """Wrapper for scipy.signal.welch for xr.DataArrays"""
    freqs, psd = scipy.signal.welch(arr, **kwargs)
    return xr.DataArray(
        psd, dims=["frequency"], coords={"frequency": freqs}, name="psd"
    )


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
    freqs = (
        arr.isel(parameter=0, lat=0, lon=0).rename("frequency").reset_coords(drop=True)
    )
    psd = arr.isel(parameter=1)
    ds = psd.to_dataset(name="psd")
    ds.coords["frequency"] = freqs
    return ds["psd"]


def butterworth(arr, n, windows, filttype="bandpass", dim="time"):
    """Apply the Butterworth filter to an xr.DataArray."""
    if filttype == "bandpass" and windows[0] > windows[1]:
        raise ValueError(
            "windows need to be in (high, low) " "frequency order; got (low, high)"
        )
    b, a = scipy.signal.butter(n, windows, filttype)
    axis_num = arr.get_axis_num(dim)
    return scipy.signal.filtfilt(b, a, arr, axis=axis_num)


# Correlations and linear regression.
def corr_detrended(arr1, arr2, dim=None, order=1):
    """Correlation coefficient of two arrays after they are detrended."""
    arr1_aligned, arr2_aligned = xr.align(arr1, arr2)
    corr = xr.corr(detrend(arr1_aligned, dim, order),
                   detrend(arr2_aligned, dim, order), dim)
    if corr.shape == tuple():
        return float(corr)
    return corr


def autocorr(arr, lag=None, dim="time", do_detrend=False):
    """Autocorrelation on xarray.DataArrays computed for specified lag(s).

    Adapted from https://stackoverflow.com/a/21863139/1706640

    Note: statsmodels implements this in its tsa.stattools.acf function,
    so that should be used instead of this.
    See https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.acf.html

    """
    if do_detrend:
        arr = detrend(arr, dim=dim)

    if np.isscalar(lag):
        if lag == 0:
            return 1.0
        return np.corrcoef(np.array([arr[:-lag], arr[lag:]]))[0, 1]
    else:
        if lag is None:
            lag = range(len(arr) - 2)
        values = [autocorr(arr, l) for l in lag]
        return xr.DataArray(
            values, dims=["lag"], coords={"lag": lag}, name="autocorrelation"
        )


def spearman(arr1, arr2, dim, do_detrend=False, **kwargs):
    """Spearman correlation coefficient using xr.apply_ufunc to broadcast."""

    def _spearman(x, y):
        """Wrapper around scipy.stats.spearmanr to use in apply_ufunc."""
        return scipy.stats.spearmanr(x, y, **kwargs)[0]

    arr1_aligned, arr2_aligned = xr.align(arr1, arr2)
    if do_detrend:
        arr1_aligned = detrend(arr1_aligned, dim)
        arr2_aligned = detrend(arr2_aligned, dim)
    return xr.apply_ufunc(
        _spearman,
        arr1_aligned,
        arr2_aligned,
        input_core_dims=[[dim], [dim]],
        vectorize=True,
        dask="parallelized",
    )


def lag_corr(arr1, arr2, lag=None, dim="time", do_align=True, do_detrend=False):
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
        return xr.DataArray(
            values, dims=["lag"], coords={"lag": lag}, name="lagged-correlation"
        )


def lin_regress(arr1, arr2, dim):
    """Use xr.apply_ufunc to broadcast scipy.stats.linregress.

    For example, over latitude and longitude.  The regression is performed
    of arr2 on arr1; i.e. arr2 corresponds to y and arr1 to x.

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
        output_dtypes=["float64"],
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


def regress_resid(arr1, arr2, dim):
    """Linear regression, outputting the residual.

    `arr1` is the array whose linear influence is to be removed from `arr2`.
    I.e. `arr1` corresponds to x, and `arr2` to y.

    """
    arr1_trunc, arr2_trunc = xr.align(arr1, arr2)
    regress = lin_regress(arr1_trunc, arr2_trunc, dim)
    y0 = regress.sel(parameter="intercept").drop("parameter")
    slope = regress.sel(parameter="slope").drop("parameter")
    predic = (y0 + arr1_trunc * slope).transpose(*arr2_trunc.dims)
    resid = (arr2_trunc - predic).transpose(*arr2_trunc.dims)
    data_vars = {
        "y0": y0,
        "slope": slope,
        "predic": predic,
        "resid": resid,
    }
    return xr.Dataset(data_vars)


def quantile_regress(arr, predictor, quantile, dim="time", solver="highs",
                     alpha=0):
    """Apply quantile regression to the non-NaN elements of the given array."""
    # Drop all-NaN slices since the calculation is expensive and they're
    # meaningless.  At the end return to the original shape and mask.
    stacked = arr.stack(location=[dim_ for dim_ in arr.dims if dim_ != dim])
    stacked_masked = stacked.where(~np.isnan(stacked), drop=True)

    def _qr(x, y):
        return linear_model.QuantileRegressor(
            quantile=quantile, solver=solver, alpha=alpha).fit(
                x[:, np.newaxis], y).coef_

    return xr.apply_ufunc(
        _qr,
        predictor,
        stacked_masked,
        input_core_dims=[[dim], [dim]],
        vectorize=True,
        dask="parallelized",
    ).unstack("location")


def rmse(arr1, arr2, dim):
    """Root mean square error using xr.apply_ufunc to broadcast.

    Adapted from
    https://github.com/pydata/xarray/issues/1815#issuecomment-614216243.

    """

    def _rmse(x, y):
        """Wrapper around scipy.stats.linregress to use in apply_ufunc."""
        return sklearn.metrics.mean_squared_error(x, y, squared=False)

    return xr.apply_ufunc(
        _rmse,
        arr1,
        arr2,
        input_core_dims=[[dim], [dim]],
        vectorize=True,
        dask="parallelized",
    )


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
    return Eof(arr.rename({time_str: "time"}).transpose("time", ...), weights=weights)


# Histograms
def xhist(arr, bin_edges, bin_centers=None, **kwargs):
    """xarray-based histograms."""
    if bin_centers is None:
        if isinstance(bin_edges, xr.DataArray):
            bin_edge_vals = bin_edges.values
        else:
            bin_edge_vals = bin_edges
        bin_center_vals = 0.5 * (bin_edge_vals[1:] + bin_edge_vals[:-1])
        bin_centers = coord_arr_1d(values=bin_center_vals, dim="bin_center")
    elif not isinstance(bin_centers, xr.DataArray):
        bin_centers = coord_arr_1d(values=bin_centers, dim="bin_center")
    counts, _ = np.histogram(arr, bins=bin_edges, **kwargs)
    return xr.ones_like(bin_centers) * counts



def hist2d(arr1, arr2, bin_edges1, bin_edges2, bin_centers1, bin_centers2,
           **kwargs):
    """2D histogram for xarray"""
    aligned1, aligned2 = xr.align(arr1, arr2)
    arr1_flat = xr.DataArray(aligned1.values.flatten(), dims=["event"])
    arr2_flat = xr.DataArray(aligned2.values.flatten(), dims=["event"])
    hist, _, _ = np.histogram2d(
        arr1_flat, arr2_flat, bins=[bin_edges1, bin_edges2], **kwargs)
    if bin_centers1.name == bin_centers2.name:
        name_out2 = f"{bin_centers2.name}2"
    else:
        name_out2 = bin_centers2.name
    return (
        xr.ones_like(bin_centers1) *
        xr.ones_like(bin_centers2.rename(**{bin_centers2.name: name_out2})) *
        hist
    ).transpose()


def hist(arr, dim, bin_edges, bin_centers=None, bin_name="bin", **kwargs):
    """Vectorized histograms using xr.apply_ufunc."""

    def _xhist(arr):
        return xhist(arr, bin_edges, bin_centers=bin_centers, **kwargs).values

    arr_hist = xr.apply_ufunc(
        _xhist,
        arr,
        input_core_dims=[[dim]],
        output_core_dims=[[bin_name]],
        vectorize=True,
        dask="parallelized",
    )
    if bin_centers is None:
        return arr_hist
    arr_hist.coords["bin"] = bin_centers.values
    return arr_hist


def normalize(arr, *args, **kwargs):
    """Normalize an array by dividing it by its sum."""
    return arr / arr.sum(*args, **kwargs)


def cdf_empirical(arr, cdf_points=None, side="left"):
    """Compute empirical cumulative distribution function."""
    try:
        vals_flat_sorted = np.sort(arr.values.flatten())
    except AttributeError:
        vals_flat_sorted = np.sort(arr.flatten())
    vals = vals_flat_sorted[~np.isnan(vals_flat_sorted)]
    if cdf_points is None:
        cdf_points = vals
    ecdf = ECDF(vals, side=side)(cdf_points)
    return xr.DataArray(
        ecdf, dims=["data"], coords={"data": cdf_points}, name="cdf")


def risk_ratio(arr1, arr2, cdf_points=None, side="left"):
    """Ratio of exceedance likelihood between two distributions."""
    if cdf_points is None:
        cdf_points = np.union1d(arr1, arr2)
    cdf1 = cdf_empirical(arr1, cdf_points=cdf_points, side=side)
    cdf2 = cdf_empirical(arr2, cdf_points=cdf_points, side=side)
    return ((1. - cdf1) / (1. - cdf2)).rename("risk_ratio")
