"""Functionality relating to statistics, timeseries, etc."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, cast

import numpy as np
import pymannkendall as mk
import scipy.signal
import scipy.stats
import sklearn.metrics
import xarray as xr
from eofs.xarray import Eof

# import ruptures as rpt  # commented due to build issues
from sklearn import linear_model
from statsmodels.distributions.empirical_distribution import ECDF

from ._typing import ArrayLike
from .interp import zero_cross_interp
from .names import LAT_STR, YEAR_STR
from .nb_utils import coord_arr_1d, cosdeg, flat_dropna, stacked_masked


# TODO: make this a decorator.
def _infer_dim_if_1d(arr: xr.DataArray, dim: str | None) -> str:
    """Helper function to get dim name of 1D arrays."""
    if dim is None:
        if arr.ndim == 1:
            dim = str(arr.dims[0])
        else:
            raise ValueError("Dimension must be specified if array isn't 1D.")
    return dim


# Trends: computing trends, detrending, etc.
def trend(
    arr: xr.DataArray,
    dim: str | None = None,
    order: int = 1,
    return_coeffs: bool = False,
) -> xr.DataArray:
    """Compute linear or higher-order polynomial fit.

    If return_coeffs is True, then coeffs.degree(sel=0) is the y-intercept,
    coeffs.degree(sel=1) is the slope, etc.

    """
    dim = _infer_dim_if_1d(arr, dim)
    coeffs = arr.polyfit(dim, order)["polyfit_coefficients"]
    if return_coeffs:
        return coeffs
    return cast(xr.DataArray, xr.polyval(arr[dim], coeffs))


def detrend(arr: xr.DataArray, dim: str | None = None, order: int = 1) -> xr.DataArray:
    """Subtract off the linear or higher order polynomial fit."""
    dim = _infer_dim_if_1d(arr, dim)
    return cast(
        xr.DataArray,
        (arr - trend(arr, dim, order) + arr.mean(dim)).transpose(*arr.dims),
    )


def xmannken(arr: xr.DataArray, dim: str) -> xr.DataArray:
    """Use xr.apply_ufunc to broadcast the Mann-Kendall test."""
    dims_bcast = [dim_ for dim_ in arr.dims if dim_ != dim]
    stacked = arr.stack(location=dims_bcast)
    stacked_masked = stacked.where(~np.isnan(stacked), drop=True)

    def _xmk(arr: np.ndarray) -> np.ndarray:
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
        ["p", "z", "tau", "s", "var_s", "slope", "intercept"], dims=["parameter"]
    )
    # The "sortby" call is because in testing an array had its first value moved to the end.
    return cast(xr.DataArray, arr.sortby(dims_bcast))


# Common timeseries transforms: anomalies, standardized anomalies, etc.
def anomaly(arr: xr.DataArray, dim: str | None = None) -> xr.DataArray:
    """Deviation at each point from the mean along a dimension."""
    dim = _infer_dim_if_1d(arr, dim)
    return cast(xr.DataArray, arr - arr.mean(dim))


def standardize(arr: xr.DataArray, dim: str | None = None) -> xr.DataArray:
    """Anomaly normalized by the standard deviation along a dimension."""
    dim = _infer_dim_if_1d(arr, dim)
    return cast(xr.DataArray, anomaly(arr, dim) / arr.std(dim))


def dt_std_anom(
    arr: xr.DataArray, dim: str | None = None, order: int = 1
) -> xr.DataArray:
    """Detrended standardized anomaly timeseries."""
    dim = _infer_dim_if_1d(arr, dim)
    return detrend(standardize(arr, dim), dim=dim, order=order)


# Centroids.
def centroid(
    arr: xr.DataArray,
    dim: str,
    weights: xr.DataArray | float | None = None,
    centroid_thresh: float = 0.5,
) -> xr.DataArray:
    """Compute centroid of a given field."""
    if weights is None:
        weights = 1.0
    arr_int = (weights * arr).cumsum(dim)
    arr_int_norm = arr_int / arr_int.max(dim)
    return zero_cross_interp(arr_int_norm - centroid_thresh, dim)


def merid_centroid(
    arr: xr.DataArray,
    lat_str: str = LAT_STR,
    centroid_thresh: float = 0.5,
    do_cos_weight: bool = True,
) -> xr.DataArray:
    """Compute centroid of a field in latitude.

    By default, includes area weighting by cos(lat).

    """
    weights: xr.DataArray | None
    if do_cos_weight:
        weights = np.abs(cosdeg(arr[lat_str]))
    else:
        weights = None
    return centroid(arr, lat_str, weights=weights, centroid_thresh=centroid_thresh)


# Filtering (time or otherwise)
def run_mean(
    arr: xr.DataArray,
    n: int = 10,
    dim: str = "time",
    center: bool = True,
    **kwargs: Any,
) -> xr.DataArray:
    """Simple running average along a dimension."""
    return cast(
        xr.DataArray,
        arr.rolling({dim: n}, center=center, **kwargs).mean().dropna(dim),
    )


def run_mean_anom(
    arr: xr.DataArray,
    n: int = 10,
    dim: str = "time",
    center: bool = True,
    **kwargs: Any,
) -> xr.DataArray:
    """Anomaly at each point from a running average along a dimension."""
    return cast(xr.DataArray, arr - run_mean(arr, n, dim, center=center, **kwargs))


def avg_monthly(arr: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Average across months weighting by number of days in each month."""
    return cast(xr.DataArray, arr.weighted(arr[dim].dt.days_in_month).mean(dim))


def rolling_avg(
    arr: xr.DataArray, weight: xr.DataArray, **rolling_kwargs: Any
) -> xr.DataArray:
    """Rolling weighted average."""
    return cast(
        xr.DataArray,
        (arr * weight).rolling(**rolling_kwargs).sum()
        / weight.rolling(**rolling_kwargs).sum(),
    )


def xwelch(arr: xr.DataArray, **kwargs: Any) -> xr.DataArray:
    """Wrapper for scipy.signal.welch for xr.DataArrays"""
    freqs, psd = scipy.signal.welch(arr, **kwargs)
    return xr.DataArray(
        psd, dims=["frequency"], coords={"frequency": freqs}, name="psd"
    )


def welch(arr: xr.DataArray, dim: str = "time", **welch_kwargs: Any) -> xr.DataArray:
    """Use xr.apply_ufunc to broadcast scipy.signal.welch.

    For example, over latitude and longitude.

    """

    def _welch(x: np.ndarray) -> np.ndarray:
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
    return cast(xr.DataArray, ds["psd"])


def butterworth(
    arr: xr.DataArray,
    n: int,
    windows: Sequence[float],
    filttype: str = "bandpass",
    dim: str = "time",
) -> np.ndarray:
    """Apply the Butterworth filter to an xr.DataArray."""
    if filttype == "bandpass" and windows[0] > windows[1]:
        raise ValueError(
            "windows need to be in (high, low) frequency order; got (low, high)"
        )
    b, a = scipy.signal.butter(n, windows, filttype)
    axis_num = arr.get_axis_num(dim)
    return cast(np.ndarray, scipy.signal.filtfilt(b, a, arr, axis=axis_num))


# Correlations and linear regression.
def corr_detrended(
    arr1: xr.DataArray,
    arr2: xr.DataArray,
    dim: str | None = None,
    order: int = 1,
) -> float | xr.DataArray:
    """Correlation coefficient of two arrays after they are detrended."""
    arr1_aligned, arr2_aligned = xr.align(arr1, arr2)
    corr = xr.corr(
        detrend(arr1_aligned, dim, order), detrend(arr2_aligned, dim, order), dim
    )
    if corr.shape == tuple():
        return float(corr)
    return corr


def autocorr(
    arr: xr.DataArray,
    lag: int | Sequence[int] | range | None = None,
    dim: str = "time",
    do_detrend: bool = False,
) -> float | xr.DataArray:
    """Autocorrelation on xarray.DataArrays computed for specified lag(s).

    Adapted from https://stackoverflow.com/a/21863139/1706640

    Note: statsmodels implements this in its tsa.stattools.acf function,
    so that should be used instead of this.
    See https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.acf.html

    """
    if do_detrend:
        arr = detrend(arr, dim=dim)

    if np.isscalar(lag):
        lag = cast(int, lag)
        if lag == 0:
            return 1.0
        return float(np.corrcoef(np.array([arr[:-lag], arr[lag:]]))[0, 1])
    if lag is None:
        lag = range(len(arr) - 2)
    values = [autocorr(arr, l) for l in cast(Iterable[int], lag)]
    return xr.DataArray(
        values, dims=["lag"], coords={"lag": lag}, name="autocorrelation"
    )


def spearman(
    arr1: xr.DataArray,
    arr2: xr.DataArray,
    dim: str,
    do_detrend: bool = False,
    **kwargs: Any,
) -> xr.DataArray:
    """Spearman correlation coefficient using xr.apply_ufunc to broadcast."""

    def _spearman(x: np.ndarray, y: np.ndarray) -> float:
        """Wrapper around scipy.stats.spearmanr to use in apply_ufunc."""
        return cast(float, scipy.stats.spearmanr(x, y, **kwargs)[0])

    arr1_aligned, arr2_aligned = xr.align(arr1, arr2)
    if do_detrend:
        arr1_aligned = detrend(arr1_aligned, dim)
        arr2_aligned = detrend(arr2_aligned, dim)
    return cast(
        xr.DataArray,
        xr.apply_ufunc(
            _spearman,
            arr1_aligned,
            arr2_aligned,
            input_core_dims=[[dim], [dim]],
            vectorize=True,
            dask="parallelized",
        ),
    )


def lag_corr(
    arr1: xr.DataArray,
    arr2: xr.DataArray,
    lag: int | Sequence[int] | range | None = None,
    dim: str = "time",
    do_align: bool = True,
    do_detrend: bool = False,
) -> float | xr.DataArray:
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
        lag = cast(int, lag)
        if lag == 0:
            arrs = [arr1.values, arr2.values]
        elif lag > 0:
            arrs = [arr1.values[lag:], arr2.values[:-lag]]
        else:  # lag < 0
            arrs = [arr1.values[:lag], arr2.values[-lag:]]
        return float(np.corrcoef(np.array(arrs))[0, 1])
    if lag is None:
        lag = range(min(len(arr1), len(arr2)) - 2)
    values = [lag_corr(arr1, arr2, _lag) for _lag in cast(Iterable[int], lag)]
    return xr.DataArray(
        values, dims=["lag"], coords={"lag": lag}, name="lagged-correlation"
    )


def lin_regress(
    arr1: xr.DataArray, arr2: xr.DataArray, dim: str, do_detrend: bool = False
) -> xr.DataArray:
    """Use xr.apply_ufunc to broadcast scipy.stats.linregress.

    For example, over latitude and longitude.  The regression is performed
    of arr2 on arr1; i.e. arr2 corresponds to y and arr1 to x.

    Adapted from
    https://github.com/pydata/xarray/issues/1815#issuecomment-614216243.

    """

    def _linregress(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Wrapper around scipy.stats.linregress to use in apply_ufunc."""
        slope, intercept, r_val, p_val, std_err = scipy.stats.linregress(x, y)
        return np.array([slope, intercept, r_val, p_val, std_err])

    arr1_trunc, arr2_trunc = xr.align(arr1, arr2)

    if do_detrend:
        arr1_trunc = detrend(arr1_trunc, dim=dim)
        arr2_trunc = detrend(arr2_trunc, dim=dim)

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
    return cast(xr.DataArray, arr)


def multi_regress(
    target: xr.DataArray, predictors: Sequence[xr.DataArray]
) -> tuple[linear_model.LinearRegression, xr.DataArray]:
    """Multiple linear regression with xarray."""
    clf_X = np.array([arr.values for arr in predictors]).transpose()
    clf = linear_model.LinearRegression()
    clf.fit(clf_X, target.values)
    return clf, xr.ones_like(target) * clf.predict(clf_X)


def regress_resid(arr1: xr.DataArray, arr2: xr.DataArray, dim: str) -> xr.Dataset:
    """Linear regression, outputting the residual.

    `arr1` is the array whose linear influence is to be removed from `arr2`.
    I.e. `arr1` corresponds to x, and `arr2` to y.

    """
    arr1_trunc, arr2_trunc = xr.align(arr1, arr2)
    regress = lin_regress(arr1_trunc, arr2_trunc, dim)
    y0 = regress.sel(parameter="intercept").drop_vars("parameter")
    slope = regress.sel(parameter="slope").drop_vars("parameter")
    predic = (y0 + arr1_trunc * slope).transpose(*arr2_trunc.dims)
    resid = (arr2_trunc - predic).transpose(*arr2_trunc.dims)
    data_vars = {
        "y0": y0,
        "slope": slope,
        "predic": predic,
        "resid": resid,
    }
    return xr.Dataset(data_vars)


def quantile_regress(
    arr: xr.DataArray,
    predictor: xr.DataArray,
    quantile: float,
    dim: str = "time",
    solver: str = "highs",
    alpha: float = 0,
) -> xr.DataArray:
    """Apply quantile regression to the non-NaN elements of the given array."""
    # Drop all-NaN slices since the calculation is expensive and they're
    # meaningless.  At the end return to the original shape and mask.
    stacked = arr.stack(location=[dim_ for dim_ in arr.dims if dim_ != dim])
    stacked_masked = stacked.where(~np.isnan(stacked), drop=True)

    def _qr(x: np.ndarray, y: np.ndarray) -> float:
        return float(
            linear_model.QuantileRegressor(
                quantile=quantile, solver=solver, alpha=alpha
            )
            .fit(x[:, np.newaxis], y)
            .coef_[0]
        )

    return cast(
        xr.DataArray,
        xr.apply_ufunc(
            _qr,
            predictor,
            stacked_masked,
            input_core_dims=[[dim], [dim]],
            vectorize=True,
            dask="parallelized",
        ).unstack("location"),
    )


def rmse(arr1: xr.DataArray, arr2: xr.DataArray, dim: str) -> xr.DataArray:
    """Root mean square error using xr.apply_ufunc to broadcast.

    Adapted from
    https://github.com/pydata/xarray/issues/1815#issuecomment-614216243.

    """

    def _rmse(x: np.ndarray, y: np.ndarray) -> float:
        """Wrapper around sklearn.metrics.mean_squared_error to use in apply_ufunc."""
        return float(np.sqrt(sklearn.metrics.mean_squared_error(x, y)))

    return cast(
        xr.DataArray,
        xr.apply_ufunc(
            _rmse,
            arr1,
            arr2,
            input_core_dims=[[dim], [dim]],
            vectorize=True,
            dask="parallelized",
        ),
    )


# Breakpoint detection
# Commenting all this out due to build errors stemming from ruptures imports.
# def detect_breakpoint(arr, dim, rpt_class=rpt.Binseg, model="l2", n_bkps=1):
#     """Use xr.apply_ufunc to broadcast breakpoint detections using ruptures"""

#     rpt_instance = rpt_class(model=model)

#     def _detect_bp(arr):
#         """Wrapper to use in apply_ufunc."""
#         return rpt_instance.fit(arr).predict(n_bkps=n_bkps)[0]

#     inds_bp = xr.apply_ufunc(
#         _detect_bp,
#         arr,
#         input_core_dims=[[dim]],
#         vectorize=True,
#         dask="parallelized",
#     )
#     return arr[dim][inds_bp]


# Empirical orthogonal functions (EOFs)
def eof_solver(
    arr: xr.DataArray, lat_str: str = LAT_STR, time_str: str = YEAR_STR
) -> Eof:
    """Generate an EOF solver for gridded lat-lon data."""
    weights = np.sqrt(cosdeg(arr[lat_str])).values[..., np.newaxis]
    # The eofs package requires that the time dimension be named "time".
    return Eof(arr.rename({time_str: "time"}).transpose("time", ...), weights=weights)


# Histograms
def xhist(
    arr: ArrayLike,
    bin_edges: ArrayLike,
    bin_centers: ArrayLike | None = None,
    **kwargs: Any,
) -> xr.DataArray:
    """xarray-based histograms."""
    if bin_centers is None:
        if isinstance(bin_edges, xr.DataArray):
            bin_edge_vals = bin_edges.values
        else:
            bin_edge_vals = np.asarray(bin_edges)
        bin_center_vals = 0.5 * (bin_edge_vals[1:] + bin_edge_vals[:-1])
        bin_centers = coord_arr_1d(values=bin_center_vals, dim="bin_center")
    elif not isinstance(bin_centers, xr.DataArray):
        bin_centers = coord_arr_1d(values=bin_centers, dim="bin_center")
    counts, _ = np.histogram(arr, bins=bin_edges, **kwargs)
    return cast(xr.DataArray, xr.ones_like(cast(xr.DataArray, bin_centers)) * counts)


def hist2d(
    arr1: xr.DataArray,
    arr2: xr.DataArray,
    bin_edges1: ArrayLike,
    bin_edges2: ArrayLike,
    bin_centers1: xr.DataArray,
    bin_centers2: xr.DataArray,
    **kwargs: Any,
) -> xr.DataArray:
    """2D histogram for xarray"""
    aligned1, aligned2 = xr.align(arr1, arr2)
    arr1_flat = xr.DataArray(aligned1.values.flatten(), dims=["event"])
    arr2_flat = xr.DataArray(aligned2.values.flatten(), dims=["event"])
    hist, _, _ = np.histogram2d(
        arr1_flat,
        arr2_flat,
        bins=[np.asarray(bin_edges1), np.asarray(bin_edges2)],
        **kwargs,
    )
    name1 = str(bin_centers1.name)
    name2 = str(bin_centers2.name)
    if name1 == name2:
        name_out2 = f"{name2}2"
    else:
        name_out2 = name2
    return cast(
        xr.DataArray,
        (
            xr.ones_like(bin_centers1)
            * xr.ones_like(bin_centers2.rename({name2: name_out2}))
            * hist
        ).transpose(),
    )


def hist(
    arr: xr.DataArray,
    dim: str,
    bin_edges: ArrayLike,
    bin_centers: xr.DataArray | None = None,
    bin_name: str = "bin",
    **kwargs: Any,
) -> xr.DataArray:
    """Vectorized histograms using xr.apply_ufunc."""

    def _xhist(arr: np.ndarray) -> np.ndarray:
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
        return cast(xr.DataArray, arr_hist)
    arr_hist.coords["bin"] = bin_centers.values
    return cast(xr.DataArray, arr_hist)


def cdf_empirical(
    arr: ArrayLike, cdf_points: ArrayLike | None = None, side: str = "left"
) -> xr.DataArray:
    """Compute empirical cumulative distribution function."""
    arr_any = cast(Any, arr)
    try:
        vals_flat_sorted = np.sort(arr_any.values.flatten())
    except AttributeError:
        vals_flat_sorted = np.sort(arr_any.flatten())
    vals = vals_flat_sorted[~np.isnan(vals_flat_sorted)]
    if cdf_points is None:
        cdf_points = vals
    ecdf = ECDF(vals, side=side)(cdf_points)
    return xr.DataArray(ecdf, dims=["data"], coords={"data": cdf_points}, name="cdf")


def risk_ratio(
    arr1: ArrayLike,
    arr2: ArrayLike,
    cdf_points: ArrayLike | None = None,
    side: str = "left",
) -> xr.DataArray:
    """Ratio of exceedance likelihood between two distributions."""
    if cdf_points is None:
        cdf_points = np.union1d(arr1, arr2)
    cdf1 = cdf_empirical(arr1, cdf_points=cdf_points, side=side)
    cdf2 = cdf_empirical(arr2, cdf_points=cdf_points, side=side)
    return cast(xr.DataArray, ((1.0 - cdf1) / (1.0 - cdf2)).rename("risk_ratio"))


# Statistical fits
def xfit(
    arr: xr.DataArray,
    dim: str,
    dist: scipy.stats.rv_continuous = scipy.stats.genextreme,
    **fit_kwargs: Any,
) -> xr.DataArray:
    """Broadcast fitting of scipy.stats distributions."""
    dims_bcast = [dim_ for dim_ in arr.dims if dim_ != dim]

    def _fit(data: np.ndarray) -> np.ndarray:
        params = dist.fit(data, **fit_kwargs)
        return np.array(list(params))

    return cast(
        xr.DataArray,
        xr.apply_ufunc(
            _fit,
            stacked_masked(arr, dim),
            input_core_dims=[[dim]],
            output_core_dims=[["parameter"]],
            vectorize=True,
            dask="parallelized",
        )
        .unstack("location")
        .sortby(dims_bcast),
    )


# False Discovery Rate
def false_disc_rate_thresh_pval(pvals: xr.DataArray, target_fdr: float = 0.05) -> float:
    """Threshold p value for significance based on False Discovery Rate.

    From Wilks 2016, BAMS, Eq. 3.

    """
    pvals_sorted = flat_dropna(pvals)
    pvals_sorted.sort()

    num_pvals = len(pvals_sorted)
    rhs = np.arange(1, num_pvals + 1) / num_pvals * target_fdr
    accepted_inds = pvals_sorted <= rhs
    return float(pvals_sorted[accepted_inds].max())
