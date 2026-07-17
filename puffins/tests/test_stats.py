"""Tests for the stats module."""

import numpy as np
import pandas as pd
import pytest
import scipy.signal
import scipy.stats
import sklearn.metrics
import xarray as xr
from sklearn import linear_model
from statsmodels.distributions.empirical_distribution import ECDF

from puffins.names import LAT_STR, LON_STR, YEAR_STR
from puffins.stats import (
    anomaly,
    autocorr,
    avg_monthly,
    butterworth,
    cdf_empirical,
    centroid,
    corr_detrended,
    detrend,
    dt_std_anom,
    eof_solver,
    false_disc_rate_thresh_pval,
    hist,
    hist2d,
    lag_corr,
    lin_regress,
    merid_centroid,
    multi_regress,
    quantile_regress,
    regress_resid,
    risk_ratio,
    rmse,
    rolling_avg,
    run_mean,
    run_mean_anom,
    spearman,
    standardize,
    trend,
    welch,
    xfit,
    xhist,
    xmannken,
    xwelch,
)


def _timeseries(values: np.ndarray, name: str = "ts") -> xr.DataArray:
    """A 1-D DataArray along ``time`` with an integer time coordinate."""
    vals = np.asarray(values, dtype=float)
    return xr.DataArray(
        vals,
        dims=["time"],
        coords={"time": np.arange(vals.size)},
        name=name,
    )


# ---------------------------------------------------------------------------
# Trends and detrending.
# ---------------------------------------------------------------------------
class TestTrend:
    """Tests for trend."""

    def test_reconstructs_slope_and_intercept(self) -> None:
        """coeffs match a raw numpy polyfit of the exact line y = 3x + 5."""
        x = np.arange(10.0)
        arr = _timeseries(3.0 * x + 5.0)
        coeffs = trend(arr, "time", return_coeffs=True)
        np.testing.assert_allclose(coeffs.sel(degree=1).item(), 3.0)
        np.testing.assert_allclose(coeffs.sel(degree=0).item(), 5.0)

    def test_fitted_line_equals_data_for_exact_line(self) -> None:
        """The evaluated fit reproduces an exactly-linear input pointwise."""
        x = np.arange(10.0)
        arr = _timeseries(-2.0 * x + 1.0)
        fit = trend(arr, "time")
        np.testing.assert_allclose(fit.values, arr.values)

    def test_infers_dim_when_1d(self) -> None:
        """A 1-D array needs no explicit dim."""
        arr = _timeseries(np.arange(6.0))
        np.testing.assert_allclose(trend(arr).values, arr.values, atol=1e-9)

    def test_raises_when_dim_ambiguous(self) -> None:
        """A multi-dim array with dim unset raises."""
        arr = xr.DataArray(np.ones((3, 4)), dims=["time", "x"])
        with pytest.raises(ValueError):
            trend(arr)

    def test_quadratic_fit_recovers_curvature(self) -> None:
        """order=2 recovers the leading coefficient of a parabola."""
        x = np.arange(12.0)
        arr = _timeseries(4.0 * x**2 - x + 2.0)
        coeffs = trend(arr, "time", order=2, return_coeffs=True)
        np.testing.assert_allclose(coeffs.sel(degree=2).item(), 4.0, atol=1e-9)


class TestDetrend:
    """Tests for detrend."""

    def test_removes_linear_trend_leaving_mean(self) -> None:
        """A pure line detrends to a flat series at its own mean."""
        x = np.arange(10.0)
        arr = _timeseries(3.0 * x + 5.0)
        result = detrend(arr, "time")
        np.testing.assert_allclose(result.values, arr.mean().item(), atol=1e-9)

    def test_preserves_mean(self) -> None:
        """Detrending leaves the mean unchanged."""
        arr = _timeseries(np.array([1.0, 4.0, 2.0, 8.0, 5.0]))
        np.testing.assert_allclose(
            detrend(arr, "time").mean().item(), arr.mean().item()
        )

    def test_preserves_dim_order(self) -> None:
        """Output keeps the input dimension order."""
        arr = xr.DataArray(
            np.random.default_rng(0).standard_normal((4, 3)),
            dims=["time", "x"],
            coords={"time": np.arange(4), "x": np.arange(3)},
        )
        assert detrend(arr, "time").dims == arr.dims


# ---------------------------------------------------------------------------
# Mann-Kendall.
# ---------------------------------------------------------------------------
class TestXMannKen:
    """Tests for xmannken."""

    def test_matches_pymannkendall_on_each_column(self) -> None:
        """Broadcast results match a direct pymannkendall call per column."""
        import pymannkendall as mk

        rng = np.random.default_rng(0)
        col_a = np.arange(20.0) + rng.standard_normal(20) * 0.1
        col_b = -np.arange(20.0) + rng.standard_normal(20) * 0.1
        arr = xr.DataArray(
            np.stack([col_a, col_b], axis=1),
            dims=["time", "x"],
            coords={"time": np.arange(20), "x": [0, 1]},
        )
        result = xmannken(arr, "time")
        for i, col in enumerate([col_a, col_b]):
            _, _, p, z, tau, s, var_s, slope, intercept = mk.original_test(col)
            got = result.isel(x=i)
            np.testing.assert_allclose(got.sel(parameter="p").item(), p)
            np.testing.assert_allclose(got.sel(parameter="tau").item(), tau)
            np.testing.assert_allclose(got.sel(parameter="slope").item(), slope)

    def test_increasing_series_has_positive_tau(self) -> None:
        """A monotone increasing column yields tau = 1."""
        arr = xr.DataArray(
            np.arange(15.0).reshape(15, 1),
            dims=["time", "x"],
            coords={"time": np.arange(15), "x": [0]},
        )
        result = xmannken(arr, "time")
        np.testing.assert_allclose(result.sel(parameter="tau").isel(x=0).item(), 1.0)


# ---------------------------------------------------------------------------
# Anomalies and standardization.
# ---------------------------------------------------------------------------
class TestAnomaly:
    """Tests for anomaly."""

    def test_subtracts_mean(self) -> None:
        arr = _timeseries(np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(anomaly(arr).values, [-1.5, -0.5, 0.5, 1.5])

    def test_zero_mean_output(self) -> None:
        arr = _timeseries(np.array([5.0, 1.0, 9.0, 3.0]))
        np.testing.assert_allclose(anomaly(arr).mean().item(), 0.0, atol=1e-12)


class TestStandardize:
    """Tests for standardize."""

    def test_zero_mean_unit_std(self) -> None:
        arr = _timeseries(np.array([2.0, 4.0, 6.0, 8.0]))
        result = standardize(arr)
        np.testing.assert_allclose(result.mean().item(), 0.0, atol=1e-12)
        np.testing.assert_allclose(result.std().item(), 1.0, atol=1e-12)

    def test_matches_raw_formula(self) -> None:
        """Output equals (x - mean) / std reconstructed from numpy."""
        vals = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
        arr = _timeseries(vals)
        expected = (vals - vals.mean()) / vals.std()
        np.testing.assert_allclose(standardize(arr).values, expected)


class TestDtStdAnom:
    """Tests for dt_std_anom."""

    def test_detrended_standardized(self) -> None:
        """A trended series becomes flat (its mean) after dt_std_anom."""
        x = np.arange(12.0)
        arr = _timeseries(2.0 * x + 1.0)
        result = dt_std_anom(arr)
        np.testing.assert_allclose(result.values, result.mean().item(), atol=1e-9)


# ---------------------------------------------------------------------------
# Centroids.
# ---------------------------------------------------------------------------
class TestCentroid:
    """Tests for centroid."""

    def test_uniform_field_centroid_at_midpoint(self) -> None:
        """Cumulative of a uniform field crosses one-half at the midpoint."""
        x = np.arange(5.0)
        arr = xr.DataArray(np.ones(5), dims=["x"], coords={"x": x})
        np.testing.assert_allclose(centroid(arr, "x").item(), 1.5)

    def test_threshold_shifts_centroid(self) -> None:
        """A larger threshold pushes the crossing to larger coordinate."""
        x = np.arange(5.0)
        arr = xr.DataArray(np.ones(5), dims=["x"], coords={"x": x})
        low = centroid(arr, "x", centroid_thresh=0.5).item()
        high = centroid(arr, "x", centroid_thresh=0.9).item()
        assert high > low

    def test_weights_bias_centroid(self) -> None:
        """Weighting the tail more pulls the centroid toward it."""
        x = np.arange(5.0)
        arr = xr.DataArray(np.ones(5), dims=["x"], coords={"x": x})
        weights = xr.DataArray(np.array([1.0, 1.0, 1.0, 5.0, 5.0]), dims=["x"])
        weighted = centroid(arr, "x", weights=weights).item()
        unweighted = centroid(arr, "x").item()
        assert weighted > unweighted


class TestMeridCentroid:
    """Tests for merid_centroid."""

    def test_symmetric_field_centroid_at_equator(self) -> None:
        """A field uniform in latitude has its cos-weighted median at the equator.

        The cumulative sum is inclusive, so the crossing lands within half a
        grid cell of 0; the symmetry claim is therefore |centroid| <= dlat/2.
        """
        lat = np.linspace(-80.0, 80.0, 33)
        spacing = lat[1] - lat[0]
        arr = xr.DataArray(np.ones_like(lat), dims=[LAT_STR], coords={LAT_STR: lat})
        assert abs(merid_centroid(arr).item()) <= 0.5 * spacing + 1e-9

    def test_cos_weight_changes_result(self) -> None:
        """Turning cos-weighting on changes the centroid of an asymmetric field."""
        lat = np.linspace(-80.0, 80.0, 33)
        vals = np.linspace(1.0, 3.0, lat.size)  # increasing toward the north
        arr = xr.DataArray(vals, dims=[LAT_STR], coords={LAT_STR: lat})
        weighted = merid_centroid(arr, do_cos_weight=True).item()
        unweighted = merid_centroid(arr, do_cos_weight=False).item()
        assert not np.isclose(weighted, unweighted)


# ---------------------------------------------------------------------------
# Filtering / running means.
# ---------------------------------------------------------------------------
class TestRunMean:
    """Tests for run_mean."""

    def test_matches_manual_window_mean(self) -> None:
        """A centered window of 3 matches hand-computed interior averages."""
        arr = _timeseries(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = run_mean(arr, n=3, dim="time")
        np.testing.assert_allclose(result.values, [2.0, 3.0, 4.0])

    def test_drops_incomplete_edges(self) -> None:
        """dropna removes the window-edge NaNs, shortening the series."""
        arr = _timeseries(np.arange(10.0))
        result = run_mean(arr, n=4, dim="time")
        assert result.sizes["time"] == 10 - 4 + 1


class TestRunMeanAnom:
    """Tests for run_mean_anom."""

    def test_equals_arr_minus_run_mean(self) -> None:
        """The anomaly equals the raw minus the running mean, on shared times."""
        arr = _timeseries(np.array([1.0, 5.0, 2.0, 8.0, 3.0]))
        rm = run_mean(arr, n=3, dim="time")
        anom = run_mean_anom(arr, n=3, dim="time")
        expected = (arr - rm).dropna("time")
        np.testing.assert_allclose(anom.dropna("time").values, expected.values)


class TestAvgMonthly:
    """Tests for avg_monthly."""

    def test_weights_by_days_in_month(self) -> None:
        """The average weights each month by its day count."""
        time = pd.date_range("2001-01-01", periods=12, freq="MS")
        vals = np.arange(1.0, 13.0)
        arr = xr.DataArray(vals, dims=["time"], coords={"time": time})
        days = arr["time"].dt.days_in_month.values
        expected = np.sum(vals * days) / np.sum(days)
        np.testing.assert_allclose(avg_monthly(arr).item(), expected)

    def test_constant_series_returns_constant(self) -> None:
        """A constant field averages to that constant regardless of weights."""
        time = pd.date_range("2001-01-01", periods=6, freq="MS")
        arr = xr.DataArray(np.full(6, 7.0), dims=["time"], coords={"time": time})
        np.testing.assert_allclose(avg_monthly(arr).item(), 7.0)


class TestRollingAvg:
    """Tests for rolling_avg."""

    def test_matches_weighted_window_sum_ratio(self) -> None:
        """Equals sum(w*x)/sum(w) within each rolling window."""
        arr = _timeseries(np.array([1.0, 2.0, 3.0, 4.0]))
        weight = _timeseries(np.array([1.0, 2.0, 3.0, 4.0]))
        result = rolling_avg(arr, weight, time=2)
        # Window over indices (1,2): (1*2 + 2*3)/(1+2)... i.e. w*x summed / w summed.
        expected_idx2 = (2.0 * 2.0 + 3.0 * 3.0) / (2.0 + 3.0)
        np.testing.assert_allclose(result.isel(time=2).item(), expected_idx2)


class TestWelch:
    """Tests for xwelch and welch."""

    def test_xwelch_matches_scipy(self) -> None:
        """xwelch reproduces scipy.signal.welch frequencies and densities."""
        rng = np.random.default_rng(0)
        arr = _timeseries(rng.standard_normal(64))
        freqs, psd = scipy.signal.welch(arr.values, nperseg=16)
        result = xwelch(arr, nperseg=16)
        np.testing.assert_allclose(result["frequency"].values, freqs)
        np.testing.assert_allclose(result.values, psd)

    def test_welch_broadcasts_over_grid(self) -> None:
        """welch returns a psd with the broadcast lat/lon dims retained."""
        rng = np.random.default_rng(1)
        data = rng.standard_normal((32, 2, 2))
        arr = xr.DataArray(
            data,
            dims=["time", LAT_STR, LON_STR],
            coords={"time": np.arange(32), LAT_STR: [0, 1], LON_STR: [0, 1]},
        )
        result = welch(arr, dim="time", nperseg=8)
        assert LAT_STR in result.dims and LON_STR in result.dims
        # A single grid cell matches the direct 1-D transform.
        _, psd = scipy.signal.welch(data[:, 0, 0], nperseg=8)
        np.testing.assert_allclose(result.isel({LAT_STR: 0, LON_STR: 0}).values, psd)


class TestButterworth:
    """Tests for butterworth."""

    def test_matches_scipy_filtfilt(self) -> None:
        """Output matches a direct scipy filtfilt with the same coefficients."""
        rng = np.random.default_rng(0)
        arr = _timeseries(rng.standard_normal(100))
        windows = [0.1, 0.4]
        result = butterworth(arr, 3, windows, filttype="bandpass", dim="time")
        b, a = scipy.signal.butter(3, windows, "bandpass")
        expected = scipy.signal.filtfilt(b, a, arr.values, axis=0)
        np.testing.assert_allclose(result, expected)

    def test_raises_on_reversed_windows(self) -> None:
        """Bandpass windows given as (low, high) raise a ValueError."""
        arr = _timeseries(np.arange(50.0))
        with pytest.raises(ValueError):
            butterworth(arr, 3, [0.4, 0.1], filttype="bandpass")


# ---------------------------------------------------------------------------
# Correlations and regression.
# ---------------------------------------------------------------------------
class TestCorrDetrended:
    """Tests for corr_detrended."""

    def test_returns_float_for_1d(self) -> None:
        """A pair of 1-D series returns a plain float."""
        rng = np.random.default_rng(0)
        a = _timeseries(rng.standard_normal(30), name="a")
        b = _timeseries(rng.standard_normal(30), name="b")
        result = corr_detrended(a, b, "time")
        assert isinstance(result, float)

    def test_removes_shared_trend(self) -> None:
        """Two series that differ only by a trend correlate near zero once detrended.

        Each series is white noise plus its own steep linear trend. Raw they
        correlate near +1 (trend dominates); after detrending only the
        independent noise remains, so the correlation collapses toward zero.
        """
        rng = np.random.default_rng(1)
        x = np.arange(60.0)
        a = _timeseries(50.0 * x + rng.standard_normal(60), name="a")
        b = _timeseries(50.0 * x + rng.standard_normal(60), name="b")
        raw = float(xr.corr(a, b, "time"))
        detrended = corr_detrended(a, b, "time")
        assert raw > 0.99
        assert abs(detrended) < 0.5


class TestAutocorr:
    """Tests for autocorr."""

    def test_lag_zero_is_one(self) -> None:
        arr = _timeseries(np.random.default_rng(0).standard_normal(20))
        assert autocorr(arr, 0) == 1.0

    def test_matches_numpy_corrcoef(self) -> None:
        """A single-lag autocorrelation matches raw np.corrcoef of shifted slices."""
        vals = np.random.default_rng(2).standard_normal(40)
        arr = _timeseries(vals)
        expected = np.corrcoef(np.array([vals[:-3], vals[3:]]))[0, 1]
        np.testing.assert_allclose(autocorr(arr, 3), expected)

    def test_array_of_lags_returns_dataarray(self) -> None:
        """A sequence of lags returns a DataArray indexed by lag."""
        arr = _timeseries(np.random.default_rng(3).standard_normal(30))
        result = autocorr(arr, [0, 1, 2])
        assert isinstance(result, xr.DataArray)
        assert result.dims == ("lag",)
        np.testing.assert_allclose(result.sel(lag=0).item(), 1.0)


class TestSpearman:
    """Tests for spearman."""

    def test_matches_scipy_spearmanr(self) -> None:
        """Broadcast Spearman equals scipy.stats.spearmanr on the same data."""
        rng = np.random.default_rng(0)
        a = _timeseries(rng.standard_normal(30), name="a")
        b = _timeseries(rng.standard_normal(30), name="b")
        expected = scipy.stats.spearmanr(a.values, b.values)[0]
        np.testing.assert_allclose(spearman(a, b, "time").item(), expected)

    def test_monotone_nonlinear_gives_unit_rank_corr(self) -> None:
        """A monotone-increasing (nonlinear) map gives Spearman = 1."""
        x = np.linspace(0.1, 3.0, 25)
        a = _timeseries(x, name="a")
        b = _timeseries(np.exp(x), name="b")
        np.testing.assert_allclose(spearman(a, b, "time").item(), 1.0)


class TestLagCorr:
    """Tests for lag_corr."""

    def test_zero_lag_matches_corrcoef(self) -> None:
        rng = np.random.default_rng(0)
        a = _timeseries(rng.standard_normal(30), name="a")
        b = _timeseries(rng.standard_normal(30), name="b")
        expected = np.corrcoef(a.values, b.values)[0, 1]
        np.testing.assert_allclose(lag_corr(a, b, 0), expected)

    def test_positive_lag_reconstructs_offset_slices(self) -> None:
        """Positive lag correlates arr1[lag:] against arr2[:-lag] (arr1 leads).

        This pins the slice offsets and the leading/lagging convention: a
        positive lag drops the first ``lag`` points of arr1 and the last
        ``lag`` of arr2 before correlating.
        """
        rng = np.random.default_rng(1)
        a = _timeseries(rng.standard_normal(40), name="a")
        b = _timeseries(rng.standard_normal(40), name="b")
        lag = 5
        expected = np.corrcoef(np.array([a.values[lag:], b.values[:-lag]]))[0, 1]
        np.testing.assert_allclose(lag_corr(a, b, lag), expected)

    def test_negative_lag_reconstructs_offset_slices(self) -> None:
        """Negative lag correlates arr1[:lag] against arr2[-lag:] (arr2 leads)."""
        rng = np.random.default_rng(2)
        a = _timeseries(rng.standard_normal(40), name="a")
        b = _timeseries(rng.standard_normal(40), name="b")
        lag = -4
        expected = np.corrcoef(np.array([a.values[:lag], b.values[-lag:]]))[0, 1]
        np.testing.assert_allclose(lag_corr(a, b, lag), expected)

    def test_shifted_copy_peaks_at_true_lag(self) -> None:
        """A series and its own forward shift peak at the shift value.

        b is a leading by 3 (b[t] = a[t+3]); the lag correlation should be
        (near) 1 at lag = 3 and smaller elsewhere, confirming the sign of the
        convention, not just its magnitude.
        """
        rng = np.random.default_rng(3)
        base = rng.standard_normal(50)
        a = _timeseries(base, name="a")
        b = _timeseries(np.roll(base, -3), name="b")
        result = lag_corr(a, b, list(range(1, 8)))
        assert isinstance(result, xr.DataArray)
        assert result.sel(lag=3).item() > 0.99
        assert result.sel(lag=3).item() == result.max().item()


class TestLinRegress:
    """Tests for lin_regress."""

    def test_matches_scipy_linregress(self) -> None:
        """All five reported parameters match scipy.stats.linregress."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(40)
        y = 2.5 * x + 1.0 + rng.standard_normal(40) * 0.1
        arr_x = _timeseries(x, name="x")
        arr_y = _timeseries(y, name="y")
        result = lin_regress(arr_x, arr_y, "time")
        slope, intercept, r_val, p_val, std_err = scipy.stats.linregress(x, y)
        np.testing.assert_allclose(result.sel(parameter="slope").item(), slope)
        np.testing.assert_allclose(result.sel(parameter="intercept").item(), intercept)
        np.testing.assert_allclose(result.sel(parameter="r_value").item(), r_val)
        np.testing.assert_allclose(result.sel(parameter="p_value").item(), p_val)
        np.testing.assert_allclose(result.sel(parameter="std_err").item(), std_err)


class TestMultiRegress:
    """Tests for multi_regress."""

    def test_matches_sklearn_and_recovers_coeffs(self) -> None:
        """Predicted values match sklearn and recover the known coefficients."""
        rng = np.random.default_rng(0)
        n = 50
        p1 = rng.standard_normal(n)
        p2 = rng.standard_normal(n)
        target_vals = 3.0 * p1 - 2.0 * p2 + 4.0
        target = _timeseries(target_vals, name="y")
        predictors = [_timeseries(p1, name="p1"), _timeseries(p2, name="p2")]
        clf, predicted = multi_regress(target, predictors)
        np.testing.assert_allclose(clf.coef_, [3.0, -2.0], atol=1e-9)
        np.testing.assert_allclose(clf.intercept_, 4.0, atol=1e-9)
        np.testing.assert_allclose(predicted.values, target_vals, atol=1e-9)


class TestRegressResid:
    """Tests for regress_resid."""

    def test_residual_uncorrelated_with_predictor(self) -> None:
        """The returned residual is orthogonal to the predictor."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(60)
        y = 1.5 * x + rng.standard_normal(60)
        arr_x = _timeseries(x, name="x")
        arr_y = _timeseries(y, name="y")
        ds = regress_resid(arr_x, arr_y, "time")
        corr = np.corrcoef(x, ds["resid"].values)[0, 1]
        np.testing.assert_allclose(corr, 0.0, atol=1e-9)

    def test_predic_plus_resid_recovers_target(self) -> None:
        """Prediction plus residual reconstructs the original y."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal(30)
        y = -0.7 * x + 2.0 + rng.standard_normal(30)
        arr_x = _timeseries(x, name="x")
        arr_y = _timeseries(y, name="y")
        ds = regress_resid(arr_x, arr_y, "time")
        np.testing.assert_allclose((ds["predic"] + ds["resid"]).values, y, atol=1e-9)

    def test_slope_matches_linregress(self) -> None:
        """The reported slope matches scipy.stats.linregress."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal(40)
        y = 0.3 * x + rng.standard_normal(40)
        ds = regress_resid(_timeseries(x), _timeseries(y), "time")
        slope = scipy.stats.linregress(x, y).slope
        np.testing.assert_allclose(ds["slope"].item(), slope)


class TestQuantileRegress:
    """Tests for quantile_regress."""

    def test_median_regression_recovers_slope(self) -> None:
        """Median (0.5) quantile regression recovers a clean linear slope."""
        x = np.linspace(-2.0, 2.0, 40)
        y = 2.0 * x + 1.0
        arr_y = xr.DataArray(
            y[:, np.newaxis],
            dims=["time", "x"],
            coords={"time": np.arange(40), "x": [0]},
        )
        predictor = _timeseries(x, name="x")
        result = quantile_regress(arr_y, predictor, 0.5)
        np.testing.assert_allclose(result.isel(x=0).item(), 2.0, atol=1e-6)


class TestRmse:
    """Tests for rmse."""

    def test_matches_raw_formula(self) -> None:
        """rmse equals sqrt(mean((x - y)**2)) reconstructed from numpy."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(30)
        y = rng.standard_normal(30)
        arr_x = _timeseries(x, name="x")
        arr_y = _timeseries(y, name="y")
        expected = np.sqrt(np.mean((x - y) ** 2))
        np.testing.assert_allclose(rmse(arr_x, arr_y, "time").item(), expected)

    def test_matches_sklearn(self) -> None:
        """rmse agrees with sklearn's mean_squared_error under a sqrt."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal(25)
        y = x + rng.standard_normal(25) * 0.3
        expected = np.sqrt(sklearn.metrics.mean_squared_error(x, y))
        np.testing.assert_allclose(
            rmse(_timeseries(x), _timeseries(y), "time").item(), expected
        )

    def test_identical_arrays_give_zero(self) -> None:
        arr = _timeseries(np.arange(10.0))
        np.testing.assert_allclose(rmse(arr, arr, "time").item(), 0.0)


# ---------------------------------------------------------------------------
# EOFs.
# ---------------------------------------------------------------------------
class TestEofSolver:
    """Tests for eof_solver."""

    def test_rank_one_field_explained_by_first_eof(self) -> None:
        """A separable (rank-1) field is captured entirely by the first EOF."""
        lat = np.linspace(-60.0, 60.0, 12)
        lon = np.linspace(0.0, 330.0, 12)
        year = np.arange(30)
        spatial = np.outer(np.cos(np.deg2rad(lat)), np.sin(np.deg2rad(lon)))
        temporal = np.sin(year / 3.0)
        data = temporal[:, None, None] * spatial[None, :, :]
        arr = xr.DataArray(
            data,
            dims=[YEAR_STR, LAT_STR, LON_STR],
            coords={YEAR_STR: year, LAT_STR: lat, LON_STR: lon},
        )
        solver = eof_solver(arr)
        var_frac = solver.varianceFraction()
        np.testing.assert_allclose(var_frac.values[0], 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Histograms.
# ---------------------------------------------------------------------------
class TestXhist:
    """Tests for xhist."""

    def test_counts_match_numpy_histogram(self) -> None:
        """Bin counts equal np.histogram on the same edges."""
        rng = np.random.default_rng(0)
        vals = rng.standard_normal(200)
        arr = xr.DataArray(vals, dims=["event"])
        edges = np.linspace(-3.0, 3.0, 7)
        expected, _ = np.histogram(vals, bins=edges)
        result = xhist(arr, edges)
        np.testing.assert_array_equal(result.values, expected)

    def test_counts_sum_to_in_range_total(self) -> None:
        """Counts sum to the number of in-range samples."""
        vals = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        arr = xr.DataArray(vals, dims=["event"])
        edges = np.array([-0.5, 0.5, 1.5, 2.5])
        result = xhist(arr, edges)
        assert result.sum().item() == 6


class TestHist2d:
    """Tests for hist2d."""

    def test_counts_match_numpy_histogram2d(self) -> None:
        """2-D counts equal np.histogram2d on the same edges."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal(300)
        b = rng.standard_normal(300)
        arr1 = xr.DataArray(a, dims=["event"])
        arr2 = xr.DataArray(b, dims=["event"])
        edges1 = np.linspace(-3.0, 3.0, 5)
        edges2 = np.linspace(-3.0, 3.0, 6)
        centers1 = xr.DataArray(
            0.5 * (edges1[1:] + edges1[:-1]), dims=["c1"], name="c1"
        )
        centers2 = xr.DataArray(
            0.5 * (edges2[1:] + edges2[:-1]), dims=["c2"], name="c2"
        )
        expected, _, _ = np.histogram2d(a, b, bins=[edges1, edges2])
        result = hist2d(arr1, arr2, edges1, edges2, centers1, centers2)
        # result is transposed relative to numpy's (nx, ny) convention.
        np.testing.assert_array_equal(result.transpose("c1", "c2").values, expected)


class TestHist:
    """Tests for hist (broadcast)."""

    def test_matches_per_row_numpy_histogram(self) -> None:
        """Each broadcast row equals np.histogram of that row."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((3, 100))
        arr = xr.DataArray(data, dims=["x", "event"], coords={"x": [0, 1, 2]})
        edges = np.linspace(-3.0, 3.0, 7)
        result = hist(arr, "event", edges)
        for i in range(3):
            expected, _ = np.histogram(data[i], bins=edges)
            np.testing.assert_array_equal(result.isel(x=i).values, expected)


# ---------------------------------------------------------------------------
# Empirical CDF and risk ratio.
# ---------------------------------------------------------------------------
class TestCdfEmpirical:
    """Tests for cdf_empirical."""

    def test_matches_statsmodels_ecdf(self) -> None:
        """Values match a direct statsmodels ECDF evaluation."""
        rng = np.random.default_rng(0)
        vals = rng.standard_normal(100)
        arr = xr.DataArray(vals, dims=["event"])
        points = np.linspace(-2.0, 2.0, 9)
        expected = ECDF(vals)(points)
        result = cdf_empirical(arr, cdf_points=points)
        np.testing.assert_allclose(result.values, expected)

    def test_known_step_values(self) -> None:
        """For 1..4, the left-continuous ECDF gives the expected step heights."""
        arr = xr.DataArray(np.array([1.0, 2.0, 3.0, 4.0]), dims=["event"])
        points = np.array([0.0, 1.0, 2.5, 4.0, 5.0])
        result = cdf_empirical(arr, cdf_points=points)
        np.testing.assert_allclose(result.values, [0.0, 0.0, 0.5, 0.75, 1.0])

    def test_accepts_bare_numpy(self) -> None:
        """A plain ndarray (no .values) is handled via the AttributeError path."""
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        result = cdf_empirical(vals, cdf_points=np.array([2.5]))
        np.testing.assert_allclose(result.values, [0.5])


class TestRiskRatio:
    """Tests for risk_ratio."""

    def test_reconstructs_exceedance_ratio(self) -> None:
        """Equals (1 - ECDF1) / (1 - ECDF2) rebuilt from raw numpy ECDFs.

        This pins both the numerator/denominator assignment (arr1 over arr2)
        and the one-minus-CDF exceedance transform.
        """
        arr1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        arr2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        points = np.array([2.0, 3.0, 4.0])
        cdf1 = ECDF(arr1, side="left")(points)
        cdf2 = ECDF(arr2, side="left")(points)
        expected = (1.0 - cdf1) / (1.0 - cdf2)
        result = risk_ratio(
            xr.DataArray(arr1, dims=["event"]),
            xr.DataArray(arr2, dims=["event"]),
            cdf_points=points,
        )
        np.testing.assert_allclose(result.values, expected)

    def test_identical_distributions_give_unity(self) -> None:
        """Two identical samples have risk ratio 1 wherever it is defined."""
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        da = xr.DataArray(arr, dims=["event"])
        result = risk_ratio(da, da, cdf_points=np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result.values, 1.0)

    def test_default_points_are_union_of_inputs(self) -> None:
        """With cdf_points unset the CDF is sampled on the sorted union."""
        arr1 = xr.DataArray(np.array([1.0, 3.0]), dims=["event"])
        arr2 = xr.DataArray(np.array([2.0, 3.0]), dims=["event"])
        result = risk_ratio(arr1, arr2)
        np.testing.assert_array_equal(result["data"].values, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Distribution fits.
# ---------------------------------------------------------------------------
class TestXfit:
    """Tests for xfit."""

    def test_matches_direct_dist_fit(self) -> None:
        """Fitted parameters match a direct scipy dist.fit on the same data."""
        rng = np.random.default_rng(0)
        data = scipy.stats.norm.rvs(loc=2.0, scale=1.5, size=200, random_state=rng)
        arr = xr.DataArray(
            data[:, np.newaxis],
            dims=["time", "x"],
            coords={"time": np.arange(200), "x": [0]},
        )
        result = xfit(arr, "time", dist=scipy.stats.norm)
        expected = np.array(scipy.stats.norm.fit(data))
        np.testing.assert_allclose(result.isel(x=0).values, expected)


# ---------------------------------------------------------------------------
# False discovery rate.
# ---------------------------------------------------------------------------
class TestFalseDiscRateThreshPval:
    """Tests for false_disc_rate_thresh_pval."""

    def test_reconstructs_wilks_threshold(self) -> None:
        """Threshold matches the raw Wilks-2016 rebuild from numpy.

        The largest p-value satisfying p_(i) <= (i/N) * target_fdr is the
        threshold; this reconstructs the whole ranked comparison, pinning the
        arange numerator, the /N, and the <= partition.
        """
        # Chosen so the deciding p-value sits on its rank threshold rhs =
        # [.01, .02, .03, .04, .05]: the fourth-ranked 0.040 is exactly at
        # rhs_4 while the fifth 0.055 sits just above rhs_5. This makes the
        # result move under both an arange-numerator offset and a /N-divisor
        # change (verified by source mutation).
        pvals = np.array([0.055, 0.005, 0.040, 0.015, 0.025])
        arr = xr.DataArray(pvals, dims=["event"])
        target_fdr = 0.05
        srt = np.sort(pvals)
        rhs = np.arange(1, srt.size + 1) / srt.size * target_fdr
        expected = srt[srt <= rhs].max()
        result = false_disc_rate_thresh_pval(arr, target_fdr=target_fdr)
        np.testing.assert_allclose(result, expected)
        # The construction is nondegenerate: some but not all pass.
        assert 0 < (srt <= rhs).sum() < srt.size

    def test_threshold_is_inclusive_at_boundary(self) -> None:
        """A p-value exactly on its rank line counts as accepted (<=, not <).

        With N=4 and target_fdr=0.04 the fourth rank line is 4/4*0.04 = 0.04
        exactly; a p-value of 0.04 must be included, making the threshold 0.04
        rather than the next-lower 0.025. This flips if the comparison is made
        strict (verified by source mutation).
        """
        pvals = np.array([0.005, 0.015, 0.025, 0.04])
        arr = xr.DataArray(pvals, dims=["event"])
        result = false_disc_rate_thresh_pval(arr, target_fdr=0.04)
        np.testing.assert_allclose(result, 0.04)

    def test_all_significant_returns_largest(self) -> None:
        """When every p-value passes, the threshold is the largest p-value."""
        pvals = np.array([0.001, 0.002, 0.003])
        arr = xr.DataArray(pvals, dims=["event"])
        result = false_disc_rate_thresh_pval(arr, target_fdr=0.5)
        np.testing.assert_allclose(result, 0.003)

    def test_drops_nans(self) -> None:
        """NaNs are ignored via flat_dropna before ranking."""
        pvals = np.array([0.001, np.nan, 0.008, 0.02, np.nan])
        arr = xr.DataArray(pvals, dims=["event"])
        clean = np.array([0.001, 0.008, 0.02])
        rhs = np.arange(1, clean.size + 1) / clean.size * 0.05
        expected = clean[clean <= rhs].max()
        result = false_disc_rate_thresh_pval(arr, target_fdr=0.05)
        np.testing.assert_allclose(result, expected)
