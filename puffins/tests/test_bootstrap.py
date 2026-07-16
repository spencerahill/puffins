"""Tests for the bootstrap module."""

import numpy as np
import pytest
import xarray as xr

from puffins.bootstrap import (
    boot_risk_ratio,
    bootstrap_samples,
    corr_bootstrap,
    corr_sig_nonzero_bootstrap,
    corr_sig_nonzero_from_full_and_boot,
)
from puffins.stats import risk_ratio


def _timeseries(values: np.ndarray, name: str = "ts") -> xr.DataArray:
    """A 1-D DataArray along ``time`` with an integer time coordinate."""
    vals = np.asarray(values, dtype=float)
    return xr.DataArray(
        vals,
        dims=["time"],
        coords={"time": np.arange(vals.size)},
        name=name,
    )


class TestBootstrapSamples:
    """Tests for bootstrap_samples."""

    def test_num_samples_matches_num_bootstraps(self) -> None:
        """With rand_states unset, the list length equals num_bootstraps."""
        arr = _timeseries(np.arange(20.0))
        assert len(bootstrap_samples(arr, "time", num_bootstraps=17)) == 17

    def test_length_matches_rand_states(self) -> None:
        """Passing rand_states overrides num_bootstraps in setting the count."""
        arr = _timeseries(np.arange(20.0))
        samples = bootstrap_samples(arr, "time", rand_states=np.array([1, 2, 3, 4]))
        assert len(samples) == 4

    def test_preserves_length_along_dim(self) -> None:
        """Each resample has the same size along the sampled dimension."""
        arr = _timeseries(np.arange(20.0))
        sample = bootstrap_samples(arr, "time", rand_states=np.array([7]))[0]
        assert sample.sizes["time"] == arr.sizes["time"]

    def test_deterministic_given_rand_states(self) -> None:
        """The same rand_states reproduce identical resamples bitwise."""
        arr = _timeseries(np.random.default_rng(0).standard_normal(25))
        states = np.array([0, 1, 2])
        first = bootstrap_samples(arr, "time", rand_states=states)
        second = bootstrap_samples(arr, "time", rand_states=states)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a.values, b.values)

    def test_resamples_are_drawn_from_original(self) -> None:
        """Resampling with replacement only ever returns original values."""
        vals = np.arange(20.0) * 1.7
        arr = _timeseries(vals)
        sample = bootstrap_samples(arr, "time", rand_states=np.array([3]))[0]
        assert np.isin(sample.values, vals).all()

    def test_resamples_whole_rows_of_multidim_array(self) -> None:
        """For a 2-D array, resampling picks whole rows along the sampled dim."""
        rows = np.arange(24.0).reshape(8, 3)
        arr = xr.DataArray(
            rows,
            dims=["time", "x"],
            coords={"time": np.arange(8), "x": np.arange(3)},
            name="field",
        )
        sample = bootstrap_samples(arr, "time", rand_states=np.array([5]))[0]
        assert sample.sizes == {"time": 8, "x": 3}
        for row in sample.values:
            assert any(np.array_equal(row, orig) for orig in rows)


class TestCorrBootstrap:
    """Tests for corr_bootstrap."""

    def test_output_has_boot_dim_of_expected_length(self) -> None:
        """The output is 1-D along dim_boot with length num_bootstraps."""
        rng = np.random.default_rng(0)
        arr1 = _timeseries(rng.standard_normal(30), name="a")
        arr2 = _timeseries(rng.standard_normal(30), name="b")
        boot = corr_bootstrap(arr1, arr2, "time", num_bootstraps=40)
        assert boot.dims == ("nboot",)
        assert boot.sizes["nboot"] == 40

    def test_custom_dim_boot_name(self) -> None:
        """The bootstrap dimension name is configurable."""
        rng = np.random.default_rng(1)
        arr1 = _timeseries(rng.standard_normal(30), name="a")
        arr2 = _timeseries(rng.standard_normal(30), name="b")
        boot = corr_bootstrap(arr1, arr2, "time", num_bootstraps=10, dim_boot="rep")
        assert boot.dims == ("rep",)

    def test_perfectly_correlated_gives_unit_correlation(self) -> None:
        """A resample of an exactly linear pair correlates at +1 every time.

        Any resample of points lying on ``y = a*x + b`` (a > 0) stays on the
        line, so the Pearson correlation is exactly 1 regardless of the draw.
        """
        x = np.random.default_rng(2).standard_normal(30)
        arr1 = _timeseries(x, name="a")
        arr2 = _timeseries(2.0 * x + 3.0, name="b")
        boot = corr_bootstrap(arr1, arr2, "time", num_bootstraps=100)
        np.testing.assert_allclose(boot.values, 1.0, atol=1e-9)

    def test_perfectly_anticorrelated_gives_negative_unit_correlation(self) -> None:
        """A negative-slope linear pair correlates at -1 for every resample."""
        x = np.random.default_rng(3).standard_normal(30)
        arr1 = _timeseries(x, name="a")
        arr2 = _timeseries(-2.0 * x + 3.0, name="b")
        boot = corr_bootstrap(arr1, arr2, "time", num_bootstraps=100)
        np.testing.assert_allclose(boot.values, -1.0, atol=1e-9)

    def test_correlations_have_no_nans(self) -> None:
        """No resample degenerates to a NaN correlation (range is definitional).

        Since a Pearson correlation is bounded to [-1, 1] by construction, the
        substantive guard here is against NaN contamination, which a resample
        collapsing to zero variance would produce; the range bound is a cheap
        secondary sanity check.
        """
        rng = np.random.default_rng(4)
        arr1 = _timeseries(rng.standard_normal(40), name="a")
        arr2 = _timeseries(rng.standard_normal(40), name="b")
        boot = corr_bootstrap(arr1, arr2, "time", num_bootstraps=100)
        assert not np.isnan(boot.values).any()
        assert float(boot.min()) >= -1.0 - 1e-9
        assert float(boot.max()) <= 1.0 + 1e-9


def _expected_sig(corr_full_val: float, boot_vals: np.ndarray, alpha: float = 0.05):
    """Reconstruct the significance decision from raw numpy.

    A positive correlation is significant iff the whole (1 - alpha) confidence
    interval lies above zero; a nonpositive one iff it lies below zero.
    """
    lower, upper = np.quantile(boot_vals, [0.5 * alpha, 1.0 - 0.5 * alpha])
    if corr_full_val > 0:
        return corr_full_val if lower > 0 else np.nan
    return corr_full_val if upper < 0 else np.nan


class TestCorrSigNonzeroFromFullAndBoot:
    """Tests for corr_sig_nonzero_from_full_and_boot (the significance formula)."""

    CASES = [
        (0.5, np.linspace(0.1, 0.9, 200)),  # positive, CI above 0 -> significant
        (0.5, np.linspace(-0.4, 0.9, 200)),  # positive, CI spans 0 -> not sig
        (-0.5, np.linspace(-0.9, -0.1, 200)),  # negative, CI below 0 -> significant
        (-0.5, np.linspace(-0.9, 0.4, 200)),  # negative, CI spans 0 -> not sig
    ]

    @pytest.mark.parametrize(
        "corr_full_val,boot_vals",
        CASES,
        ids=["pos-sig", "pos-notsig", "neg-sig", "neg-notsig"],
    )
    def test_matches_raw_numpy_reconstruction(
        self, corr_full_val: float, boot_vals: np.ndarray
    ) -> None:
        """The output matches a raw-numpy rebuild of the CI significance test."""
        corrs_boot = xr.DataArray(boot_vals, dims=["nboot"])
        result = corr_sig_nonzero_from_full_and_boot(
            xr.DataArray(corr_full_val), corrs_boot
        )
        expected = _expected_sig(corr_full_val, boot_vals)
        if np.isnan(expected):
            assert np.isnan(result.item())
        else:
            np.testing.assert_allclose(result.item(), expected)

    def test_respects_alpha_as_two_sided_half_width(self) -> None:
        """The alpha argument enters as 0.5*alpha per tail.

        The bootstrap distribution is built so its lower 2.5th percentile is
        below zero but its lower 5th percentile is above zero. A positive
        correlation is then not significant at alpha=0.05 but is significant at
        alpha=0.10, which only holds if alpha is halved for the lower tail.
        """
        boot_vals = np.concatenate(
            [np.linspace(-0.3, -0.01, 7), np.linspace(0.01, 0.9, 193)]
        )
        corrs_boot = xr.DataArray(boot_vals, dims=["nboot"])
        corr_full = xr.DataArray(0.5)
        # Confirm the construction actually produces the flip.
        assert np.isnan(_expected_sig(0.5, boot_vals, alpha=0.05))
        assert _expected_sig(0.5, boot_vals, alpha=0.10) == 0.5
        res_05 = corr_sig_nonzero_from_full_and_boot(corr_full, corrs_boot, alpha=0.05)
        res_10 = corr_sig_nonzero_from_full_and_boot(corr_full, corrs_boot, alpha=0.10)
        assert np.isnan(res_05.item())
        np.testing.assert_allclose(res_10.item(), 0.5)

    def test_broadcasts_over_extra_dimension(self) -> None:
        """Significance is evaluated independently along non-bootstrap dims."""
        boot = xr.DataArray(
            np.stack([np.linspace(0.1, 0.9, 100), np.linspace(-0.4, 0.9, 100)]),
            dims=["loc", "nboot"],
        )
        corr_full = xr.DataArray([0.5, 0.5], dims=["loc"])
        result = corr_sig_nonzero_from_full_and_boot(corr_full, boot)
        assert dict(result.sizes) == {"loc": 2}
        np.testing.assert_allclose(result.isel(loc=0).item(), 0.5)
        assert np.isnan(result.isel(loc=1).item())


class TestCorrSigNonzeroBootstrap:
    """Tests for the end-to-end corr_sig_nonzero_bootstrap."""

    def test_perfectly_correlated_is_significant_positive(self) -> None:
        """An exactly linear positive pair is flagged significant at its value."""
        x = np.random.default_rng(5).standard_normal(30)
        arr1 = _timeseries(x, name="a")
        arr2 = _timeseries(2.0 * x + 3.0, name="b")
        sig = corr_sig_nonzero_bootstrap(arr1, arr2, "time", num_bootstraps=50)
        np.testing.assert_allclose(sig.item(), 1.0, atol=1e-9)

    def test_perfectly_anticorrelated_is_significant_negative(self) -> None:
        """An exactly linear negative pair is flagged significant at -1."""
        x = np.random.default_rng(6).standard_normal(30)
        arr1 = _timeseries(x, name="a")
        arr2 = _timeseries(-2.0 * x + 3.0, name="b")
        sig = corr_sig_nonzero_bootstrap(arr1, arr2, "time", num_bootstraps=50)
        np.testing.assert_allclose(sig.item(), -1.0, atol=1e-9)


class TestBootRiskRatio:
    """Tests for boot_risk_ratio."""

    def test_output_has_expected_boot_dim(self) -> None:
        """The output carries an ``nboot`` dimension of length num_bootstraps."""
        arr = _timeseries(np.random.default_rng(7).standard_normal(20))
        boot = boot_risk_ratio(
            arr, 5, 5, "time", cdf_points=np.array([-0.5, 0.0, 0.5]), num_bootstraps=15
        )
        assert boot.sizes["nboot"] == 15

    def test_seeded_output_reconstructs_manual_split(self) -> None:
        """With a seed, the output matches a manual permute-split-risk_ratio rebuild.

        Reproducing the seeded generator externally pins the whole permutation
        pipeline: the per-bootstrap permutation, the disjoint numerator and
        denominator split with its slice offsets, the risk_ratio call, and the
        concatenation. (risk_ratio itself is the dependency being orchestrated,
        so it is exercised, not re-derived, here.)
        """
        arr = _timeseries(np.arange(12.0) * 0.5)
        cdf_points = np.array([1.0, 2.0, 3.0])
        num_numer, num_denom, num_boot, seed = 4, 5, 6, 0
        result = boot_risk_ratio(
            arr,
            num_numer,
            num_denom,
            "time",
            cdf_points=cdf_points,
            num_bootstraps=num_boot,
            seed=seed,
        )
        rng = np.random.default_rng(seed)
        expected = []
        for _ in range(num_boot):
            perm = rng.permutation(arr["time"])
            numer = perm[:num_numer]
            denom = perm[num_numer : num_numer + num_denom]
            expected.append(
                risk_ratio(
                    arr.sel({"time": numer}),
                    arr.sel({"time": denom}),
                    cdf_points=cdf_points,
                    side="left",
                )
            )
        expected_boot = xr.concat(expected, dim="nboot")
        np.testing.assert_allclose(result.values, expected_boot.values, equal_nan=True)

    def test_seed_makes_output_reproducible(self) -> None:
        """The same seed reproduces the bootstrap draw; a different seed differs."""
        arr = _timeseries(np.arange(15.0) * 0.3)
        cdf_points = np.array([1.0, 2.0])
        first = boot_risk_ratio(
            arr, 4, 5, "time", cdf_points=cdf_points, num_bootstraps=8, seed=0
        )
        again = boot_risk_ratio(
            arr, 4, 5, "time", cdf_points=cdf_points, num_bootstraps=8, seed=0
        )
        other = boot_risk_ratio(
            arr, 4, 5, "time", cdf_points=cdf_points, num_bootstraps=8, seed=1
        )
        np.testing.assert_array_equal(first.values, again.values)
        assert not np.array_equal(first.values, other.values)

    def test_constant_array_gives_unit_risk_ratio(self) -> None:
        """A constant field has identical exceedance in both groups: ratio 1.

        With every value equal to 5 and CDF points below 5, the exceedance
        probability is 1 for both the numerator and denominator groups, so the
        risk ratio is exactly 1 for every permutation.
        """
        arr = _timeseries(np.full(12, 5.0))
        boot = boot_risk_ratio(
            arr, 4, 4, "time", cdf_points=np.array([3.0, 4.0]), num_bootstraps=20
        )
        np.testing.assert_allclose(boot.values, 1.0)

    def test_raises_when_groups_exceed_sample_size(self) -> None:
        """Requesting more numerator+denominator members than exist is rejected."""
        arr = _timeseries(np.arange(10.0))
        with pytest.raises(AssertionError):
            boot_risk_ratio(
                arr, 6, 6, "time", cdf_points=np.array([1.0]), num_bootstraps=2
            )
