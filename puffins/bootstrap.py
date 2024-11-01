"""Bootstrap sampling"""
import numpy as np
import sklearn.utils
import xarray as xr

from .stats import risk_ratio


def bootstrap_samples(arr, dim, num_bootstraps=1000, rand_states=None):
    """Return specified number of bootstrap-resampled arrays from the given array."""
    if rand_states is None:
        rand_states = np.random.randint(0, 4e9, size=num_bootstraps)
    return [
        sklearn.utils.resample(arr.transpose(dim, ...), random_state=rand_state)
        for rand_state in rand_states]


def corr_bootstrap(arr1, arr2, dim, num_bootstraps=1000, dim_boot="nboot"):
    """Return specified number of bootstrap-resampled correlation coefficients."""
    arr1_aligned, arr2_aligned = xr.align(arr1, arr2)
    rand_states = np.random.randint(0, 4e9, size=num_bootstraps)
    corrs_boot_arrs = []
    for rand_state in rand_states:
        arr1_subsamp = sklearn.utils.resample(arr1_aligned.transpose(dim, ...), random_state=rand_state)
        arr2_subsamp = sklearn.utils.resample(arr2_aligned.transpose(dim, ...), random_state=rand_state)
        corrs_boot_arrs.append(xr.corr(arr1_subsamp, arr2_subsamp, dim))
    return xr.concat(corrs_boot_arrs, dim=dim_boot)


def corr_sig_nonzero_from_full_and_boot(corr_full, corrs_boot, alpha=0.05, dim_boot="nboot"):
    """Significance of nonzero correlation, given full and bootstrap correlations."""
    quantiles = [0.5 * alpha, 1 - 0.5 * alpha]
    conf_intvl = corrs_boot.quantile(quantiles, dim=dim_boot)
    return xr.where(
        corr_full > 0, 
        corr_full.where(conf_intvl.isel(quantile=0) > 0),
        corr_full.where(conf_intvl.isel(quantile=1) < 0),
    )


def corr_sig_nonzero_bootstrap(arr1, arr2, dim, num_bootstraps=1000, alpha=0.05, dim_boot="nboot"):
    """Significance of nonzero correlation between two arrays estimated from bootstrap."""
    corrs_boot = corr_bootstrap(arr1, arr2, dim, num_bootstraps=num_bootstraps, dim_boot=dim_boot)
    corr_full = xr.corr(arr1, arr2, dim)
    return corr_sig_nonzero_from_full_and_boot(corr_full, corrs_boot, alpha=alpha, dim_boot=dim_boot)


def boot_risk_ratio(arr, num_numer, num_denom, dim, cdf_points, n_samples=100, side="left"):
    """Bootstrap risk ratio."""
    assert num_numer + num_denom <= len(arr[dim])

    def _rr_one_sample():
        dim_randomized = np.random.default_rng().permutation(arr[dim])
        rand_numer = dim_randomized[:num_numer]
        rand_denom = dim_randomized[num_numer:num_numer + num_denom]
        return risk_ratio(
            arr.sel(**{dim: rand_numer}),
            arr.sel(**{dim: rand_denom}),
            cdf_points=cdf_points,
            side=side,
        )

    boot_rr_vals = [_rr_one_sample() for _ in range(n_samples)]
    return xr.concat(boot_rr_vals, dim="nboot")
