"""Bootstrap sampling"""

from __future__ import annotations

from typing import cast

import numpy as np
import sklearn.utils
import xarray as xr

from ._typing import ArrayLike
from .stats import risk_ratio


def bootstrap_samples(
    arr: xr.DataArray,
    dim: str,
    num_bootstraps: int = 1000,
    rand_states: np.ndarray | None = None,
) -> list[xr.DataArray]:
    """Return specified number of bootstrap-resampled arrays from the given array."""
    if rand_states is None:
        rand_states = np.random.default_rng().integers(
            0, 4_000_000_000, size=num_bootstraps
        )
    return [
        sklearn.utils.resample(arr.transpose(dim, ...), random_state=rand_state)
        for rand_state in rand_states
    ]


def corr_bootstrap(
    arr1: xr.DataArray,
    arr2: xr.DataArray,
    dim: str,
    num_bootstraps: int = 1000,
    dim_boot: str = "nboot",
) -> xr.DataArray:
    """Return specified number of bootstrap-resampled correlation coefficients."""
    arr1_aligned, arr2_aligned = xr.align(arr1, arr2)
    rand_states = np.random.default_rng().integers(
        0, 4_000_000_000, size=num_bootstraps
    )
    corrs_boot_arrs: list[xr.DataArray] = []
    for rand_state in rand_states:
        arr1_subsamp = sklearn.utils.resample(
            arr1_aligned.transpose(dim, ...), random_state=rand_state
        )
        arr2_subsamp = sklearn.utils.resample(
            arr2_aligned.transpose(dim, ...), random_state=rand_state
        )
        corrs_boot_arrs.append(xr.corr(arr1_subsamp, arr2_subsamp, dim))
    return xr.concat(corrs_boot_arrs, dim=dim_boot)


def corr_sig_nonzero_from_full_and_boot(
    corr_full: xr.DataArray,
    corrs_boot: xr.DataArray,
    alpha: float = 0.05,
    dim_boot: str = "nboot",
) -> xr.DataArray:
    """Significance of nonzero correlation, given full and bootstrap correlations."""
    quantiles = [0.5 * alpha, 1 - 0.5 * alpha]
    conf_intvl = corrs_boot.quantile(quantiles, dim=dim_boot)
    return cast(
        xr.DataArray,
        xr.where(
            corr_full > 0,
            corr_full.where(conf_intvl.isel(quantile=0) > 0),
            corr_full.where(conf_intvl.isel(quantile=1) < 0),
        ),
    )


def corr_sig_nonzero_bootstrap(
    arr1: xr.DataArray,
    arr2: xr.DataArray,
    dim: str,
    num_bootstraps: int = 1000,
    alpha: float = 0.05,
    dim_boot: str = "nboot",
) -> xr.DataArray:
    """Significance of nonzero correlation between two arrays estimated from bootstrap."""
    corrs_boot = corr_bootstrap(
        arr1, arr2, dim, num_bootstraps=num_bootstraps, dim_boot=dim_boot
    )
    corr_full = xr.corr(arr1, arr2, dim)
    return corr_sig_nonzero_from_full_and_boot(
        corr_full, corrs_boot, alpha=alpha, dim_boot=dim_boot
    )


def boot_risk_ratio(
    arr: xr.DataArray,
    num_numer: int,
    num_denom: int,
    dim: str,
    cdf_points: ArrayLike,
    num_bootstraps: int = 1000,
    side: str = "left",
) -> xr.DataArray:
    """Bootstrap risk ratio.

    Note that this is slow: each risk ratio calculation can be slow, especially if
    there are a lot of points sampled along the CDF, and then you're repeating it
    a large number of times.

    """
    assert num_numer + num_denom <= len(arr[dim])

    def _rr_one_sample() -> xr.DataArray:
        dim_randomized = np.random.default_rng().permutation(arr[dim])
        rand_numer = dim_randomized[:num_numer]
        rand_denom = dim_randomized[num_numer : num_numer + num_denom]
        return cast(
            xr.DataArray,
            risk_ratio(
                arr.sel({dim: rand_numer}),
                arr.sel({dim: rand_denom}),
                cdf_points=cdf_points,
                side=side,
            ),
        )

    boot_rr_vals = [_rr_one_sample() for _ in range(num_bootstraps)]
    return xr.concat(boot_rr_vals, dim="nboot")
