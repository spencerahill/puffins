"""Functionality relating to calendars, times, and dates."""

from typing import cast, overload

import numpy as np
import xarray as xr

from .names import TIME_STR

months = dict(
    jan=1,
    feb=2,
    mar=3,
    apr=4,
    may=5,
    jun=6,
    jul=7,
    aug=8,
    sep=9,
    oct=10,
    nov=11,
    dec=12,
)
seasons_2mon = {
    "jf": [1, 2],
    "fm": [2, 3],
    "ma": [3, 4],
    "am": [4, 5],
    "mj": [5, 6],
    "jj": [6, 7],
    "ja": [7, 8],
    "as": [8, 9],
    "so": [9, 10],
    "on": [10, 11],
    "nd": [11, 12],
}
seasons_3mon = dict(
    jfm=[1, 2, 3],
    fma=[2, 3, 4],
    mam=[3, 4, 5],
    amj=[4, 5, 6],
    mjj=[5, 6, 7],
    jja=[6, 7, 8],
    jas=[7, 8, 9],
    aso=[8, 9, 10],
    son=[9, 10, 11],
    ond=[10, 11, 12],
)
seasons_4mon = dict(
    jfma=[1, 2, 3, 4],
    fmam=[2, 3, 4, 5],
    mamj=[3, 4, 5, 6],
    amjj=[4, 5, 6, 7],
    mjja=[5, 6, 7, 8],
    jjas=[6, 7, 8, 9],
    jaso=[7, 8, 9, 10],
    ason=[8, 9, 10, 11],
    sond=[9, 10, 11, 12],
)
seasons_5mon = dict(
    jfmam=[1, 2, 3, 4, 5],
    fmamj=[2, 3, 4, 5, 6],
    mamjj=[3, 4, 5, 6, 7],
    amjja=[4, 5, 6, 7, 8],
    mjjas=[5, 6, 7, 8, 9],
    jjaso=[6, 7, 8, 9, 10],
    jason=[7, 8, 9, 10, 11],
    asond=[8, 9, 10, 11, 12],
)
seasons_6mon = dict(
    jfmamj=[1, 2, 3, 4, 5, 6],
    fmamjj=[2, 3, 4, 5, 6, 7],
    mamjja=[3, 4, 5, 6, 7, 8],
    amjjas=[4, 5, 6, 7, 8, 9],
    mjjaso=[5, 6, 7, 8, 9, 10],
    jjason=[6, 7, 8, 9, 10, 11],
    jasond=[7, 8, 9, 10, 11, 12],
)
ann = dict(ann=range(1, 13))
_ann_subs = (
    months,
    seasons_2mon,
    seasons_3mon,
    seasons_4mon,
    seasons_5mon,
    seasons_6mon,
    ann,
)
ann_subsets: dict[str, int | list[int] | range] = {}
[ann_subsets.update(d) for d in _ann_subs]


def subset_ann(
    arr: xr.DataArray,
    months: str | list[int] | int,
    dim_time: str = TIME_STR,
    drop: bool = False,
) -> xr.DataArray:
    """Restrict array values to a subset of each calendar year."""
    months_vals: int | list[int] | range
    if isinstance(months, str):
        if months == "ann":
            return arr
        months_vals = ann_subsets[months.lower()]
    else:
        months_vals = months
    time = arr[dim_time]
    return cast(
        xr.DataArray,
        arr.where(
            (time.dt.month >= np.min(months_vals))
            & (time.dt.month <= np.max(months_vals)),
            drop=drop,
        ),
    )


def ann_subset_ts(
    arr: xr.DataArray,
    months: str | list[int] | int,
    reduction: str = "mean",
    dim_time: str = TIME_STR,
) -> xr.DataArray:
    """Annually resolved timeseries of array reduced over the given month.

    Default reduction is an average.

    """
    grouped = subset_ann(arr, months, dim_time).groupby(dim_time + ".year")
    func = getattr(grouped, reduction)
    return cast(xr.DataArray, func())


def ann_ts_djf(arr: xr.DataArray) -> xr.DataArray:
    """Annual timeseries of DJF values, accounting for year wrapping.

    Simple grouping by year would put the J, F, and D values in the same
    calendar year together, when what is desired is the December of one year
    grouped with the January and February of the following year.

    """
    arr_dec = ann_subset_ts(arr, 12)
    arr_jf = ann_subset_ts(arr, "jf")

    # Want previous year's December and this year's Jan-Feb.  So first year is
    # just Jan-Feb.  Accomplish this by just shifting the December years by 1,
    # then averaging.
    years_for_dec = arr_dec["year"].copy()
    years_for_dec = years_for_dec + 1
    arr_dec.coords["year"] = years_for_dec

    weights_dec = 31 * xr.ones_like(arr_dec["year"])
    weights_jf = (31 + 28.25) * xr.ones_like(arr_jf["year"])

    arr_djf_no_yr1 = (weights_dec * arr_dec + weights_jf * arr_jf) / (
        weights_dec + weights_jf
    )

    arr_djf = cast(
        xr.DataArray,
        xr.concat([arr_jf.isel(year=0), arr_djf_no_yr1], dim="year"),
    )
    if arr.ndim > 1:
        ind_time = arr.dims.index("time")
        dims_out = list(arr.dims)
        dims_out[ind_time] = "year"
        return cast(xr.DataArray, arr_djf.transpose(*dims_out))
    return arr_djf


@overload
def ann_harm(
    arr: xr.DataArray,
    num_harm: int = ...,
    normalize: bool = ...,
    do_sum: bool = ...,
) -> xr.DataArray: ...


@overload
def ann_harm(
    arr: np.ndarray,
    num_harm: int = ...,
    normalize: bool = ...,
    do_sum: bool = ...,
) -> np.ndarray: ...


def ann_harm(
    arr: xr.DataArray | np.ndarray,
    num_harm: int = 1,
    normalize: bool = False,
    do_sum: bool = True,
) -> xr.DataArray | np.ndarray:
    """Compute annual harmonics.

    Adapted from https://stackoverflow.com/a/69424590/1706640.

    """
    if num_harm == len(arr):
        return arr
    arr_mean = arr.mean()
    arr_anom = arr - arr_mean
    if isinstance(arr, xr.DataArray):
        mfft = np.fft.fft(arr_anom.values)
    else:
        mfft = np.fft.fft(arr_anom)
    mask = np.zeros_like(mfft)
    if do_sum:
        mask[-num_harm:] = 1
    else:
        mask[-num_harm] = 1
    vals = float(arr_mean) + 2.0 * np.real(np.fft.ifft(mfft * mask))
    if normalize:
        vals /= np.abs(vals).max()
    if isinstance(arr, xr.DataArray):
        return cast(xr.DataArray, xr.ones_like(arr) * vals)
    return np.asarray(vals)
    # These two lines below are for if you want the approximation at
    # whatever frequency has the most power.  What I want is the
    # approximation using just the lowest `n` frequencies.
    # imax = np.argmax(np.absolute(mfft))
    # mask[[imax]] = 1


def time_to_year_and_day(arr: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Split time index into two, one for the year, one for the day of year"""
    return cast(
        xr.DataArray,
        arr.groupby(f"{dim}.year").map(lambda x: x.groupby(f"{dim}.dayofyear").first()),
    )
