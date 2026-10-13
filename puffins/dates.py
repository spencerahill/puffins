"""Functionality relating to calendars, times, and dates."""

from typing import cast, overload

import numpy as np
import xarray as xr

from .names import TIME_STR, YEAR_STR

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
ann_subsets: dict[str, int | list[int] | range] = {
    key: val for subs in _ann_subs for key, val in subs.items()
}


def _months_list(months: str | list[int] | int) -> list[int]:
    """The calendar months that ``months`` names, as a list."""
    vals = ann_subsets[months.lower()] if isinstance(months, str) else months
    return [vals] if isinstance(vals, int) else list(vals)


def subset_ann(
    arr: xr.DataArray,
    months: str | list[int] | int,
    dim_time: str = TIME_STR,
    drop: bool = False,
) -> xr.DataArray:
    """Restrict array values to a subset of each calendar year.

    Exactly the listed months are kept, so ``[12, 1, 2]`` keeps December,
    January and February, and ``[1, 3]`` keeps January and March.  With
    ``drop`` False the other months are set to NaN; with ``drop`` True they
    are removed.
    """
    if months == "ann":
        return arr
    in_months = arr[dim_time].dt.month.isin(_months_list(months))
    return cast(xr.DataArray, arr.where(in_months, drop=drop))


def _season_start(months: list[int]) -> int | None:
    """First month of the unbroken run ``months``, or None for all twelve.

    The run is cyclic, so ``[11, 12, 1]`` starts in November.

    Raises
    ------
    ValueError
        If ``months`` repeats a month, holds a value outside 1-12, or is not
        one unbroken run of consecutive months.
    """
    month_set = set(months)
    if len(month_set) != len(months) or not month_set <= set(range(1, 13)):
        raise ValueError(f"months must be distinct calendar months 1-12, got {months}")
    if len(month_set) == 12:
        return None
    # A month whose predecessor is absent begins a run; one run has one.
    starts = [m for m in month_set if (m - 2) % 12 + 1 not in month_set]
    if len(starts) != 1:
        raise ValueError(
            f"months must be one unbroken run of consecutive months, got {months}"
        )
    return starts[0]


def ann_subset_ts(
    arr: xr.DataArray,
    months: str | list[int] | int,
    reduction: str = "mean",
    dim_time: str = TIME_STR,
) -> xr.DataArray:
    """Annually resolved timeseries of array reduced over the given months.

    Default reduction is an average, with every month weighted equally.

    A year's value is NaN wherever any of the season's months is missing,
    whether the month lies outside the record or its value is NaN, and this
    holds for every reduction.  A season that crosses the year end belongs to
    the year of its January: with ``months=[12, 1, 2]`` the 2001 value is
    December 2000 through February 2001.  The output has one entry per
    calendar year of ``arr``.

    Parameters
    ----------
    arr : xarray.DataArray
        Monthly data, with one time step per calendar month.
    months : str, int or list of int
        A key of ``ann_subsets``, one month, or a list of months forming one
        unbroken run, in any order.  The run is cyclic, so ``[11, 12, 1]`` is
        allowed.
    reduction : str
        Name of the groupby reduction, such as "mean" or "max".
    dim_time : str
        Name of the time dimension.

    Returns
    -------
    xarray.DataArray
        The reduced values, with a ``year`` dimension in place of ``dim_time``.

    Raises
    ------
    ValueError
        If ``months`` is not one unbroken run of distinct months, if ``arr``
        has more than one time step in a calendar month, or if ``arr`` holds
        none of the months.
    """
    month_list = _months_list(months)
    start = _season_start(month_list)
    time = arr[dim_time]
    year_month = time.dt.year * 12 + time.dt.month
    if np.unique(year_month).size != year_month.size:
        raise ValueError("arr must be monthly, with one time step per calendar month")
    sub = arr.isel({dim_time: time.dt.month.isin(month_list)})
    if sub.sizes[dim_time] == 0:
        raise ValueError(f"arr holds none of the months {month_list}")
    sub_month = sub[dim_time].dt.month
    season_year = sub[dim_time].dt.year
    if start is not None and start + len(month_list) - 1 > 12:
        # The season crosses the year end: its months from `start` through
        # December belong to the following year's season.
        season_year = season_year + (sub_month >= start)
    season_year = season_year.rename(YEAR_STR)
    reduced = getattr(sub.groupby(season_year), reduction)(dim_time)
    n_valid = sub.notnull().groupby(season_year).sum(dim_time)
    complete = reduced.where(n_valid == len(month_list))
    return cast(xr.DataArray, complete.reindex({YEAR_STR: np.unique(time.dt.year)}))


def ann_ts_djf(arr: xr.DataArray) -> xr.DataArray:
    """Annual timeseries of December-February means.

    Each winter belongs to the year of its January, so the 2001 value is the
    mean of December 2000, January 2001 and February 2001, each month weighted
    equally.  A winter missing any of the three months is NaN, which includes
    the first year of a record that starts in January.  See ``ann_subset_ts``.
    """
    return ann_subset_ts(arr, [12, 1, 2])


def monthly_anom(
    arr: xr.DataArray,
    dim_time: str = TIME_STR,
    base: tuple[int, int] | None = None,
) -> xr.DataArray:
    """Anomaly of each month from its calendar-month climatology.

    For each calendar month, the climatological mean of that month is
    subtracted from every occurrence of that month, removing the mean seasonal
    cycle.  The climatology spans every year of ``arr``, or the years of
    ``base`` when it is given.  Any non-time dimensions are retained, with the
    climatology computed separately at each of their points.  The output
    gains a ``month`` coordinate along ``dim_time``.

    Parameters
    ----------
    arr : xarray.DataArray
        Data along a datetime ``dim_time`` dimension.
    dim_time : str
        Name of the time dimension.
    base : tuple of int, optional
        First and last years of the climatology, inclusive.

    Raises
    ------
    ValueError
        If ``base`` is given and ``arr`` lacks any month of any of its years.
    """
    grouped = arr.groupby(dim_time + ".month")
    if base is None:
        clim = grouped.mean(dim_time)
    else:
        years = arr[dim_time].dt.year
        in_base = (years >= base[0]) & (years <= base[1])
        base_times = arr[dim_time][in_base]
        n_months = np.unique(base_times.dt.year * 12 + base_times.dt.month).size
        if n_months != 12 * (base[1] - base[0] + 1):
            raise ValueError(
                f"the data do not cover every month of the base years "
                f"{base[0]}-{base[1]}"
            )
        clim = arr.isel({dim_time: in_base}).groupby(dim_time + ".month").mean(dim_time)
    return cast(xr.DataArray, grouped - clim)


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
    mfft = np.fft.fft(np.asarray(arr_anom))
    mask = np.zeros_like(mfft)
    if do_sum:
        mask[-num_harm:] = 1
    else:
        mask[-num_harm] = 1
    # These two lines below are for if you want the approximation at
    # whatever frequency has the most power.  What I want is the
    # approximation using just the lowest `n` frequencies.
    # imax = np.argmax(np.absolute(mfft))
    # mask[[imax]] = 1
    vals = float(arr_mean) + 2.0 * np.real(np.fft.ifft(mfft * mask))
    if normalize:
        vals /= np.abs(vals).max()
    if isinstance(arr, xr.DataArray):
        return cast(xr.DataArray, xr.ones_like(arr) * vals)
    return np.asarray(vals)


def time_to_year_and_day(arr: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Split time index into two, one for the year, one for the day of year"""
    return cast(
        xr.DataArray,
        arr.groupby(f"{dim}.year").map(lambda x: x.groupby(f"{dim}.dayofyear").first()),
    )
