"""Functionality relating to calendars, times, and dates."""
import numpy as np

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
ann_subsets = {}
[ann_subsets.update(d) for d in _ann_subs]


def subset_ann(arr, months, dim_time=TIME_STR, drop=False):
    """Restrict array values to a subset of each calendar year.

    """
    if isinstance(months, str):
        if months == "ann":
            return arr
        months = ann_subsets[months]
    time = arr[dim_time]
    return arr.where((time.dt.month >= np.min(months)) &
                     (time.dt.month <= np.max(months)), drop=drop)


def ann_subset_ts(arr, months, reduction="mean", dim_time=TIME_STR):
    """Annually resolved timeseries of array reduced over the given month.

    Default reduction is an average.

    """
    grouped = subset_ann(arr, months, dim_time).groupby(dim_time + ".year")
    func = getattr(grouped, reduction)
    return func()
