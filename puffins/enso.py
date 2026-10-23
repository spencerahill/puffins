"""ENSO and tropical-ocean SST indices from a gridded monthly SST field.

The centerpiece is the Relative Oceanic Nino Index (RONI) that NOAA's Climate
Prediction Center (CPC) publishes, following L'Heureux et al. (2024, J. Climate
37, 1197-1211, doi:10.1175/JCLI-D-23-0406.1).  Their page 1200 defines it as
``(ONI - TropAve) * sigma_ONI / sigma_(ONI - TropAve)``:

1. the NINO3.4 SST anomaly minus the 20S-20N ocean-mean SST anomaly, both from
   a monthly climatology over a base period (1991-2020);
2. as 3-month running means;
3. rescaled separately for each calendar month by the ratio of the standard
   deviation of the NINO3.4 running mean to that of the difference, over
   1950-2020.

Subtracting the tropical mean removes the part of NINO3.4's change that the
whole tropical ocean shares, which is what makes the index "relative"; the idea
is van Oldenborgh et al. (2021, Environ. Res. Lett. 16, 044003), and the
rescaling by a ratio is L'Heureux et al.'s simplification of it.

CPC's main RONI is computed from ERSSTv6 SST (as of 2026-10), and CPC also
posts an ERSSTv5 version, so compare this module's output with the CPC table
built from the same SST product as its input.

``relative_nino34`` is the same quantity month by month, without the running
mean, and ``relative_nino34_jjas`` is its June-September mean, taken with
``dates.ann_subset_ts``.  The anomalies come from ``dates.monthly_anom``.
``enso_phase`` labels each value of an index El Niño, La Niña or neutral.  Its
defaults are CPC's rules for an index of overlapping 3-month seasons, one per
month, such as RONI; with ``min_run=1`` it labels each value of a monthly or
yearly index, such as these two, alone.

The module also gives the plain NINO3.4 box mean and the tropical Indian Ocean
(TIO) box of Yu et al. (2021, Geophys. Res. Lett. 48, e2021GL092873), 20S-20N
and 50-110E, as their Figure 3a caption states it.

All box means are cos(latitude)-weighted over the gridpoint centers inside
inclusive bounds, with land (NaN in an SST product) dropped from the weights as
well as from the sum.  Longitudes are compared on 0-360, whichever convention
the field and the bounds use.  A box that selects no gridpoint, or only NaN
cells, raises a ValueError, as does a field that repeats a longitude, such as
a copy of the 0E column at 360E.

These functions moved here on 2026-09-30 from the lps-enso-grl project's
``enso_indices.py``, so that the projects using them share one definition.
"""

import numbers
from typing import cast

import numpy as np
import xarray as xr

from .dates import ann_subset_ts, monthly_anom
from .longitude import lon_to_0360
from .names import LAT_STR, LON_STR, TIME_STR

NINO34_LAT: tuple[float, float] = (-5.0, 5.0)
NINO34_LON: tuple[float, float] = (190.0, 240.0)
TROPICS_LAT: tuple[float, float] = (-20.0, 20.0)
TIO_LAT: tuple[float, float] = (-20.0, 20.0)
TIO_LON: tuple[float, float] = (50.0, 110.0)

RONI_BASE: tuple[int, int] = (1991, 2020)
"""The years of the monthly climatology anomalies are taken from."""

RONI_SIGMA_YEARS: tuple[int, int] = (1950, 2020)
"""The years over which RONI's rescaling ratio is computed."""

ENSO_THRESHOLD: float = 0.5
"""The index magnitude, in K, that CPC's thresholds for El Niño and La Niña use."""

_ROUND_TOL = 1e-4
"""How far below a half a value may sit and still round up in ``enso_phase``,
in units of the last decimal kept.  Stored in float32, a two-decimal half
below 4 in magnitude sits up to 1e-6 below it in those units; 0.45, stored as
0.44999999, sits 1.2e-7 below."""


def box_mean(
    sst: xr.DataArray,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float] | None = None,
    lat_str: str = LAT_STR,
    lon_str: str = LON_STR,
) -> xr.DataArray:
    """cos(latitude)-weighted mean of ``sst`` over a latitude-longitude box.

    Parameters
    ----------
    sst : xarray.DataArray
        Field with latitude and longitude dimensions, plus any others.  NaN
        cells (land) are left out of both the sum and the weights.
    lat_bounds : tuple of float
        Inclusive latitude bounds, in either order.
    lon_bounds : tuple of float, optional
        Inclusive western and eastern longitude bounds.  They are compared
        with the field's longitudes on 0-360, whichever convention either
        uses, so (190, 240) and (-170, -120) select the same box.  A western
        bound east of the eastern one, such as (350, 10), wraps through 0E.
        So reversing the bounds selects the complementary arc: (240, 190) is
        the 310-degree arc from 240E east through 0E to 190E.  With None, or
        bounds 360 degrees or more apart, the mean spans every longitude.
    lat_str, lon_str : str
        Names of the latitude and longitude dimensions.

    Returns
    -------
    xarray.DataArray
        The box mean, with the latitude and longitude dimensions removed.  It
        is float64 whatever the input dtype.

    Raises
    ------
    ValueError
        If no gridpoint center falls inside the box, if every cell inside it
        is NaN, or if two of the field's longitudes coincide on 0-360.
    """
    lon360 = np.asarray(lon_to_0360(sst[lon_str]))
    repeated, counts = np.unique(lon360, return_counts=True)
    if (counts > 1).any():
        raise ValueError(
            f"longitudes {repeated[counts > 1].tolist()} (on 0-360) appear more "
            "than once; drop the duplicated column, such as a copy of 0E at 360E"
        )
    lat = sst[lat_str]
    lat_lo, lat_hi = sorted(lat_bounds)
    sub = sst.isel({lat_str: (lat >= lat_lo) & (lat <= lat_hi)})
    if sub.sizes[lat_str] == 0:
        raise ValueError(f"no gridpoint inside lat {lat_bounds}")
    if lon_bounds is not None and lon_bounds[1] - lon_bounds[0] < 360:
        west, east = lon_to_0360(lon_bounds[0]), lon_to_0360(lon_bounds[1])
        if west <= east:
            in_box = (lon360 >= west) & (lon360 <= east)
        else:
            in_box = (lon360 >= west) | (lon360 <= east)
        sub = sub.isel({lon_str: in_box})
        if sub.sizes[lon_str] == 0:
            raise ValueError(f"no gridpoint inside lon {lon_bounds}")
    if not sub.notnull().any():
        raise ValueError(f"every cell inside lat {lat_bounds}, lon {lon_bounds} is NaN")
    # Float64 weights make the weighted sums float64.  SST products are
    # stored as float32, whose summation order otherwise shows up at the 1e-6
    # relative level.
    coslat = cast(xr.DataArray, np.cos(np.deg2rad(sub[lat_str].astype(np.float64))))
    return cast(xr.DataArray, sub.weighted(coslat).mean((lat_str, lon_str)))


def nino34(sst: xr.DataArray, **kwargs: str) -> xr.DataArray:
    """NINO3.4 SST: the box mean over 5S-5N, 190-240E.

    ``kwargs`` pass coordinate names through to ``box_mean``.
    """
    return cast(
        xr.DataArray, box_mean(sst, NINO34_LAT, NINO34_LON, **kwargs).rename("nino34")
    )


def tropical_mean(sst: xr.DataArray, **kwargs: str) -> xr.DataArray:
    """20S-20N ocean-mean SST, all longitudes."""
    return cast(
        xr.DataArray, box_mean(sst, TROPICS_LAT, None, **kwargs).rename("tropical_mean")
    )


def tio(sst: xr.DataArray, **kwargs: str) -> xr.DataArray:
    """Tropical Indian Ocean SST on Yu et al. (2021)'s box, 20S-20N, 50-110E."""
    return cast(xr.DataArray, box_mean(sst, TIO_LAT, TIO_LON, **kwargs).rename("tio"))


def _in_long_run(flag: xr.DataArray, min_run: int, time_str: str) -> xr.DataArray:
    """Whether each value lies in a run of ``min_run`` or more True values."""
    if flag.sizes[time_str] < min_run:
        return xr.zeros_like(flag)
    # True at the last value of each window of min_run values that are all True.
    ends = flag.rolling({time_str: min_run}).sum() == min_run
    # A value is in such a window if one ends at it or at one of the next
    # min_run - 1 values.
    covered = ends
    for lag in range(1, min_run):
        covered = covered | ends.shift({time_str: -lag}, fill_value=False)
    return covered


def _require_monthly(times: xr.DataArray, why: str) -> None:
    """Raise a ValueError, ending with ``why``, unless ``times`` steps by one month."""
    year_month = (times.dt.year * 12 + times.dt.month).to_numpy()
    if not np.all(np.diff(year_month) == 1):
        raise ValueError(f"the time axis must be monthly with no gaps, since {why}")


def enso_phase(
    index: xr.DataArray,
    threshold: float = ENSO_THRESHOLD,
    *,
    inclusive: bool = True,
    decimals: int | None = 1,
    min_run: int = 5,
    time_str: str = TIME_STR,
) -> xr.DataArray:
    """The ENSO phase of each value of an index: El Niño, La Niña or neutral.

    1 marks El Niño, a value at or above ``threshold``; -1 marks La Niña, a
    value at or below ``-threshold``; 0 marks neutral; and NaN stays NaN.  By
    default the function also applies CPC's two further rules, described
    below, and so gives CPC's labels.  ``decimals=None, min_run=1`` labels
    each value alone and as given.

    CPC's operational definitions are inclusive: El Niño is "a positive RONI
    greater than or equal to +0.5ºC" and La Niña "a negative RONI less than
    or equal to -0.5ºC", in the "NOAA Operational Definitions for El Niño and
    La Niña" of CPC's weekly "ENSO: Recent Evolution, Current Status and
    Predictions" (issue of 19 October 2026; the file at
    https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/lanina/enso_evolution-status-fcsts-web.pdf
    is replaced each week).  ``inclusive=False`` gives the strict test some
    studies use instead.

    First, CPC applies the threshold to the index rounded to one decimal, as
    its tables show it: RONI for February to April 1951, 0.48 in CPC's
    two-decimal ``RONI.ascii.txt``, is shown as 0.5 and counted as El Niño.
    ``decimals`` rounds a value halfway between up, so 0.45 becomes 0.5 and
    -0.45 becomes -0.4.  CPC's tables round that way 80 of the 87 RONI values
    and 87 of the 89 ONI values in its two-decimal text files that end in 5.

    Second, CPC's RONI and ONI tables color a season El Niño or La Niña only
    "when the threshold is met for a minimum of five (5) consecutive
    overlapping seasons"
    (https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/).
    ``min_run`` applies this along ``time_str``.  A run shorter than
    ``min_run`` is neutral, including one at the end of the index's record
    that later values may lengthen.

    With its defaults, the function reproduces CPC's coloring of all 912
    seasons from DJF 1950 to NDJ 2025 in its RONI and ONI tables, as downloaded
    on 23 October 2026, from the one-decimal values the tables show.  From the
    two-decimal text files it reproduces every ONI season, and every RONI
    season but four, FMA to MJJ 1984: CPC shows FMA's -0.45 as -0.5, which
    joins the three seasons after it to the La Niña run before it.

    Parameters
    ----------
    index : xarray.DataArray
        An ENSO index such as RONI or ONI, in K.
    threshold : float
        The positive magnitude separating neutral from El Niño and La Niña, in
        the units of ``index``; for a standardized index, pass it in standard
        deviations.  With ``decimals``, it must be a multiple of
        ``10**-decimals``, the only values the rounded index takes; pass
        ``decimals=None`` for any other threshold.
    inclusive : bool
        Whether a value exactly at ``threshold`` or ``-threshold`` counts as
        El Niño or La Niña rather than neutral.
    decimals : int, optional
        The index is rounded to this many decimals, halves up, before it is
        compared with the threshold.  The default, 1, is CPC's; None leaves
        the index as given.  The rounding is done in float64, and a value
        less than 1e-4 of a step below a half, as a half stored in float32
        is, also rounds up.
    min_run : int
        The fewest consecutive values along ``time_str`` that must all pass
        the same threshold for any of them to count.  The default, 5, is
        CPC's, for an index of overlapping 3-month seasons, one per month; 1
        labels each value alone.  A NaN ends a run.  Where ``time_str`` has a
        coordinate of dates, they must step by one month, with no gaps;
        otherwise the values along it are taken to be consecutive months.
    time_str : str
        Name of the time dimension, used only when ``min_run`` exceeds 1.

    Returns
    -------
    xarray.DataArray
        The phase, as floats so that NaN can stand for a missing value, on
        ``index``'s coordinates, named ``"enso_phase"``, without ``index``'s
        attributes, since a phase has no units.

    Raises
    ------
    ValueError
        If ``threshold`` is not positive, or is off the grid of
        ``10**-decimals``; if ``min_run`` is less than 1; or if ``min_run``
        exceeds 1 and ``index`` has no ``time_str`` dimension, or has dates
        along it that do not step by one month.
    TypeError
        If ``decimals`` is not an integer or None, or ``min_run`` is not an
        integer.
    """
    if not threshold > 0:
        raise ValueError(f"threshold must be positive; got {threshold}")
    if not isinstance(min_run, numbers.Integral):
        raise TypeError(f"min_run must be an integer; got {min_run!r}")
    if min_run < 1:
        raise ValueError(f"min_run must be at least 1; got {min_run}")
    if min_run > 1:
        if time_str not in index.dims:
            raise ValueError(
                f"index has no {time_str!r} dimension to count runs along; pass "
                "min_run=1 to label each value alone, as for a yearly index, or "
                "time_str to name a monthly time dimension"
            )
        times = index.coords.get(time_str)
        if times is not None and (
            np.issubdtype(times.dtype, np.datetime64)
            or isinstance(index.indexes.get(time_str), xr.CFTimeIndex)
        ):
            _require_monthly(
                times,
                "min_run counts runs of consecutive months; pass min_run=1 to "
                "label each value alone",
            )
    value = index
    if decimals is not None:
        if not isinstance(decimals, numbers.Integral):
            raise TypeError(f"decimals must be an integer or None; got {decimals!r}")
        steps = threshold * 10**decimals
        if abs(steps - round(steps)) > _ROUND_TOL:
            raise ValueError(
                f"threshold {threshold} is not a multiple of 10**-{decimals}, so "
                f"rounding the index to {decimals} decimals would move it; pass "
                "decimals=None, or a threshold on that grid"
            )
        # In float64, so that the rounded value equals a float64 threshold on
        # the same grid exactly.
        scaled = index.astype(np.float64) * 10**decimals
        value = np.floor(scaled + 0.5 + _ROUND_TOL) / 10**decimals
    if inclusive:
        warm, cold = value >= threshold, value <= -threshold
    else:
        warm, cold = value > threshold, value < -threshold
    if min_run > 1:
        warm = _in_long_run(warm, min_run, time_str)
        cold = _in_long_run(cold, min_run, time_str)
    phase = xr.where(warm, 1.0, xr.where(cold, -1.0, 0.0)).where(index.notnull())
    return cast(xr.DataArray, phase.rename("enso_phase"))


def _box_anomalies(
    sst: xr.DataArray, base: tuple[int, int], time_str: str, **kwargs: str
) -> tuple[xr.DataArray, xr.DataArray]:
    """NINO3.4 and tropical-mean SST anomalies from the ``base`` climatology."""
    n34 = monthly_anom(nino34(sst, **kwargs), time_str, base)
    trop = monthly_anom(tropical_mean(sst, **kwargs), time_str, base)
    return n34.drop_vars("month"), trop.drop_vars("month")


def _running_means(
    nino34_anom: xr.DataArray, tropical_anom: xr.DataArray, time_str: str
) -> tuple[xr.DataArray, xr.DataArray]:
    """ONI and the running mean of the relative anomaly, centered 3-month.

    Raises
    ------
    ValueError
        If the time axis is not monthly or has a gap, since the running mean
        takes its neighbors by position.
    """
    _require_monthly(
        nino34_anom[time_str],
        "the 3-month running mean takes its neighbors by position",
    )
    oni = nino34_anom.rolling({time_str: 3}, center=True).mean()
    diff = (nino34_anom - tropical_anom).rolling({time_str: 3}, center=True).mean()
    return oni, diff


def _ratio(
    oni: xr.DataArray, diff: xr.DataArray, years: tuple[int, int], time_str: str
) -> xr.DataArray:
    """sigma(oni) / sigma(diff) by calendar month, over ``years`` inclusive."""
    year = oni[time_str].dt.year
    keep = (year >= years[0]) & (year <= years[1])
    # The axis is gap-free monthly here, so a count of 12 per year is coverage.
    if keep.sum().item() != 12 * (years[1] - years[0] + 1):
        raise ValueError(
            f"the data do not cover every month of sigma_years {years[0]}-{years[1]}"
        )
    sd_oni = oni.isel({time_str: keep}).groupby(f"{time_str}.month").std(ddof=1)
    sd_diff = diff.isel({time_str: keep}).groupby(f"{time_str}.month").std(ddof=1)
    return cast(xr.DataArray, (sd_oni / sd_diff).rename("roni_ratio"))


def roni_ratio(
    nino34_anom: xr.DataArray,
    tropical_anom: xr.DataArray,
    years: tuple[int, int] = RONI_SIGMA_YEARS,
    time_str: str = TIME_STR,
) -> xr.DataArray:
    """sigma(ONI) / sigma(ONI - tropical mean), one value per calendar month.

    ONI is the 3-month running mean of the NINO3.4 anomaly, centered on the
    month, and the tropical mean is smoothed the same way.  Each month's two
    standard deviations are taken over the running means centered on that
    calendar month within ``years``, inclusive.  With no NaN in either
    anomaly, both are taken over the same running means, so the n/(n-1) factor
    of ``ddof=1`` cancels in the ratio.

    Returns
    -------
    xarray.DataArray
        Ratio along a ``month`` dimension, 1 through 12.

    Raises
    ------
    ValueError
        If the time axis is not monthly or has a gap, or does not cover every
        month of ``years``.
    """
    oni, diff = _running_means(nino34_anom, tropical_anom, time_str)
    return _ratio(oni, diff, years, time_str)


def _ratio_by_month(ratio: xr.DataArray, times: xr.DataArray) -> xr.DataArray:
    """``ratio`` looked up for the calendar month of each time."""
    return cast(xr.DataArray, ratio.sel(month=times.dt.month).drop_vars("month"))


def roni(
    sst: xr.DataArray,
    base: tuple[int, int] = RONI_BASE,
    sigma_years: tuple[int, int] = RONI_SIGMA_YEARS,
    time_str: str = TIME_STR,
    **kwargs: str,
) -> xr.DataArray:
    """CPC's RONI: one value per overlapping 3-month season.

    Indexed by each season's center month, so the value at July is CPC's JJA.

    Parameters
    ----------
    sst : xarray.DataArray
        Monthly SST field on a gap-free monthly ``time_str`` axis, since the
        running mean takes neighbors by position.
    base, sigma_years : tuple of int
        Climatology years and the years of the rescaling ratio.
    time_str : str
        Name of the time dimension.
    kwargs
        Coordinate names passed to ``box_mean``.

    Raises
    ------
    ValueError
        If the time axis is not monthly or has a gap, or does not cover every
        month of ``base`` or of ``sigma_years``.
    """
    n34, trop = _box_anomalies(sst, base, time_str, **kwargs)
    oni, diff = _running_means(n34, trop, time_str)
    ratio = _ratio(oni, diff, sigma_years, time_str)
    return cast(
        xr.DataArray, (diff * _ratio_by_month(ratio, diff[time_str])).rename("roni")
    )


def relative_nino34(
    sst: xr.DataArray,
    base: tuple[int, int] = RONI_BASE,
    sigma_years: tuple[int, int] = RONI_SIGMA_YEARS,
    time_str: str = TIME_STR,
    *,
    scaled: bool = True,
    **kwargs: str,
) -> xr.DataArray:
    """The monthly relative NINO3.4 anomaly: RONI without the running mean.

    The NINO3.4 anomaly minus the tropical-mean anomaly, month by month.  With
    ``scaled``, each month is multiplied by RONI's ratio for its calendar
    month (``roni_ratio``, which is computed from running means, exactly as in
    ``roni``).  A seasonal mean of this series includes no month from outside
    the season, where a mean of ``roni`` would.

    The positional parameters match ``roni``'s; ``scaled`` is keyword-only.

    Raises
    ------
    ValueError
        If the time axis does not cover every month of ``base``, or, with
        ``scaled``, is not monthly, has a gap, or does not cover every month
        of ``sigma_years``.
    """
    n34, trop = _box_anomalies(sst, base, time_str, **kwargs)
    rel = n34 - trop
    if scaled:
        ratio = roni_ratio(n34, trop, sigma_years, time_str)
        rel = rel * _ratio_by_month(ratio, rel[time_str])
    return cast(xr.DataArray, rel.rename("relative_nino34"))


def relative_nino34_jjas(
    sst: xr.DataArray,
    base: tuple[int, int] = RONI_BASE,
    sigma_years: tuple[int, int] = RONI_SIGMA_YEARS,
    time_str: str = TIME_STR,
    *,
    scaled: bool = True,
    **kwargs: str,
) -> xr.DataArray:
    """June-September mean of ``relative_nino34``: RONI's summer counterpart.

    One value per calendar year of ``sst``, NaN for a year missing any of the
    four months (see ``dates.ann_subset_ts``).  The arguments are those of
    ``relative_nino34``.
    """
    rel = relative_nino34(sst, base, sigma_years, time_str, scaled=scaled, **kwargs)
    out = ann_subset_ts(rel, "jjas", dim_time=time_str)
    return cast(xr.DataArray, out.rename("relative_nino34_jjas"))
