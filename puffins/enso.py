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

``relative_nino34`` is the same quantity month by month, without the running
mean, and ``season_mean`` averages any monthly series over a season, so a
June-September mean of the monthly relative index is
``season_mean(relative_nino34(sst), JJAS)``.

The module also gives the plain NINO3.4 box mean and the tropical Indian Ocean
(TIO) box of Yu et al. (2021, Geophys. Res. Lett. 48, e2021GL092873), 20S-20N
and 50-110E, as their Figure 3a caption states it.

All box means are cos(latitude)-weighted over the gridpoint centers inside
inclusive bounds, with land (NaN in an SST product) dropped from the weights as
well as from the sum.  Longitudes must be on the same 0-360 or -180-180
convention as the bounds given; a box that selects no gridpoint raises rather
than returning an all-NaN series.

These functions moved here on 2026-09-30 from the lps-enso-grl project's
``enso_indices.py``, so that the projects using them share one definition.
"""

from typing import cast

import numpy as np
import xarray as xr

from .names import LAT_STR, LON_STR, TIME_STR, YEAR_STR

NINO34_LAT: tuple[float, float] = (-5.0, 5.0)
NINO34_LON: tuple[float, float] = (190.0, 240.0)
TROPICS_LAT: tuple[float, float] = (-20.0, 20.0)
TIO_LAT: tuple[float, float] = (-20.0, 20.0)
TIO_LON: tuple[float, float] = (50.0, 110.0)

RONI_BASE: tuple[int, int] = (1991, 2020)
"""The years of the monthly climatology anomalies are taken from."""

RONI_SIGMA_YEARS: tuple[int, int] = (1950, 2020)
"""The years over which RONI's rescaling ratio is computed."""

JJAS: tuple[int, ...] = (6, 7, 8, 9)
DJF: tuple[int, ...] = (12, 1, 2)


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
    lat_bounds, lon_bounds : tuple of float
        Inclusive bounds, in the field's own coordinate convention.  With
        ``lon_bounds`` None the mean spans every longitude.
    lat_str, lon_str : str
        Names of the latitude and longitude dimensions.

    Returns
    -------
    xarray.DataArray
        The box mean, with the latitude and longitude dimensions removed.

    Raises
    ------
    ValueError
        If no gridpoint center falls inside the box.
    """
    lat = sst[lat_str]
    sub = sst.isel({lat_str: (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])})
    if lon_bounds is not None:
        lon = sub[lon_str]
        sub = sub.isel({lon_str: (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])})
    if sub.sizes[lat_str] == 0 or sub.sizes[lon_str] == 0:
        raise ValueError(
            f"no gridpoint inside lat {lat_bounds}, lon {lon_bounds}; check the "
            "longitude convention against the bounds"
        )
    # Float64 weights make the products and sums float64.  SST products are
    # stored as float32, whose summation order otherwise shows up at the 1e-6
    # relative level.
    coslat = cast(xr.DataArray, np.cos(np.deg2rad(sub[lat_str].astype(np.float64))))
    weights = coslat.broadcast_like(sub).where(sub.notnull())
    dims = (lat_str, lon_str)
    return cast(xr.DataArray, (sub * weights).sum(dims) / weights.sum(dims))


def nino34(sst: xr.DataArray, **kwargs: str) -> xr.DataArray:
    """NINO3.4 SST: the box mean over 5S-5N, 190-240E.

    ``kwargs`` pass coordinate names through to ``box_mean``.
    """
    return box_mean(sst, NINO34_LAT, NINO34_LON, **kwargs).rename("nino34")


def tropical_mean(sst: xr.DataArray, **kwargs: str) -> xr.DataArray:
    """20S-20N ocean-mean SST, all longitudes."""
    return box_mean(sst, TROPICS_LAT, None, **kwargs).rename("tropical_mean")


def tio(sst: xr.DataArray, **kwargs: str) -> xr.DataArray:
    """Tropical Indian Ocean SST on Yu et al. (2021)'s box, 20S-20N, 50-110E."""
    return box_mean(sst, TIO_LAT, TIO_LON, **kwargs).rename("tio")


def monthly_anomaly(
    series: xr.DataArray,
    base: tuple[int, int] = RONI_BASE,
    time_str: str = TIME_STR,
) -> xr.DataArray:
    """Departure from the ``base`` years' mean for the same calendar month.

    Parameters
    ----------
    series : xarray.DataArray
        Monthly values along a datetime ``time_str`` dimension.
    base : tuple of int
        First and last years of the climatology, inclusive.

    Raises
    ------
    ValueError
        If the base years do not cover all twelve calendar months.
    """
    years = series[time_str].dt.year
    in_base = series.isel({time_str: (years >= base[0]) & (years <= base[1])})
    if np.unique(in_base[time_str].dt.month).size != 12:
        raise ValueError(f"base period {base} does not cover all twelve months")
    clim = in_base.groupby(f"{time_str}.month").mean(time_str)
    month = series[time_str].dt.month
    return series - clim.sel(month=month).drop_vars("month")


def roni_ratio(
    nino34_anom: xr.DataArray,
    tropical_anom: xr.DataArray,
    years: tuple[int, int] = RONI_SIGMA_YEARS,
    time_str: str = TIME_STR,
) -> xr.DataArray:
    """sigma(ONI) / sigma(ONI - tropical mean), one value per calendar month.

    ONI is the 3-month running mean of the NINO3.4 anomaly, centered on the
    month, and the tropical mean is smoothed the same way.  Each month's two
    standard deviations (``ddof=1``) are taken over the running means centered
    on that calendar month within ``years``, inclusive.

    Returns
    -------
    xarray.DataArray
        Ratio along a ``month`` dimension, 1 through 12.
    """
    oni = nino34_anom.rolling({time_str: 3}, center=True).mean()
    diff = oni - tropical_anom.rolling({time_str: 3}, center=True).mean()
    year = oni[time_str].dt.year
    keep = (year >= years[0]) & (year <= years[1])
    sd_oni = oni.isel({time_str: keep}).groupby(f"{time_str}.month").std(ddof=1)
    sd_diff = diff.isel({time_str: keep}).groupby(f"{time_str}.month").std(ddof=1)
    return (sd_oni / sd_diff).rename("roni_ratio")


def _ratio_by_month(ratio: xr.DataArray, times: xr.DataArray) -> xr.DataArray:
    """``ratio`` looked up for the calendar month of each time."""
    return ratio.sel(month=times.dt.month).drop_vars("month")


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
        Monthly SST field on a contiguous monthly ``time_str`` axis, since the
        running mean takes neighbors by position.
    base, sigma_years : tuple of int
        Climatology years and the years of the rescaling ratio.
    kwargs
        Coordinate names passed to ``box_mean``.
    """
    n34 = monthly_anomaly(nino34(sst, **kwargs), base, time_str)
    trop = monthly_anomaly(tropical_mean(sst, **kwargs), base, time_str)
    ratio = roni_ratio(n34, trop, sigma_years, time_str)
    diff = (
        n34.rolling({time_str: 3}, center=True).mean()
        - trop.rolling({time_str: 3}, center=True).mean()
    )
    return (diff * _ratio_by_month(ratio, diff[time_str])).rename("roni")


def relative_nino34(
    sst: xr.DataArray,
    scaled: bool = True,
    base: tuple[int, int] = RONI_BASE,
    sigma_years: tuple[int, int] = RONI_SIGMA_YEARS,
    time_str: str = TIME_STR,
    **kwargs: str,
) -> xr.DataArray:
    """The monthly relative NINO3.4 anomaly: RONI without the running mean.

    The NINO3.4 anomaly minus the tropical-mean anomaly, month by month.  With
    ``scaled``, each month is multiplied by RONI's ratio for its calendar
    month (``roni_ratio``, which is computed from running means, exactly as in
    ``roni``).  A seasonal mean of this series carries no month from outside
    the season, where a mean of ``roni`` would.
    """
    n34 = monthly_anomaly(nino34(sst, **kwargs), base, time_str)
    trop = monthly_anomaly(tropical_mean(sst, **kwargs), base, time_str)
    rel = n34 - trop
    if scaled:
        ratio = roni_ratio(n34, trop, sigma_years, time_str)
        rel = rel * _ratio_by_month(ratio, rel[time_str])
    return rel.rename("relative_nino34")


def season_mean(
    monthly: xr.DataArray,
    months: tuple[int, ...],
    time_str: str = TIME_STR,
    year_str: str = YEAR_STR,
) -> xr.DataArray:
    """Mean over ``months`` by year, for seasons with every month finite.

    A season that crosses the new year belongs to the year of its last
    months: with ``months=(12, 1, 2)`` the 1998 value is December 1997 through
    February 1998.  Months listed before the point where the sequence wraps
    are the ones moved forward a year.

    Returns
    -------
    xarray.DataArray
        Season means along a ``year_str`` dimension.  A season missing any
        month, or holding a NaN month, is left out.
    """
    month = monthly[time_str].dt.month
    sub = monthly.isel({time_str: month.isin(months)})
    wrap = next((i for i in range(1, len(months)) if months[i] < months[i - 1]), None)
    sub_month = sub[time_str].dt.month
    season_year = sub[time_str].dt.year
    if wrap is not None:
        season_year = season_year + sub_month.isin(months[:wrap]).astype(int)
    season_year = season_year.rename(year_str)
    mean = sub.groupby(season_year).mean(time_str)
    count = sub.notnull().groupby(season_year).sum(time_str)
    return mean.where(count == len(months), drop=True)


def relative_nino34_jjas(sst: xr.DataArray, scaled: bool = True) -> xr.DataArray:
    """June-September mean of ``relative_nino34``: RONI's summer counterpart."""
    return season_mean(relative_nino34(sst, scaled=scaled), JJAS).rename(
        "relative_nino34_jjas"
    )
