#! /usr/bin/env python
"""Tropopause diagnostics.

Functions for computing the tropopause using various definitions: WMO
lapse-rate criterion, cold-point temperature minimum, maximum vertical
curvature of temperature, fixed temperature, and fixed height. Each
diagnostic has an internal implementation and a public wrapper that
handles groupby operations over additional dimensions.
"""

import numpy as np

from .interp import drop_nans_and_interp
from .names import LAT_STR, LEV_STR
from .nb_utils import apply_maybe_groupby


def tropo_wmo(
    temp,
    height,
    p_str=LEV_STR,
    threshold=-2e-3,
    max_pressure=600,
    do_interp=True,
    interp_vals=None,
):
    """Tropopause height using the WMO lapse-rate definition.

    The tropopause is identified as the lowest level above ``max_pressure``
    where the lapse rate (dT/dz) exceeds the threshold (i.e. is less
    negative than -2 K/km by default). Optionally interpolates to finer
    vertical resolution before computing.

    Parameters
    ----------
    temp : xarray.DataArray
        Temperature field on pressure levels.
    height : xarray.DataArray
        Geopotential height field on pressure levels.
    p_str : str, optional
        Name of the pressure/level dimension. Default: 'level'.
    threshold : float, optional
        Lapse-rate threshold (K/m). Default: -2e-3 (i.e. -2 K/km).
    max_pressure : float, optional
        Maximum pressure level (hPa) to consider; levels with higher
        pressure are excluded. Default: 600.
    do_interp : bool, optional
        Whether to interpolate to finer vertical resolution before
        computing. Default: True.
    interp_vals : array-like, optional
        Custom pressure values to interpolate to. If None and
        ``do_interp`` is True, uses ``np.arange(max_pressure, 20, -0.1)``.

    Returns
    -------
    xarray.DataArray
        Tropopause height, named 'tropopause_wmo'.
    """
    # Fill nans with zeros so that cubic interpolation works.
    temp_arr = temp.fillna(0.0)
    z_arr = height.fillna(0.0)
    # Interpolate if desired.
    if do_interp:
        if interp_vals is None:
            interp_vals = np.arange(max_pressure, 20, -0.1)
        dict_interp = {p_str: interp_vals, "method": "cubic"}
        t_interp = temp_arr.interp(**dict_interp)
        z_interp = z_arr.interp(**dict_interp)
    else:
        t_interp = temp_arr
        z_interp = z_arr
    # Compute the tropopause level and corresponding height.
    dtemp_dz = t_interp.diff(p_str) / z_interp.diff(p_str)
    dtemp_dz = dtemp_dz.where(z_interp[p_str] < max_pressure, drop=True)
    above_thresh = dtemp_dz[p_str].where(dtemp_dz > threshold)
    tropo_lev = above_thresh.idxmax(p_str)
    tropo_height = z_interp.sel(level=tropo_lev)
    return tropo_height.rename("tropopause_wmo")


# SAH note 2024-02-23: the logic for all of the functions below, which was
# written in 2019 or 2020 I believe, is no longer working.  I've defined
# the "tropo_wmo" function above which does work, but I haven't attempted
# to update the others.
def _tropo_wmo(
    temp,
    z,
    p_str=LEV_STR,
    lat_str=LAT_STR,
    threshold=-2e-3,
    max_pressure=600,
    interpolate=True,
):
    """WMO definition of tropopause: lapse rate < 2 K / km."""
    temp_arr, z_arr = drop_nans_and_interp(
        [temp, z], do_interp=interpolate, p_str=p_str
    )
    dtemp_dz = temp_arr.diff(p_str) / z_arr.diff(p_str)
    dtemp_dz = dtemp_dz.where(z_arr[p_str] < max_pressure, drop=True)
    above_thresh = dtemp_dz[p_str].where(dtemp_dz > threshold)
    p_tropo_ind = above_thresh.dropna(p_str, how="all").argmax(p_str)
    arr = temp_arr[{p_str: p_tropo_ind}].interp(**{lat_str: temp[lat_str]})
    arr.name = "tropopause_wmo"
    return arr


def tropopause_wmo(
    temp,
    z,
    p_str=LEV_STR,
    lat_str=LAT_STR,
    threshold=-2e-3,
    max_pressure=600,
    interpolate=True,
):
    """Tropopause temperature using the WMO lapse-rate definition.

    Wrapper around the internal ``_tropo_wmo`` that handles groupby
    operations over extra dimensions (e.g. time).

    Parameters
    ----------
    temp : xarray.DataArray
        Temperature field on pressure levels.
    z : xarray.DataArray
        Geopotential height field on pressure levels.
    p_str : str, optional
        Name of the pressure/level dimension. Default: 'level'.
    lat_str : str, optional
        Name of the latitude dimension. Default: 'lat'.
    threshold : float, optional
        Lapse-rate threshold (K/m). Default: -2e-3.
    max_pressure : float, optional
        Maximum pressure level (hPa) to consider. Default: 600.
    interpolate : bool, optional
        Whether to interpolate to finer vertical resolution. Default: True.

    Returns
    -------
    xarray.DataArray
        Tropopause temperature at each latitude, named 'tropopause_wmo'.
    """
    kwargs = dict(
        p_str=p_str,
        lat_str=lat_str,
        threshold=threshold,
        max_pressure=max_pressure,
        interpolate=interpolate,
    )
    return apply_maybe_groupby(_tropo_wmo, [p_str, lat_str], [temp, z], kwargs=kwargs)


def _tropo_cold_point(temp, interpolate=True, p_str=LEV_STR, lat_str=LAT_STR):
    """Tropopause defined as the coldest point in the column."""
    (temp_arr,) = drop_nans_and_interp([temp], do_interp=interpolate, p_str=p_str)
    cold_point = temp_arr.min(p_str)
    cold_point_lev = temp_arr[p_str][temp_arr.argmin(p_str)]
    cold_point[p_str] = cold_point_lev
    return cold_point.interp(**{lat_str: temp[lat_str]})


def tropopause_cold_point(temp, interpolate=True, p_str=LEV_STR, lat_str=LAT_STR):
    """Tropopause defined as the coldest point in the column.

    Identifies the pressure level with the minimum temperature in each
    column. Wrapper that handles groupby over extra dimensions.

    Parameters
    ----------
    temp : xarray.DataArray
        Temperature field on pressure levels.
    interpolate : bool, optional
        Whether to interpolate to finer vertical resolution. Default: True.
    p_str : str, optional
        Name of the pressure/level dimension. Default: 'level'.
    lat_str : str, optional
        Name of the latitude dimension. Default: 'lat'.

    Returns
    -------
    xarray.DataArray
        Cold-point tropopause temperature at each latitude.
    """
    kwargs = dict(interpolate=interpolate, p_str=p_str, lat_str=lat_str)
    return apply_maybe_groupby(
        _tropo_cold_point, [p_str, lat_str], [temp], kwargs=kwargs
    )


def _tropo_max_vert_curv(
    temp, z, interpolate=True, max_pressure=500, p_str=LEV_STR, lat_str=LAT_STR
):
    """Tropopause defined as where d^2T/dz^2 maximizes."""
    temp_arr, z_arr = drop_nans_and_interp(
        [temp, z], do_interp=interpolate, p_str=p_str
    )
    temp_arr["z"] = z_arr
    d2temp_dz2 = temp_arr.differentiate("z").differentiate("z")
    d2temp_dz2 = d2temp_dz2.where(z_arr[p_str] < max_pressure, drop=True)
    d2temp_dz2_max_ind = d2temp_dz2.argmax(p_str)
    return temp_arr[{p_str: d2temp_dz2_max_ind}].interp(**{lat_str: temp[lat_str]})


def tropopause_max_vert_curv(
    temp, z, interpolate=True, max_pressure=500, p_str=LEV_STR, lat_str=LAT_STR
):
    """Tropopause defined as the level of maximum vertical curvature of temperature.

    Identifies the pressure level where :math:`d^2T/dz^2` is maximized,
    above ``max_pressure``. Wrapper that handles groupby over extra
    dimensions.

    Parameters
    ----------
    temp : xarray.DataArray
        Temperature field on pressure levels.
    z : xarray.DataArray
        Geopotential height field on pressure levels.
    interpolate : bool, optional
        Whether to interpolate to finer vertical resolution. Default: True.
    max_pressure : float, optional
        Maximum pressure level (hPa) to consider. Default: 500.
    p_str : str, optional
        Name of the pressure/level dimension. Default: 'level'.
    lat_str : str, optional
        Name of the latitude dimension. Default: 'lat'.

    Returns
    -------
    xarray.DataArray
        Tropopause temperature at each latitude.
    """
    kwargs = dict(
        interpolate=interpolate, p_str=p_str, lat_str=lat_str, max_pressure=max_pressure
    )
    return apply_maybe_groupby(
        _tropo_max_vert_curv, [p_str, lat_str], [temp, z], kwargs=kwargs
    )


# NOTE: this one still needs work.  Sometimes oscillates between near-surface
# and upper troposphere if specified tropopause temperature is sufficiently
# warm.  Conversely, if specified temperature is too cold, some columns never
# come close to it, yet this will take the level nearest to it regardless.  So
# need to introduce some threshold, and apply it from the mid-troposphere up,
# akin to how the WMO version is done.
def _tropo_fixed_temp(
    temp, temp_tropo=200.0, interpolate=True, p_str=LEV_STR, lat_str=LAT_STR
):
    """Tropopause defined as a fixed temperature."""
    (temp_arr,) = drop_nans_and_interp([temp], do_interp=interpolate, p_str=p_str)
    temp_closest_ind = np.abs(temp_arr - temp_tropo).argmin(p_str)
    return temp_arr[{p_str: temp_closest_ind}].interp(**{lat_str: temp[lat_str]})


def tropopause_fixed_temp(
    temp, temp_tropo=200.0, interpolate=True, p_str=LEV_STR, lat_str=LAT_STR
):
    """Tropopause defined as the level closest to a fixed temperature.

    Finds the pressure level where the temperature is closest to the
    specified tropopause temperature. Wrapper that handles groupby over
    extra dimensions.

    Parameters
    ----------
    temp : xarray.DataArray
        Temperature field on pressure levels.
    temp_tropo : float, optional
        Target tropopause temperature (K). Default: 200.0.
    interpolate : bool, optional
        Whether to interpolate to finer vertical resolution. Default: True.
    p_str : str, optional
        Name of the pressure/level dimension. Default: 'level'.
    lat_str : str, optional
        Name of the latitude dimension. Default: 'lat'.

    Returns
    -------
    xarray.DataArray
        Temperature at the identified tropopause level.
    """
    kwargs = dict(
        interpolate=interpolate, p_str=p_str, lat_str=lat_str, temp_tropo=temp_tropo
    )
    return apply_maybe_groupby(
        _tropo_fixed_temp, [p_str, lat_str], [temp], kwargs=kwargs
    )


def _tropo_fixed_height(
    temp, z, height_tropo=1e4, interpolate=True, p_str=LEV_STR, lat_str=LAT_STR
):
    """Tropopause defined as a fixed height."""
    temp_arr, z_arr = drop_nans_and_interp(
        [temp, z], do_interp=interpolate, p_str=p_str
    )
    z_closest_ind = np.abs(z_arr - height_tropo).argmin(p_str)
    return temp_arr[{p_str: z_closest_ind}].interp(**{lat_str: temp[lat_str]})


def tropopause_fixed_height(
    temp, z, height_tropo=1e4, interpolate=True, p_str=LEV_STR, lat_str=LAT_STR
):
    """Tropopause defined as the level closest to a fixed height.

    Finds the pressure level where the geopotential height is closest to
    the specified tropopause height. Wrapper that handles groupby over
    extra dimensions.

    Parameters
    ----------
    temp : xarray.DataArray
        Temperature field on pressure levels.
    z : xarray.DataArray
        Geopotential height field on pressure levels.
    height_tropo : float, optional
        Target tropopause height (m). Default: 1e4 (10 km).
    interpolate : bool, optional
        Whether to interpolate to finer vertical resolution. Default: True.
    p_str : str, optional
        Name of the pressure/level dimension. Default: 'level'.
    lat_str : str, optional
        Name of the latitude dimension. Default: 'lat'.

    Returns
    -------
    xarray.DataArray
        Temperature at the identified tropopause level.
    """
    kwargs = dict(
        interpolate=interpolate, p_str=p_str, lat_str=lat_str, height_tropo=height_tropo
    )
    return apply_maybe_groupby(
        _tropo_fixed_height, [p_str, lat_str], [temp, z], kwargs=kwargs
    )


if __name__ == "__main__":
    pass
