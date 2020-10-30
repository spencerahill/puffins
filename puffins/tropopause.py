#! /usr/bin/env python
"""Calculations of the tropopause."""

import numpy as np
import xarray as xr

from .constants import GRAV_EARTH, R_D
from .names import LAT_STR, LEV_STR
from .nb_utils import apply_maybe_groupby
from .interp import drop_nans_and_interp


def z_from_hypso(temp, p_sfc=1000., p_top=0.1, p_str=LEV_STR,
                 r_d=R_D, grav=GRAV_EARTH):
    """Height computed from hypsometric equation."""
    # Ensure all pressures have same horizontal dimensions as temperature.
    non_vert_coords = xr.ones_like(
        temp.isel(**{p_str: 0})).drop(p_str)
    if np.isscalar(p_sfc):
        p_sfc_val = p_sfc
        p_sfc = p_sfc_val*non_vert_coords
        p_sfc[p_str] = p_sfc_val
    p_top_val = p_top
    p_top = xr.zeros_like(p_sfc)
    p_top[p_str] = p_top_val
    pressure = (non_vert_coords*temp[p_str]).transpose(*temp.dims)

    # Compute half-level pressure values as averages of full levels.
    p_half_inner = 0.5*(
        pressure.isel(**{p_str: slice(1, None)}).values +
        pressure.isel(**{p_str: slice(None, -1)}).values
    )
    p_axis_num = temp.get_axis_num(p_str)
    p_half = np.concatenate(
        [np.expand_dims(p_top, p_axis_num), p_half_inner,
         np.expand_dims(p_sfc, p_axis_num)], axis=p_axis_num
    )

    # Convert from hPa to Pa if necessary.
    if p_half.max() < 2000:
        log_p_half = np.log(p_half*1e2)
    else:
        log_p_half = np.log(p_half)
    dlog_p_half_values = np.diff(log_p_half, axis=p_axis_num)
    dlog_p_half = xr.ones_like(pressure)*dlog_p_half_values
    temp_dlog_p = temp*dlog_p_half

    # Integrate vertically.
    height = r_d / grav*temp_dlog_p.isel(
        **{p_str: slice(-1, None, -1)}).cumsum(p_str)
    height = height.isel(**{p_str: slice(-1, None, -1)})

    # Replace 'inf' values at TOA with NaNs and mask where
    # input temperature array is masked.
    return xr.where(np.isfinite(height) & temp.notnull(),
                    height, np.nan)


# TODO: define a decorator that handles this maybe-groupby logic,
# so that it's unnecessary to define the two separate functions
# for each case.  It seems like pretty simple boilerplate.

def _tropo_wmo(temp, z, p_str=LEV_STR, lat_str=LAT_STR,
               threshold=-2e-3, max_pressure=600,
               interpolate=True):
    """WMO definition of tropopause: lapse rate < 2 K / km."""
    temp_arr, z_arr = drop_nans_and_interp(
        [temp, z],  do_interp=interpolate, p_str=p_str)
    dtemp_dz = temp_arr.diff(p_str) / z_arr.diff(p_str)
    dtemp_dz = dtemp_dz.where(z_arr[p_str] < max_pressure, drop=True)
    above_thresh = dtemp_dz[p_str].where(dtemp_dz > threshold)
    p_tropo_ind = above_thresh.dropna(p_str, how='all').argmax(p_str)
    arr = temp_arr[{p_str: p_tropo_ind}].interp(**{lat_str: temp[lat_str]})
    arr.name = 'tropopause_wmo'
    return arr


def tropopause_wmo(temp, z, p_str=LEV_STR, lat_str=LAT_STR,
                   threshold=-2e-3, max_pressure=600,
                   interpolate=True):
    """WMO definition of tropopause: lapse rate < 2 K / km."""
    kwargs = dict(p_str=p_str, lat_str=lat_str, threshold=threshold,
                  max_pressure=max_pressure, interpolate=interpolate)
    return apply_maybe_groupby(_tropo_wmo, [p_str, lat_str], [temp, z],
                               kwargs=kwargs)


def _tropo_cold_point(temp, interpolate=True,
                      p_str=LEV_STR, lat_str=LAT_STR):
    """Tropopause defined as the coldest point in the column."""
    temp_arr, = drop_nans_and_interp(
        [temp],  do_interp=interpolate, p_str=p_str)
    cold_point = temp_arr.min(p_str)
    cold_point_lev = temp_arr[p_str][temp_arr.argmin(p_str)]
    cold_point[p_str] = cold_point_lev
    return cold_point.interp(**{lat_str: temp[lat_str]})


def tropopause_cold_point(temp, interpolate=True,
                          p_str=LEV_STR, lat_str=LAT_STR):
    """Tropopause defined as the coldest point in the column."""
    kwargs = dict(interpolate=interpolate, p_str=p_str,
                  lat_str=lat_str)
    return apply_maybe_groupby(_tropo_cold_point, [p_str, lat_str], [temp],
                               kwargs=kwargs)


def _tropo_max_vert_curv(temp, z, interpolate=True, max_pressure=500,
                         p_str=LEV_STR, lat_str=LAT_STR):
    """Tropopause defined as where d^2T/dz^2 maximizes."""
    temp_arr, z_arr = drop_nans_and_interp([temp, z], do_interp=interpolate,
                                           p_str=p_str)
    temp_arr["z"] = z_arr
    d2temp_dz2 = temp_arr.differentiate("z").differentiate("z")
    d2temp_dz2 = d2temp_dz2.where(z_arr[p_str] < max_pressure, drop=True)
    d2temp_dz2_max_ind = d2temp_dz2.argmax(p_str)
    return temp_arr[{p_str: d2temp_dz2_max_ind}].interp(**{lat_str:
                                                           temp[lat_str]})


def tropopause_max_vert_curv(temp, z, interpolate=True, max_pressure=500,
                             p_str=LEV_STR, lat_str=LAT_STR):
    """Tropopause defined as where d^2T/dz^2 maximizes."""
    kwargs = dict(interpolate=interpolate, p_str=p_str,
                  lat_str=lat_str, max_pressure=max_pressure)
    return apply_maybe_groupby(_tropo_max_vert_curv, [p_str, lat_str],
                               [temp, z], kwargs=kwargs)


# NOTE: this one still needs work.  Sometimes oscillates between near-surface
# and upper troposphere if specified tropopause temperature is sufficiently
# warm.  Conversely, if specified temperature is too cold, some columns never
# come close to it, yet this will take the level nearest to it regardless.  So
# need to introduce some threshold, and apply it from the mid-troposphere up,
# akin to how the WMO version is done.
def _tropo_fixed_temp(temp, temp_tropo=200., interpolate=True,
                      p_str=LEV_STR, lat_str=LAT_STR):
    """Tropopause defined as a fixed temperature."""
    temp_arr, = drop_nans_and_interp(
        [temp],  do_interp=interpolate, p_str=p_str)
    temp_closest_ind = np.abs(temp_arr - temp_tropo).argmin(p_str)
    return temp_arr[{p_str: temp_closest_ind}].interp(**{lat_str:
                                                         temp[lat_str]})


def tropopause_fixed_temp(temp, temp_tropo=200., interpolate=True,
                          p_str=LEV_STR, lat_str=LAT_STR):
    """Tropopause defined as a fixed temperature."""
    kwargs = dict(interpolate=interpolate, p_str=p_str,
                  lat_str=lat_str, temp_tropo=temp_tropo)
    return apply_maybe_groupby(_tropo_fixed_temp, [p_str, lat_str],
                               [temp], kwargs=kwargs)


def _tropo_fixed_height(temp, z, height_tropo=1e4, interpolate=True,
                        p_str=LEV_STR, lat_str=LAT_STR):
    """Tropopause defined as a fixed height."""
    temp_arr, z_arr = drop_nans_and_interp(
        [temp, z],  do_interp=interpolate, p_str=p_str)
    z_closest_ind = np.abs(z_arr - height_tropo).argmin(p_str)
    return temp_arr[{p_str: z_closest_ind}].interp(**{lat_str: temp[lat_str]})


def tropopause_fixed_height(temp, z, height_tropo=1e4, interpolate=True,
                            p_str=LEV_STR, lat_str=LAT_STR):
    """Tropopause defined as a fixed height."""
    kwargs = dict(interpolate=interpolate, p_str=p_str,
                  lat_str=lat_str, height_tropo=height_tropo)
    return apply_maybe_groupby(_tropo_fixed_height, [p_str, lat_str],
                               [temp, z], kwargs=kwargs)


if __name__ == '__main__':
    pass
