#! /usr/bin/env python
"""Derivatives, integrals, and averages."""

from indiff.diff import BwdDiff, CenDiff
import numpy as np
import xarray as xr

from .constants import GRAV_EARTH, RAD_EARTH
from .names import LAT_STR, LEV_STR, Z_STR
from .nb_utils import cosdeg, sindeg, to_pascal


# Derivatives.
def lat_deriv(arr, lats=None, lat_str=LAT_STR):
    if lats is None:
        lats = arr[lat_str]
    return (CenDiff(arr, lat_str, fill_edge='both').diff() /
            CenDiff(np.deg2rad(lats), lat_str, fill_edge='both').diff())


def z_deriv(arr, z, z_str=Z_STR):
    return (CenDiff(arr, z_str, fill_edge='both').diff() /
            CenDiff(z, z_str, fill_edge='both').diff())


def bwd_deriv(arr):
    return BwdDiff(arr, LAT_STR).diff() / BwdDiff(arr[LAT_STR], LAT_STR).diff()


# Vertical integrals and averages.
def integrate(arr, ddim, dim="plev"):
    """Integrate along the given dimension."""
    return (arr * ddim).sum(dim=dim)


def int_dp_g(arr, dp, dim="plev", grav=GRAV_EARTH):
    """Mass weighted integral."""
    return integrate(arr, to_pascal(dp, is_dp=True), dim=dim) / grav


def subtract_col_avg(arr, dp, dim="plev", grav=GRAV_EARTH):
    """Impoze zero column integral by subtracting column average at each level.

    Used e.g. for computing the zonally integrated mass flux.  In the time-mean
    and neglecting tendencies in column mass, the column integrated meridional
    mass transport should be zero at each latitude; otherwise there would be a
    build up of mass on one side.

    """
    col_avg = (int_dp_g(arr, dp, dim=dim, grav=grav) /
               int_dp_g(1.0, dp, dim=dim, grav=grav))
    return arr - col_avg


# Meridional integrals and averages.
def merid_integral_point_data(arr, min_lat=-90, max_lat=90, lat_str=LAT_STR):
    """Area-weighted meridional integral for data defined at single lats.

    As opposed to e.g. gridded climate model output, wherein the quantity at
    the given latitude corresponds to the value of a cell of finite area.  In
    that case, a discrete form of the summing operation should be used and is
    implemented in the function ``merid_integral_grid_data``.

    """
    lat = arr[lat_str]
    masked = arr.where((lat > min_lat) & (lat < max_lat), drop=True)
    dlat = lat.diff(lat_str)
    if not np.isclose(dlat.min(), dlat.max()):
        raise ValueError("Uniform latitude spacing required; not uniform.")
    return (masked*cosdeg(lat)*np.deg2rad(dlat)).sum(lat_str)


def merid_avg_point_data(arr, min_lat=-90, max_lat=90, lat_str=LAT_STR):
    """Area-weighted meridional average for data defined at single lats.

    As opposed to e.g. gridded climate model output, wherein the quantity at
    the given latitude corresponds to the value of a cell of finite area.  In
    that case, a discrete form of the summing operation should be used and is
    implemented in the function ``merid_average_grid_data``.

    """
    return (merid_integral_point_data(arr, min_lat, max_lat, lat_str) /
            merid_integral_point_data(xr.ones_like(arr), min_lat,
                                      max_lat, lat_str))


def merid_integral_grid_data(arr, min_lat=-90, max_lat=90, lat_str=LAT_STR,
                             dlat_var_tol=0.01, radius=RAD_EARTH):
    """Area-weighted meridional integral for data on finite grid cells.

    As opposed to data defined at individual latitudes, wherein the quantity at
    the given latitude corresponds to exactly that latitude only, not to a cell
    of finite area surrounding that latitude.  In that case, the function
    ``merid_integral_point_data`` should be used.

    """
    lat = arr[lat_str]
    arr_masked = arr.where((lat > min_lat) & (lat < max_lat),
                           drop=True)

    dlat = lat.diff(lat_str)
    dlat_mean = dlat.mean(lat_str)
    dlat_frac_var = (dlat - dlat_mean) / dlat_mean
    if np.any(np.abs(dlat_frac_var) > dlat_var_tol):
        raise ValueError("Uniform latitude spacing required; given "
                         "latitudes not sufficiently uniform.")

    # Given uniform latitude spacing, find bounding latitudes.
    assert lat[0] < lat[1]
    lat_above = lat + 0.5*dlat_mean
    if lat_above[-1] > 90:
        lat_above[-1] = 90.
    lat_below = lat - 0.5*dlat_mean
    if lat_below[0] < -90:
        lat_below[0] = -90.

    sinlat_diff = sindeg(lat_above.values) - sindeg(lat_below.values)
    area = xr.ones_like(lat)*2.*np.pi*radius**2*sinlat_diff
    area_masked = area.where((lat > min_lat) & (lat < max_lat),
                             drop=True)
    return (arr_masked*area_masked).sum(lat_str)


def merid_avg_grid_data(arr, min_lat=-90, max_lat=90, lat_str=LAT_STR):
    """Area-weighted meridional average for data on finite grid cells.

    As opposed to data defined at individual latitudes, wherein the quantity at
    the given latitude corresponds to exactly that latitude only, not to a cell
    of finite area surrounding that latitude.  In that case, the function
    ``merid_avg_point_data`` should be used.

    """
    return (merid_integral_point_data(arr, min_lat, max_lat, lat_str) /
            merid_integral_point_data(xr.ones_like(arr), min_lat,
                                      max_lat, lat_str))


# Pressure spacing and averages.
def dp_from_p_half(p_half, pressure):
    """Pressure thickness of vertical levels given interface pressures."""
    dp_vals = p_half.values[1:] - p_half.values[:-1]
    return xr.ones_like(pressure)*dp_vals


def dlogp_from_p_half(p_half, pressure):
    """Pressure thickness of vertical levels given interface pressures."""
    dlogp_vals = np.log(p_half.values[1:]/p_half.values[:-1])
    return xr.ones_like(pressure)*dlogp_vals


def _flip_dim(arr, dim):
    return arr.isel(**{dim: slice(None, None, -1)})


def avg_p_weighted(arr, p_half, pressure, p_str=LEV_STR):
    """Pressure-weighted vertical average."""
    dp = np.abs(dp_from_p_half(p_half, pressure))
    if p_half[0] > p_half[1]:
        arr_out = _flip_dim(arr, p_str)
        dp_out = _flip_dim(dp, p_str)
    else:
        arr_out = arr
        dp_out = dp
    return (arr_out*dp_out).cumsum(p_str) / dp_out.cumsum(p_str)


def avg_logp_weighted(arr, p_half, pressure, p_str=LEV_STR):
    """Log-pressure-weighted vertical average."""
    dlogp = dlogp_from_p_half(p_half, pressure)
    return (arr*dlogp).cumsum(p_str) / dlogp.cumsum(p_str)


def col_extrema(arr, p_str=LEV_STR):
    """Locations and values of local extrema within each column."""
    darr_dp = z_deriv(arr, arr[p_str], z_str=p_str)
    sign_change = np.sign(darr_dp).diff(p_str)
    return arr.where(sign_change)


if __name__ == '__main__':
    pass
