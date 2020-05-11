#! /usr/bin/env python
"""Derivatives, integrals, and averages."""

from indiff.diff import BwdDiff, CenDiff
import numpy as np
import xarray as xr

from .constants import GRAV_EARTH, P0, RAD_EARTH
from .names import LAT_STR, LEV_STR, PHALF_STR, Z_STR
from .nb_utils import cosdeg, sindeg, to_pascal


# Derivatives.
def lat_deriv(arr, lats=None, lat_str=LAT_STR):
    """Meridional derivative approximated by centered differencing."""
    if lats is None:
        lats = arr[lat_str]
    return (
        CenDiff(arr, lat_str, fill_edge='both').diff() /
        CenDiff(np.deg2rad(lats), lat_str, fill_edge='both').diff()
    ).transpose(*arr.dims)


def z_deriv(arr, z=None, z_str=Z_STR):
    """Vertical derivative approximated by centered differencing."""
    if z is None:
        z = arr[z_str]
    return (
        CenDiff(arr, z_str, fill_edge='both').diff() /
        CenDiff(z, z_str, fill_edge='both').diff()
    ).transpose(*arr.dims)


def bwd_deriv(arr, dim):
    """Derivative approximated by one-sided backwards differencing."""
    return BwdDiff(arr, dim).diff() / BwdDiff(arr[dim], dim).diff()


def flux_div(arr_merid_flux, arr_vert_flux, vert_coord=None, vert_str=LEV_STR,
             lat_str=LAT_STR, radius=RAD_EARTH):
    """Horizontal plus vertical flux divergence of a given field."""
    if vert_coord is None:
        vert_coord = arr_vert_flux[vert_str]
    merid_flux_div = (lat_deriv(arr_merid_flux, lat_str=lat_str) /
                               (radius*cosdeg(arr_merid_flux[lat_str])))
    vert_flux_div = z_deriv(arr_vert_flux, vert_coord, z_str=vert_str)
    return merid_flux_div + vert_flux_div


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
def merid_integral_point_data(arr, min_lat=-90, max_lat=90, unif_thresh=0.01,
                              lat_str=LAT_STR):
    """Area-weighted meridional integral for data defined at single lats.

    As opposed to e.g. gridded climate model output, wherein the quantity at
    the given latitude corresponds to the value of a cell of finite area.  In
    that case, a discrete form of the summing operation should be used and is
    implemented in the function ``merid_integral_grid_data``.

    """
    lat = arr[lat_str]
    masked = arr.where((lat > min_lat) & (lat < max_lat), drop=True)
    dlat = lat.diff(lat_str)
    if (dlat.max() - dlat.min())/dlat.mean() > unif_thresh:
        raise ValueError("Uniform latitude spacing required; given values "
                         "are not sufficiently uniform.")
    return (masked*cosdeg(lat)*np.deg2rad(dlat)).sum(lat_str)


def merid_avg_point_data(arr, min_lat=-90, max_lat=90, unif_thresh=0.01,
                         lat_str=LAT_STR):
    """Area-weighted meridional average for data defined at single lats.

    As opposed to e.g. gridded climate model output, wherein the quantity at
    the given latitude corresponds to the value of a cell of finite area.  In
    that case, a discrete form of the summing operation should be used and is
    implemented in the function ``merid_average_grid_data``.

    """
    return (merid_integral_point_data(arr, min_lat, max_lat,
                                      unif_thresh, lat_str) /
            merid_integral_point_data(xr.ones_like(arr), min_lat,
                                      max_lat, unif_thresh, lat_str))


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
def dp_from_pfull(pfull, p_str="plev", p_top=0., p_bot=1012.5e2):
    """Pressure thickness of levels given pressures at level centers."""
    if pfull[0] < pfull[1]:
        p_first = p_top
        p_last = p_bot
    else:
        p_first = p_bot
        p_last = p_top
    p_half_inner_vals = 0.5*(pfull.values[1:] + pfull.values[:-1])
    p_half_vals = np.concatenate([[p_first], p_half_inner_vals, [p_last]])
    return np.abs(xr.ones_like(pfull) * np.diff(p_half_vals))


def dp_from_p_half(p_half, pressure):
    """Pressure thickness of vertical levels given interface pressures."""
    dp_vals = p_half.values[1:] - p_half.values[:-1]
    return xr.ones_like(pressure)*dp_vals


def dlogp_from_p_half(p_half, pressure):
    """Pressure thickness of vertical levels given interface pressures."""
    dlogp_vals = np.log(p_half.values[1:]/p_half.values[:-1])
    return xr.ones_like(pressure)*dlogp_vals


def pfull_vals_simm_burr(phalf, phalf_ref, pfull_ref, phalf_str=PHALF_STR):
    """Compute pressure at full levels using Simmons-Burridge spacing.

    See Simmons and Burridge, 1981, "An Energy and Angular-Momentum Conserving
    Vertical Finite-Difference Scheme and Hybrid Vertical Coordinates."
    Monthly Weather Review, 109(4), 758-766.

    """
    dp_vals = phalf.diff(phalf_str).values
    # Above means vertically above (i.e. lower pressure).
    phalf_above = phalf.isel(phalf=slice(None, -1))
    phalf_below = phalf.isel(phalf=slice(1, None))

    dlog_phalf_vals = np.log(phalf_below.values / phalf_above.values)
    phalf_over_dp_vals = phalf_above.values / dp_vals

    alpha_vals = 1. - phalf_over_dp_vals*dlog_phalf_vals

    ln_pfull_vals = np.log(phalf_below.values) - alpha_vals
    pfull_vals = np.exp(ln_pfull_vals)
    top_lev_factor = float(pfull_ref[0] / phalf_ref[1])
    pfull_vals[0] = phalf.isel(phalf=1)*top_lev_factor
    return pfull_vals


def pfull_simm_burr(arr_template, phalf, phalf_ref, pfull_ref,
                    phalf_str=PHALF_STR):
    """Compute pressure at full levels using Simmons-Burridge spacing.

    See Simmons and Burridge, 1981, "An Energy and Angular-Momentum Conserving
    Vertical Finite-Difference Scheme and Hybrid Vertical Coordinates."
    Monthly Weather Review, 109(4), 758-766.

    """
    pfull_vals = pfull_vals_simm_burr(phalf, phalf_ref, pfull_ref, phalf_str)
    return xr.ones_like(arr_template)*pfull_vals


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
