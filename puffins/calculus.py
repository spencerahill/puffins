#! /usr/bin/env python
"""Derivatives, integrals, and averages."""
import logging

import numpy as np
import xarray as xr

from .constants import GRAV_EARTH, MEAN_SLP_EARTH, RAD_EARTH
from .names import (
    BOUNDS_STR,
    LAT_BOUNDS_STR,
    LAT_STR,
    LON_BOUNDS_STR,
    LON_STR,
    LEV_STR,
    PFULL_STR,
    PHALF_STR,
    SFC_AREA_STR,
)
from .nb_utils import coord_arr_1d, cosdeg, sindeg, to_pascal


# Derivatives.
def lat_deriv(arr, lat_str=LAT_STR):
    """Meridional derivative approximated by centered differencing."""
    # Latitude is in degrees but in the denominator, so using `np.rad2deg`
    # gives the correct conversion from degrees to radians.
    return np.rad2deg(arr.differentiate(lat_str))


def flux_div(arr_merid_flux, arr_vert_flux, vert_str=LEV_STR,
             lat_str=LAT_STR, radius=RAD_EARTH):
    """Horizontal plus vertical flux divergence of a given field."""
    merid_flux_div = (lat_deriv(arr_merid_flux, lat_str) /
                               (radius*cosdeg(arr_merid_flux[lat_str])))
    vert_flux_div = arr_vert_flux.differentiate(vert_str)
    return merid_flux_div + vert_flux_div


# Vertical integrals and averages.
def integrate(arr, ddim, dim=LEV_STR):
    """Integrate along the given dimension."""
    return (arr * ddim).sum(dim=dim)


def int_dp_g(arr, dp, dim=LEV_STR, grav=GRAV_EARTH):
    """Mass weighted integral."""
    return integrate(arr, to_pascal(dp, is_dp=True), dim=dim) / grav


def int_dlogp(arr, p_top=0., p_bot=MEAN_SLP_EARTH, pfull_str=LEV_STR,
              phalf_str=PHALF_STR):
    """Integral of array on pressure levels but weighted by log(pressure)."""
    dlogp = dlogp_from_pfull(arr[pfull_str], p_top=p_top, p_bot=p_bot,
                             phalf_str=phalf_str)
    return integrate(arr, dlogp, dim=pfull_str)


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
    arr_masked = arr.where((lat > min_lat) & (lat < max_lat), drop=True)

    dlat = lat.diff(lat_str)
    dlat_mean = dlat.mean(lat_str)
    dlat_frac_var = (dlat - dlat_mean) / dlat_mean
    if np.any(np.abs(dlat_frac_var) > dlat_var_tol):
        max_frac_var = float(np.max(np.abs(dlat_frac_var)))
        raise ValueError(
            f"Uniform latitude spacing required to within {dlat_var_tol}.  "
            f"Actual max fractional deviation from uniform: {max_frac_var}"
        )

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
    return (arr_masked * area_masked).sum(lat_str)


def merid_avg_grid_data(arr, min_lat=-90, max_lat=90, lat_str=LAT_STR):
    """Area-weighted meridional average for data on finite grid cells.

    As opposed to data defined at individual latitudes, wherein the quantity at
    the given latitude corresponds to exactly that latitude only, not to a cell
    of finite area surrounding that latitude.  In that case, the function
    ``merid_avg_point_data`` should be used.

    """
    return (merid_integral_grid_data(arr, min_lat, max_lat, lat_str) /
            merid_integral_grid_data(xr.ones_like(arr), min_lat,
                                     max_lat, lat_str))


def global_avg_grid_data(arr, lat_str=LAT_STR, lon_str=LON_STR,
                         sfc_area_str=SFC_AREA_STR):
    """Area-weighted global average for data on finite-area grid cells."""
    if sfc_area_str in arr:
        sfc_area = arr[sfc_area_str]
        return ((arr * sfc_area).sum([lon_str, lat_str]) /
                sfc_area.sum([lon_str, lat_str]))
    # TODO: this assumes uniform longitude, which isn't strictly guaranteed.
    return merid_avg_grid_data(arr.mean(lon_str), min_lat=-90, max_lat=90,
                               lat_str=lat_str)


def merid_avg_sinlat_data(arr, min_lat=-90, max_lat=90, sinlat=None,
                          lat_str=LAT_STR, dsinlat_var_tol=0.001):
    """Area-weighted meridional average for data evenly spaced in sin(lat).

    Data spaced uniformly by sin(lat) is already area-weighted, so just
    average, but first check that the spacing really is uniform (enough).

    """
    lat = arr[lat_str]
    arr_masked = arr.where((lat > min_lat) & (lat < max_lat), drop=True)

    if sinlat is not None:
        dsinlat = sinlat.diff(lat_str)
    else:
        dsinlat = sindeg(lat).diff(lat_str)

    dsinlat_mean = dsinlat.mean(lat_str)
    dsinlat_frac_var = (dsinlat - dsinlat_mean) / dsinlat_mean
    if np.any(np.abs(dsinlat_frac_var) > dsinlat_var_tol):
        max_frac_var = float(np.max(np.abs(dsinlat_frac_var)))
        raise ValueError(
            f"Uniform sin(lat) spacing required to within {dsinlat_var_tol}.  "
            f"Actual max fractional deviation from uniform: {max_frac_var}"
        )
    return arr_masked.mean(lat_str)


# Surface area of lat-lon data.
# TODO: add check that spacing is (nearly) uniform.
def infer_bounds(arr, dim, dim_bounds=None, bounds_str=BOUNDS_STR):
    """Requires that array be evenly spaced (up to an error threshold)."""
    arr_vals = arr.values
    midpoint_vals = 0.5*(arr_vals[:-1] + arr_vals[1:])

    bound_left = arr_vals[0] - (midpoint_vals[0] - arr_vals[0])
    bound_right = arr_vals[-1] + (arr_vals[-1] - midpoint_vals[-1])

    bounds_left_vals = np.concatenate(([bound_left], midpoint_vals))
    bounds_right_vals = np.concatenate((midpoint_vals, [bound_right]))

    bounds_vals = np.array([bounds_left_vals, bounds_right_vals]).transpose()

    if dim_bounds is None:
        bounds_arr_name = dim + '_bounds'
    else:
        bounds_arr_name = dim_bounds
    return xr.DataArray(bounds_vals, dims=[dim, bounds_str],
                        coords={dim: arr}, name=bounds_arr_name)


def add_lat_lon_bounds(arr, lat_str=LAT_STR, lon_str=LON_STR,
                       lat_bounds_str=LAT_BOUNDS_STR,
                       lon_bounds_str=LON_BOUNDS_STR):
    """Add bounding arrays to lat and lon arrays."""
    if isinstance(arr, xr.DataArray):
        ds = arr.to_dataset()
    else:
        ds = arr
    lon_bounds = infer_bounds(ds[lon_str], lon_str, lon_bounds_str)
    lat_bounds = infer_bounds(ds[lat_str], lat_str, lat_bounds_str)
    ds.coords[lon_bounds_str] = lon_bounds
    ds.coords[lat_bounds_str] = lat_bounds
    return ds


def to_radians(arr, is_delta=False):
    """Force data with units either degrees or radians to be radians."""
    # Infer the units from embedded metadata, if it's there.
    try:
        units = arr.units
    except AttributeError:
        pass
    else:
        if units.lower().startswith('degrees'):
            warn_msg = f"Conversion applied: degrees->radians to array: {arr}"
            logging.debug(warn_msg)
            return np.deg2rad(arr)
    # Otherwise, assume degrees if the values are sufficiently large.
    threshold = 0.1*np.pi if is_delta else 4*np.pi
    if np.max(np.abs(arr)) > threshold:
        warn_msg = f"Conversion applied: degrees->radians to array: {arr}"
        logging.debug(warn_msg)
        return np.deg2rad(arr)
    return arr


def _bounds_from_array(arr, dim, bounds_dim=BOUNDS_STR):
    """Get the bounds of an array given its center values.

    E.g. if lat-lon grid center lat/lon values are known, but not the
    bounds of each grid box.  The algorithm assumes that the bounds
    are simply halfway between each pair of center values.

    """
    # TODO: don't assume needed dimension is in axis=0
    spacing = arr.diff(dim).values
    lower = xr.DataArray(np.empty_like(arr), dims=arr.dims, coords=arr.coords)
    lower.values[:-1] = arr.values[:-1] - 0.5*spacing
    lower.values[-1] = arr.values[-1] - 0.5*spacing[-1]
    upper = xr.DataArray(np.empty_like(arr), dims=arr.dims, coords=arr.coords)
    upper.values[:-1] = arr.values[:-1] + 0.5*spacing
    upper.values[-1] = arr.values[-1] + 0.5*spacing[-1]
    bounds = xr.concat([lower, upper], dim=bounds_dim)
    return bounds.T


def _diff_bounds(bounds, coord):
    """Get grid spacing by subtracting upper and lower bounds."""
    try:
        return bounds[:, 1] - bounds[:, 0]
    except IndexError:
        diff = np.diff(bounds, axis=0)
        return xr.DataArray(diff, dims=coord.dims, coords=coord.coords)


def _grid_sfc_area(lon, lat, lon_bounds=None, lat_bounds=None, lon_str=LON_STR,
                   lat_str=LAT_STR, lon_bounds_str=LON_BOUNDS_STR,
                   lat_bounds_str=LAT_BOUNDS_STR, sfc_area_str=SFC_AREA_STR,
                   radius=RAD_EARTH):
    # Compute the bounds if not given.
    if lon_bounds is None:
        lon_bounds = _bounds_from_array(lon, lon_str, lon_bounds_str)
    if lat_bounds is None:
        lat_bounds = _bounds_from_array(lat, lat_str, lat_bounds_str)
    # Compute the surface area.
    dlon = _diff_bounds(to_radians(lon_bounds, is_delta=True), lon)
    sinlat_bounds = np.sin(to_radians(lat_bounds, is_delta=True))
    dsinlat = np.abs(_diff_bounds(sinlat_bounds, lat))
    sfc_area = dlon*dsinlat*(radius**2)
    # Rename the coordinates such that they match the actual lat / lon.
    try:
        sfc_area = sfc_area.rename({lat_bounds_str: lat_str,
                                    lon_bounds_str: lon_str})
    except ValueError:
        pass
    # Clean up: correct names and dimension order.
    sfc_area = sfc_area.rename(sfc_area_str)
    sfc_area[lat_str] = lat
    sfc_area[lon_str] = lon
    return sfc_area.transpose()


def sfc_area_latlon_box(ds, lat_str=LAT_STR, lon_str=LON_STR,
                        lat_bounds_str=LAT_BOUNDS_STR,
                        lon_bounds_str=LON_BOUNDS_STR,
                        sfc_area_str=SFC_AREA_STR, radius=RAD_EARTH):
    """Calculate surface area of each grid cell in a lon-lat grid."""
    lon = ds[lon_str]
    lat = ds[lat_str]
    lon_bounds = ds[lon_bounds_str]
    lat_bounds = ds[lat_bounds_str]
    return _grid_sfc_area(
        lon,
        lat,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
        lon_str=lon_str,
        lat_str=lat_str,
        lon_bounds_str=lon_bounds_str,
        lat_bounds_str=lat_bounds_str,
        sfc_area_str=sfc_area_str,
        radius=radius,
    )


# Pressure spacing and averages.
def phalf_from_pfull(pfull, p_top=0., p_bot=MEAN_SLP_EARTH,
                     phalf_str=PHALF_STR):
    """Pressure at half levels given pressures at level centers."""
    if pfull[0] < pfull[1]:
        p_first = p_top
        p_last = p_bot
    else:
        p_first = p_bot
        p_last = p_top
    phalf_inner_vals = 0.5*(pfull.values[1:] + pfull.values[:-1])
    phalf_vals = np.concatenate([[p_first], phalf_inner_vals, [p_last]])
    return coord_arr_1d(values=phalf_vals, dim=phalf_str)


def dp_from_pfull(pfull, p_top=0., p_bot=MEAN_SLP_EARTH):
    """Pressure thickness of levels given pressures at level centers."""
    phalf = phalf_from_pfull(pfull, p_top=p_top, p_bot=p_bot)
    return np.abs(xr.ones_like(pfull) * np.diff(phalf.values))


def dp_from_phalf(phalf, pfull_ref, phalf_str=PHALF_STR, pfull_str=PFULL_STR):
    """Pressure thickness of vertical levels given interface pressures."""
    dp_vals = np.abs(phalf.isel(**{phalf_str: slice(None, -1)}).values -
                     phalf.isel(**{phalf_str: slice(1, None)}).values)
    dims_out = []
    for dim in phalf.dims:
        if dim == "phalf":
            dims_out.append(pfull_str)
        else:
            dims_out.append(dim)

    vals_template = ([phalf[dim] for dim in phalf.dims if dim != phalf_str] +
                     [pfull_ref])
    arr_template = xr.ones_like(np.product(vals_template)).transpose(*dims_out)
    return (arr_template * dp_vals).rename("dp")


def dlogp_from_phalf(phalf, pressure):
    """Pressure thickness of vertical levels given interface pressures."""
    # Avoid divide-by-zero error by overwriting if top pressure is zero.
    phalf_vals = phalf.copy().values
    if phalf_vals[0] == 0:
        phalf_vals[0] = 0.5 * phalf_vals[1]
    elif phalf_vals[-1] == 0:
        phalf_vals[-1] = 0.5 * phalf_vals[-2]
    dlogp_vals = np.log(phalf_vals[1:] / phalf_vals[:-1])
    return xr.ones_like(pressure) * dlogp_vals


def dlogp_from_pfull(pfull, p_top=0., p_bot=MEAN_SLP_EARTH,
                     phalf_str=PHALF_STR):
    """Thickness in log(p) of vertical levels given level-center pressures."""
    phalf = phalf_from_pfull(pfull, p_top=p_top, p_bot=p_bot,
                             phalf_str=phalf_str)
    return dlogp_from_phalf(phalf, pfull)


def phalf_from_psfc(bk, pk, p_sfc):
    """Compute pressure of half levels of hybrid sigma-pressure coordinates."""
    return p_sfc * bk + pk


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
    return xr.ones_like(arr_template) * pfull_vals


def _flip_dim(arr, dim):
    return arr.isel(**{dim: slice(None, None, -1)})


def avg_p_weighted(arr, phalf, pressure, p_str=LEV_STR):
    """Pressure-weighted vertical average."""
    dp = np.abs(dp_from_phalf(phalf, pressure))
    if phalf[0] > phalf[1]:
        arr_out = _flip_dim(arr, p_str)
        dp_out = _flip_dim(dp, p_str)
    else:
        arr_out = arr
        dp_out = dp
    return (arr_out*dp_out).cumsum(p_str) / dp_out.cumsum(p_str)


def avg_logp_weighted(arr, phalf, pressure, p_str=LEV_STR):
    """Log-pressure-weighted vertical average."""
    dlogp = dlogp_from_phalf(phalf, pressure)
    return (arr * dlogp).cumsum(p_str) / dlogp.cumsum(p_str)


def col_extrema(arr, p_str=LEV_STR):
    """Locations and values of local extrema within each column."""
    darr_dp = arr.differentiate(p_str)
    sign_change = np.sign(darr_dp).diff(p_str)
    return arr.where(sign_change)


if __name__ == "__main__":
    pass
