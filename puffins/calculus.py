#! /usr/bin/env python
"""Derivatives, integrals, and averages."""
import logging

import numpy as np
import xarray as xr

from .constants import RAD_EARTH
from .names import (
    BOUNDS_STR,
    LAT_BOUNDS_STR,
    LAT_STR,
    LON_BOUNDS_STR,
    LON_STR,
    LEV_STR,
    SFC_AREA_STR,
)
from .nb_utils import coord_arr_1d, cosdeg, sindeg


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
    return integrate(arr, dp, dim=dim) / grav


def int_dlogp(arr, p_top=0., p_bot=MEAN_SLP_EARTH, pfull_str=LEV_STR,
              phalf_str=PHALF_STR):
    """Integral of array on pressure levels but weighted by log(pressure)."""
    dlogp = dlogp_from_pfull(arr[pfull_str], p_top=p_top, p_bot=p_bot,
                             phalf_str=phalf_str)
    return integrate(arr, dlogp, dim=pfull_str)


def col_avg(arr, dp, dim=LEV_STR):
    """Pressure-weighted column average."""
    return integrate(arr, dp, dim=dim) / integrate(1.0, dp, dim=dim)


def subtract_col_avg(arr, dp, dim=LEV_STR):
    """Impoze zero column integral by subtracting column average at each level.

    Used e.g. for computing the zonally integrated mass flux.  In the time-mean
    and neglecting tendencies in column mass, the column integrated meridional
    mass transport should be zero at each latitude; otherwise there would be a
    build up of mass on one side.

    """
    return arr - col_avg(arr, dp, dim=dim)


# Meridional integrals and averages.
def merid_integral_point_data(arr, min_lat=-90, max_lat=90, unif_thresh=0.01,
                              do_cumsum=False, lat_str=LAT_STR):
    """Area-weighted meridional integral for data defined at single lats.

    As opposed to e.g. gridded climate model output, wherein the quantity at
    the given latitude corresponds to the value of a cell of finite area.  In
    that case, a discrete form of the summing operation should be used and is
    implemented in the function ``merid_integral_grid_data``.

    """
    lat = arr[lat_str]
    masked = arr.where((lat > min_lat) & (lat < max_lat), drop=True)
    dlat = lat.diff(lat_str)
    if (dlat.max() - dlat.min()) / dlat.mean() > unif_thresh:
        raise ValueError("Uniform latitude spacing required; given values "
                         "are not sufficiently uniform.")
    integrand = masked * cosdeg(lat) * np.deg2rad(dlat)
    if do_cumsum:
        return integrand.cumsum(lat_str)
    return integrand.sum(lat_str)


def merid_avg_point_data(arr, min_lat=-90, max_lat=90, unif_thresh=0.01,
                         lat_str=LAT_STR):
    """Area-weighted meridional average for data defined at single lats.

    As opposed to e.g. gridded climate model output, wherein the quantity at
    the given latitude corresponds to the value of a cell of finite area.  In
    that case, a discrete form of the summing operation should be used and is
    implemented in the function ``merid_average_grid_data``.

    """
    return (merid_integral_point_data(arr, min_lat=min_lat, max_lat=max_lat,
                                      unif_thresh=unif_thresh, do_cumsum=False,
                                      lat_str=lat_str) /
            merid_integral_point_data(xr.ones_like(arr),
                                      min_lat=min_lat, max_lat=max_lat,
                                      unif_thresh=unif_thresh,
                                      do_cumsum=False, lat_str=lat_str))


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
    area = xr.ones_like(lat) * 2. * np.pi * radius ** 2 * sinlat_diff
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


def lat_circumf(lat, radius=RAD_EARTH):
    """Circumference of a latitude circle."""
    return 2 * np.pi * radius * cosdeg(lat)


def lat_circumf_weight(arr, lat=None, lat_str=LAT_STR, radius=RAD_EARTH):
    """Multiply an array by the latitude circumference.

    For e.g. poleward tracer fluxes.

    """
    if lat is None:
        lat = arr[lat_str]
    return arr * lat_circumf(lat, radius=radius)


if __name__ == "__main__":
    pass
