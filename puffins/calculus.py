#! /usr/bin/env python
"""Derivatives, integrals, and averages."""

from __future__ import annotations

import contextlib
import logging
from typing import cast, overload

import numpy as np
import xarray as xr

from ._typing import ArrayLike, Scalar, XarrayObj
from .constants import GRAV_EARTH, RAD_EARTH
from .names import (
    BOUNDS_STR,
    LAT_BOUNDS_STR,
    LAT_STR,
    LEV_STR,
    LON_BOUNDS_STR,
    LON_STR,
    SFC_AREA_STR,
)
from .nb_utils import cosdeg, sindeg
from .vert_coords import int_dp_g


# Derivatives.
def lat_deriv(arr: xr.DataArray, lat_str: str = LAT_STR) -> xr.DataArray:
    """Meridional derivative approximated by centered differencing."""
    # Latitude is in degrees but in the denominator, so using `np.rad2deg`
    # gives the correct conversion from degrees to radians.
    return cast(xr.DataArray, np.rad2deg(arr.differentiate(lat_str)))


def flux_div(
    arr_merid_flux: xr.DataArray,
    arr_vert_flux: xr.DataArray,
    vert_str: str = LEV_STR,
    lat_str: str = LAT_STR,
    radius: float = RAD_EARTH,
) -> xr.DataArray:
    """Horizontal plus vertical flux divergence of a given field."""
    merid_flux_div = lat_deriv(arr_merid_flux, lat_str) / (
        radius * cosdeg(arr_merid_flux[lat_str])
    )
    vert_flux_div = arr_vert_flux.differentiate(vert_str)
    return cast(xr.DataArray, merid_flux_div + vert_flux_div)


# Meridional integrals and averages.
def merid_integral_point_data(
    arr: xr.DataArray,
    min_lat: float = -90,
    max_lat: float = 90,
    unif_thresh: float = 0.01,
    do_cumsum: bool = False,
    centered: bool = False,
    lat_str: str = LAT_STR,
) -> xr.DataArray:
    """Area-weighted meridional integral for data defined at single lats.

    As opposed to e.g. gridded climate model output, wherein the quantity at
    the given latitude corresponds to the value of a cell of finite area.  In
    that case, a discrete form of the summing operation should be used and is
    implemented in the function ``merid_integral_grid_data``.

    Parameters
    ----------
    arr : xarray.DataArray
        Field to integrate.
    min_lat, max_lat : float, optional
        Latitude bounds of the integral.  Default: the full sphere.
    unif_thresh : float, optional
        Maximum fractional spread in latitude spacing tolerated before
        raising.  Default: 0.01.
    do_cumsum : bool, optional
        Return the cumulative integral as a function of latitude rather than
        the total.  Default: False.
    centered : bool, optional
        Use the midpoint rather than the one-sided rectangle rule for the
        cumulative integral.  Only meaningful when ``do_cumsum`` is True.
        Default: False, which preserves the one-sided rule.  See Notes.
    lat_str : str, optional
        Name of the latitude dimension.  Default: 'lat'.

    Returns
    -------
    xarray.DataArray
        The integral, reduced over latitude, or the cumulative integral as a
        function of latitude when ``do_cumsum`` is True.

    Raises
    ------
    ValueError
        If the latitude spacing is not uniform to within ``unif_thresh``, or
        if ``centered`` is True without ``do_cumsum``.

    Notes
    -----
    A plain ``cumsum`` accumulates the whole cell centered on each latitude,
    so it approximates the integral up to half a grid cell *past* that
    latitude.  That makes it first-order accurate in the grid spacing.
    Backing off half a cell, which is what ``centered=True`` does, is the
    midpoint rule and is second-order accurate; measured convergence is 1.00
    and 2.00 respectively.

    The centered value at the final latitude is the full integral less half
    that latitude's cell.  For a zero-mean field on a grid whose endpoints are
    the poles, ``cos`` vanishes there and the cumulative integral still closes
    to zero to machine precision.  On a cell-centred grid it does not, and
    should not: the last cell centre is not the pole, and there is still half
    a cell of area north of it.  Measured residuals for a zero-mean field are
    4e-4 of the peak on a 1 degree cell-centred grid and 2e-5 on a 0.25 degree
    one.
    """
    if centered and not do_cumsum:
        raise ValueError("`centered` is only meaningful when `do_cumsum` is True.")
    lat = arr[lat_str]
    lat_mask = (lat >= min_lat) & (lat <= max_lat)
    dlat_arr = lat.where(lat_mask, drop=True).diff(lat_str)
    # ``abs`` on the denominator so the check also fires for a descending
    # latitude coordinate, where the mean spacing is negative.
    if (dlat_arr.max() - dlat_arr.min()) / np.abs(dlat_arr.mean()) > unif_thresh:
        raise ValueError(
            "Uniform latitude spacing required; given values "
            "are not sufficiently uniform."
        )
    else:
        dlat = dlat_arr.mean()
    integrand = arr.where(lat_mask, drop=True) * cosdeg(lat) * np.deg2rad(dlat)
    if do_cumsum:
        cumulative = integrand.cumsum(lat_str)
        if centered:
            cumulative = cumulative - 0.5 * integrand
        return cast(xr.DataArray, cumulative)
    return cast(xr.DataArray, integrand.sum(lat_str))


def merid_avg_point_data(
    arr: xr.DataArray,
    min_lat: float = -90,
    max_lat: float = 90,
    unif_thresh: float = 0.01,
    do_cumsum: bool = False,
    lat_str: str = LAT_STR,
) -> xr.DataArray:
    """Area-weighted meridional average for data defined at single lats.

    As opposed to e.g. gridded climate model output, wherein the quantity at
    the given latitude corresponds to the value of a cell of finite area.  In
    that case, a discrete form of the summing operation should be used and is
    implemented in the function ``merid_average_grid_data``.

    """
    return cast(
        xr.DataArray,
        merid_integral_point_data(
            arr,
            min_lat=min_lat,
            max_lat=max_lat,
            unif_thresh=unif_thresh,
            do_cumsum=do_cumsum,
            lat_str=lat_str,
        )
        / merid_integral_point_data(
            xr.ones_like(arr),
            min_lat=min_lat,
            max_lat=max_lat,
            unif_thresh=unif_thresh,
            do_cumsum=do_cumsum,
            lat_str=lat_str,
        ),
    )


def merid_integral_grid_data(
    arr: xr.DataArray,
    min_lat: float = -90,
    max_lat: float = 90,
    lat_str: str = LAT_STR,
    dlat_var_tol: float = 0.01,
    radius: float = RAD_EARTH,
) -> xr.DataArray:
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
    lat_above = lat + 0.5 * dlat_mean
    if lat_above[-1] > 90:
        lat_above[-1] = 90.0
    lat_below = lat - 0.5 * dlat_mean
    if lat_below[0] < -90:
        lat_below[0] = -90.0

    sinlat_diff = sindeg(lat_above.values) - sindeg(lat_below.values)
    area = xr.ones_like(lat) * 2.0 * np.pi * radius**2 * sinlat_diff
    area_masked = area.where((lat > min_lat) & (lat < max_lat), drop=True)
    return cast(xr.DataArray, (arr_masked * area_masked).sum(lat_str))


def merid_avg_grid_data(
    arr: xr.DataArray,
    min_lat: float = -90,
    max_lat: float = 90,
    lat_str: str = LAT_STR,
) -> xr.DataArray:
    """Area-weighted meridional average for data on finite grid cells.

    As opposed to data defined at individual latitudes, wherein the quantity at
    the given latitude corresponds to exactly that latitude only, not to a cell
    of finite area surrounding that latitude.  In that case, the function
    ``merid_avg_point_data`` should be used.

    """
    return cast(
        xr.DataArray,
        merid_integral_grid_data(arr, min_lat, max_lat, lat_str)
        / merid_integral_grid_data(xr.ones_like(arr), min_lat, max_lat, lat_str),
    )


def global_avg_grid_data(
    arr: xr.DataArray,
    lat_str: str = LAT_STR,
    lon_str: str = LON_STR,
    sfc_area_str: str = SFC_AREA_STR,
) -> xr.DataArray:
    """Area-weighted global average for data on finite-area grid cells."""
    if sfc_area_str in arr:
        sfc_area = arr[sfc_area_str]
        return cast(
            xr.DataArray,
            (arr * sfc_area).sum([lon_str, lat_str]) / sfc_area.sum([lon_str, lat_str]),
        )
    # TODO: this assumes uniform longitude, which isn't strictly guaranteed.
    return merid_avg_grid_data(
        cast(xr.DataArray, arr.mean(lon_str)),
        min_lat=-90,
        max_lat=90,
        lat_str=lat_str,
    )


def merid_avg_sinlat_data(
    arr: xr.DataArray,
    min_lat: float = -90,
    max_lat: float = 90,
    sinlat: xr.DataArray | None = None,
    lat_str: str = LAT_STR,
    dsinlat_var_tol: float = 0.001,
) -> xr.DataArray:
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
    return cast(xr.DataArray, arr_masked.mean(lat_str))


# Surface area of lat-lon data.
def _check_uniform_spacing(
    coord: xr.DataArray,
    dim: str,
    name: str,
    tol: float = 0.01,
) -> None:
    """Raise ValueError if coordinate spacing is not nearly uniform.

    Parameters
    ----------
    coord : xr.DataArray
        The coordinate array to check.
    dim : str
        The dimension name along which to compute differences.
    name : str
        Human-readable name used in error messages.
    tol : float
        Maximum allowed fractional deviation from uniform spacing.

    """
    diff = coord.diff(dim)
    mean_diff = diff.mean(dim)
    frac_var = (diff - mean_diff) / mean_diff
    if np.any(np.abs(frac_var) > tol):
        max_frac_var = float(np.max(np.abs(frac_var)))
        raise ValueError(
            f"Uniform {name} spacing required to within {tol}. "
            f"Actual max fractional deviation from uniform: {max_frac_var}"
        )


def infer_bounds(
    arr: xr.DataArray,
    dim: str,
    dim_bounds: str | None = None,
    bounds_str: str = BOUNDS_STR,
    spacing_tol: float = 0.01,
) -> xr.DataArray:
    """Infer bounding values from evenly spaced coordinate centers.

    Parameters
    ----------
    arr : xarray.DataArray
        Coordinate array whose bounds are to be inferred.
    dim : str
        Name of the dimension along which to infer bounds.
    dim_bounds : str or None
        Name to assign to the resulting bounds DataArray.
    bounds_str : str
        Name of the bounds dimension.
    spacing_tol : float
        Maximum allowed fractional deviation from uniform spacing.

    Raises
    ------
    ValueError
        If the spacing along ``dim`` is not nearly uniform.

    """
    if not isinstance(dim, str):
        raise TypeError(f"dim must be a str, got {type(dim).__name__!r}")
    arr_vals = arr.values
    spacing = np.diff(arr_vals)
    if spacing.mean() == 0:
        raise ValueError("Array values are all identical; cannot infer bounds.")
    _check_uniform_spacing(arr, dim, dim, tol=spacing_tol)

    midpoint_vals = 0.5 * (arr_vals[:-1] + arr_vals[1:])

    bound_left = arr_vals[0] - (midpoint_vals[0] - arr_vals[0])
    bound_right = arr_vals[-1] + (arr_vals[-1] - midpoint_vals[-1])

    bounds_left_vals = np.concatenate(([bound_left], midpoint_vals))
    bounds_right_vals = np.concatenate((midpoint_vals, [bound_right]))

    bounds_vals = np.array([bounds_left_vals, bounds_right_vals]).transpose()

    if dim_bounds is None:
        bounds_arr_name = dim + "_bounds"
    else:
        bounds_arr_name = dim_bounds
    return xr.DataArray(
        bounds_vals, dims=[dim, bounds_str], coords={dim: arr}, name=bounds_arr_name
    )


def add_lat_lon_bounds(
    arr: XarrayObj,
    lat_str: str = LAT_STR,
    lon_str: str = LON_STR,
    lat_bounds_str: str = LAT_BOUNDS_STR,
    lon_bounds_str: str = LON_BOUNDS_STR,
) -> xr.Dataset:
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


@overload
def to_radians(arr: xr.DataArray, is_delta: bool = ...) -> xr.DataArray: ...
@overload
def to_radians(arr: np.ndarray, is_delta: bool = ...) -> np.ndarray: ...
@overload
def to_radians(arr: Scalar, is_delta: bool = ...) -> Scalar: ...
def to_radians(arr: ArrayLike, is_delta: bool = False) -> ArrayLike:
    """Force data with units either degrees or radians to be radians."""
    # Infer the units from embedded metadata, if it's there.
    try:
        units = arr.units  # type: ignore[union-attr]
    except AttributeError:
        pass
    else:
        if units.lower().startswith("degrees"):
            warn_msg = f"Conversion applied: degrees->radians to array: {arr}"
            logging.debug(warn_msg)
            return np.deg2rad(arr)
    # Otherwise, assume degrees if the values are sufficiently large.
    threshold = 0.1 * np.pi if is_delta else 4 * np.pi
    if np.max(np.abs(arr)) > threshold:
        warn_msg = f"Conversion applied: degrees->radians to array: {arr}"
        logging.debug(warn_msg)
        return np.deg2rad(arr)
    return arr


def _bounds_from_array(
    arr: xr.DataArray, dim: str, bounds_dim: str = BOUNDS_STR
) -> xr.DataArray:
    """Get the bounds of an array given its center values.

    E.g. if lat-lon grid center lat/lon values are known, but not the
    bounds of each grid box.  The algorithm assumes that the bounds
    are simply halfway between each pair of center values.

    """
    spacing = arr.diff(dim)
    last_spacing = spacing.isel({dim: -1})
    spacing_padded = xr.concat([spacing, last_spacing.expand_dims(dim)], dim=dim)
    spacing_padded[dim] = arr[dim]
    lower = arr - 0.5 * spacing_padded
    upper = arr + 0.5 * spacing_padded
    bounds = xr.concat([lower, upper], dim=bounds_dim)
    return cast(xr.DataArray, bounds.T)


def _diff_bounds(bounds: xr.DataArray, coord: xr.DataArray) -> xr.DataArray:
    """Get grid spacing by subtracting upper and lower bounds."""
    try:
        return cast(xr.DataArray, bounds[:, 1] - bounds[:, 0])
    except IndexError:
        diff = np.diff(bounds, axis=0)
        return xr.DataArray(diff, dims=coord.dims, coords=coord.coords)


def _grid_sfc_area(
    lon: xr.DataArray,
    lat: xr.DataArray,
    lon_bounds: xr.DataArray | None = None,
    lat_bounds: xr.DataArray | None = None,
    lon_str: str = LON_STR,
    lat_str: str = LAT_STR,
    lon_bounds_str: str = LON_BOUNDS_STR,
    lat_bounds_str: str = LAT_BOUNDS_STR,
    sfc_area_str: str = SFC_AREA_STR,
    radius: float = RAD_EARTH,
) -> xr.DataArray:
    # Compute the bounds if not given.
    if lon_bounds is None:
        lon_bounds = _bounds_from_array(lon, lon_str, lon_bounds_str)
    if lat_bounds is None:
        lat_bounds = _bounds_from_array(lat, lat_str, lat_bounds_str)
    # Compute the surface area.
    dlon = _diff_bounds(cast(xr.DataArray, to_radians(lon_bounds, is_delta=True)), lon)
    sinlat_bounds = cast(xr.DataArray, np.sin(to_radians(lat_bounds, is_delta=True)))
    dsinlat = np.abs(_diff_bounds(sinlat_bounds, lat))
    sfc_area = dlon * dsinlat * (radius**2)
    # Rename the coordinates such that they match the actual lat / lon.
    with contextlib.suppress(ValueError):
        sfc_area = sfc_area.rename({lat_bounds_str: lat_str, lon_bounds_str: lon_str})
    # Clean up: correct names and dimension order.
    sfc_area = sfc_area.rename(sfc_area_str)
    sfc_area[lat_str] = lat
    sfc_area[lon_str] = lon
    return cast(xr.DataArray, sfc_area.transpose())


def sfc_area_latlon_box(
    ds: xr.Dataset,
    lat_str: str = LAT_STR,
    lon_str: str = LON_STR,
    lat_bounds_str: str = LAT_BOUNDS_STR,
    lon_bounds_str: str = LON_BOUNDS_STR,
    sfc_area_str: str = SFC_AREA_STR,
    radius: float = RAD_EARTH,
) -> xr.DataArray:
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


@overload
def lat_circumf(lat: xr.DataArray, radius: float = ...) -> xr.DataArray: ...
@overload
def lat_circumf(lat: np.ndarray, radius: float = ...) -> np.ndarray: ...
@overload
def lat_circumf(lat: Scalar, radius: float = ...) -> Scalar: ...
def lat_circumf(lat: ArrayLike, radius: float = RAD_EARTH) -> ArrayLike:
    """Circumference of a latitude circle."""
    return cast(ArrayLike, 2 * np.pi * radius * cosdeg(lat))


def lat_circumf_weight(
    arr: xr.DataArray,
    lat: xr.DataArray | None = None,
    lat_str: str = LAT_STR,
    radius: float = RAD_EARTH,
) -> xr.DataArray:
    """Multiply an array by the latitude circumference.

    For e.g. poleward tracer fluxes.

    """
    if lat is None:
        lat = arr[lat_str]
    return cast(xr.DataArray, arr * lat_circumf(lat, radius=radius))


def col_int_merid_flux(
    v: xr.DataArray,
    arr: xr.DataArray,
    dp: xr.DataArray,
    vert_str: str = LEV_STR,
    lat_str: str = LAT_STR,
    radius: float = RAD_EARTH,
    grav: float = GRAV_EARTH,
) -> xr.DataArray:
    """Total meridional flux of a mass-specific quantity across a lat circle.

    The mass-weighted vertical integral of ``v * arr``, multiplied by the
    circumference of the latitude circle.  When ``arr`` is an energy per unit
    mass such as moist static energy, the result is the total northward energy
    transport in Watts.

    Parameters
    ----------
    v : xarray.DataArray
        Meridional velocity (m/s).  For a column energy budget this should
        usually be the mass-corrected velocity, i.e. the output of
        ``vert_coords.subtract_col_avg``, so that a spurious net column mass
        flux does not contaminate the transport.
    arr : xarray.DataArray
        The transported quantity, per unit mass (e.g. J/kg for an energy).
    dp : xarray.DataArray
        Pressure thickness of each level (Pa).
    vert_str : str, optional
        Name of the vertical dimension.  Default: 'plev'.
    lat_str : str, optional
        Name of the latitude dimension.  Default: 'lat'.
    radius : float, optional
        Planetary radius (m).  Default: Earth's.
    grav : float, optional
        Gravitational acceleration (m/s^2).  Default: Earth's.

    Returns
    -------
    xarray.DataArray
        Flux across the full latitude circle, in units of ``arr`` times kg/s,
        i.e. Watts when ``arr`` is in J/kg.

    See Also
    --------
    inferred_merid_flux : The same transport inferred from boundary fluxes.
    puffins.vert_coords.subtract_col_avg : Mass correction for ``v``.

    Notes
    -----
    ``dp`` must carry the same below-ground mask as ``v``.  The vertical sum
    skips NaN but sums every ``dp``, so an unmasked ``dp`` paired with a
    masked ``v`` dilutes the column mean that ``subtract_col_avg`` removes and
    leaves a large spurious column mass flux behind.  Building ``dp`` from the
    level coordinate alone and masking only the fields is the way this goes
    wrong.

    The vertical sum skips NaN, so a column with missing levels is integrated
    over the levels that remain.  That is correct for genuinely below-ground
    levels, whose mass is absent, but it also means an unintended NaN silently
    reduces the transport, and a wholly missing column returns 0 rather than
    NaN.
    """
    col_int = int_dp_g(v * arr, dp, dim=vert_str, grav=grav)
    return lat_circumf_weight(col_int, lat_str=lat_str, radius=radius)


def inferred_merid_flux(
    boundary_fluxes: xr.DataArray,
    do_remove_global_mean: bool = True,
    lat_str: str = LAT_STR,
    radius: float = RAD_EARTH,
) -> xr.DataArray:
    r"""Meridional flux implied by the net energy input to each column.

    Integrates the net input per unit area northward from the south pole,
    weighting by area:

    .. math::

        F(\phi) = 2 \pi a^2 \int_{-\pi/2}^{\phi}
                  \left[ Q - \langle Q \rangle \right] \cos\phi' \, d\phi'

    where :math:`Q` is ``boundary_fluxes`` and :math:`\langle Q \rangle` is its
    global, area-weighted mean.  Removing that mean is what forces the implied
    flux back to zero at the north pole.  Without it, any global imbalance
    accumulates into a spurious flux that grows with latitude.

    Standard uses: net top-of-atmosphere radiation gives the total (atmosphere
    plus ocean) transport; net surface flux gives the ocean transport; net
    column energy input minus column energy tendency gives the atmospheric
    transport.

    Parameters
    ----------
    boundary_fluxes : xarray.DataArray
        Net energy input per unit area (W/m^2), signed positive into the
        column.
    do_remove_global_mean : bool, optional
        Subtract the global area-weighted mean before integrating.
        Default: True.
    lat_str : str, optional
        Name of the latitude dimension.  Default: 'lat'.
    radius : float, optional
        Planetary radius (m).  Default: Earth's.

    Returns
    -------
    xarray.DataArray
        Northward flux (W) as a function of latitude.

    See Also
    --------
    col_int_merid_flux : The same transport computed directly from the winds.

    Notes
    -----
    Requires uniformly spaced latitudes; see ``merid_integral_point_data``.

    ``boundary_fluxes`` should span pole to pole.  The mean that is removed is
    the mean over whatever latitudes are supplied, so on a hemispheric or
    tropical subset this removes a domain mean rather than a global one, and
    the result is then forced to zero at the northern edge of that subset.
    That is the same property this function relies on for correctness on a
    global array, so a subset produces a plausible-looking profile with no
    warning.

    The cumulative integral uses the midpoint rule, which is second-order
    accurate in the latitude spacing.  It begins half a grid cell south of the
    first latitude, so the value returned there is the contribution of that
    half cell rather than zero.
    """
    if do_remove_global_mean:
        integrand = boundary_fluxes - merid_avg_point_data(
            boundary_fluxes, lat_str=lat_str
        )
    else:
        integrand = boundary_fluxes
    return cast(
        xr.DataArray,
        2.0
        * np.pi
        * radius**2
        * merid_integral_point_data(
            integrand, do_cumsum=True, centered=True, lat_str=lat_str
        ),
    )


def effective_diffusivity(
    flux: xr.DataArray,
    arr: xr.DataArray,
    radius: float = RAD_EARTH,
    lat_str: str = LAT_STR,
) -> xr.DataArray:
    r"""Bulk diffusivity implied by a meridional flux and a meridional gradient.

    Defined by requiring that the flux be down-gradient:

    .. math::

        D_{\mathrm{eff}} = -F \left/ \frac{\partial m}{\partial y} \right.

    with :math:`\partial / \partial y = a^{-1} \partial / \partial \phi`.

    Parameters
    ----------
    flux : xarray.DataArray
        Meridional flux, e.g. the output of ``col_int_merid_flux`` (Watts, for
        an energy flux across the full latitude circle).
    arr : xarray.DataArray
        Field whose meridional gradient sets the down-gradient direction, e.g.
        near-surface moist static energy (J/kg).
    radius : float, optional
        Planetary radius (m).  Default: Earth's.
    lat_str : str, optional
        Name of the latitude dimension.  Default: 'lat'.

    Returns
    -------
    xarray.DataArray
        Effective diffusivity, in units of ``flux`` divided by units of ``arr``
        per meter.  For ``flux`` in Watts and ``arr`` in J/kg that is kg m / s,
        which is *not* the m^2/s of a true eddy diffusivity: the two differ by
        the column mass along the latitude circle per unit meridional distance,
        :math:`2 \pi a \cos\phi \, p_s / g`.

    Notes
    -----
    The denominator vanishes at extrema of ``arr``, and the result there is
    not usable.  Which way it fails depends on the grid, and neither way is
    announced.  When a latitude sits exactly on the extremum the result is
    a signed infinity, emitted without a numpy divide-by-zero warning, which
    then propagates through a later ``mean`` and survives ``nanmax``.  When
    the extremum falls between latitudes the result is instead a large finite
    spike, which no warning or non-finite check will catch.  Both ``flux`` and
    ``arr`` are usually smoothed in latitude first, and results within a few
    grid points of an extremum of ``arr`` should be masked.
    """
    grad = lat_deriv(arr, lat_str=lat_str) / radius
    return cast(xr.DataArray, -1 * flux / grad)


if __name__ == "__main__":
    pass
