#! /usr/bin/env python
"""Fundamental atmospheric dynamical quantities."""

from typing import cast

import numpy as np
import xarray as xr

from ._typing import ArrayLike
from .calculus import lat_deriv
from .constants import (
    DELTA_V,
    GRAV_EARTH,
    HEIGHT_TROPO,
    MEAN_SLP_EARTH,
    R_D,
    RAD_EARTH,
    ROT_RATE_EARTH,
    THETA_REF,
)
from .names import LAT_STR, LEV_STR
from .nb_utils import cosdeg, sindeg, tandeg


def coriolis_param(lat: ArrayLike, rot_rate: float = ROT_RATE_EARTH) -> ArrayLike:
    """Coriolis parameter, i.e. 'f'."""
    return cast(ArrayLike, 2.0 * rot_rate * sindeg(lat))


def plan_burg_num(
    height: float = HEIGHT_TROPO,
    grav: float = GRAV_EARTH,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
) -> float:
    """Planetary Burger number"""
    return height * grav / (rot_rate * radius) ** 2


def therm_ross_num(
    delta_h: ArrayLike,
    height: float = HEIGHT_TROPO,
    lat_max: float = 90,
    grav: float = GRAV_EARTH,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
) -> ArrayLike:
    """Thermal Rossby number."""
    return cast(
        ArrayLike,
        delta_h
        * sindeg(lat_max)
        * plan_burg_num(height, grav=grav, rot_rate=rot_rate, radius=radius),
    )


def abs_ang_mom(
    u: xr.DataArray,
    lat: ArrayLike | None = None,
    radius: float = RAD_EARTH,
    rot_rate: float = ROT_RATE_EARTH,
    lat_str: str = LAT_STR,
) -> xr.DataArray:
    """Absolute angular momentum."""
    if lat is None:
        lat = u[lat_str]
    coslat = cosdeg(lat)
    return cast(xr.DataArray, radius * coslat * (rot_rate * radius * coslat + u))


def abs_vort_vert_comp(
    abs_ang_mom: xr.DataArray,
    radius: float = RAD_EARTH,
    lat_str: str = LAT_STR,
) -> xr.DataArray:
    """Vertical component of absolute vorticity (in axisymmetric case)."""
    return cast(
        xr.DataArray,
        -1
        * lat_deriv(abs_ang_mom, lat_str)
        / (radius**2 * cosdeg(abs_ang_mom[lat_str])),
    )


def abs_vort_from_u(
    u: xr.DataArray,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
    lat_str: str = LAT_STR,
) -> xr.DataArray:
    """Vertical component of absolute vorticity computed from zonal wind."""
    lats = u[lat_str]
    sinlat = sindeg(lats)
    coslat = cosdeg(lats)
    return cast(
        xr.DataArray,
        (u * sinlat) / (radius * coslat)
        - lat_deriv(u, lat_str) / radius
        + 2 * rot_rate * sinlat,
    )


def rel_vort_from_u(
    uwind: xr.DataArray,
    radius: float = RAD_EARTH,
    lat_str: str = LAT_STR,
) -> xr.DataArray:
    """Vertical component of relative vorticity computed from zonal wind."""
    lat = uwind[lat_str]
    coslat = cosdeg(lat)
    return cast(xr.DataArray, -lat_deriv(uwind * coslat, lat_str) / (radius * coslat))


def ross_num_from_uwind(
    uwind: xr.DataArray,
    lat: ArrayLike | None = None,
    radius: float = RAD_EARTH,
    rot_rate: float = ROT_RATE_EARTH,
    lat_str: str = LAT_STR,
) -> xr.DataArray:
    """Rossby number computed from zonal wind.

    Traditional Rossby number definition in terms of only relative vorticity
    and the Coriolis parameter, as opposed to the "3D" version introduced by
    Singh 2019.

    """
    if lat is None:
        lat = uwind[lat_str]
    rel_vort = rel_vort_from_u(uwind, radius=radius, lat_str=lat_str)
    coriolis = coriolis_param(lat, rot_rate=rot_rate)
    return cast(xr.DataArray, -rel_vort / coriolis)


def ross_num_gen(
    uwind: xr.DataArray,
    vwind: xr.DataArray,
    omega: xr.DataArray,
    lat: ArrayLike | None = None,
    hpa_to_pa: bool = False,
    radius: float = RAD_EARTH,
    rot_rate: float = ROT_RATE_EARTH,
    lat_str: str = LAT_STR,
    lev_str: str = LEV_STR,
) -> xr.DataArray:
    """Generalized Rossby number from Singh 2019, JAS.

    Unnumbered equation from pg. 1997.  One minus the ratio of advection of
    angular momentum to advection of planetary angular momentum.  Zonal
    advection is neglected here b/c assuming zonal-mean quantities.

    """
    if lat is None:
        lat = uwind[lat_str]
    coriolis = coriolis_param(lat, rot_rate=rot_rate)
    du_dp = uwind.differentiate(lev_str)
    if hpa_to_pa:
        du_dp *= 1e-2
    return cast(
        xr.DataArray,
        (
            lat_deriv(uwind, lat_str) / radius
            - uwind * tandeg(lat) / radius
            + omega * du_dp / vwind
        )
        / coriolis,
    )


def brunt_vaisala_freq(
    dtheta_dz: ArrayLike,
    theta_ref: float = THETA_REF,
    grav: float = GRAV_EARTH,
) -> ArrayLike:
    """Brunt Vaisala frequency."""
    return cast(ArrayLike, (grav * dtheta_dz / theta_ref) ** 0.5)


def rossby_radius(
    lat: ArrayLike,
    dtheta_dz: ArrayLike,
    height: float = HEIGHT_TROPO,
    theta_ref: float = THETA_REF,
    grav: float = GRAV_EARTH,
    rot_rate: float = ROT_RATE_EARTH,
) -> ArrayLike:
    """Rossby radius of deformation"""
    return cast(
        ArrayLike,
        brunt_vaisala_freq(dtheta_dz, theta_ref=theta_ref, grav=grav)
        * height
        / (2 * rot_rate * sindeg(lat)),
    )


def zonal_fric_inferred_steady(
    u_merid_flux: xr.DataArray,
    u_vert_flux: xr.DataArray,
    vwind: xr.DataArray,
    radius: float = RAD_EARTH,
    rot_rate: float = ROT_RATE_EARTH,
    vert_str: str = LEV_STR,
    lat_str: str = LAT_STR,
) -> xr.DataArray:
    """Steady-state zonal friction inferred from flux div + Coriolis."""
    lats = u_merid_flux[lat_str]
    coslat = cosdeg(lats)
    duv_dlat_term = lat_deriv(u_merid_flux * coslat, lat_str) / (radius * coslat**2)
    duw_dvert_term = u_vert_flux.differentiate(vert_str)

    u_flux_div = duv_dlat_term + duw_dvert_term
    cor_param = coriolis_param(lats, rot_rate=rot_rate)
    return cast(xr.DataArray, u_flux_div - cor_param * vwind)


def z_from_hypso(
    temp: xr.DataArray,
    p_sfc: float = MEAN_SLP_EARTH,
    p_top: float = 1,
    r_d: float = R_D,
    grav: float = GRAV_EARTH,
    p_str: str = LEV_STR,
) -> xr.DataArray:
    """Height computed from hypsometric equation.

    Pressure is assumed in Pa, not hPa.

    """
    # Ensure all pressures have same horizontal dimensions as temperature.
    non_vert_coords = xr.ones_like(temp.isel({p_str: 0})).drop_vars(p_str)
    p_sfc_da: xr.DataArray = p_sfc * non_vert_coords
    p_sfc_da[p_str] = p_sfc
    p_top_da: xr.DataArray = xr.zeros_like(p_sfc_da)
    p_top_da[p_str] = p_top
    pressure = (non_vert_coords * temp[p_str]).transpose(*temp.dims)

    # Compute half-level pressure values as averages of full levels.
    p_half_inner = 0.5 * (
        pressure.isel({p_str: slice(1, None)}).values
        + pressure.isel({p_str: slice(None, -1)}).values
    )
    p_axis_num = temp.get_axis_num(p_str)
    p_half = np.concatenate(
        [
            np.expand_dims(p_top_da, p_axis_num),
            p_half_inner,
            np.expand_dims(p_sfc_da, p_axis_num),
        ],
        axis=p_axis_num,
    )

    log_p_half = np.log(p_half)

    dlog_p_half_values = np.diff(log_p_half, axis=p_axis_num)
    dlog_p_half = xr.ones_like(pressure) * dlog_p_half_values
    temp_dlog_p = temp * dlog_p_half

    # Integrate vertically.
    height = r_d / grav * temp_dlog_p.isel({p_str: slice(-1, None, -1)}).cumsum(p_str)
    height = height.isel({p_str: slice(-1, None, -1)})

    # Replace 'inf' values at TOA with NaNs and mask where
    # input temperature array is masked.
    return cast(
        xr.DataArray, xr.where(np.isfinite(height) & temp.notnull(), height, np.nan)
    )


def u_bci_2layer_qg(
    lat: ArrayLike,
    height: float = HEIGHT_TROPO,
    delta_v: float = DELTA_V,
    grav: float = GRAV_EARTH,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
) -> ArrayLike:
    """Critical zonal wind shear in 2-layer QG model for baroclinic instability."""
    return cast(
        ArrayLike,
        grav
        * height
        * delta_v
        * cosdeg(lat)
        / (2 * rot_rate * radius * sindeg(lat) ** 2),
    )


def bulk_stat_stab(
    pot_temp: xr.DataArray,
    lev_upper: float = 500,
    lev_lower: float = 850,
    p_str: str = LEV_STR,
    pot_temp_ref: float = 300.0,
) -> xr.DataArray:
    """Bulk (dry) static stability."""
    return cast(
        xr.DataArray,
        (
            pot_temp.sel({p_str: lev_upper}, method="nearest")
            - pot_temp.sel({p_str: lev_lower}, method="nearest")
        )
        / pot_temp_ref,
    )


if __name__ == "__main__":
    pass
