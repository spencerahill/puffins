#! /usr/bin/env python
"""Gradient balance and thermal wind balance."""

from typing import cast, overload

import numpy as np
import xarray as xr

from ._typing import ArrayLike, Scalar
from .calculus import lat_deriv
from .constants import (
    C_P,
    GRAV_EARTH,
    HEIGHT_TROPO,
    L_V,
    MEAN_SLP_EARTH,
    P0,
    R_D,
    R_V,
    RAD_EARTH,
    ROT_RATE_EARTH,
    THETA_REF,
)
from .dynamics import abs_vort_from_u, plan_burg_num, z_from_hypso
from .interp import zero_cross_interp
from .names import LAT_STR, LEV_STR
from .nb_utils import cosdeg, sindeg
from .thermodynamics import temp_from_equiv_pot_temp


# Angular momentum conserving wind.
def u_ang_mom_cons(
    lats: ArrayLike,
    lat_ascent: ArrayLike,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
) -> ArrayLike:
    """Angular momentum conserving zonal wind."""
    coslat = cosdeg(lats)
    return cast(
        ArrayLike, rot_rate * radius * ((cosdeg(lat_ascent) ** 2 - coslat**2) / coslat)
    )


def u_ang_mom_cons_small_ang(
    lats: ArrayLike,
    lat_ascent: ArrayLike,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
) -> ArrayLike:
    """Angular momentum conserving zonal wind in the small-angle limit"""
    return cast(
        ArrayLike,
        rot_rate * radius * (np.deg2rad(lats) ** 2 - np.deg2rad(lat_ascent) ** 2),
    )


# Fields corresponding to a specified, meridionally uniform Rossby number.
def u_unif_ro(
    lat: ArrayLike,
    lat_ascent: ArrayLike,
    ross_num: float,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
) -> ArrayLike:
    """Zonal wind for a specified uniform local Rossby number"""
    return ross_num * u_ang_mom_cons(lat, lat_ascent, rot_rate=rot_rate, radius=radius)


def u_unif_ro_small_ang(
    lat: ArrayLike,
    lat_ascent: ArrayLike,
    ross_num: float,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
) -> ArrayLike:
    """Zonal wind for a specified uniform local Rossby number, small-angle limit"""
    return ross_num * u_ang_mom_cons_small_ang(
        lat, lat_ascent, rot_rate=rot_rate, radius=radius
    )


def abs_ang_mom_unif_ro(
    lat: ArrayLike,
    lat_ascent: ArrayLike,
    ross_num: float,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
) -> ArrayLike:
    """Absolute angular momentum for a given Rossby number."""
    return cast(
        ArrayLike,
        rot_rate
        * radius**2
        * ((1 - ross_num) * cosdeg(lat) ** 2 + ross_num * cosdeg(lat_ascent) ** 2),
    )


def pot_temp_avg_unif_ro(
    lat: ArrayLike,
    lat_ascent: ArrayLike,
    pot_temp_ascent: float,
    ross_num: float,
    height: float = 10e3,
    theta_ref: float = THETA_REF,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
    grav: float = GRAV_EARTH,
) -> ArrayLike:
    """Potential temperature in gradient balance with fixed-Ro u wind."""
    prefactor = ross_num * rot_rate**2 * radius**2 / (2 * grav * height)
    coslat = cosdeg(lat)
    cosascent = cosdeg(lat_ascent)
    cos_ratio = cosascent / coslat
    return cast(
        ArrayLike,
        pot_temp_ascent
        - theta_ref
        * prefactor
        * (
            (2 - ross_num) * coslat**2
            + cosascent**2
            * (4 * (1 - ross_num) * np.log(cos_ratio) + ross_num * (cos_ratio) ** 2 - 2)
        ),
    )


def pot_temp_avg_unif_ro_small_ang(
    lat: ArrayLike,
    lat_ascent: ArrayLike,
    pot_temp_ascent: float,
    ross_num: float,
    burg_num: float | None = None,
    height: float = 10e3,
    theta_ref: float = THETA_REF,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
    grav: float = GRAV_EARTH,
) -> ArrayLike:
    """Small-angle approx for pot. temp. in balance w/ fixed-Ro u wind."""
    if burg_num is None:
        burg_num = plan_burg_num(height, grav, rot_rate, radius)
    prefactor = 0.5 * ross_num / burg_num
    lat_sq = lat**2
    lata_sq = lat_ascent**2
    return pot_temp_ascent - theta_ref * prefactor * (
        (2 - ross_num) * (1 - lat_sq)
        + (1 - lata_sq) * (ross_num * (1 - lat_sq) / (1 - lata_sq) - 2)
    )


def pot_temp_avg_unif_ro_small_ang_eq_ascent(
    lat: ArrayLike,
    pot_temp_equator: float,
    ross_num: float = 1,
    theta_ref: float = THETA_REF,
    burg_num: float | None = None,
    height: float = 10e3,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
    grav: float = GRAV_EARTH,
) -> ArrayLike:
    """Small-angle, ann. mean. pot. temp. balanced w/ fixed-Ro u"""
    if burg_num is None:
        burg_num = plan_burg_num(height, grav, rot_rate, radius)
    return pot_temp_equator - (
        theta_ref * 0.5 * ross_num / burg_num * np.deg2rad(lat) ** 4
    )


# Fields corresponding to a Rossby number varying linearly in latitude.
def u_lin_ro(
    lat: ArrayLike,
    lat_ascent: ArrayLike,
    lat_descent: ArrayLike,
    ross_ascent: float,
    ross_descent: float,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
) -> ArrayLike:
    r"""Zonal wind for Rossby number linear in sin(latitude).

    Note that, though lat_descent (\lat_d) and ross_descent (\Ro_d) are
    inputted separately, they only appear together in the combination
    (\Ro_a-\Ro_d)/lat_d, resulting in only three rather than four free
    parameters (for fixed planetary rotation rate and radius).

    """
    u_ro_a = u_unif_ro(lat, lat_ascent, ross_ascent, rot_rate=rot_rate, radius=radius)
    sinlat = sindeg(lat)
    return cast(
        ArrayLike,
        u_ro_a
        - 2
        * rot_rate
        * radius
        * (ross_ascent - ross_descent)
        / (3 * sindeg(lat_descent) * cosdeg(lat))
        * (sinlat**3 - sindeg(lat_ascent) ** 3),
    )


def u_lin_ro_small_ang(
    lat: ArrayLike,
    lat_ascent: ArrayLike,
    lat_descent: ArrayLike,
    ross_ascent: float,
    ross_descent: float,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
) -> ArrayLike:
    r"""Small-angle zonal wind for Rossby number linear in latitude.

    Note that, though lat_descent (\lat_d) and ross_descent (\Ro_d) are
    inputted separately, they only appear together in the combination
    (\Ro_a-\Ro_d)/lat_d, resulting in only three rather than four free
    parameters (for fixed planetary rotation rate and radius).

    """
    u_ro_a = u_unif_ro_small_ang(
        lat, lat_ascent, ross_ascent, rot_rate=rot_rate, radius=radius
    )
    delta_ro = ross_ascent - ross_descent
    latrad = np.deg2rad(lat)
    return cast(
        ArrayLike,
        u_ro_a
        - 2
        * delta_ro
        * rot_rate
        * radius
        / (3 * np.deg2rad(lat_descent))
        * (latrad**3 - np.deg2rad(lat_ascent) ** 3),
    )


def pot_temp_lin_ro_lata0_small_ang(
    lat: ArrayLike,
    lat_descent: float,
    ross_ascent: float,
    ross_descent: float,
    pot_temp_lat0: float,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
    theta_ref: float = THETA_REF,
    grav: float = GRAV_EARTH,
    height: float = HEIGHT_TROPO,
    burg_num: float | None = None,
) -> ArrayLike:
    """Column pot. temp. for Rossby number linear in latitude, small-angle.

    If ``burg_num`` is given it overrides the planetary Burger number
    computed from ``grav``, ``height``, ``rot_rate``, and ``radius``; pass
    ``burg_num = therm_ross_num / delta_h`` for consistency with a
    thermal-Rossby-number parameterized equal-area solution.

    """
    if burg_num is None:
        burg_num = plan_burg_num(
            grav=grav, height=height, rot_rate=rot_rate, radius=radius
        )
    delro = ross_ascent - ross_descent
    latrad = np.deg2rad(lat)
    latd_rad = np.deg2rad(lat_descent)
    pot_temp: ArrayLike = pot_temp_lat0 - theta_ref * (latrad**4) / burg_num * (
        ross_ascent / 2
        - 4 * delro * latrad / (15 * latd_rad)
        + (ross_ascent**2) * (latrad**2) / 6
        - 4 * delro * ross_ascent * (latrad**3) / (21 * latd_rad)
        + (delro**2) * (latrad**4) / (18 * latd_rad**2)
    )
    return pot_temp


# Boussinesq atmospheres.
@overload
def grad_wind_bouss(
    lats: xr.DataArray,
    height: float,
    theta_ref: float,
    dtheta_dlat: ArrayLike,
    grav: float = ...,
    rot_rate: float = ...,
    radius: float = ...,
) -> xr.DataArray: ...
@overload
def grad_wind_bouss(
    lats: ArrayLike,
    height: float,
    theta_ref: float,
    dtheta_dlat: xr.DataArray,
    grav: float = ...,
    rot_rate: float = ...,
    radius: float = ...,
) -> xr.DataArray: ...
@overload
def grad_wind_bouss(
    lats: np.ndarray,
    height: float,
    theta_ref: float,
    dtheta_dlat: np.ndarray | Scalar,
    grav: float = ...,
    rot_rate: float = ...,
    radius: float = ...,
) -> np.ndarray: ...
@overload
def grad_wind_bouss(
    lats: Scalar,
    height: float,
    theta_ref: float,
    dtheta_dlat: np.ndarray,
    grav: float = ...,
    rot_rate: float = ...,
    radius: float = ...,
) -> np.ndarray: ...
@overload
def grad_wind_bouss(
    lats: Scalar,
    height: float,
    theta_ref: float,
    dtheta_dlat: Scalar,
    grav: float = ...,
    rot_rate: float = ...,
    radius: float = ...,
) -> Scalar: ...
def grad_wind_bouss(
    lats: ArrayLike,
    height: float,
    theta_ref: float,
    dtheta_dlat: ArrayLike,
    grav: float = GRAV_EARTH,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
) -> ArrayLike:
    """Gradient wind in balance with a given potential temperature profile."""
    coslat = cosdeg(lats)
    sqrt_fac = (
        grav
        * height
        * dtheta_dlat
        / (theta_ref * rot_rate**2 * radius**2 * coslat * sindeg(lats))
    )
    return cast(ArrayLike, rot_rate * radius * coslat * ((1 - sqrt_fac) ** 0.5 - 1))


def pot_temp_avg_amc_bouss(
    lat: ArrayLike,
    lat_max: ArrayLike,
    pot_temp_max: float,
    pot_temp_ref: float,
    height: float,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
    grav: float = GRAV_EARTH,
    extra_factor: float = 1.0,
) -> ArrayLike:
    """Thermal field in gradient balance with angular momentum conserving wind.

    I.e. Eq. (7) of Lindzen and Hou 1988.  Holds for Boussinesq atmosphere.

    """
    chi = -1 * extra_factor * ((rot_rate**2 * radius**2) / (2.0 * grav * height))
    numerator = (sindeg(lat) ** 2 - sindeg(lat_max) ** 2) ** 2
    arr: ArrayLike = pot_temp_max + pot_temp_ref * chi * numerator / cosdeg(lat) ** 2
    return arr


def u_rce_minus_u_amc_bouss(
    lats: ArrayLike,
    lat_max: ArrayLike,
    height: float,
    theta_ref: float,
    dtheta_dlat: ArrayLike,
    grav: float = GRAV_EARTH,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
) -> ArrayLike:
    """RCE minus AMC zonal wind for Boussinesq fluid."""
    coslat = cosdeg(lats)
    sqrt_fac = (
        grav
        * height
        * dtheta_dlat
        / (coslat * sindeg(lats) * theta_ref * rot_rate**2 * radius**2)
    )
    return cast(
        ArrayLike,
        rot_rate
        * radius
        * coslat
        * ((1 - sqrt_fac) ** 0.5 - cosdeg(lat_max) ** 2 / coslat**2),
    )


# Convective quasi-equilibrium (CQE) atmospheres.
def grad_wind_cqe(
    theta_b: xr.DataArray,
    temp_tropo: float | None = None,
    const_stab: float | bool = False,
    c_p: float = C_P,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
    compute_temp_sfc: bool = False,
    rel_hum: float = 0.7,
    pressure: float = P0,
    p0: float = P0,
    tot_wat_mix_ratio: float | None = None,
    c_liq: float = 4185.5,
    l_v: float = L_V,
    r_d: float = R_D,
    r_v: float = R_V,
    lat_str: str = LAT_STR,
) -> xr.DataArray:
    """Gradient balanced zonal wind in convective quasi-equilibrium atmosphere.

    ``const_stab`` selects the branch by truthiness: a nonzero value is used
    directly as the (constant) stability, while ``False`` (or ``0``/``0.0``)
    routes to the ``temp_tropo`` branch, which then requires ``temp_tropo``.
    Pass a nonzero float to specify a constant stability.
    """
    numer: ArrayLike
    if const_stab:
        numer = c_p * const_stab
    else:
        if temp_tropo is None:
            raise ValueError("`temp_tropo` is required when `const_stab` is False.")
        if compute_temp_sfc:
            temp_sfc = temp_from_equiv_pot_temp(
                theta_b,
                rel_hum=rel_hum,
                pressure=p0,
                tot_wat_mix_ratio=tot_wat_mix_ratio,
                p0=p0,
                c_p=c_p,
                c_liq=c_liq,
                l_v=l_v,
                r_d=r_d,
                r_v=r_v,
            )
        else:
            temp_sfc = theta_b
        numer = c_p * (temp_sfc - temp_tropo)
    lats = theta_b[lat_str]
    coslat = cosdeg(lats)
    denom = coslat * sindeg(lats) * rot_rate**2 * radius**2
    log_theta = cast(xr.DataArray, np.log(theta_b))
    sqrt_term = (1 - (numer / denom) * lat_deriv(log_theta, lat_str)) ** 0.5
    return cast(xr.DataArray, rot_rate * radius * coslat * (-1 + sqrt_term))


def abs_vort_zero_cross_cqe(
    theta_b: xr.DataArray,
    temp_tropo: float | None = None,
    const_stab: float | bool = False,
    c_p: float = C_P,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
    compute_temp_sfc: bool = False,
    rel_hum: float = 0.7,
    pressure: float = P0,
    p0: float = P0,
    tot_wat_mix_ratio: float | None = None,
    c_liq: float = 4185.5,
    l_v: float = L_V,
    r_d: float = R_D,
    r_v: float = R_V,
    lat_str: str = LAT_STR,
) -> xr.DataArray:
    """Zero crossing of absolute vorticity in CQE atmosphere."""
    u = grad_wind_cqe(
        theta_b,
        temp_tropo=temp_tropo,
        const_stab=const_stab,
        c_p=c_p,
        rot_rate=rot_rate,
        radius=radius,
        compute_temp_sfc=compute_temp_sfc,
        rel_hum=rel_hum,
        pressure=pressure,
        p0=p0,
        tot_wat_mix_ratio=tot_wat_mix_ratio,
        c_liq=c_liq,
        l_v=l_v,
        r_d=r_d,
        r_v=r_v,
        lat_str=lat_str,
    )
    eta = abs_vort_from_u(
        u,
        rot_rate=rot_rate,
        radius=radius,
        lat_str=lat_str,
    )
    return zero_cross_interp(eta.where(eta[lat_str] > 0), lat_str)


def pot_temp_amc_cqe(
    lat: ArrayLike,
    lat_max: ArrayLike,
    pot_temp_max: float,
    vert_temp_diff: float,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
    c_p: float = C_P,
    extra_factor: float = 1.0,
) -> ArrayLike:
    """Subcloud equivalent potential temperature field in gradient balance with
    angular momentum conserving wind at the tropopause.

    I.e. Eq. (11) of Emanuel 1995, but note missing 1/2 factor in that original
    expression.

    """
    chi = extra_factor * (rot_rate**2 * radius**2) / (c_p * vert_temp_diff)
    numerator = (cosdeg(lat_max) ** 2 - cosdeg(lat) ** 2) ** 2
    arr: ArrayLike = pot_temp_max * np.exp(-0.5 * chi * numerator / cosdeg(lat) ** 2)
    return arr


def u_rce_minus_u_amc_cqe(
    lat_max: ArrayLike,
    theta_b: xr.DataArray,
    temp_tropo: float | None = None,
    const_stab: float | bool = False,
    c_p: float = C_P,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
    lat_str: str = LAT_STR,
) -> xr.DataArray:
    """Criticality in terms of zonal wind for continuous, CQE fluid.

    Assumes surface temperature equals theta_b.

    """
    lats = theta_b[lat_str]
    u_rce = grad_wind_cqe(
        theta_b,
        temp_tropo=temp_tropo,
        const_stab=const_stab,
        c_p=c_p,
        rot_rate=rot_rate,
        radius=radius,
        lat_str=lat_str,
    )
    u_amc = u_ang_mom_cons(lats, lat_max, rot_rate, radius)
    return cast(xr.DataArray, u_rce - u_amc)


# Non-Boussinesq, non-CQE atmospheres, in pressure coordinates.
def thermal_wind_shear_p_coords(
    temp: xr.DataArray,
    pressure: ArrayLike,
    radius: float = RAD_EARTH,
    rot_rate: float = ROT_RATE_EARTH,
    r_d: float = R_D,
    lat_str: str = LAT_STR,
) -> xr.DataArray:
    """Thermal wind shear for data on the sphere in pressure coordinates.

    Assumes that pressure is in Pa, not hPa.

    Returns thermal wind shear in pressure coordinates, i.e. du/dp, with units
    m s^-1 Pa^-1.


    """
    return cast(
        xr.DataArray,
        -1
        * r_d
        * lat_deriv(temp, lat_str)
        / (2 * rot_rate * radius * pressure * sindeg(temp[lat_str])),
    )


def thermal_wind_p_coords(
    height: xr.DataArray | None = None,
    temp: xr.DataArray | None = None,
    p_sfc: float = MEAN_SLP_EARTH,
    p_top: float = 1.0,
    radius: float = RAD_EARTH,
    rot_rate: float = ROT_RATE_EARTH,
    r_d: float = R_D,
    grav: float = GRAV_EARTH,
    lat_str: str = LAT_STR,
    p_str: str = LEV_STR,
) -> xr.DataArray:
    """Thermal wind for data on the sphere in pressure coordinates.

    Assumes that pressure is in Pa, not hPa.

    Returns thermal wind, in m/s.

    """
    if height is None:
        if temp is None:
            raise ValueError("One of 'height' or 'temp' must be provided.")
        height = z_from_hypso(
            temp, p_sfc=p_sfc, p_top=p_top, r_d=r_d, grav=grav, p_str=p_str
        )
    sinlat = sindeg(height[lat_str])
    return cast(
        xr.DataArray,
        -1 * grav * lat_deriv(height, lat_str) / (2 * rot_rate * radius * sinlat),
    )


def grad_wind_p_coords(
    height: xr.DataArray | None = None,
    temp: xr.DataArray | None = None,
    p_sfc: float = MEAN_SLP_EARTH,
    p_top: float = 1.0,
    radius: float = RAD_EARTH,
    rot_rate: float = ROT_RATE_EARTH,
    grav: float = GRAV_EARTH,
    r_d: float = R_D,
    lat_str: str = LAT_STR,
    p_str: str = LEV_STR,
) -> xr.DataArray:
    """Gradient wind for data on the sphere in pressure coordinates.

    Assumes that pressure is in Pa, not hPa.

    """
    if height is None:
        if temp is None:
            raise ValueError("One of 'height' or 'temp' must be provided.")
        height = z_from_hypso(
            temp, p_sfc=p_sfc, p_top=p_top, r_d=r_d, grav=grav, p_str=p_str
        )
    lat = height[lat_str]
    sinlat = sindeg(lat)
    coslat = cosdeg(lat)

    sqrt_arg = 1 - grav * lat_deriv(height, lat_str) / (
        sinlat * coslat * (rot_rate * radius) ** 2
    )
    return cast(
        xr.DataArray,
        (rot_rate * radius * coslat * (np.sqrt(sqrt_arg) - 1)).transpose(*height.dims),
    )


if __name__ == "__main__":
    pass
