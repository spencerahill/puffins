#! /usr/bin/env python
"""Gradient balance and thermal wind balance."""

import numpy as np

from .constants import (
    C_P,
    GRAV_EARTH,
    L_V,
    P0,
    R_D,
    R_V,
    RAD_EARTH,
    ROT_RATE_EARTH,
    THETA_REF,
)
from .names import LAT_STR, LEV_STR
from .nb_utils import cosdeg, sindeg
from .calculus import avg_logp_weighted, lat_deriv
from .dynamics import abs_vort_from_u
from .thermodynamics import temp_from_equiv_pot_temp


# Angular momentum conserving wind.
def u_ang_mom_cons(lats, max_lat, rot_rate=ROT_RATE_EARTH, radius=RAD_EARTH):
    """Angular momentum conserving zonal wind."""
    coslat = cosdeg(lats)
    return rot_rate*radius*coslat*((cosdeg(max_lat) / coslat)**2 - 1)


def u_given_ro(lat, lat_ascent, ross_num,
               rot_rate=ROT_RATE_EARTH, radius=RAD_EARTH):
    """Absolute angular momentum for a given Rossby number."""
    return ross_num*u_ang_mom_cons(lat, lat_ascent,
                                   rot_rate=rot_rate,
                                   radius=radius)


def abs_ang_mom_given_ro(lat, lat_ascent, ross_num,
                         rot_rate=ROT_RATE_EARTH, radius=RAD_EARTH):
    """Absolute angular momentum for a given Rossby number."""
    return rot_rate*radius**2*((1 - ross_num)*cosdeg(lat)**2 +
                               ross_num*cosdeg(lat_ascent)**2)


def pot_temp_avg_given_ro(lat, lat_ascent, pot_temp_ascent, ross_num,
                          height=10e3, theta_ref=THETA_REF,
                          rot_rate=ROT_RATE_EARTH, radius=RAD_EARTH,
                          grav=GRAV_EARTH):
    """Potential temperature in gradient balance with fixed-Ro u wind."""
    prefactor = ross_num*rot_rate**2*radius**2 / (2*grav*height)
    coslat = cosdeg(lat)
    cosascent = cosdeg(lat_ascent)
    cos_ratio = cosascent / coslat
    return pot_temp_ascent - theta_ref*prefactor*(
        (2 - ross_num)*coslat**2 + cosascent**2*(
            4*(1 - ross_num)*np.log(cos_ratio) +
            ross_num*(cos_ratio)**2 - 2))


# Boussinesq atmospheres.
def grad_wind_bouss(lats, height, theta_ref, dtheta_dlat,
                    grav=GRAV_EARTH, rot_rate=ROT_RATE_EARTH,
                    radius=RAD_EARTH):
    """Gradient wind in balance with a given potential temperature profile."""
    coslat = cosdeg(lats)
    sqrt_fac = (grav*height*dtheta_dlat /
                (theta_ref*rot_rate**2*radius**2*coslat*sindeg(lats)))
    return rot_rate*radius*coslat*((1 - sqrt_fac)**0.5 - 1)


def pot_temp_avg_amc_bouss(lat, lat_max, pot_temp_max, pot_temp_ref, height,
                           rot_rate=ROT_RATE_EARTH, radius=RAD_EARTH,
                           grav=GRAV_EARTH, extra_factor=1.0):
    """Thermal field in gradient balance with angular momentum conserving wind.

    I.e. Eq. (7) of Lindzen and Hou 1988.  Holds for Boussinesq atmosphere.

    """
    chi = -1 * extra_factor * ((rot_rate ** 2 * radius ** 2) /
                               (2.0 * grav * height))
    numerator = (sindeg(lat) ** 2 - sindeg(lat_max) ** 2) ** 2
    arr = pot_temp_max + pot_temp_ref * chi * numerator / cosdeg(lat) ** 2
    return arr


def u_rce_minus_u_amc_bouss(lats, lat_max, height, theta_ref, dtheta_dlat,
                            grav=GRAV_EARTH, rot_rate=ROT_RATE_EARTH,
                            radius=RAD_EARTH):
    """RCE minus AMC zonal wind for Boussinesq fluid."""
    coslat = cosdeg(lats)
    sqrt_fac = (grav*height*dtheta_dlat /
                (coslat*sindeg(lats)*theta_ref*rot_rate**2*radius**2))
    return rot_rate*radius*coslat * ((1 - sqrt_fac)**0.5 -
                                     cosdeg(lat_max)**2 / coslat**2)


# Convective quasi-equilibrium (CQE) atmospheres.
def grad_wind_cqe(
    theta_b,
    temp_tropo=None,
    const_stab=False,
    c_p=C_P,
    rot_rate=ROT_RATE_EARTH,
    radius=RAD_EARTH,
    compute_temp_sfc=False,
    rel_hum=0.7,
    pressure=P0,
    p0=P0,
    tot_wat_mix_ratio=None,
    c_liq=4185.5,
    l_v=L_V,
    r_d=R_D,
    r_v=R_V,
    lat_str=LAT_STR,
):
    """Gradient balanced zonal wind in convective quasi-equilibrium atmosphere.

    """
    if const_stab:
        numer = c_p*const_stab
    else:
        if compute_temp_sfc:
            temp_sfc = temp_from_equiv_pot_temp(
                theta_b, rel_hum=rel_hum, pressure=p0,
                tot_wat_mix_ratio=tot_wat_mix_ratio, p0=p0, c_p=c_p,
                c_liq=c_liq, l_v=l_v, r_d=r_d, r_v=r_v)
        else:
            temp_sfc = theta_b
        numer = c_p*(temp_sfc - temp_tropo)
    lats = theta_b[lat_str]
    coslat = cosdeg(lats)
    denom = coslat*sindeg(lats)*rot_rate**2*radius**2
    sqrt_term = (1 - (numer/denom)*lat_deriv(np.log(theta_b), lat_str))**0.5
    return rot_rate*radius*coslat*(-1 + sqrt_term)


def abs_vort_zero_cross_cqe(theta_b, temp_tropo=None, const_stab=False,
                            c_p=C_P, rot_rate=ROT_RATE_EARTH, radius=RAD_EARTH,
                            compute_temp_sfc=False, rel_hum=0.7, pressure=P0,
                            p0=P0, tot_wat_mix_ratio=None, c_liq=4185.5,
                            l_v=L_V, r_d=R_D, r_v=R_V, lat_str=LAT_STR):
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
        lat_str=lat_str
    )
    eta = abs_vort_from_u(
        u,
        rot_rate=rot_rate,
        radius=radius,
        lat_str=lat_str,
    )
    return zero_cross_nh(eta, lat_str=lat_str)


def pot_temp_amc_cqe(lat, lat_max, pot_temp_max, vert_temp_diff,
                     rot_rate=ROT_RATE_EARTH, radius=RAD_EARTH,
                     c_p=C_P, extra_factor=1.):
    """Subcloud equivalent potential temperature field in gradient balance with
    angular momentum conserving wind at the tropopause.

    I.e. Eq. (11) of Emanuel 1995, but note missing 1/2 factor in that original
    expression.

    """
    chi = extra_factor * (rot_rate**2 * radius**2) / (c_p * vert_temp_diff)
    numerator = (cosdeg(lat_max)**2 - cosdeg(lat)**2)**2
    arr = pot_temp_max * np.exp(-0.5 * chi * numerator / cosdeg(lat)**2)
    return arr


def u_rce_minus_u_amc_cqe(lat_max, theta_b, temp_tropo=None,
                          const_stab=False, c_p=C_P, rot_rate=ROT_RATE_EARTH,
                          radius=RAD_EARTH, plus_solution=True,
                          lat_str=LAT_STR):
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
        plus_solution=plus_solution,
        lat_str=lat_str,
    )
    u_amc = u_ang_mom_cons(lats, lat_max, rot_rate, radius)
    return u_rce - u_amc


# Non-Boussinesq, non-CQE atmospheres, in pressure coordinates.
def thermal_wind_p_coords(temp_avg, pressure, p0=1e3, lat_str=LAT_STR,
                          radius=RAD_EARTH, rot_rate=ROT_RATE_EARTH, r_d=R_D):
    if np.max(pressure) > 2e3:
        pressure *= 1e-2
    return (-r_d*lat_deriv(temp_avg, lat_str)*np.log(p0/pressure) /
            (2*rot_rate*radius*sindeg(temp_avg[lat_str])))


def grad_wind_p_coords(temp, p_half, pressure, p0=1e3, lat_str=LAT_STR,
                       p_str=LEV_STR, radius=RAD_EARTH,
                       rot_rate=ROT_RATE_EARTH, r_d=R_D):
    lat = temp[lat_str]
    coslat = cosdeg(lat)
    temp_avg = avg_logp_weighted(temp, p_half, pressure, p_str=p_str)
    dtemp_avg_dlat = lat_deriv(temp_avg, lat_str)
    if pressure.max() > 1e4:
        raise ValueError("Expected pressures in hPa; got Pa")
    sqrt_arg = 1 - (r_d*np.log(p0/pressure)*dtemp_avg_dlat /
                    (coslat*sindeg(lat)*(rot_rate*radius)**2))
    return rot_rate * radius * coslat * (np.sqrt(sqrt_arg) - 1)


if __name__ == '__main__':
    pass
