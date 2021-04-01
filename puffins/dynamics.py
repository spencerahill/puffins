#! /usr/bin/env python
"""Fundamental atmospheric dynamical quantities."""

from .constants import GRAV_EARTH, RAD_EARTH, ROT_RATE_EARTH, THETA_REF
from .names import LAT_STR, LEV_STR
from .nb_utils import cosdeg, sindeg
from .calculus import lat_deriv


def plan_burg_num(height, grav=GRAV_EARTH, rot_rate=ROT_RATE_EARTH,
                  radius=RAD_EARTH):
    """Planetary Burger number"""
    return height * grav / (rot_rate * radius)**2


def therm_ross_num(delta_h, height, grav=GRAV_EARTH,
                   rot_rate=ROT_RATE_EARTH, radius=RAD_EARTH):
    """Thermal Rossby number."""
    return delta_h * plan_burg_num(height, grav=grav, rot_rate=rot_rate,
                                   radius=radius)


def abs_ang_mom(u, lat=None, radius=RAD_EARTH, rot_rate=ROT_RATE_EARTH,
                lat_str=LAT_STR):
    """Absolute angular momentum."""
    if lat is None:
        lat = u[lat_str]
    coslat = cosdeg(lat)
    return radius*coslat*(rot_rate*radius*coslat + u)


def abs_vort_vert_comp(abs_ang_mom, radius=RAD_EARTH, lat_str=LAT_STR):
    """Vertical component of absolute vorticity (in axisymmetric case)."""
    return (-1*lat_deriv(abs_ang_mom, lat_str) /
            (radius**2 * cosdeg(abs_ang_mom[lat_str])))


def abs_vort_from_u(u, rot_rate=ROT_RATE_EARTH, radius=RAD_EARTH,
                    lat_str=LAT_STR):
    """Absolute vorticity computed from zonal wind."""
    lats = u[lat_str]
    sinlat = sindeg(lats)
    coslat = cosdeg(lats)
    return ((u*sinlat)/(radius*coslat) - lat_deriv(u, lat_str)/radius +
            2*rot_rate*sinlat)


def brunt_vaisala_freq(lat, dtheta_dz, theta_ref=THETA_REF, grav=GRAV_EARTH):
    return (grav * dtheta_dz / theta_ref)**0.5


def rossby_radius(lat, dtheta_dz, height, theta_ref=THETA_REF,
                  grav=GRAV_EARTH, rot_rate=ROT_RATE_EARTH):
    """Rossby radius of deformation"""
    return brunt_vaisala_freq(lat, dtheta_dz, theta_ref=theta_ref,
                              grav=grav) * height / (2*rot_rate*sindeg(lat))


def coriolis_param(lat, rot_rate=ROT_RATE_EARTH):
    """Coriolis parameter, i.e. 'f'."""
    return 2*rot_rate*sindeg(lat)


def zonal_fric_inferred_steady(u_merid_flux, u_vert_flux, vwind,
                               radius=RAD_EARTH, rot_rate=ROT_RATE_EARTH,
                               vert_str=LEV_STR, lat_str=LAT_STR):
    """Steady-state zonal friction inferred from flux div + Coriolis."""
    lats = u_merid_flux[lat_str]
    coslat = cosdeg(lats)
    duv_dlat_term = (lat_deriv(u_merid_flux*coslat, lat_str) /
                     (radius*coslat**2))
    duw_dvert_term = u_vert_flux.differentiate(vert_str)

    u_flux_div = duv_dlat_term + duw_dvert_term
    f = coriolis_param(lats, rot_rate=rot_rate)
    return u_flux_div - f*vwind


if __name__ == '__main__':
    pass
