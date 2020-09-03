#! /usr/bin/env python
"""Atmospheres with fixed lapse rate and/or tropopause temperature."""

import numpy as np
from scipy.optimize import brentq
import xarray as xr

from .constants import (
    C_P,
    GRAV_EARTH,
    HEIGHT_TROPO,
    P0,
    REL_HUM,
    RAD_EARTH,
    ROT_RATE_EARTH,
    TEMP_TROPO,
    THETA_REF,
)
from .names import LAT_STR
from .nb_utils import cosdeg, sindeg
from .calculus import lat_deriv
from .num_solver import brentq_solver_sweep_param
from .grad_bal import pot_temp_amc_cqe
from .thermodynamics import temp_from_equiv_pot_temp


DTHETA_DTS_MOIST = 1.4
GAMMA_MOIST = 0.65


# Tropopause height.
def trop_height_fixed_tt(temp_sfc, temp_tropo=TEMP_TROPO, gamma=GAMMA_MOIST,
                         grav=GRAV_EARTH, c_p=C_P):
    """Tropopause height for fixed tropopause temperature and lapse rate."""
    return c_p*(temp_sfc - temp_tropo) / (grav*gamma)


# Potential temperature and surface temperature fields.
def pot_temp_fixed_lapse_rate(temp_sfc, height, gamma=GAMMA_MOIST,
                              grav=GRAV_EARTH, c_p=C_P):
    """Potential temperature at a given height if lapse rate is constant."""
    invgam = 1.0 / gamma
    return (temp_sfc**invgam *
            (temp_sfc - gamma*grav/c_p*height)**(1 - invgam))


def pot_temp_tropopause_fixed_tt(temp_sfc, temp_tropo=TEMP_TROPO,
                                 gamma=GAMMA_MOIST):
    """Potential temperature at the tropopause, given a fixed lapse rate."""
    invgam = 1. / gamma
    return temp_sfc**invgam * temp_tropo**(1 - invgam)


def pot_temp_avg_fixed_lapse_rate(temp_sfc, height, gamma=GAMMA_MOIST,
                                  grav=GRAV_EARTH, c_p=C_P):
    """Vertically averaged potential temperature from surface to given height.

    Assumes a fixed lapse rate from the given surface temperature to the given
    height.

    """
    invgam = 1.0 / gamma
    gamma_dry = grav / c_p
    return ((temp_sfc**invgam) / ((2 - invgam)*gamma*gamma_dry*height) *
            (temp_sfc**(2 - invgam) -
             (temp_sfc - gamma*gamma_dry*height)**(2 - invgam)))


def pot_temp_avg_fixed_lapse_rate_temp_tropo(temp_sfc, temp_tropo=TEMP_TROPO,
                                             gamma=GAMMA_MOIST):
    """Troposphere-averaged potential temperature.

    Assumes tropopause occurs at fixed temperature and that the lapse rate =
    gamma*Gamma_dry.

    """
    invgam = 1.0 / gamma
    return (
        (gamma / (2 * gamma - 1))
        * (temp_sfc**invgam / (temp_sfc - temp_tropo))
        * (temp_sfc**(2 - invgam) - temp_tropo**(2 - invgam))
    )


def temp_sfc_fixed_lapse_rate_height_tropo(pot_temp_avg,
                                           height_tropo=HEIGHT_TROPO,
                                           gamma=GAMMA_MOIST,
                                           dtheta_dts=DTHETA_DTS_MOIST,
                                           grav=GRAV_EARTH, c_p=C_P):
    """Surface temperature distribution given a troposphere-averaged
    potential temperature distribution and assuming a uniform
    lapse rate and uniform tropopause height.

    """

    if gamma == 1.0:
        return pot_temp_avg

    def func(tsfc, theta):
        invgam = 1.0 / gamma
        gamma_dry = grav / c_p
        return theta*(2 - invgam)*gamma*gamma_dry*height_tropo - (
            (tsfc**invgam) * (tsfc**(2 - invgam) - (
                tsfc - gamma*gamma_dry*height_tropo)**(2 - invgam)))

    pot_temp_is_scalar = np.isscalar(pot_temp_avg)
    if pot_temp_is_scalar:
        pot_temp_array = [pot_temp_avg]
    else:
        pot_temp_array = pot_temp_avg
    solutions = []
    for pta in pot_temp_array:
        guess_lower = 0.5 * pta / dtheta_dts
        guess_upper = 1.5 * pta / dtheta_dts
        solutions.append(brentq(func, guess_lower, guess_upper, args=(pta,)))
    if pot_temp_is_scalar:
        return solutions[0]
    return np.ones_like(pot_temp_avg)*solutions


def temp_sfc_fixed_lapse_rate_temp_tropo(pot_temp_avg, temp_tropo=TEMP_TROPO,
                                         gamma=GAMMA_MOIST,
                                         dtheta_dts=DTHETA_DTS_MOIST,
                                         grav=GRAV_EARTH, c_p=C_P):
    """Surface temperature distribution given a troposphere-averaged
    potential temperature distribution and assuming a uniform
    lapse rate and uniform tropopause temperature.

    """
    if gamma == 1.0:
        return pot_temp_avg

    def func(tsfc, theta):
        return theta - pot_temp_avg_fixed_lapse_rate_temp_tropo(
            tsfc,
            temp_tropo=temp_tropo,
            gamma=gamma,
        )

    pot_temp_is_scalar = np.isscalar(pot_temp_avg)
    if pot_temp_is_scalar:
        pot_temp_array = [pot_temp_avg]
    else:
        pot_temp_array = pot_temp_avg

    solutions = []
    for pta in pot_temp_array:
        guess_lower = 0.5 * pta / dtheta_dts
        guess_upper = 1.5 * pta / dtheta_dts
        solutions.append(brentq(func, guess_lower, guess_upper, args=(pta,)))
    if pot_temp_is_scalar:
        return solutions[0]
    return xr.ones_like(pot_temp_avg)*solutions


def dpot_temp_avg_dtemp_sfc_fixed_lapse_rate(temp_sfc, gamma=GAMMA_MOIST,
                                             temp_tropo=TEMP_TROPO):
    """Derivative of troposphere-averaged potential temperature
     w/r/t surface temperature, assuming fixed lapse rate equal
     to gamma*(dry adiabatic lapse rate)

    """
    ts = temp_sfc
    tt = temp_tropo
    invgam = 1.0 / gamma
    leading_factor = ts**(invgam - 1) / ((2 * gamma - 1) * (ts - tt))
    term1 = (2*gamma - 1)*ts**(2 - invgam)
    term2 = ((1 - gamma)*ts - tt)*(ts**(2 - invgam) -
                                   tt**(2 - invgam)) / (ts - tt)
    return leading_factor * (term1 + term2)


# Boussinesq atmospheres.
def grad_wind_bouss_fixed_temp_tropo(
    lat,
    pot_temp_avg,
    theta_ref=THETA_REF,
    temp_tropo=TEMP_TROPO,
    gamma=GAMMA_MOIST,
    dtheta_dts=DTHETA_DTS_MOIST,
    rot_rate=ROT_RATE_EARTH,
    radius=RAD_EARTH,
    grav=GRAV_EARTH,
    c_p=C_P,
    lat_str=LAT_STR,
    compute_temp_sfc=True,
):
    """Gradient wind if tropopause temperature and lapse rate are uniform.

    Assumes fixed lapse rate equal to `gamma` times the dry adiabatic lapse
    rate.  If `compute_temp_sfc` is `True`, compute the surface temperature
    explicitly from the average potential temperature and lapse rate.  If
    `False`, approximate the surface temperature as equal to the average
    potential temperature.

    """
    if compute_temp_sfc:
        temp_sfc = temp_sfc_fixed_lapse_rate_temp_tropo(
            pot_temp_avg,
            temp_tropo=temp_tropo,
            gamma=gamma,
            dtheta_dts=dtheta_dts,
            grav=grav,
            c_p=c_p,
        )
        temp_drop = temp_sfc - temp_tropo
    else:
        temp_sfc = pot_temp_avg
        temp_drop = pot_temp_avg - temp_tropo
    dtemp_sfc_dlat = lat_deriv(temp_sfc, lat_str)

    pot_temp_tropo = pot_temp_tropopause_fixed_tt(
        temp_sfc,
        temp_tropo=temp_tropo,
        gamma=gamma,
    )
    pot_temp_drop = pot_temp_tropo - pot_temp_avg
    dtheta_dlat = lat_deriv(pot_temp_avg, lat_str)

    coslat = cosdeg(lat)
    leading_factor = c_p / (
        coslat * sindeg(lat) * gamma * theta_ref * rot_rate**2 * radius**2
    )

    sqrt_term = 1 - leading_factor * (temp_drop*dtheta_dlat -
                                      pot_temp_drop*dtemp_sfc_dlat)
    return rot_rate*radius*coslat*(np.sqrt(sqrt_term) - 1)


def temp_sfc_amc_fixed_tt_bouss(lat, lat_ascent, temp_sfc_ascent,
                                theta_ref=THETA_REF, gamma=GAMMA_MOIST,
                                dtheta_dts=DTHETA_DTS_MOIST,
                                temp_tropo=TEMP_TROPO,
                                rot_rate=ROT_RATE_EARTH, radius=RAD_EARTH,
                                c_p=C_P):
    """AMC surface temperature, assuming fixed lapse rate and tropopause temp.

    Lapse rate is equal to gamma*(dry lapse rate).

    Treats d(\hat\theta)/d(\Tsfc) as a constant, as it only varies from roughly
    1.3 to 1.5 for Tsfc ranging from 200 to 500 K.  But this value decreases
    with gamma.  For gamma=1, it is exactly 1 for all Tsfc.

    """
    if gamma == 1.0:
        dtheta_dts = 1.0
    chi = gamma * theta_ref * (rot_rate * radius)**2 / (c_p * dtheta_dts)
    lat_factor = (cosdeg(lat_ascent)**2 - cosdeg(lat)**2)**2 / cosdeg(lat)**2
    return temp_tropo + np.sqrt((temp_sfc_ascent - temp_tropo)**2 -
                                chi*lat_factor)


def pot_temp_avg_amc_fixed_tt_bouss(lat, lat_ascent, temp_sfc_ascent,
                                    theta_ref=THETA_REF, gamma=GAMMA_MOIST,
                                    dtheta_dts=DTHETA_DTS_MOIST,
                                    temp_tropo=TEMP_TROPO,
                                    rot_rate=ROT_RATE_EARTH, radius=RAD_EARTH,
                                    c_p=C_P):
    """Angular momentum conserving, column-integrated, potential temperature
    assuming fixed tropopause temperature and fixed lapse rate.

    """
    temp_sfc = temp_sfc_amc_fixed_tt_bouss(
        lat,
        lat_ascent,
        temp_sfc_ascent,
        theta_ref=theta_ref,
        gamma=gamma,
        temp_tropo=temp_tropo,
        dtheta_dts=dtheta_dts,
        rot_rate=rot_rate,
        radius=radius,
        c_p=c_p,
    )
    if gamma == 1:
        return temp_sfc
    return pot_temp_avg_fixed_lapse_rate_temp_tropo(
        temp_sfc, temp_tropo=temp_tropo, gamma=gamma
    )


# CQE atmosphere.
def _theta_b_amc_root(guess, lat, theta_ascent, lat_ascent, temp_tropo,
                      rot_rate=ROT_RATE_EARTH, radius=RAD_EARTH, c_p=C_P):
    """Solve to get critical temperature with fixed tropopause temperature."""
    cos2lat = cosdeg(lat)**2
    lat_term = ((cosdeg(lat_ascent)**2 - cos2lat)**2) / cos2lat
    rhs = theta_ascent - (((rot_rate**2 * radius**2) / (2.*c_p)) * lat_term)
    return guess - temp_tropo*np.log(guess / theta_ascent) - rhs


def _theta_b_amc_root_compute_temp_sfc(guess, lat, theta_ascent, lat_ascent,
                                       temp_tropo, rel_hum=REL_HUM,
                                       pressure=P0, rot_rate=ROT_RATE_EARTH,
                                       radius=RAD_EARTH, c_p=C_P):
    """Solve to get critical temperature with fixed tropopause temperature."""
    cos2lat = cosdeg(lat)**2
    rhs = -(((rot_rate**2 * radius**2) / (2.*c_p)) *
            ((cosdeg(lat_ascent)**2 - cos2lat)**2) / cos2lat)
    temp_sfc = temp_from_equiv_pot_temp(guess, rel_hum=1., pressure=700e2)
    lhs = (temp_sfc*(1 - theta_ascent / guess) +
           temp_tropo*np.log(theta_ascent / guess))
    return lhs - rhs


def pot_temp_amc_cqe_fixed_tt(lats, bound_guess_range, pot_temp_ascent,
                              lat_ascent, temp_tropo=TEMP_TROPO,
                              rel_hum=REL_HUM, pressure=P0,
                              rot_rate=ROT_RATE_EARTH, radius=RAD_EARTH,
                              c_p=C_P, compute_temp_sfc=False):
    """Angular momentum conserving thermal field, fixed tropopause temperature.

    Solved numerically using the Brent (1973) root finding algorithm, as
    implemented in scipy's ``optimize.brentq`` function.

    Parameters
    ----------

    lats : array-like
        Latitudes for which to solve for the critical temperature.
    bound_guess_range : array-like
        Temperatures to sample as initial guesses for the root-finder
    pot_temp_ascent : float
        Temperature value at its global maximum
    lat_ascent : float
        Latitude, in degrees, where `pot_temp_ascent` occurs.
    temp_tropo : float
        Tropopause temperature, which is taken to be constant in latitude.

    Returns
    -------

    crit_temps : xarray.DataArray
        Array of the numerical solution at each latitude value in `lats`

    """
    if np.isscalar(lats):
        lats = [lats]
    init_guess = pot_temp_amc_cqe(
        lats[0],
        lat_ascent,
        pot_temp_ascent,
        pot_temp_ascent - temp_tropo,
        rot_rate=rot_rate,
        radius=radius,
        c_p=c_p,
    )
    if compute_temp_sfc:
        funcargs = (
            pot_temp_ascent,
            lat_ascent,
            temp_tropo,
            rel_hum,
            pressure,
            rot_rate,
            radius,
            c_p,
        )
        func = _theta_b_amc_root_compute_temp_sfc
    else:
        funcargs = (
            pot_temp_ascent,
            lat_ascent,
            temp_tropo,
            rot_rate,
            radius,
            c_p,
        )
        func = _theta_b_amc_root
    return brentq_solver_sweep_param(func, lats, init_guess, bound_guess_range,
                                     funcargs=funcargs)


if __name__ == '__main__':
    pass
