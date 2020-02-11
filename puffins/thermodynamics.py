#! /usr/bin/env python
"""Thermodynamic quantities."""

import numpy as np
from scipy.optimize import brentq

from .constants import (
    C_P,
    C_PV,
    EPSILON,
    GRAV_EARTH,
    L_V,
    P0,
    R_D,
    R_V,
    REL_HUM,
)


def sat_vap_press_tetens_kelvin(temp):
    """Saturation vapor pressure using Tetens equation.

    Note: unlike original Tetens expression, temperature should be in Kelvin,
    NOT degrees Celsius.  And result has units Pa, not kPa as in original
    version.

    """
    a = 61.078
    b = 17.27
    c = -35.85
    return a*np.exp(b*(temp - 273.15) / (temp + c))


def saturation_entropy(temp, pressure=P0, sat_vap_press=None,
                       c_p=C_P, r_d=R_D, l_v=L_V):
    """Saturation entropy, from Emanuel and Rotunno 2011, JAS.

    Saturation vapor pressure can be provided as `sat_vap_press`, otherwise it
    is computed using the Tetens equation.

    If `pressure` is not provided (units Pa), it is assumed to be 1e5 Pa,
    i.e. 1000 hPa.

    Neglects difference between mixing ratio and specific humidity.

    Note that this expression is not identical to moist entropy computed as
    c_p*log(theta_e_sat), where theta_e_sat is equivalent potential temperature
    computed at saturation (i.e. with relative humidity = 1).  This expression
    is much lower, i.e. around 2500 J/kg/K for Earth-like near-surface
    conditions, compared to 5500-6000 J/kg/K for the cp*log(theta_e_sat)
    version.

    """
    if sat_vap_press is None:
        sat_vap_press = sat_vap_press_tetens_kelvin(temp)
    sat_spec_hum = sat_vap_press / pressure
    return (c_p*np.log(temp) - r_d*np.log(pressure) +
            l_v*sat_spec_hum / temp)


def dsat_entrop_dtemp_approx(temp, pressure=P0, c_p=C_P, r_v=R_V, l_v=L_V):
    sat_vap_press = sat_vap_press_tetens_kelvin(temp)
    sat_spec_hum = sat_vap_press / pressure
    return (c_p + l_v*sat_spec_hum*(l_v/(r_v*temp) - 1)/temp) / temp


def water_vapor_mixing_ratio(vapor_press, pressure, epsilon=EPSILON):
    return epsilon*vapor_press / (pressure - vapor_press)


def equiv_pot_temp(temp, rel_hum, pressure,
                   tot_wat_mix_ratio=None, p0=P0, c_p=C_P, c_liq=4185.5,
                   l_v=L_V, r_d=R_D, r_v=R_V):
    """Equivalent potential temperature."""
    sat_vap_press = sat_vap_press_tetens_kelvin(temp)
    vapor_pressure = rel_hum*sat_vap_press
    pressure_dry = pressure - vapor_pressure
    vap_mix_ratio = water_vapor_mixing_ratio(vapor_pressure, pressure)
    if tot_wat_mix_ratio is None:
        denom = c_p
    else:
        denom = c_p + c_liq*tot_wat_mix_ratio
    return (temp*(p0/pressure_dry)**(r_d/denom) *
            rel_hum**(r_v*vap_mix_ratio / denom) *
            np.exp(l_v*vap_mix_ratio / (denom*temp)))


def temp_from_equiv_pot_temp(theta_e, rel_hum=0.7, pressure=P0,
                             tot_wat_mix_ratio=None, p0=P0, c_p=C_P,
                             c_liq=4185.5, l_v=L_V, r_d=R_D, r_v=R_V):
    """Temperature, given the equivalent potential temperature."""
    def func(temp, theta):
        sat_vap_press = sat_vap_press_tetens_kelvin(temp)
        vapor_pressure = rel_hum*sat_vap_press
        vap_mix_ratio = water_vapor_mixing_ratio(vapor_pressure, pressure)
        if tot_wat_mix_ratio is None:
            denom = c_p
        else:
            denom = c_p + c_liq*tot_wat_mix_ratio
        return (theta*(pressure/p0)**(r_d/denom) /
                rel_hum**(r_v*vap_mix_ratio / denom) -
                temp*np.exp(l_v*vap_mix_ratio / (denom*temp)))

    pot_temp_is_scalar = np.isscalar(theta_e)
    pot_temp_is_len0_arr = not pot_temp_is_scalar and not theta_e.shape
    if pot_temp_is_scalar:
        pot_temp_array = [theta_e]
    elif pot_temp_is_len0_arr:
        pot_temp_array = [float(theta_e)]
    else:
        pot_temp_array = theta_e

    solutions = []
    for pta in pot_temp_array:
        # Start with guess range narrowly bounding the the theta_e value, and
        # then progressively widen if the function doesn't change sign within
        # the bound.  Ensures that the zero crossing is as close as possible
        # to the neighborhood of the theta_e value, so that the algorithm will
        # converge and we don't accidentally catch another irrelevant zero
        # crossing by mistake.
        for factor in np.arange(0.01, 0.99, 0.01):
            guess_lower = (1 - factor)*pta
            guess_upper = (1 + factor)*pta
            try:
                sol = brentq(func, guess_lower, guess_upper, args=(pta,))
            except ValueError:
                pass
            else:
                # Temperature is always less than equiv. pot. temp., meaning
                # that the procedure failed if the opposite occurs.  Mask it.
                if sol > pta:
                    solutions.append(np.nan)
                else:
                    solutions.append(sol)
                break
    # If no solution found, just mask.  Otherwise, return same type/shape as
    # original input data.
    if len(solutions) == 0:
        return np.nan
    elif pot_temp_is_scalar or pot_temp_is_len0_arr:
        return solutions[0]
    else:
        return np.ones_like(theta_e)*solutions


def moist_entropy(temp, rel_hum, pressure, tot_wat_mix_ratio=None, p0=P0,
                  c_p=C_P, c_liq=4185.5, l_v=L_V, r_d=R_D, r_v=R_V):
    """Moist entropy."""
    return c_p*np.log(equiv_pot_temp(
        temp,
        rel_hum,
        pressure,
        tot_wat_mix_ratio=tot_wat_mix_ratio,
        p0=p0,
        c_p=c_p,
        c_liq=c_liq,
        l_v=l_v,
        r_d=r_d,
        r_v=r_v,
    ))


def pseudoadiabatic_lapse_rate(temp, pressure, rel_hum=REL_HUM,
                               grav=GRAV_EARTH, c_p=C_P, r_d=R_D, l_v=L_V,
                               r_v=R_V, c_pv=C_PV):
    """Pseudoadiabatic lapse rate."""
    sat_vap_press = sat_vap_press_tetens_kelvin(temp)
    vapor_pressure = rel_hum*sat_vap_press
    vap_mix_ratio = water_vapor_mixing_ratio(vapor_pressure, pressure)
    numer = grav*(1 + vap_mix_ratio)*(1 + l_v*vap_mix_ratio / (r_d*temp))
    epsilon = r_d / r_v
    denom = c_p + c_pv*vap_mix_ratio + (
        l_v**2*vap_mix_ratio * (epsilon + vap_mix_ratio) / (r_d*temp**2)
    )
    return numer / denom


if __name__ == '__main__':
    pass
