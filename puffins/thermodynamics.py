#! /usr/bin/env python
"""Thermodynamic quantities."""

import numpy as np
from scipy.optimize import brentq

from .constants import (
    C_P,
    C_PV,
    C_VL,
    EPSILON,
    GRAV_EARTH,
    L_V,
    P0,
    R_D,
    R_V,
    REL_HUM,
)


def exner_func(pressure, p0=1000., r_d=R_D, c_p=C_P):
    """Exner function."""
    return (pressure / p0) ** (r_d / c_p)


def pot_temp(temp, pressure, p0=1000., r_d=R_D, c_p=C_P):
    """Potential temperature."""
    return temp / exner_func(pressure, p0=p0, r_d=r_d, c_p=c_p)


def moist_enthalpy(temp, sphum, c_p=C_P, l_v=L_V):
    """Moist enthalpy in units of Kelvin."""
    return temp + l_v * sphum / c_p


def water_vapor_mixing_ratio(vapor_press, pressure, epsilon=EPSILON):
    """Water vapor mixing ratio.

    Both the vapor pressure and pressure must have the same units, e.g. both
    Pascals or both hectopascals (hPa).

    E.g. https://glossary.ametsoc.org/wiki/Mixing_ratio

    """
    return epsilon * vapor_press / (pressure - vapor_press)


def vap_press_from_mix_ratio(mix_ratio, pressure, epsilon=EPSILON):
    """Water vapor pressure given mixing ration and pressure."""
    return mix_ratio * pressure / (epsilon + mix_ratio)


def specific_humidity(mixing_ratio):
    """Specific humidity computed from water vapor mixing ratio.

    E.g. https://glossary.ametsoc.org/wiki/Specific_humidity

    """
    return mixing_ratio / (1. + mixing_ratio)


def mixing_ratio(spec_hum):
    """Mixing ratio (of water vapor) computed from specific humidity."""
    return spec_hum / (1. - spec_hum)


def sat_vap_press_tetens_kelvin(temp):
    """Saturation vapor pressure using Tetens equation.

    E.g. https://en.wikipedia.org/wiki/Tetens_equation

    Note: unlike original Tetens expression, temperature should be in Kelvin,
    NOT degrees Celsius.  And result has units Pa, not kPa as in original
    version.

    """
    a = 610.78
    b = 17.27
    c = 237.3 - 273.15
    return a * np.exp(b * (temp - 273.15) / (temp + c))


def saturation_mixing_ratio(pressure, temp, epsilon=EPSILON):
    """Saturation mixing ratio.

    Pressure must be in Pascals, not hPa.  Temperature must be in Kelvin.
    """
    sat_vap_press = sat_vap_press_tetens_kelvin(temp)
    return water_vapor_mixing_ratio(sat_vap_press, pressure, epsilon=epsilon)


def saturation_specific_humidity(pressure, temp, epsilon=EPSILON):
    """Saturation specific humidity."""
    sat_mix_ratio = saturation_mixing_ratio(pressure, temp, epsilon=epsilon)
    return specific_humidity(sat_mix_ratio)


def relative_humidity(vapor_pressure, sat_vap_press):
    """Relative humidity.

    C.f. https://glossary.ametsoc.org/wiki/Relative_humidity.

    """
    return vapor_pressure / sat_vap_press


def rel_hum_from_temp_dewpoint(temp, temp_dew):
    """Relative humidity, given the temperature and dewpoint.

    Conceptually, relative humidity is the ratio of the actual vapor pressure
    to the saturation vapor pressure.  Tetens equation or the
    August-Roche-Magnus equation can be used to compute the saturation vapor
    pressure from the actual temperature.  Separately, the actual vapor
    pressure can be computed from the dewpoint temperature using the same
    Tetens etc.  Then you simply take the ratio of the computed vapor pressures
    to get the RH.

    C.f. https://bmcnoldy.rsmas.miami.edu/Humidity.html.

    """
    return (sat_vap_press_tetens_kelvin(temp_dew) /
            sat_vap_press_tetens_kelvin(temp))


def moist_static_energy(temp, height, spec_hum, c_p=C_P, grav=GRAV_EARTH,
                        l_v=L_V):
    """Moist static energy."""
    return c_p * temp + grav * height + l_v * spec_hum


def saturation_mse(temp, height, pressure=P0, c_p=C_P, grav=GRAV_EARTH,
                   l_v=L_V, epsilon=EPSILON):
    """Saturation moist static energy.

    I.e. MSE if the specific humidity was at its saturation value.

    """
    sat_spec_hum = saturation_specific_humidity(pressure, temp,
                                                epsilon=epsilon)
    return moist_static_energy(temp, height, sat_spec_hum, c_p=c_p,
                               grav=grav, l_v=l_v)


def saturation_entropy(temp, pressure=P0, sat_vap_press=None,
                       c_p=C_P, r_d=R_D, l_v=L_V, epsilon=EPSILON):
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
    sat_q = saturation_specific_humidity(pressure, temp, epsilon=epsilon)
    return (c_p * np.log(temp) - r_d * np.log(pressure) +
            l_v * sat_q / temp)


def dsat_entrop_dtemp_approx(temp, pressure=P0, c_p=C_P, r_v=R_V, l_v=L_V):
    sat_vap_press = sat_vap_press_tetens_kelvin(temp)
    sat_spec_hum = sat_vap_press / pressure
    return (c_p + l_v*sat_spec_hum*(l_v/(r_v*temp) - 1)/temp) / temp


def equiv_pot_temp(temp, rel_hum, pressure, tot_wat_mix_ratio=0., p0=P0,
                   c_p=C_P, c_liq=C_VL, l_v=L_V, r_d=R_D, r_v=R_V):
    """Equivalent potential temperature.

    Note that pressure must be in Pascals, not hPa.

    """
    sat_vap_press = sat_vap_press_tetens_kelvin(temp)
    vapor_pressure = rel_hum * sat_vap_press
    pressure_dry = pressure - vapor_pressure
    vap_mix_ratio = water_vapor_mixing_ratio(vapor_pressure, pressure)
    denom = c_p + c_liq * tot_wat_mix_ratio
    return (temp * (p0 / pressure_dry) ** (r_d / denom) *
            rel_hum ** (-1 * r_v * vap_mix_ratio / denom) *
            np.exp(l_v * vap_mix_ratio / (denom * temp)))


def sat_equiv_pot_temp(temp, pressure, tot_wat_mix_ratio=0., p0=P0,
                       c_p=C_P, c_liq=4185.5, l_v=L_V, r_d=R_D, r_v=R_V):
    """Saturation equivalent potential temperature.

    Note that pressure must be in Pascals, not hPa.

    """
    return equiv_pot_temp(
        temp,
        1.,
        pressure,
        tot_wat_mix_ratio=tot_wat_mix_ratio,
        p0=p0,
        c_p=c_p,
        c_liq=c_liq,
        l_v=l_v,
        r_d=r_d,
        r_v=r_v,
    )


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


def exner_func(pressure, p0=1000., r_d=R_D, c_p=C_P):
    """Exner function."""
    return (pressure / p0) ** (r_d / c_p)


def pot_temp(temp, pressure, p0=1000., r_d=R_D, c_p=C_P):
    """Potential temperature."""
    return temp / exner_func(pressure, p0=p0, r_d=r_d, c_p=c_p)


def moist_enthalpy(temp, sphum, c_p=C_P, l_v=L_V):
    """Moist enthalpy in units of Kelvin."""
    return temp + l_v * sphum / c_p


if __name__ == '__main__':
    pass
