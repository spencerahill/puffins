#! /usr/bin/env python
"""Lifting condensation level, following Romps 2017.

This module deliberately defines its own physical constants (below) with
Romps's optimized values rather than importing from :mod:`puffins.constants`;
see the comment on the constant block for the rationale.

Romps, David M. 2017. "Exact Expression for the Lifting Condensation Level."
Journal of the Atmospheric Sciences 74 (12): 3891-3900.
https://doi.org/10.1175/JAS-D-17-0102.1.

"""

from typing import cast

import numpy as np
import scipy.special

from ._typing import ArrayLike

# Physical constants, following Romps (2017) exactly.
#
# These DELIBERATELY differ from ``puffins.constants``.  Romps jointly optimized
# R_V and C_VL (along with the ice values, unused here), together with the fixed
# triple-point quantities, so that the analytic saturation-vapor-pressure
# expression [Eq. (4)] matches the Wagner and Pruss (2002) laboratory data; that
# fit is what gives the LCL its ~5 m accuracy.  Substituting the general-purpose
# textbook values from ``puffins.constants`` (e.g. C_VL = 4186 vs. 4119 here, or
# R_V = 461.4 vs. 461) would silently degrade that accuracy, so this module
# keeps its own self-consistent set.  Equation numbers below are from Romps (2017).
P_TRIP = 611.65  # Triple-point vapor pressure (Pa); Eq. (7).
T_TRIP = 273.16  # Triple-point temperature (K); Eq. (8).
E_0V = 2.3740e6  # Vapor-liquid internal-energy difference at T_TRIP (J/kg); Eq. (9).
R_D = 287.04  # Specific gas constant of dry air (J/kg/K); Eq. (18).
R_V = 461.0  # Specific gas constant of water vapor (J/kg/K); Eq. (11).
C_VD = 719.0  # Specific heat of dry air at constant volume (J/kg/K); Eq. (19).
C_VV = 1418.0  # Specific heat of water vapor at constant volume (J/kg/K); Eq. (6).
C_VL = 4119.0  # Specific heat of liquid water (J/kg/K); Eq. (12).
C_PD = C_VD + R_D  # Dry-air specific heat at constant pressure (= 1006.04 J/kg/K).
C_PV = C_VV + R_V  # Water-vapor specific heat at constant pressure (= 1879.0 J/kg/K).
GRAV = 9.81  # Gravitational acceleration (m/s^2); Romps (2017).


def sat_vap_press_liq_wat(
    temp: ArrayLike,
    p_trip: float = P_TRIP,
    t_trip: float = T_TRIP,
    e_0v: float = E_0V,
    r_v: float = R_V,
    c_pv: float = C_PV,
    c_vv: float = C_VV,
    c_vl: float = C_VL,
) -> ArrayLike:
    """Saturation vapor pressure over liquid water.  Romps 2017."""
    return (
        p_trip
        * (temp / t_trip) ** ((c_pv - c_vl) / r_v)
        * np.exp((e_0v - (c_vv - c_vl) * t_trip) / r_v * (1 / t_trip - 1 / temp))
    )


def _q_v(
    press: ArrayLike,
    vap_press: ArrayLike,
    r_d: float = R_D,
    r_v: float = R_V,
) -> ArrayLike:
    return r_d * vap_press / (r_v * press + (r_d - r_v) * vap_press)


def gas_const_moist_air(
    press: ArrayLike,
    vap_press: ArrayLike,
    r_d: float = R_D,
    r_v: float = R_V,
) -> ArrayLike:
    """Gas constant of moist air.  Romps 2017."""
    q_v = _q_v(press, vap_press, r_d=r_d, r_v=r_v)
    return (1 - q_v) * r_d + q_v * r_v


def spec_heat_const_press_moist_air(
    press: ArrayLike,
    vap_press: ArrayLike,
    r_d: float = R_D,
    r_v: float = R_V,
    c_pd: float = C_PD,
    c_pv: float = C_PV,
) -> ArrayLike:
    """Specific heat at constant pressure of moist air."""
    q_v = _q_v(press, vap_press, r_d=r_d, r_v=r_v)
    return (1 - q_v) * c_pd + q_v * c_pv


def temp_lift_cond_level(
    press: ArrayLike,
    temp: ArrayLike,
    rel_hum: ArrayLike,
    p_trip: float = P_TRIP,
    t_trip: float = T_TRIP,
    r_d: float = R_D,
    r_v: float = R_V,
    c_pd: float = C_PD,
    c_pv: float = C_PV,
    c_vv: float = C_VV,
    c_vl: float = C_VL,
    e_0v: float = E_0V,
) -> ArrayLike:
    """Temperature at lifting condensation level.  C.f. Romps 2017.

    Romps, David M. 2017. "Exact Expression for the Lifting Condensation Level."
    Journal of the Atmospheric Sciences 74 (12): 3891-3900.
    https://doi.org/10.1175/JAS-D-17-0102.1.

    Modified from scripts provided by D. Romps at
    https://romps.berkeley.edu/papers/pubdata/2016/lcl/lcl.py

    """
    sat_vap_press = sat_vap_press_liq_wat(
        temp,
        p_trip=p_trip,
        t_trip=t_trip,
        e_0v=e_0v,
        r_v=r_v,
        c_pv=c_pv,
        c_vv=c_vv,
        c_vl=c_vl,
    )
    vap_press = rel_hum * sat_vap_press

    r_m = gas_const_moist_air(press, vap_press, r_d=r_d, r_v=r_v)
    c_pm = spec_heat_const_press_moist_air(
        press, vap_press, r_d=r_d, r_v=r_v, c_pd=c_pd, c_pv=c_pv
    )

    term_a = (c_vl - c_pv) / r_v + c_pm / r_m
    term_b = -(e_0v - (c_vv - c_vl) * t_trip) / (r_v * temp)
    term_c = term_b / term_a
    return cast(
        ArrayLike,
        term_c
        * temp
        / scipy.special.lambertw(
            rel_hum ** (1.0 / term_a) * term_c * np.exp(term_c), -1
        ).real,
    )


def pres_lift_cond_level(
    press: ArrayLike,
    temp: ArrayLike,
    rel_hum: ArrayLike,
    p_trip: float = P_TRIP,
    t_trip: float = T_TRIP,
    r_d: float = R_D,
    r_v: float = R_V,
    c_pd: float = C_PD,
    c_pv: float = C_PV,
    c_vv: float = C_VV,
    c_vl: float = C_VL,
    e_0v: float = E_0V,
) -> ArrayLike:
    """Pressure at lifting condensation level.  C.f. Romps 2017, Eq. (22b).

    Obtained from potential-temperature conservation during adiabatic ascent:
    ``p_lcl = press * (temp_lcl / temp) ** (c_pm / r_m)``.

    Romps, David M. 2017. "Exact Expression for the Lifting Condensation Level."
    Journal of the Atmospheric Sciences 74 (12): 3891-3900.
    https://doi.org/10.1175/JAS-D-17-0102.1.

    """
    sat_vap_press = sat_vap_press_liq_wat(
        temp,
        p_trip=p_trip,
        t_trip=t_trip,
        e_0v=e_0v,
        r_v=r_v,
        c_pv=c_pv,
        c_vv=c_vv,
        c_vl=c_vl,
    )
    vap_press = rel_hum * sat_vap_press
    r_m = gas_const_moist_air(press, vap_press, r_d=r_d, r_v=r_v)
    c_pm = spec_heat_const_press_moist_air(
        press, vap_press, r_d=r_d, r_v=r_v, c_pd=c_pd, c_pv=c_pv
    )
    temp_lcl = temp_lift_cond_level(
        press,
        temp,
        rel_hum,
        p_trip=p_trip,
        t_trip=t_trip,
        r_d=r_d,
        r_v=r_v,
        c_pd=c_pd,
        c_pv=c_pv,
        c_vv=c_vv,
        c_vl=c_vl,
        e_0v=e_0v,
    )
    return cast(ArrayLike, press * (temp_lcl / temp) ** (c_pm / r_m))


def height_lift_cond_level(
    press: ArrayLike,
    temp: ArrayLike,
    rel_hum: ArrayLike,
    z_0: float = 0.0,
    p_trip: float = P_TRIP,
    t_trip: float = T_TRIP,
    r_d: float = R_D,
    r_v: float = R_V,
    c_pd: float = C_PD,
    c_pv: float = C_PV,
    c_vv: float = C_VV,
    c_vl: float = C_VL,
    e_0v: float = E_0V,
    grav: float = GRAV,
) -> ArrayLike:
    """Height of the lifting condensation level.  C.f. Romps 2017, Eq. (22c).

    ``z_lcl = z_0 + (c_pm / grav) * (temp - temp_lcl)``, valid in a well-mixed
    layer where the parcel's dry static energy is conserved.

    Romps, David M. 2017. "Exact Expression for the Lifting Condensation Level."
    Journal of the Atmospheric Sciences 74 (12): 3891-3900.
    https://doi.org/10.1175/JAS-D-17-0102.1.

    Modified from scripts provided by D. Romps at
    https://romps.berkeley.edu/papers/pubdata/2016/lcl/lcl.py

    """
    sat_vap_press = sat_vap_press_liq_wat(
        temp,
        p_trip=p_trip,
        t_trip=t_trip,
        e_0v=e_0v,
        r_v=r_v,
        c_pv=c_pv,
        c_vv=c_vv,
        c_vl=c_vl,
    )
    vap_press = rel_hum * sat_vap_press
    c_pm = spec_heat_const_press_moist_air(
        press, vap_press, r_d=r_d, r_v=r_v, c_pd=c_pd, c_pv=c_pv
    )
    temp_lcl = temp_lift_cond_level(
        press,
        temp,
        rel_hum,
        p_trip=p_trip,
        t_trip=t_trip,
        r_d=r_d,
        r_v=r_v,
        c_pd=c_pd,
        c_pv=c_pv,
        c_vv=c_vv,
        c_vl=c_vl,
        e_0v=e_0v,
    )
    return z_0 + c_pm / grav * (temp - temp_lcl)
