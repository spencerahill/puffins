#! /usr/bin/env python
"""Lifting condensation level, following Romps 2017.

Romps, David M. 2017. "Exact Expression for the Lifting Condensation Level."
Journal of the Atmospheric Sciences 74 (12): 3891-3900.
https://doi.org/10.1175/JAS-D-17-0102.1.

"""

from typing import cast

import numpy as np
import scipy.special

from ._typing import ArrayLike
from .constants import (
    C_PD,
    C_PV,
    C_VL,
    C_VV,
    E_0V,
    GRAV_EARTH,
    P_TRIP,
    R_D,
    R_V,
    T_TRIP,
)

# Parameters
GRAV = GRAV_EARTH


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


def lift_cond_level(
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
    """Lifting condensation level.  C.f. Romps 2017.

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
