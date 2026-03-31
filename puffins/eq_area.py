#! /usr/bin/env python
"""Equal-area model analytical solutions and numerical solvers.

Implements the equal-area constraint for Hadley cell models following
Held & Hou (1980), Lindzen & Hou (1988), and extensions. Includes
analytical solutions for the small-angle, uniform-Rossby-number case,
linear-Rossby-number profiles, and numerical solvers for Boussinesq,
convective quasi-equilibrium (CQE), and fixed-tropopause-temperature
variants.

References
----------
.. [1] Held, I. M. & Hou, A. Y. (1980). "Nonlinear Axially Symmetric
   Circulations in a Nearly Inviscid Atmosphere." J. Atmos. Sci., 37,
   515-533.
.. [2] Lindzen, R. S. & Hou, A. Y. (1988). "Hadley Circulations for
   Zonally Averaged Heating Centered off the Equator." J. Atmos. Sci.,
   45, 2416-2427.
"""

import numpy as np
import scipy.integrate
import scipy.optimize

from .constants import (
    C_P,
    DELTA_H,
    DELTA_V,
    GRAV_EARTH,
    HEIGHT_TROPO,
    P0,
    RAD_EARTH,
    REL_HUM,
    ROT_RATE_EARTH,
    TEMP_TROPO,
    THETA_REF,
)
from .fixed_temp_tropo import (
    pot_temp_amc_cqe_fixed_tt,
    pot_temp_avg_amc_fixed_tt_bouss,
)
from .grad_bal import (
    pot_temp_amc_cqe,
    pot_temp_avg_unif_ro_small_ang_eq_ascent,
    pot_temp_lin_ro_lata0_small_ang,
)
from .lindzen_hou_1988 import pot_temp_rce_lh88
from .nb_utils import sin2deg


# Analytical solutions for annual-mean, uniform-Ro, small-angle case
def eq_pot_temp_mean_ro(
    ross_num,
    therm_ross_num,
    delta_h,
    theta_ref=THETA_REF,
):
    """Equatorial potential temperature for the fixed-Ro, small-angle, equal-area solution.

    Parameters
    ----------
    ross_num : float
        Rossby number.
    therm_ross_num : float
        Thermal Rossby number.
    delta_h : float
        Fractional horizontal temperature difference.
    theta_ref : float, optional
        Reference potential temperature (K). Default: THETA_REF.

    Returns
    -------
    float
        Equatorial potential temperature (K).
    """
    return theta_ref * (
        1 + delta_h / 3 - 5 * therm_ross_num * delta_h / (18 * ross_num)
    )


def pot_temp_mean_ro(
    lat,
    ross_num,
    therm_ross_num,
    delta_h,
    theta_ref=THETA_REF,
):
    """Potential temperature profile for the fixed-Ro, small-angle, equal-area solution.

    Parameters
    ----------
    lat : array-like
        Latitude (degrees).
    ross_num : float
        Rossby number.
    therm_ross_num : float
        Thermal Rossby number.
    delta_h : float
        Fractional horizontal temperature difference.
    theta_ref : float, optional
        Reference potential temperature (K). Default: THETA_REF.

    Returns
    -------
    array-like
        Potential temperature at each latitude (K).
    """
    pot_temp_eq = eq_pot_temp_mean_ro(
        ross_num,
        therm_ross_num,
        delta_h,
        theta_ref=theta_ref,
    )
    return pot_temp_avg_unif_ro_small_ang_eq_ascent(
        lat,
        pot_temp_eq,
        ross_num=ross_num,
        theta_ref=theta_ref,
        burg_num=therm_ross_num / delta_h,
    )


def cell_edge_mean_ro(
    ross_num,
    therm_ross_num,
):
    """Hadley cell edge for the fixed-Ro, small-angle, equatorial-ascent case.

    Parameters
    ----------
    ross_num : float
        Rossby number.
    therm_ross_num : float
        Thermal Rossby number.

    Returns
    -------
    float
        Cell edge latitude (degrees).
    """
    return np.rad2deg((5 * therm_ross_num / (3 * ross_num)) ** 0.5)


def heat_flux_mean_ro(
    lat,
    therm_ross_num,
    ross_num,
    theta_ref=THETA_REF,
    radius=RAD_EARTH,
    height=HEIGHT_TROPO,
    delta_h=DELTA_H,
    tau=20 * 86400,
):
    """Meridional heat flux for the fixed-Ro, small-angle, equatorial-ascent case.

    Parameters
    ----------
    lat : array-like
        Latitude (degrees).
    therm_ross_num : float
        Thermal Rossby number.
    ross_num : float
        Rossby number.
    theta_ref : float, optional
        Reference potential temperature (K). Default: THETA_REF.
    radius : float, optional
        Planetary radius (m). Default: Earth.
    height : float, optional
        Tropopause height (m). Default: HEIGHT_TROPO.
    delta_h : float, optional
        Fractional horizontal temperature difference. Default: DELTA_H.
    tau : float, optional
        Radiative relaxation timescale (s). Default: 20 days.

    Returns
    -------
    array-like
        Meridional heat flux (K m^2/s).
    """
    prefac = (
        (5 / 18)
        * (5 / 3) ** 0.5
        * radius
        * height
        * delta_h
        / tau
        * (therm_ross_num / ross_num) ** 1.5
    )
    lat_div_lat_ro = lat / cell_edge_mean_ro(ross_num, therm_ross_num)
    return (
        theta_ref
        * prefac
        * (lat_div_lat_ro - 2 * lat_div_lat_ro**3 + lat_div_lat_ro**5)
    )


def mom_flux_mean_ro(
    lat,
    therm_ross_num,
    ross_num,
    radius=RAD_EARTH,
    rot_rate=ROT_RATE_EARTH,
    height=HEIGHT_TROPO,
    delta_h=DELTA_H,
    delta_v=DELTA_V,
    tau=20 * 86400,
):
    """Meridional momentum flux for the fixed-Ro, small-angle, equatorial-ascent case.

    Parameters
    ----------
    lat : array-like
        Latitude (degrees).
    therm_ross_num : float
        Thermal Rossby number.
    ross_num : float
        Rossby number.
    radius : float, optional
        Planetary radius (m). Default: Earth.
    rot_rate : float, optional
        Planetary rotation rate (rad/s). Default: Earth.
    height : float, optional
        Tropopause height (m). Default: HEIGHT_TROPO.
    delta_h : float, optional
        Fractional horizontal temperature difference. Default: DELTA_H.
    delta_v : float, optional
        Fractional vertical temperature difference. Default: DELTA_V.
    tau : float, optional
        Radiative relaxation timescale (s). Default: 20 days.

    Returns
    -------
    array-like
        Meridional momentum flux (m^2/s^2).
    """
    prefac = rot_rate * radius**2 * height * delta_h / (6 * tau * delta_v)
    latrad = np.deg2rad(lat)
    latrad2 = latrad**2
    return (
        prefac
        * latrad**3
        * (
            5 * therm_ross_num / 3
            - ross_num * latrad2 * (2 - 3 * ross_num * latrad2 / (5 * therm_ross_num))
        )
    )


def u_sfc_mean_ro(
    lat,
    therm_ross_num,
    ross_num,
    radius=RAD_EARTH,
    rot_rate=ROT_RATE_EARTH,
    height=HEIGHT_TROPO,
    delta_h=DELTA_H,
    delta_v=DELTA_V,
    tau=20 * 86400,
    drag_coeff=0.005,
):
    """Surface zonal wind for the fixed-Ro, small-angle, equatorial-ascent case.

    Parameters
    ----------
    lat : array-like
        Latitude (degrees).
    therm_ross_num : float
        Thermal Rossby number.
    ross_num : float
        Rossby number.
    radius : float, optional
        Planetary radius (m). Default: Earth.
    rot_rate : float, optional
        Planetary rotation rate (rad/s). Default: Earth.
    height : float, optional
        Tropopause height (m). Default: HEIGHT_TROPO.
    delta_h : float, optional
        Fractional horizontal temperature difference. Default: DELTA_H.
    delta_v : float, optional
        Fractional vertical temperature difference. Default: DELTA_V.
    tau : float, optional
        Radiative relaxation timescale (s). Default: 20 days.
    drag_coeff : float, optional
        Surface drag coefficient (dimensionless). Default: 0.005.

    Returns
    -------
    array-like
        Surface zonal wind (m/s).
    """
    prefac = (
        -25 * rot_rate * radius * height * delta_h / (18 * drag_coeff * tau * delta_v)
    )
    lat_div_lat_ro = lat / cell_edge_mean_ro(ross_num, therm_ross_num)
    return (
        prefac
        * (therm_ross_num / ross_num) ** 2
        * (
            lat_div_lat_ro**2
            - (10 / 3) * lat_div_lat_ro**4
            + (7 / 3) * lat_div_lat_ro**6
        )
    )


# Solutions for small-angle, linear meridional profile in Ro
def cell_edge_lin_ro_lata0_full(therm_ross, ross_ascent, ross_descent):
    """Cell edge for the linear-Ro, equatorial-ascent, small-angle case (full solution).

    Solves the quadratic polynomial in latitude-squared for the cell
    edge when the Rossby number varies linearly from ascent to descent.

    Parameters
    ----------
    therm_ross : float
        Thermal Rossby number.
    ross_ascent : float
        Rossby number at the ascent latitude.
    ross_descent : float
        Rossby number at the descent (cell edge) latitude.

    Returns
    -------
    float
        Cell edge latitude (degrees).
    """
    delro = ross_ascent - ross_descent

    def _term_latd4(ross_ascent, delro):
        return ross_ascent**2 / 7 - ross_ascent * delro / 6 + 4 * delro**2 / 81

    def _term_latd2(ross_ascent, delro):
        return 2 * ross_ascent / 5 - 2 * delro / 9

    def _term_latd0(therm_ro):
        return -2 * therm_ro / 3

    poly_obj = np.polynomial.Polynomial(
        [
            _term_latd0(therm_ross),
            _term_latd2(ross_ascent, delro),
            _term_latd4(ross_ascent, delro),
        ]
    )
    return np.rad2deg(poly_obj.roots()[1] ** 0.5)


def cell_edge_lin_ro_lata0_approx(therm_ross, ross_ascent, ross_descent):
    """Leading-order approximation for cell edge with a linear Rossby number profile.

    Parameters
    ----------
    therm_ross : float
        Thermal Rossby number.
    ross_ascent : float
        Rossby number at the ascent latitude.
    ross_descent : float
        Rossby number at the descent (cell edge) latitude.

    Returns
    -------
    float
        Approximate cell edge latitude (degrees).

    See Also
    --------
    cell_edge_lin_ro_lata0_full : Full (non-approximate) solution.
    """
    delta_ross = ross_ascent - ross_descent
    return np.rad2deg((15 * therm_ross / (9 * ross_ascent - 5 * delta_ross)) ** 0.5)


def eq_pot_temp_lin_ro_lata0_small_ang(
    therm_ross,
    ross_ascent,
    ross_descent,
    delta_h=DELTA_H,
    theta_ref=THETA_REF,
):
    """Equatorial potential temperature for the small-angle, linear-Ro, equal-area case.

    Parameters
    ----------
    therm_ross : float
        Thermal Rossby number.
    ross_ascent : float
        Rossby number at the ascent latitude.
    ross_descent : float
        Rossby number at the descent (cell edge) latitude.
    delta_h : float, optional
        Fractional horizontal temperature difference. Default: DELTA_H.
    theta_ref : float, optional
        Reference potential temperature (K). Default: THETA_REF.

    Returns
    -------
    float
        Equatorial potential temperature (K).
    """
    delro = ross_ascent - ross_descent
    burg_num = therm_ross / delta_h
    latd2 = (
        np.deg2rad(cell_edge_lin_ro_lata0_full(therm_ross, ross_ascent, ross_descent))
        ** 2
    )
    return theta_ref * (
        1
        + delta_h / 3
        - latd2
        * (
            delta_h
            + latd2
            / burg_num
            * (
                4 * delro / 15
                - ross_ascent / 2
                + (
                    -(ross_ascent**2) / 6
                    + 4 * ross_ascent * delro / 21
                    - (delro**2) / 18
                )
                * latd2
            )
        )
    )


def pot_temp_lin_ro_eq_area(
    lat,
    therm_ross,
    ross_ascent,
    ross_descent,
    delta_h=DELTA_H,
    theta_ref=THETA_REF,
    rot_rate=ROT_RATE_EARTH,
    radius=RAD_EARTH,
    grav=GRAV_EARTH,
    height=HEIGHT_TROPO,
):
    """Potential temperature profile for the linear-Ro, small-angle, equal-area case.

    Combines the equatorial temperature and cell edge solutions to compute
    the full meridional potential temperature profile.

    Parameters
    ----------
    lat : array-like
        Latitude (degrees).
    therm_ross : float
        Thermal Rossby number.
    ross_ascent : float
        Rossby number at the ascent latitude.
    ross_descent : float
        Rossby number at the descent (cell edge) latitude.
    delta_h : float, optional
        Fractional horizontal temperature difference. Default: DELTA_H.
    theta_ref : float, optional
        Reference potential temperature (K). Default: THETA_REF.
    rot_rate : float, optional
        Planetary rotation rate (rad/s). Default: Earth.
    radius : float, optional
        Planetary radius (m). Default: Earth.
    grav : float, optional
        Gravitational acceleration (m/s^2). Default: Earth.
    height : float, optional
        Tropopause height (m). Default: HEIGHT_TROPO.

    Returns
    -------
    array-like
        Potential temperature at each latitude (K).
    """
    pot_temp_eq = eq_pot_temp_lin_ro_lata0_small_ang(
        therm_ross=therm_ross,
        ross_ascent=ross_ascent,
        ross_descent=ross_descent,
        delta_h=delta_h,
        theta_ref=theta_ref,
    )
    lat_descent = cell_edge_lin_ro_lata0_full(
        therm_ross=therm_ross,
        ross_ascent=ross_ascent,
        ross_descent=ross_descent,
    )
    return pot_temp_lin_ro_lata0_small_ang(
        lat=lat,
        lat_descent=lat_descent,
        ross_ascent=ross_ascent,
        ross_descent=ross_descent,
        pot_temp_lat0=pot_temp_eq,
        rot_rate=rot_rate,
        radius=radius,
        theta_ref=theta_ref,
        grav=grav,
        height=height,
    )


# Original Lindzen and Hou 1988.
def _theta_hat_amc_lh88(sinlat, sinlat_1, theta_hat_1, theta_ref, del_h_over_ro):
    """Lindzen and Hou 1988 Eq. 7."""
    return theta_hat_1 - 0.5 * theta_ref * del_h_over_ro * (
        sinlat**2 - sinlat_1**2
    ) ** 2 / (1 - sinlat**2)


def _theta_hat_rce_lh88(sinlat, sinlat_0, theta_ref, delta_h):
    """Lindzen and Hou 1988 Eq. 1b."""
    return theta_ref * (1 + delta_h / 3.0 - delta_h * (sinlat - sinlat_0) ** 2)


def _lh88_amc_rce_theta_hat_diff(
    sinlat, sinlat_0, sinlat_1, theta_hat_1, theta_ref, delta_h, thermal_ro
):
    """Difference between AMC and RCE depth-averaged potential temperatures.

    At the cell edges, this difference is zero by construction.

    """
    return _theta_hat_amc_lh88(
        sinlat, sinlat_1, theta_hat_1, theta_ref, delta_h / thermal_ro
    ) - _theta_hat_rce_lh88(sinlat, sinlat_0, theta_ref, delta_h)


def _lh88_cons_energy_integral(
    sinlat_edge, sinlat_0, sinlat_1, theta_hat_1, theta_ref, delta_h, thermal_ro
):
    return scipy.integrate.quad(
        _lh88_amc_rce_theta_hat_diff,
        sinlat_1,
        sinlat_edge,
        args=(sinlat_0, sinlat_1, theta_hat_1, theta_ref, delta_h, thermal_ro),
    )[0]


def _lh88_model(x, sinlat_0, theta_ref, delta_h, thermal_ro):
    """Matrix representation of the Lindzen-Hou 1988 equal-area model.

    Parameters
    ----------
    x : length-4 array
        Values correspond, in order, to:

        - $\\sin\varphi_w$: sine of the poleward edge of the winter cell
        - $\\sin\varphi_1$: sine of the cells' shared inner edge
        - $\\sin\varphi_s$: sine of the poleward edge of the summer cell
        - $\theta_hat_1$: potential temperature at the cells' shared inner edge

    sinlat_0 : sine of latitude of the prescribed heating maximum
    theta_ref : the reference potential temperature ($\theta_hat_0$ in LH88)
    delta_h : the prescribed fractional horizontal temperature difference
    thermal_ro : the prescribed thermal rossby number

    """
    sinlat_south, sinlat_1, sinlat_north, theta_hat_1 = x
    args = (sinlat_0, sinlat_1, theta_hat_1, theta_ref, delta_h, thermal_ro)
    x0 = _lh88_amc_rce_theta_hat_diff(sinlat_south, *args)
    x1 = _lh88_amc_rce_theta_hat_diff(sinlat_north, *args)
    x2 = _lh88_cons_energy_integral(sinlat_south, *args)
    x3 = _lh88_cons_energy_integral(sinlat_north, *args)
    return np.array([x0, x1, x2, x3])


def equal_area_lh88(
    init_guess, sinlat_0, theta_ref=THETA_REF, delta_h=DELTA_H, thermal_ro=0.15
):
    """Solve the Lindzen-Hou 1988 equal-area model numerically.

    Finds the winter cell edge, shared inner edge, summer cell edge,
    and inner-edge potential temperature by solving the system of four
    equal-area constraints.

    Parameters
    ----------
    init_guess : array-like of length 4
        Initial guess for [sin(lat_winter), sin(lat_1), sin(lat_summer),
        theta_hat_1].
    sinlat_0 : float
        Sine of the latitude of maximum heating.
    theta_ref : float, optional
        Reference potential temperature (K). Default: THETA_REF.
    delta_h : float, optional
        Fractional horizontal temperature difference. Default: DELTA_H.
    thermal_ro : float, optional
        Thermal Rossby number. Default: 0.15.

    Returns
    -------
    numpy.ndarray of length 4
        Solution array: [sin(lat_winter), sin(lat_1), sin(lat_summer),
        theta_hat_1].

    References
    ----------
    .. [1] Lindzen, R. S. & Hou, A. Y. (1988). J. Atmos. Sci., 45,
       2416-2427.
    """
    sol = scipy.optimize.root(
        _lh88_model, init_guess, args=(sinlat_0, theta_ref, delta_h, thermal_ro)
    )
    return sol.x


# Modified LH88: fixed lapse rate and tropopause temperature.
def _lh88_amc_rce_theta_diff_fixed_tt(
    sinlat,
    sinlat_0,
    sinlat_1,
    temp_sfc_1,
    theta_ref,
    delta_h,
    gamma,
    dtheta_dts,
    temp_tropo,
    rot_rate,
    radius,
    c_p,
):
    """Difference between AMC and RCE wind in Lindzen and Hou 1988 model,
    but assuming that the tropopause temperature is fixed, rather than
    the standard Boussinesq assumption that tropopause height is fixed.

    """
    theta_avg_amc = pot_temp_avg_amc_fixed_tt_bouss(
        sin2deg(sinlat),
        sin2deg(sinlat_1),
        temp_sfc_1,
        theta_ref=theta_ref,
        gamma=gamma,
        dtheta_dts=dtheta_dts,
        temp_tropo=temp_tropo,
        rot_rate=rot_rate,
        radius=radius,
        c_p=c_p,
    )
    theta_avg_rce = pot_temp_rce_lh88(
        sin2deg(sinlat),
        sin2deg(sinlat_0),
        z=0.5,
        theta_ref=theta_ref,
        height=1.0,
        delta_h=delta_h,
    )
    return theta_avg_amc - theta_avg_rce


def _lh88_cons_energy_integral_fixed_tt(
    sinlat_edge,
    sinlat_0,
    sinlat_1,
    temp_sfc_1,
    theta_ref,
    delta_h,
    gamma,
    dtheta_dts,
    temp_tropo,
    rot_rate,
    radius,
    c_p,
):
    return scipy.integrate.quad(
        _lh88_amc_rce_theta_diff_fixed_tt,
        sinlat_1,
        sinlat_edge,
        args=(
            sinlat_0,
            sinlat_1,
            temp_sfc_1,
            theta_ref,
            delta_h,
            gamma,
            dtheta_dts,
            temp_tropo,
            rot_rate,
            radius,
            c_p,
        ),
    )[0]


def _lh88_model_fixed_tt(
    x,
    sinlat_0,
    theta_ref,
    delta_h,
    gamma,
    dtheta_dts,
    temp_tropo,
    rot_rate,
    radius,
    c_p,
):
    """LH88 equal area, assuming fixed lapse rate and tropopause temperature.

    Parameters
    ----------
    x : length-4 array
        Values correspond, in order, to:

        - $\\sin\varphi_w$: sine of the poleward edge of the winter cell
        - $\\sin\varphi_1$: sine of the cells' shared inner edge
        - $\\sin\varphi_s$: sine of the poleward edge of the summer cell
        - $T_{s1}$: surface temperature at the cells' shared inner edge

    sinlat_0 : sine of latitude of the prescribed heating maximum
    theta_ref : the reference potential temperature ($\theta_0$ in LH88)
    delta_h : the prescribed fractional horizontal temperature difference
    gamma : this times dry adiabatic lapse rate gives the actual lapse rate
    dtheta_dts : Approximation to d(\\hat\theta)/d(T_sfc)
    temp_tropo : tropopause temperature
    rot_rate, radius : planetary rotation rate and planetary radius
    c_p : specific heat of dry air at constant pressure

    """
    sinlat_south, sinlat_1, sinlat_north, temp_sfc_1 = x
    args = (
        sinlat_0,
        sinlat_1,
        temp_sfc_1,
        theta_ref,
        delta_h,
        gamma,
        dtheta_dts,
        temp_tropo,
        rot_rate,
        radius,
        c_p,
    )
    x0 = _lh88_amc_rce_theta_diff_fixed_tt(sinlat_south, *args)
    x1 = _lh88_amc_rce_theta_diff_fixed_tt(sinlat_north, *args)
    x2 = _lh88_cons_energy_integral_fixed_tt(sinlat_south, *args)
    x3 = _lh88_cons_energy_integral_fixed_tt(sinlat_north, *args)
    return np.array([x0, x1, x2, x3])


def equal_area_lh88_fixed_temp_tropo(
    init_guess,
    sinlat_0,
    theta_ref=THETA_REF,
    delta_h=DELTA_H,
    gamma=1.0,
    dtheta_dts=1.0,
    temp_tropo=TEMP_TROPO,
    rot_rate=ROT_RATE_EARTH,
    radius=RAD_EARTH,
    c_p=C_P,
):
    """Solve the LH88 equal-area model with fixed tropopause temperature.

    Variant of the Lindzen-Hou 1988 model that assumes a fixed lapse rate
    and tropopause temperature rather than the standard Boussinesq
    assumption of fixed tropopause height.

    Parameters
    ----------
    init_guess : array-like of length 4
        Initial guess for [sin(lat_winter), sin(lat_1), sin(lat_summer),
        T_sfc_1].
    sinlat_0 : float
        Sine of the latitude of maximum heating.
    theta_ref : float, optional
        Reference potential temperature (K). Default: THETA_REF.
    delta_h : float, optional
        Fractional horizontal temperature difference. Default: DELTA_H.
    gamma : float, optional
        Ratio of actual lapse rate to dry adiabatic. Default: 1.0.
    dtheta_dts : float, optional
        Approximation to d(theta_hat)/d(T_sfc). Default: 1.0.
    temp_tropo : float, optional
        Tropopause temperature (K). Default: TEMP_TROPO.
    rot_rate : float, optional
        Planetary rotation rate (rad/s). Default: Earth.
    radius : float, optional
        Planetary radius (m). Default: Earth.
    c_p : float, optional
        Specific heat at constant pressure (J/kg/K). Default: C_P.

    Returns
    -------
    numpy.ndarray of length 4
        Solution array: [sin(lat_winter), sin(lat_1), sin(lat_summer),
        T_sfc_1].
    """
    sol = scipy.optimize.root(
        _lh88_model_fixed_tt,
        init_guess,
        args=(
            sinlat_0,
            theta_ref,
            delta_h,
            gamma,
            dtheta_dts,
            temp_tropo,
            rot_rate,
            radius,
            c_p,
        ),
    )
    return sol.x


# Boussinesq, generic RCE field (rather than that from Lindzen and Hou 1988)
def _amc_rce_theta_diff_bouss(
    sinlat, sinlat_1, theta_1, theta_ref, del_h_over_ro, _theta_rce_func
):
    """Difference between AMC and RCE wind for a generic RCE wind field

    At the cell edges, this difference is zero by construction.

    """
    return _theta_hat_amc_lh88(
        sinlat, sinlat_1, theta_1, theta_ref, del_h_over_ro
    ) - _theta_rce_func(sinlat)


def _cons_energy_integral_bouss(
    sinlat_edge, sinlat_1, theta_1, theta_ref, del_h_over_ro, _theta_rce_func
):
    return scipy.integrate.quad(
        _amc_rce_theta_diff_bouss,
        sinlat_1,
        sinlat_edge,
        args=(sinlat_1, theta_1, theta_ref, del_h_over_ro, _theta_rce_func),
    )[0]


def _equal_area_model_bouss(x, theta_ref, del_h_over_ro, _theta_rce_func):
    """Matrix representation of the equal area model for any given RCE field

    Parameters
    ----------
    x : length-4 array
        Values correspond, in order, to:

        - $\\sin\varphi_w$: sine of the poleward edge of the winter cell
        - $\\sin\varphi_1$: sine of the cells' shared inner edge
        - $\\sin\varphi_s$: sine of the poleward edge of the summer cell
        - $\theta_1$: potential temperature at the cells' shared inner edge

    sinlat_0 : sine of latitude of the prescribed heating maximum
    theta_ref : the reference potential temperature ($\theta_0$ in LH88)
    del_h_over_ro : equal to Omega^2 a^2 / (gH)

    """
    sinlat_south, sinlat_1, sinlat_north, theta_1 = x
    args = (sinlat_1, theta_1, theta_ref, del_h_over_ro, _theta_rce_func)
    x0 = _amc_rce_theta_diff_bouss(sinlat_south, *args)
    x1 = _amc_rce_theta_diff_bouss(sinlat_north, *args)
    x2 = _cons_energy_integral_bouss(sinlat_south, *args)
    x3 = _cons_energy_integral_bouss(sinlat_north, *args)
    return np.array([x0, x1, x2, x3])


def equal_area_bouss(init_guess, theta_ref, del_h_over_ro, _theta_rce_func):
    """Boussinesq equal-area model for arbitrary RCE potential temperatures.

    Solves the equal-area constraint system for the Boussinesq case with
    a user-supplied RCE potential temperature function.

    Parameters
    ----------
    init_guess : array-like of length 4
        Initial guess for [sin(lat_winter), sin(lat_1), sin(lat_summer),
        theta_1].
    theta_ref : float
        Reference potential temperature (K).
    del_h_over_ro : float
        Ratio of delta_h to the thermal Rossby number, equal to
        Omega^2 a^2 / (gH).
    _theta_rce_func : callable
        Function of sin(latitude) returning the RCE depth-averaged
        potential temperature.

    Returns
    -------
    numpy.ndarray of length 4
        Solution array: [sin(lat_winter), sin(lat_1), sin(lat_summer),
        theta_1].
    """
    sol = scipy.optimize.root(
        _equal_area_model_bouss,
        init_guess,
        args=(theta_ref, del_h_over_ro, _theta_rce_func),
    )
    return sol.x


# Convective quasi-equilibrium atmosphere, general RCE profile
def _amc_rce_theta_diff_cqe(
    sinlat, sinlat_1, theta_1, sfc_trop_diff, c_p, radius, rot_rate, _theta_rce_func
):
    """Difference between AMC and RCE wind for a generic RCE wind field

    At the cell edges, this difference is zero by construction.

    """
    return pot_temp_amc_cqe(
        sin2deg(sinlat),
        sin2deg(sinlat_1),
        theta_1,
        sfc_trop_diff,
        c_p=c_p,
        radius=radius,
        rot_rate=rot_rate,
    ) - _theta_rce_func(sinlat)


def _cons_energy_integral_cqe(
    sinlat_edge,
    sinlat_1,
    theta_1,
    sfc_trop_diff,
    c_p,
    radius,
    rot_rate,
    _theta_rce_func,
):
    return scipy.integrate.quad(
        _amc_rce_theta_diff_cqe,
        sinlat_1,
        sinlat_edge,
        args=(sinlat_1, theta_1, sfc_trop_diff, c_p, radius, rot_rate, _theta_rce_func),
    )[0]


def _equal_area_model_cqe(x, sfc_trop_diff, c_p, radius, rot_rate, _theta_rce_func):
    """Matrix representation of the equal area model for any given RCE field

    Parameters
    ----------
    x : length-4 array
        Values correspond, in order, to:

        - $\\sin\varphi_w$: sine of the poleward edge of the winter cell
        - $\\sin\varphi_1$: sine of the cells' shared inner edge
        - $\\sin\varphi_s$: sine of the poleward edge of the summer cell
        - $\theta_1$: potential temperature at the cells' shared inner edge

    """
    sinlat_south, sinlat_1, sinlat_north, theta_1 = x
    args = (sinlat_1, theta_1, sfc_trop_diff, c_p, radius, rot_rate, _theta_rce_func)
    x0 = _amc_rce_theta_diff_cqe(sinlat_south, *args)
    x1 = _amc_rce_theta_diff_cqe(sinlat_north, *args)
    x2 = _cons_energy_integral_cqe(sinlat_south, *args)
    x3 = _cons_energy_integral_cqe(sinlat_north, *args)
    return np.array([x0, x1, x2, x3])


def equal_area_cqe(init_guess, sfc_trop_diff, c_p, radius, rot_rate, _theta_rce_func):
    """CQE equal-area model for arbitrary RCE potential temperatures.

    Solves the equal-area constraint system under the convective
    quasi-equilibrium (CQE) assumption with a user-supplied RCE
    potential temperature function.

    Parameters
    ----------
    init_guess : array-like of length 4
        Initial guess for [sin(lat_winter), sin(lat_1), sin(lat_summer),
        theta_1].
    sfc_trop_diff : float
        Surface-to-tropopause temperature difference (K).
    c_p : float
        Specific heat at constant pressure (J/kg/K).
    radius : float
        Planetary radius (m).
    rot_rate : float
        Planetary rotation rate (rad/s).
    _theta_rce_func : callable
        Function of sin(latitude) returning the RCE depth-averaged
        potential temperature.

    Returns
    -------
    numpy.ndarray of length 4
        Solution array: [sin(lat_winter), sin(lat_1), sin(lat_summer),
        theta_1].
    """
    sol = scipy.optimize.root(
        _equal_area_model_cqe,
        init_guess,
        args=(sfc_trop_diff, c_p, radius, rot_rate, _theta_rce_func),
    )
    return sol.x


# CQE, fixed tropopause temperature.
def _theta_hat_amc_rce_diff_cqe_fixed_tt(
    sinlat,
    sinlat_1,
    theta_1,
    theta_rce_func,
    theta_guesses,
    temp_tropo,
    rel_hum,
    pressure,
    c_p,
    radius,
    rot_rate,
):
    """Difference between AMC and RCE wind for a generic RCE wind field

    At the cell edges, this difference is zero by construction.

    """
    return pot_temp_amc_cqe_fixed_tt(
        sin2deg(sinlat),
        theta_guesses,
        theta_1,
        sin2deg(sinlat_1),
        temp_tropo,
        rel_hum=rel_hum,
        pressure=pressure,
        c_p=c_p,
        radius=radius,
        rot_rate=rot_rate,
    ) - theta_rce_func(sinlat)


def _cons_energy_integral_cqe_fixed_tt(
    sinlat_edge,
    sinlat_1,
    theta_1,
    theta_rce_func,
    theta_guesses,
    temp_tropo,
    rel_hum,
    pressure,
    c_p,
    radius,
    rot_rate,
):
    return scipy.integrate.quad(
        _theta_hat_amc_rce_diff_cqe_fixed_tt,
        sinlat_1,
        sinlat_edge,
        args=(
            sinlat_1,
            theta_1,
            theta_rce_func,
            theta_guesses,
            temp_tropo,
            rel_hum,
            pressure,
            c_p,
            radius,
            rot_rate,
        ),
    )[0]


def _eq_area_cqe_fixed_tt(
    x,
    theta_rce_func,
    theta_guesses,
    temp_tropo,
    rel_hum,
    pressure,
    c_p,
    radius,
    rot_rate,
):
    """Matrix representation of the equal area model for any given RCE field

    Parameters
    ----------
    x : length-4 array
        Values correspond, in order, to:

        - $\\sin\varphi_w$: sine of the poleward edge of the winter cell
        - $\\sin\varphi_1$: sine of the cells' shared inner edge
        - $\\sin\varphi_s$: sine of the poleward edge of the summer cell
        - $\theta_1$: potential temperature at the cells' shared inner edge

    """
    sinlat_south, sinlat_1, sinlat_north, theta_1 = x
    args = (
        sinlat_1,
        theta_1,
        theta_rce_func,
        theta_guesses,
        temp_tropo,
        rel_hum,
        pressure,
        c_p,
        radius,
        rot_rate,
    )
    x0 = _theta_hat_amc_rce_diff_cqe_fixed_tt(sinlat_south, *args)
    x1 = _theta_hat_amc_rce_diff_cqe_fixed_tt(sinlat_north, *args)
    x2 = _cons_energy_integral_cqe_fixed_tt(sinlat_south, *args)
    x3 = _cons_energy_integral_cqe_fixed_tt(sinlat_north, *args)
    return np.array([float(x0), float(x1), x2, x3])


def equal_area_cqe_fixed_tt(
    init_guess,
    theta_rce_func,
    theta_guesses,
    temp_tropo=TEMP_TROPO,
    rel_hum=REL_HUM,
    pressure=P0,
    c_p=C_P,
    radius=RAD_EARTH,
    rot_rate=ROT_RATE_EARTH,
):
    """CQE equal-area model with fixed tropopause temperature.

    Solves the equal-area constraint system under the convective
    quasi-equilibrium (CQE) assumption with a fixed tropopause
    temperature.

    Parameters
    ----------
    init_guess : array-like of length 4
        Initial guess for [sin(lat_winter), sin(lat_1), sin(lat_summer),
        theta_1].
    theta_rce_func : callable
        Function of sin(latitude) returning the RCE depth-averaged
        potential temperature.
    theta_guesses : array-like
        Initial guesses for potential temperature in the CQE solver.
    temp_tropo : float, optional
        Tropopause temperature (K). Default: TEMP_TROPO.
    rel_hum : float, optional
        Relative humidity. Default: REL_HUM.
    pressure : float, optional
        Reference pressure (Pa). Default: P0.
    c_p : float, optional
        Specific heat at constant pressure (J/kg/K). Default: C_P.
    radius : float, optional
        Planetary radius (m). Default: Earth.
    rot_rate : float, optional
        Planetary rotation rate (rad/s). Default: Earth.

    Returns
    -------
    numpy.ndarray of length 4
        Solution array: [sin(lat_winter), sin(lat_1), sin(lat_summer),
        theta_1].
    """
    sol = scipy.optimize.root(
        _eq_area_cqe_fixed_tt,
        init_guess,
        args=(
            theta_rce_func,
            theta_guesses,
            temp_tropo,
            rel_hum,
            pressure,
            c_p,
            radius,
            rot_rate,
        ),
    )
    return sol.x


if __name__ == "__main__":
    pass
