#! /usr/bin/env python
"""Equal area model solvers."""

import numpy as np
import scipy.integrate
import scipy.optimize

from .constants import (
    C_P,
    DELTA_H,
    P0,
    RAD_EARTH,
    REL_HUM,
    ROT_RATE_EARTH,
    TEMP_TROPO,
    THETA_REF,
)
from .nb_utils import sin2deg
from .grad_bal import pot_temp_amc_cqe
from .fixed_temp_tropo import (
    pot_temp_amc_cqe_fixed_tt,
    pot_temp_avg_amc_fixed_tt_bouss,
)
from .lindzen_hou_1988 import pot_temp_rce_lh88


# Original Lindzen and Hou 1988.
def _theta_hat_amc_lh88(sinlat, sinlat_1, theta_hat_1, theta_ref,
                        del_h_over_ro):
    """Lindzen and Hou 1988 Eq. 7."""
    return (theta_hat_1 - 0.5*theta_ref*del_h_over_ro *
            (sinlat**2 - sinlat_1**2)**2 / (1 - sinlat**2))


def _theta_hat_rce_lh88(sinlat, sinlat_0, theta_ref, delta_h):
    """Lindzen and Hou 1988 Eq. 1b."""
    return theta_ref*(1 + delta_h / 3. - delta_h * (sinlat - sinlat_0)**2)


def _lh88_amc_rce_theta_hat_diff(sinlat, sinlat_0, sinlat_1, theta_hat_1,
                                 theta_ref, delta_h, thermal_ro):
    """Difference between AMC and RCE depth-averaged potential temperatures.

    At the cell edges, this difference is zero by construction.

    """
    return (_theta_hat_amc_lh88(sinlat, sinlat_1, theta_hat_1,
                                theta_ref, delta_h / thermal_ro) -
            _theta_hat_rce_lh88(sinlat, sinlat_0, theta_ref, delta_h))


def _lh88_cons_energy_integral(sinlat_edge, sinlat_0, sinlat_1, theta_hat_1,
                               theta_ref, delta_h, thermal_ro):
    return scipy.integrate.quad(
        _lh88_amc_rce_theta_hat_diff, sinlat_1, sinlat_edge,
        args=(sinlat_0, sinlat_1, theta_hat_1, theta_ref,
              delta_h, thermal_ro))[0]


def _lh88_model(x, sinlat_0, theta_ref, delta_h, thermal_ro):
    """Matrix representation of the Lindzen-Hou 1988 equal-area model.

    Parameters
    ----------
    x : length-4 array
        Values correspond, in order, to:

        - $\sin\varphi_w$: sine of the poleward edge of the winter cell
        - $\sin\varphi_1$: sine of the cells' shared inner edge
        - $\sin\varphi_s$: sine of the poleward edge of the summer cell
        - $\theta_hat_1$: potential temperature at the cells' shared inner edge

    sinlat_0 : sine of latitude of the prescribed heating maximum
    theta_ref : the reference potential temperature ($\theta_hat_0$ in LH88)
    delta_h : the prescribed fractional horizontal temperature difference
    thermal_ro : the prescribed thermal rossby number

    """
    sinlat_south, sinlat_1, sinlat_north, theta_hat_1 = x
    args = (sinlat_0, sinlat_1, theta_hat_1,
            theta_ref, delta_h, thermal_ro)
    x0 = _lh88_amc_rce_theta_hat_diff(sinlat_south, *args)
    x1 = _lh88_amc_rce_theta_hat_diff(sinlat_north, *args)
    x2 = _lh88_cons_energy_integral(sinlat_south, *args)
    x3 = _lh88_cons_energy_integral(sinlat_north, *args)
    return np.array([x0, x1, x2, x3])


def equal_area_lh88(init_guess, sinlat_0, theta_ref=THETA_REF, delta_h=DELTA_H,
                    thermal_ro=0.15):
    sol = scipy.optimize.root(
        _lh88_model, init_guess, args=(sinlat_0, theta_ref,
                                       delta_h, thermal_ro))
    return sol.x


# Modified LH88: fixed lapse rate and tropopause temperature.
def _lh88_amc_rce_theta_diff_fixed_tt(sinlat, sinlat_0, sinlat_1, temp_sfc_1,
                                      theta_ref, delta_h, gamma, dtheta_dts,
                                      temp_tropo, rot_rate, radius, c_p):
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
        height=1.,
        delta_h=delta_h,
    )
    return theta_avg_amc - theta_avg_rce


def _lh88_cons_energy_integral_fixed_tt(sinlat_edge, sinlat_0, sinlat_1,
                                        temp_sfc_1, theta_ref, delta_h, gamma,
                                        dtheta_dts, temp_tropo, rot_rate,
                                        radius, c_p):
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


def _lh88_model_fixed_tt(x, sinlat_0, theta_ref, delta_h, gamma, dtheta_dts,
                         temp_tropo, rot_rate, radius, c_p):
    """LH88 equal area, assuming fixed lapse rate and tropopause temperature.

    Parameters
    ----------
    x : length-4 array
        Values correspond, in order, to:

        - $\sin\varphi_w$: sine of the poleward edge of the winter cell
        - $\sin\varphi_1$: sine of the cells' shared inner edge
        - $\sin\varphi_s$: sine of the poleward edge of the summer cell
        - $T_{s1}$: surface temperature at the cells' shared inner edge

    sinlat_0 : sine of latitude of the prescribed heating maximum
    theta_ref : the reference potential temperature ($\theta_0$ in LH88)
    delta_h : the prescribed fractional horizontal temperature difference
    gamma : this times dry adiabatic lapse rate gives the actual lapse rate
    dtheta_dts : Approximation to d(\hat\theta)/d(T_sfc)
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


def equal_area_lh88_fixed_temp_tropo(init_guess, sinlat_0, theta_ref=THETA_REF,
                                     delta_h=DELTA_H, gamma=1., dtheta_dts=1.,
                                     temp_tropo=TEMP_TROPO,
                                     rot_rate=ROT_RATE_EARTH, radius=RAD_EARTH,
                                     c_p=C_P):
    sol = scipy.optimize.root(
        _lh88_model_fixed_tt, init_guess,
        args=(sinlat_0, theta_ref, delta_h, gamma, dtheta_dts,
              temp_tropo, rot_rate, radius, c_p),
    )
    return sol.x


# Boussinesq, generic RCE field (rather than that from Lindzen and Hou 1988)
def _amc_rce_theta_diff_bouss(sinlat, sinlat_1, theta_1, theta_ref,
                              del_h_over_ro, _theta_rce_func):
    """Difference between AMC and RCE wind for a generic RCE wind field

    At the cell edges, this difference is zero by construction.

    """
    return _theta_hat_amc_lh88(sinlat, sinlat_1, theta_1, theta_ref,
                               del_h_over_ro) - _theta_rce_func(sinlat)


def _cons_energy_integral_bouss(sinlat_edge, sinlat_1, theta_1, theta_ref,
                                del_h_over_ro, _theta_rce_func):
    return scipy.integrate.quad(
        _amc_rce_theta_diff_bouss, sinlat_1, sinlat_edge,
        args=(sinlat_1, theta_1, theta_ref, del_h_over_ro,
              _theta_rce_func))[0]


def _equal_area_model_bouss(x, theta_ref, del_h_over_ro, _theta_rce_func):
    """Matrix representation of the equal area model for any given RCE field

    Parameters
    ----------
    x : length-4 array
        Values correspond, in order, to:

        - $\sin\varphi_w$: sine of the poleward edge of the winter cell
        - $\sin\varphi_1$: sine of the cells' shared inner edge
        - $\sin\varphi_s$: sine of the poleward edge of the summer cell
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
    """Boussinesq equal area model for arbitrary RCE potential temperatures."""
    sol = scipy.optimize.root(
        _equal_area_model_bouss, init_guess,
        args=(theta_ref, del_h_over_ro, _theta_rce_func))
    return sol.x


# Convective quasi-equilibrium atmosphere, general RCE profile
def _amc_rce_theta_diff_cqe(sinlat, sinlat_1, theta_1, sfc_trop_diff,
                            c_p, radius, rot_rate, _theta_rce_func):
    """Difference between AMC and RCE wind for a generic RCE wind field

    At the cell edges, this difference is zero by construction.

    """
    return pot_temp_amc_cqe(sin2deg(sinlat), sin2deg(sinlat_1), theta_1,
                            sfc_trop_diff, c_p=c_p, radius=radius,
                            rot_rate=rot_rate) - _theta_rce_func(sinlat)


def _cons_energy_integral_cqe(sinlat_edge, sinlat_1, theta_1,
                              sfc_trop_diff, c_p, radius, rot_rate,
                              _theta_rce_func):
    return scipy.integrate.quad(
        _amc_rce_theta_diff_cqe, sinlat_1, sinlat_edge,
        args=(sinlat_1, theta_1, sfc_trop_diff, c_p, radius,
              rot_rate, _theta_rce_func))[0]


def _equal_area_model_cqe(x, sfc_trop_diff, c_p, radius, rot_rate,
                          _theta_rce_func):
    """Matrix representation of the equal area model for any given RCE field

    Parameters
    ----------
    x : length-4 array
        Values correspond, in order, to:

        - $\sin\varphi_w$: sine of the poleward edge of the winter cell
        - $\sin\varphi_1$: sine of the cells' shared inner edge
        - $\sin\varphi_s$: sine of the poleward edge of the summer cell
        - $\theta_1$: potential temperature at the cells' shared inner edge

    """
    sinlat_south, sinlat_1, sinlat_north, theta_1 = x
    args = (sinlat_1, theta_1, sfc_trop_diff, c_p,
            radius, rot_rate, _theta_rce_func)
    x0 = _amc_rce_theta_diff_cqe(sinlat_south, *args)
    x1 = _amc_rce_theta_diff_cqe(sinlat_north, *args)
    x2 = _cons_energy_integral_cqe(sinlat_south, *args)
    x3 = _cons_energy_integral_cqe(sinlat_north, *args)
    return np.array([x0, x1, x2, x3])


def equal_area_cqe(init_guess, sfc_trop_diff, c_p, radius,
                   rot_rate, _theta_rce_func):
    """cqeinesq equal area model for arbitrary RCE potential temperatures."""
    sol = scipy.optimize.root(_equal_area_model_cqe, init_guess,
                              args=(sfc_trop_diff, c_p, radius,
                                    rot_rate, _theta_rce_func))
    return sol.x


# CQE, fixed tropopause temperature.
def _theta_hat_amc_rce_diff_cqe_fixed_tt(sinlat, sinlat_1, theta_1,
                                         theta_rce_func, theta_guesses,
                                         temp_tropo, rel_hum, pressure, c_p,
                                         radius, rot_rate):
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


def _cons_energy_integral_cqe_fixed_tt(sinlat_edge, sinlat_1, theta_1,
                                       theta_rce_func, theta_guesses,
                                       temp_tropo, rel_hum, pressure, c_p,
                                       radius, rot_rate):
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
        )
    )[0]


def _eq_area_cqe_fixed_tt(x, theta_rce_func, theta_guesses, temp_tropo,
                          rel_hum, pressure, c_p, radius, rot_rate):
    """Matrix representation of the equal area model for any given RCE field

    Parameters
    ----------
    x : length-4 array
        Values correspond, in order, to:

        - $\sin\varphi_w$: sine of the poleward edge of the winter cell
        - $\sin\varphi_1$: sine of the cells' shared inner edge
        - $\sin\varphi_s$: sine of the poleward edge of the summer cell
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


def equal_area_cqe_fixed_tt(init_guess, theta_rce_func, theta_guesses,
                            temp_tropo=TEMP_TROPO, rel_hum=REL_HUM,
                            pressure=P0, c_p=C_P, radius=RAD_EARTH,
                            rot_rate=ROT_RATE_EARTH):
    """CQE equal area model with fixed tropopause temperature."""
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
        )
    )
    return sol.x


if __name__ == '__main__':
    pass
