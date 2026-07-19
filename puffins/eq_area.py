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

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
import scipy.integrate
import scipy.optimize

from ._typing import ArrayLike
from .constants import (
    C_P,
    DELTA_H,
    DELTA_V,
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
    ross_num: float,
    therm_ross_num: float,
    delta_h: float,
    theta_ref: float = THETA_REF,
) -> float:
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
    lat: ArrayLike,
    ross_num: float,
    therm_ross_num: float,
    delta_h: float,
    theta_ref: float = THETA_REF,
) -> ArrayLike:
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
    ross_num: float,
    therm_ross_num: float,
) -> float:
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
    return float(np.rad2deg((5 * therm_ross_num / (3 * ross_num)) ** 0.5))


def heat_flux_mean_ro(
    lat: ArrayLike,
    therm_ross_num: float,
    ross_num: float,
    theta_ref: float = THETA_REF,
    radius: float = RAD_EARTH,
    height: float = HEIGHT_TROPO,
    delta_h: float = DELTA_H,
    tau: float = 20 * 86400,
) -> ArrayLike:
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
    return cast(
        ArrayLike,
        theta_ref
        * prefac
        * (lat_div_lat_ro - 2 * lat_div_lat_ro**3 + lat_div_lat_ro**5),
    )


def mom_flux_mean_ro(
    lat: ArrayLike,
    therm_ross_num: float,
    ross_num: float,
    radius: float = RAD_EARTH,
    rot_rate: float = ROT_RATE_EARTH,
    height: float = HEIGHT_TROPO,
    delta_h: float = DELTA_H,
    delta_v: float = DELTA_V,
    tau: float = 20 * 86400,
) -> ArrayLike:
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
        Meridional momentum flux (m^3/s^2).
    """
    prefac = rot_rate * radius**2 * height * delta_h / (6 * tau * delta_v)
    latrad = np.deg2rad(lat)
    latrad2 = latrad**2
    return cast(
        ArrayLike,
        prefac
        * latrad**3
        * (
            5 * therm_ross_num / 3
            - ross_num * latrad2 * (2 - 3 * ross_num * latrad2 / (5 * therm_ross_num))
        ),
    )


def u_sfc_mean_ro(
    lat: ArrayLike,
    therm_ross_num: float,
    ross_num: float,
    radius: float = RAD_EARTH,
    rot_rate: float = ROT_RATE_EARTH,
    height: float = HEIGHT_TROPO,
    delta_h: float = DELTA_H,
    delta_v: float = DELTA_V,
    tau: float = 20 * 86400,
    drag_coeff: float = 0.005,
) -> ArrayLike:
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
        Surface drag coefficient (m/s). Default: 0.005.

    Returns
    -------
    array-like
        Surface zonal wind (m/s).
    """
    prefac = (
        -25 * rot_rate * radius * height * delta_h / (18 * drag_coeff * tau * delta_v)
    )
    lat_div_lat_ro = lat / cell_edge_mean_ro(ross_num, therm_ross_num)
    return cast(
        ArrayLike,
        prefac
        * (therm_ross_num / ross_num) ** 2
        * (
            lat_div_lat_ro**2
            - (10 / 3) * lat_div_lat_ro**4
            + (7 / 3) * lat_div_lat_ro**6
        ),
    )


# Solutions for small-angle, linear meridional profile in Ro
def cell_edge_lin_ro_lata0_full(
    therm_ross: float, ross_ascent: float, ross_descent: float
) -> float:
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

    def _term_latd4(ross_ascent: float, delro: float) -> float:
        return ross_ascent**2 / 7 - ross_ascent * delro / 6 + 4 * delro**2 / 81

    def _term_latd2(ross_ascent: float, delro: float) -> float:
        return 2 * ross_ascent / 5 - 2 * delro / 9

    def _term_latd0(therm_ro: float) -> float:
        return -2 * therm_ro / 3

    poly_obj = np.polynomial.Polynomial(
        [
            _term_latd0(therm_ross),
            _term_latd2(ross_ascent, delro),
            _term_latd4(ross_ascent, delro),
        ]
    )
    roots = poly_obj.roots()
    real_roots = roots[np.isclose(roots.imag, 0.0)].real
    # The polynomial is quadratic in latitude-squared with a positive
    # leading and non-positive constant coefficient, so the physical
    # (non-negative) root is the larger of the two real roots.
    if real_roots.size == 0 or real_roots.max() < -1e-12:
        raise ValueError(
            "no non-negative real root for the cell-edge polynomial in "
            f"latitude-squared; got roots {roots}"
        )
    return float(np.rad2deg(max(real_roots.max(), 0.0) ** 0.5))


def cell_edge_lin_ro_lata0_approx(
    therm_ross: float, ross_ascent: float, ross_descent: float
) -> float:
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
    return float(
        np.rad2deg((15 * therm_ross / (9 * ross_ascent - 5 * delta_ross)) ** 0.5)
    )


def eq_pot_temp_lin_ro_lata0_small_ang(
    therm_ross: float,
    ross_ascent: float,
    ross_descent: float,
    delta_h: float = DELTA_H,
    theta_ref: float = THETA_REF,
) -> float:
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
    return float(
        theta_ref
        * (
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
    )


def pot_temp_lin_ro_eq_area(
    lat: ArrayLike,
    therm_ross: float,
    ross_ascent: float,
    ross_descent: float,
    delta_h: float = DELTA_H,
    theta_ref: float = THETA_REF,
) -> ArrayLike:
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

    Returns
    -------
    array-like
        Potential temperature at each latitude (K).

    Notes
    -----
    The profile shape is computed with the Burger number implied by the
    given thermal Rossby number, ``therm_ross / delta_h``, so that it is
    consistent with the cell edge and equatorial temperature (which are
    parameterized by ``therm_ross``).  This function formerly accepted
    planetary parameters that set an independent (and generally
    inconsistent) Burger number for the profile shape.
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
        theta_ref=theta_ref,
        burg_num=therm_ross / delta_h,
    )


def _checked_root_solve(
    func: Callable[..., np.ndarray],
    init_guess: Sequence[float] | np.ndarray,
    args: tuple,
) -> np.ndarray:
    """Solve func(x, *args) = 0, raising if scipy reports non-convergence.

    Without this check a failed solve silently returns the (possibly
    unmodified) iterate, which is indistinguishable from a solution.
    """
    sol = scipy.optimize.root(func, init_guess, args=args)
    if not sol.success:
        raise RuntimeError(f"equal-area solver did not converge: {sol.message}")
    return cast(np.ndarray, sol.x)


# Original Lindzen and Hou 1988.
def _theta_hat_amc_lh88(
    sinlat: float,
    sinlat_1: float,
    theta_hat_1: float,
    theta_ref: float,
    del_h_over_ro: float,
) -> float:
    """Lindzen and Hou 1988 Eq. 7."""
    return theta_hat_1 - 0.5 * theta_ref * del_h_over_ro * (
        sinlat**2 - sinlat_1**2
    ) ** 2 / (1 - sinlat**2)


def _theta_hat_rce_lh88(
    sinlat: float, sinlat_0: float, theta_ref: float, delta_h: float
) -> float:
    """Lindzen and Hou 1988 Eq. 1b."""
    return theta_ref * (1 + delta_h / 3.0 - delta_h * (sinlat - sinlat_0) ** 2)


def _lh88_amc_rce_theta_hat_diff(
    sinlat: float,
    sinlat_0: float,
    sinlat_1: float,
    theta_hat_1: float,
    theta_ref: float,
    delta_h: float,
    thermal_ro: float,
) -> float:
    """Difference between AMC and RCE depth-averaged potential temperatures.

    At the cell edges, this difference is zero by construction.

    """
    return _theta_hat_amc_lh88(
        sinlat, sinlat_1, theta_hat_1, theta_ref, delta_h / thermal_ro
    ) - _theta_hat_rce_lh88(sinlat, sinlat_0, theta_ref, delta_h)


def _lh88_cons_energy_integral(
    sinlat_edge: float,
    sinlat_0: float,
    sinlat_1: float,
    theta_hat_1: float,
    theta_ref: float,
    delta_h: float,
    thermal_ro: float,
) -> float:
    return float(
        scipy.integrate.quad(
            _lh88_amc_rce_theta_hat_diff,
            sinlat_1,
            sinlat_edge,
            args=(sinlat_0, sinlat_1, theta_hat_1, theta_ref, delta_h, thermal_ro),
        )[0]
    )


def _lh88_model(
    x: np.ndarray,
    sinlat_0: float,
    theta_ref: float,
    delta_h: float,
    thermal_ro: float,
) -> np.ndarray:
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
    init_guess: Sequence[float] | np.ndarray,
    sinlat_0: float,
    theta_ref: float = THETA_REF,
    delta_h: float = DELTA_H,
    thermal_ro: float = 0.15,
) -> np.ndarray:
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
    return _checked_root_solve(
        _lh88_model, init_guess, (sinlat_0, theta_ref, delta_h, thermal_ro)
    )


# Modified LH88: fixed lapse rate and tropopause temperature.
def _lh88_amc_rce_theta_diff_fixed_tt(
    sinlat: float,
    sinlat_0: float,
    sinlat_1: float,
    temp_sfc_1: float,
    theta_ref: float,
    delta_h: float,
    gamma: float,
    dtheta_dts: float,
    temp_tropo: float,
    rot_rate: float,
    radius: float,
    c_p: float,
) -> float:
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
    return float(theta_avg_amc - theta_avg_rce)


def _lh88_cons_energy_integral_fixed_tt(
    sinlat_edge: float,
    sinlat_0: float,
    sinlat_1: float,
    temp_sfc_1: float,
    theta_ref: float,
    delta_h: float,
    gamma: float,
    dtheta_dts: float,
    temp_tropo: float,
    rot_rate: float,
    radius: float,
    c_p: float,
) -> float:
    return float(
        scipy.integrate.quad(
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
    )


def _lh88_model_fixed_tt(
    x: np.ndarray,
    sinlat_0: float,
    theta_ref: float,
    delta_h: float,
    gamma: float,
    dtheta_dts: float,
    temp_tropo: float,
    rot_rate: float,
    radius: float,
    c_p: float,
) -> np.ndarray:
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
    init_guess: Sequence[float] | np.ndarray,
    sinlat_0: float,
    theta_ref: float = THETA_REF,
    delta_h: float = DELTA_H,
    gamma: float = 1.0,
    dtheta_dts: float = 1.0,
    temp_tropo: float = TEMP_TROPO,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
    c_p: float = C_P,
) -> np.ndarray:
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
    return _checked_root_solve(
        _lh88_model_fixed_tt,
        init_guess,
        (
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


# Boussinesq, generic RCE field (rather than that from Lindzen and Hou 1988)
def _amc_rce_theta_diff_bouss(
    sinlat: float,
    sinlat_1: float,
    theta_1: float,
    theta_ref: float,
    del_h_over_ro: float,
    _theta_rce_func: Callable[[float], float],
) -> float:
    """Difference between AMC and RCE wind for a generic RCE wind field

    At the cell edges, this difference is zero by construction.

    """
    return _theta_hat_amc_lh88(
        sinlat, sinlat_1, theta_1, theta_ref, del_h_over_ro
    ) - _theta_rce_func(sinlat)


def _cons_energy_integral_bouss(
    sinlat_edge: float,
    sinlat_1: float,
    theta_1: float,
    theta_ref: float,
    del_h_over_ro: float,
    _theta_rce_func: Callable[[float], float],
) -> float:
    return float(
        scipy.integrate.quad(
            _amc_rce_theta_diff_bouss,
            sinlat_1,
            sinlat_edge,
            args=(sinlat_1, theta_1, theta_ref, del_h_over_ro, _theta_rce_func),
        )[0]
    )


def _equal_area_model_bouss(
    x: np.ndarray,
    theta_ref: float,
    del_h_over_ro: float,
    _theta_rce_func: Callable[[float], float],
) -> np.ndarray:
    """Matrix representation of the equal area model for any given RCE field

    Parameters
    ----------
    x : length-4 array
        Values correspond, in order, to:

        - $\\sin\varphi_w$: sine of the poleward edge of the winter cell
        - $\\sin\varphi_1$: sine of the cells' shared inner edge
        - $\\sin\varphi_s$: sine of the poleward edge of the summer cell
        - $\theta_1$: potential temperature at the cells' shared inner edge

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


def equal_area_bouss(
    init_guess: Sequence[float] | np.ndarray,
    theta_ref: float,
    del_h_over_ro: float,
    _theta_rce_func: Callable[[float], float],
) -> np.ndarray:
    """Boussinesq equal-area model for arbitrary RCE potential temperatures.

    Solves the equal-area constraint system for the Boussinesq case with
    a user-supplied RCE potential temperature function.  The circulation
    temperature field is the angular-momentum-conserving (Ro = 1) one.
    To solve the fixed-Ro generalization in the small-angle limit,
    multiply ``del_h_over_ro`` by Ro (the small-angle fixed-Ro field is
    the AMC field scaled by Ro); note this scaling is NOT exact at full
    angle, where the fixed-Ro gradient-balanced field has a different
    functional form.

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
    return _checked_root_solve(
        _equal_area_model_bouss,
        init_guess,
        (theta_ref, del_h_over_ro, _theta_rce_func),
    )


# Convective quasi-equilibrium atmosphere, general RCE profile
def _amc_rce_theta_diff_cqe(
    sinlat: float,
    sinlat_1: float,
    theta_1: float,
    sfc_trop_diff: float,
    c_p: float,
    radius: float,
    rot_rate: float,
    _theta_rce_func: Callable[[float], float],
) -> float:
    """Difference between AMC and RCE wind for a generic RCE wind field

    At the cell edges, this difference is zero by construction.

    """
    return float(
        pot_temp_amc_cqe(
            sin2deg(sinlat),
            sin2deg(sinlat_1),
            theta_1,
            sfc_trop_diff,
            c_p=c_p,
            radius=radius,
            rot_rate=rot_rate,
        )
        - _theta_rce_func(sinlat)
    )


def _cons_energy_integral_cqe(
    sinlat_edge: float,
    sinlat_1: float,
    theta_1: float,
    sfc_trop_diff: float,
    c_p: float,
    radius: float,
    rot_rate: float,
    _theta_rce_func: Callable[[float], float],
) -> float:
    return float(
        scipy.integrate.quad(
            _amc_rce_theta_diff_cqe,
            sinlat_1,
            sinlat_edge,
            args=(
                sinlat_1,
                theta_1,
                sfc_trop_diff,
                c_p,
                radius,
                rot_rate,
                _theta_rce_func,
            ),
        )[0]
    )


def _equal_area_model_cqe(
    x: np.ndarray,
    sfc_trop_diff: float,
    c_p: float,
    radius: float,
    rot_rate: float,
    _theta_rce_func: Callable[[float], float],
) -> np.ndarray:
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


def equal_area_cqe(
    init_guess: Sequence[float] | np.ndarray,
    sfc_trop_diff: float,
    c_p: float,
    radius: float,
    rot_rate: float,
    _theta_rce_func: Callable[[float], float],
) -> np.ndarray:
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
    return _checked_root_solve(
        _equal_area_model_cqe,
        init_guess,
        (sfc_trop_diff, c_p, radius, rot_rate, _theta_rce_func),
    )


# CQE, fixed tropopause temperature.
def _theta_hat_amc_rce_diff_cqe_fixed_tt(
    sinlat: float,
    sinlat_1: float,
    theta_1: float,
    theta_rce_func: Callable[[float], float],
    theta_guesses: np.ndarray,
    temp_tropo: float,
    rel_hum: float,
    pressure: float,
    c_p: float,
    radius: float,
    rot_rate: float,
) -> float:
    """Difference between AMC and RCE wind for a generic RCE wind field

    At the cell edges, this difference is zero by construction.

    """
    theta_amc = pot_temp_amc_cqe_fixed_tt(
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
    )
    # pot_temp_amc_cqe_fixed_tt returns a 1-element DataArray for scalar
    # input, but scipy.integrate.quad needs a plain float from its
    # integrand (numpy >= 2.0 forbids float() on size-1 arrays of ndim >= 1).
    return float(np.asarray(theta_amc).squeeze()) - float(theta_rce_func(sinlat))


def _cons_energy_integral_cqe_fixed_tt(
    sinlat_edge: float,
    sinlat_1: float,
    theta_1: float,
    theta_rce_func: Callable[[float], float],
    theta_guesses: np.ndarray,
    temp_tropo: float,
    rel_hum: float,
    pressure: float,
    c_p: float,
    radius: float,
    rot_rate: float,
) -> float:
    return float(
        scipy.integrate.quad(
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
    )


def _eq_area_cqe_fixed_tt(
    x: np.ndarray,
    theta_rce_func: Callable[[float], float],
    theta_guesses: np.ndarray,
    temp_tropo: float,
    rel_hum: float,
    pressure: float,
    c_p: float,
    radius: float,
    rot_rate: float,
) -> np.ndarray:
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
    init_guess: Sequence[float] | np.ndarray,
    theta_rce_func: Callable[[float], float],
    theta_guesses: np.ndarray,
    temp_tropo: float = TEMP_TROPO,
    rel_hum: float = REL_HUM,
    pressure: float = P0,
    c_p: float = C_P,
    radius: float = RAD_EARTH,
    rot_rate: float = ROT_RATE_EARTH,
) -> np.ndarray:
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
    return _checked_root_solve(
        _eq_area_cqe_fixed_tt,
        init_guess,
        (
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


if __name__ == "__main__":
    pass
