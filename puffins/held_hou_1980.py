#! /usr/bin/env python
"""Expressions from Held and Hou 1980."""

import numpy as np

from .constants import RAD_EARTH, ROT_RATE_EARTH
from .nb_utils import cosdeg, sindeg
from .num_solver import brentq_solver_sweep_param


def pot_temp_rce_hh80(lats, z, theta_ref, height, delta_h, delta_v):
    """Eq. (2) of Held Hou 1980 (slightly rearranged)."""
    return theta_ref*(1 + delta_h*(cosdeg(lats)**2 - 2/3) +
                      (z/height - 0.5)*delta_v)


def u_rce_hh80(lats, thermal_ro, rot_rate=ROT_RATE_EARTH, radius=RAD_EARTH):
    """Zonal wind in gradient balance with equilibrium temperatures."""
    return rot_rate*radius*cosdeg(lats)*((1 + 2*thermal_ro)**0.5 - 1)


def dpot_temp_rce_hh80_dlat(lats, delta_h):
    """Meridional derivative of RCE potential temperature."""
    return -2*delta_h*sindeg(lats)*cosdeg(lats)


def u_crit_switch_lat_hh80(thermal_ro):
    """Where RCE and AMC winds are equal in Held Hou 1980 model."""
    return np.rad2deg(np.arccos((((1 + 2*thermal_ro)**-0.25))))


def u_crit_switch_lat_hh80_small_angle(thermal_ro):
    """Where RCE and AMC winds are equal in Held Hou 1980; small angle."""
    return np.rad2deg(thermal_ro**0.5)


def hc_edge_hh80_small_angle(thermal_ro):
    """Eq. 16 of Held Hou 1980."""
    return np.rad2deg((5*thermal_ro/3)**0.5)


def _hc_edge_hh80_lhs(lat, thermal_ro):
    """Left hand side of Eq. 17 of Held Hou 1980 (right hand side is zero)."""
    y = sindeg(lat)
    return ((1/3)*(4*thermal_ro - 1)*y**3 - y**5/(1-y**2) -
            y + 0.5*np.log((1 + y)/(1 - y)))


def hc_edge_hh80(thermal_ro, init_guess=0.1,
                 bound_guess_range=np.arange(0.1, 90.1, 10)):
    """Hadley cell edge according to Held and Hou 1980, Eq. 17.

    Solved numerically using the Brent (1973) root finding algorithm, as
    implemented in scipy's ``scipy.optimize.brentq`` function.

    Parameters
    ----------

    thermal_ro : scalar or array-like
        Thermal rossby number value(s) for which to solve.

    Returns
    -------

    hc_edge : xarray.DataArray
        Array of the numerical solution for each thermal Rossby number value in
        `thermal_ro`

    """
    return brentq_solver_sweep_param(_hc_edge_hh80_lhs,
                                     thermal_ro, init_guess,
                                     bound_guess_range)


if __name__ == '__main__':
    pass
