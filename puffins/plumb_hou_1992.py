#! /usr/bin/env python
"""Expressions from Plumb and Hou 1992."""

import numpy as np
import xarray as xr

from .constants import GRAV_EARTH, RAD_EARTH, ROT_RATE_EARTH, THETA_REF
from .nb_utils import cosdeg, sindeg
from .grad_bal import grad_wind_bouss


def pot_temp_rce_ph92(lat, theta_max_factor, lat_max=25,
                      nonzero_width=15, temp_ref=THETA_REF):
    """Vertical average of Eq. 9 of Plumb and Hou 1992."""
    nonzero_region = temp_ref + theta_max_factor*cosdeg(
        90.*(lat - lat_max) / nonzero_width)**2
    flat_region = xr.ones_like(lat)*temp_ref
    return xr.where((lat < lat_max + nonzero_width) &
                    (lat > lat_max - nonzero_width),
                    nonzero_region, flat_region)
    return nonzero_region


def dtheta_rce_ph92_dlat(lat, theta_max, lat_max, nonzero_width):
    """Meridional derivative of vert. averaged Eq. 9 of Plumb and Hou 1992."""
    sincos_arg = 90*(lat - lat_max) / nonzero_width
    nonzero_region = (-np.pi*theta_max / np.deg2rad(nonzero_width) *
                      cosdeg(sincos_arg)*sindeg(sincos_arg))
    return xr.where((lat < lat_max + nonzero_width) &
                    (lat > lat_max - nonzero_width),
                    nonzero_region, 0.)


def u_ph92_rce(lat, theta_max, lat_max, nonzero_width, height, temp_ref,
               grav=GRAV_EARTH, rot_rate=ROT_RATE_EARTH, radius=RAD_EARTH):
    """RCE wind corresponding to Eq. 9 of Plumb and Hou 1992."""
    dtheta_dlat = dtheta_rce_ph92_dlat(lat, theta_max, lat_max, nonzero_width)
    return grad_wind_bouss(lat, height, temp_ref, dtheta_dlat,
                           grav=grav, rot_rate=rot_rate, radius=radius,
                           plus_solution=True)


if __name__ == '__main__':
    pass
