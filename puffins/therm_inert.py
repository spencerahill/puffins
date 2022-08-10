"""Surface heat capacity effects on seasonality.

Mostly from Mitchell, Vallis, and Potter 2014, JAS.

"""
import numpy as np
from .constants import (
    C_VL,
    DENS_LIQ_WAT,
    ORB_FREQ_EARTH,
    STEF_BOLTZ_CONST,
    THETA_REF,
)
from .nb_utils import sindeg


def mixed_layer_heat_cap(depth, density=DENS_LIQ_WAT, spec_heat=C_VL):
    """Heat capacity of a given depth of liquid water, in J/m^2/K."""
    return depth * density * spec_heat


def therm_inert_timescale(heat_cap, temp):
    """Thermal inertia timescale.

    Essentially, ratio of heat capacity to radiative restoring rate.

    """
    return heat_cap / (4. * STEF_BOLTZ_CONST * temp ** 3)


def seas_therm_inert_ratio(orb_period, therm_inert_timescale):
    """Ratio of thermal inertial and orbital timescales.

    I.e. `\alpha` parameter from Mitchell, Vallis, and Potter 2014.

    """
    seas_timescale = orb_period / (2. * np.pi)
    return seas_timescale / therm_inert_timescale


def seas_damp_factor(alpha):
    """Damping of rad. eq. temps. ann. cycle given therm. inertia."""
    return alpha / np.sqrt(1 + alpha**2)


def seas_lag(alpha, orb_freq=None):
    """Phase lag of rad. eq. temps. ann cycle given therm. inertia.

    If no orbital frequency is given, this is the nondimensional version.
    If orbital frequency is provided, this is the dimensional version.

    """
    lag = np.arctan(1. / alpha)
    if orb_freq is None:
        return lag
    return lag / orb_freq


def temp_rad_eq_eff(lat, time, alpha, maxlat_ann=44., delta_h=1/6.,
                    orb_freq=ORB_FREQ_EARTH, temp_ref=THETA_REF):
    """Effective rad-eq temperatures given thermal inertia, from MVP14.

    Note that `time` must be relative to northern spring equinox and have
    units of seconds.

    """
    sinlat = sindeg(lat)
    damping = seas_damp_factor(alpha)
    lag = seas_lag(alpha, orb_freq)
    ann_mean_term = 1 + delta_h / 3 * (1 - 3 * sinlat ** 2)
    ann_cyc_term = (2. * delta_h * damping * sindeg(maxlat_ann) * sinlat *
                    np.sin(orb_freq * (time - lag)))
    return temp_ref * (ann_mean_term + ann_cyc_term)
