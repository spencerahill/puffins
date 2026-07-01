"""Surface heat capacity effects on seasonality.

Mostly from Mitchell, Vallis, and Potter 2014, JAS.

"""

from __future__ import annotations

from typing import cast

import numpy as np

from ._typing import ArrayLike
from .constants import (
    C_VL,
    DENS_LIQ_WAT,
    ORB_FREQ_EARTH,
    STEF_BOLTZ_CONST,
    THETA_REF,
)
from .nb_utils import sindeg


def mixed_layer_heat_cap(
    depth: ArrayLike, density: float = DENS_LIQ_WAT, spec_heat: float = C_VL
) -> ArrayLike:
    """Heat capacity of a given depth of liquid water, in J/m^2/K."""
    return cast(ArrayLike, depth * density * spec_heat)


def therm_inert_timescale(heat_cap: ArrayLike, temp: ArrayLike) -> ArrayLike:
    """Thermal inertia timescale.

    Essentially, ratio of heat capacity to radiative restoring rate.

    """
    return cast(ArrayLike, heat_cap / (4.0 * STEF_BOLTZ_CONST * temp**3))


def seas_therm_inert_ratio(
    orb_period: ArrayLike, therm_inert_timescale: ArrayLike
) -> ArrayLike:
    """Ratio of thermal inertial and orbital timescales.

    I.e. `\alpha` parameter from Mitchell, Vallis, and Potter 2014.

    """
    seas_timescale = orb_period / (2.0 * np.pi)
    return cast(ArrayLike, seas_timescale / therm_inert_timescale)


def seas_damp_factor(alpha: ArrayLike) -> ArrayLike:
    """Damping of rad. eq. temps. ann. cycle given therm. inertia."""
    return cast(ArrayLike, alpha / np.sqrt(1 + alpha**2))


def seas_lag(alpha: ArrayLike, orb_freq: ArrayLike | None = None) -> ArrayLike:
    """Phase lag of rad. eq. temps. ann cycle given therm. inertia.

    If no orbital frequency is given, this is the nondimensional version.
    If orbital frequency is provided, this is the dimensional version.

    """
    lag = np.arctan(1.0 / alpha)
    if orb_freq is None:
        return cast(ArrayLike, lag)
    return cast(ArrayLike, lag / orb_freq)


def temp_rad_eq_eff(
    lat: ArrayLike,
    time: ArrayLike,
    alpha: ArrayLike,
    maxlat_ann: float = 44.0,
    delta_h: float = 1 / 6.0,
    orb_freq: float = ORB_FREQ_EARTH,
    temp_ref: float = THETA_REF,
) -> ArrayLike:
    """Effective rad-eq temperatures given thermal inertia, from MVP14.

    Note that `time` must be relative to northern spring equinox and have
    units of seconds.

    """
    sinlat = sindeg(lat)
    damping = seas_damp_factor(alpha)
    lag = seas_lag(alpha, orb_freq)
    ann_mean_term = 1 + delta_h / 3 * (1 - 3 * sinlat**2)
    ann_cyc_term = (
        2.0
        * delta_h
        * damping
        * sindeg(maxlat_ann)
        * sinlat
        * np.sin(orb_freq * (time - lag))
    )
    return cast(ArrayLike, temp_ref * (ann_mean_term + ann_cyc_term))
