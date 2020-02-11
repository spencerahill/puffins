"""Physical constants."""

import math

# Earth's dry atmosphere.
C_PD = 1003.5  # Specific heat of dry air at constant pressure.
C_P = C_PD  # Shorthand for c_pd.
R_D = 287.06  # Gas constant of dry air.
C_VD = C_PD - R_D  # Specific heat of dry air at constant volume.
KAPPA = R_D / C_PD

# Water vapor.
L_V = 2.5e6  # Latent heat of vaporization.
C_VV = 1418.  # Specific heat of water vapor at constant volume.
R_V = 461.4  # Gas constant of water vapor.
C_PV = C_VV + R_V  # Specific heat of water vapor at constant pressure.
EPSILON = R_D / R_V  # Ratio of dry air and vapor gas constants.

# Earth's solid body and orbit.
GRAV_EARTH = 9.81
RAD_EARTH = 6.371e6
ROT_RATE_EARTH = 2 * math.pi / 86400.0
SOLAR_CONST = 1365.2
STEF_BOLTZ_CONST = 5.67e-8

# Default values for parameters that appear in many functions.
DELTA_H = 1./6.
DELTA_V = 1./8.
HEIGHT_TROPO = 10.0e3
P0 = 1.0e5
REL_HUM = 0.7
TEMP_TROPO = 200.0
THETA_REF = 290.0
