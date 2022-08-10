"""Physical constants."""

import math

# Earth's dry atmosphere.
C_PD = 1003.5  # Specific heat of dry air at constant pressure.
C_P = C_PD  # Shorthand for c_pd.
R_D = 287.06  # Gas constant of dry air.
C_VD = C_PD - R_D  # Specific heat of dry air at constant volume.
KAPPA = R_D / C_PD
MEAN_SLP_EARTH = 101325  # Earth's mean sea level pressure (Pa)

# Water.
L_F = 3.34e5  # Latent heat of fusion.
L_V = 2.5e6  # Latent heat of vaporization.
C_VL = 4186.  # Specific heat of liquid water at constant volume.
C_VS = 2106.  # Specific heat of solid water at constant volume.
C_VV = 1418.  # Specific heat of water vapor at constant volume.
R_V = 461.4  # Gas constant of water vapor.
C_PV = C_VV + R_V  # Specific heat of water vapor at constant pressure.
EPSILON = R_D / R_V  # Ratio of dry air and vapor gas constants.
P_TRIP = 611.65  # Pressure of water triple point.
T_TRIP = 273.16  # Temperature of water triple point.
DENS_LIQ_WAT = 1000.  # Density of liquid water.

# Earth's solid body and orbit.
GRAV_EARTH = 9.81
RAD_EARTH = 6.371e6
ROT_RATE_EARTH = 2 * math.pi / 86400.0
OBLIQ_EARTH = 23.5
ORB_PERIOD_EARTH = 365.25 * 86400.
ORB_FREQ_EARTH = 2. * math.pi / ORB_PERIOD_EARTH

SOLAR_CONST = 1365.2
STEF_BOLTZ_CONST = 5.67e-8

# Default values for parameters that appear in many functions.
DELTA_H = 1./6.
DELTA_V = 1./8.
HEIGHT_TROPO = 10.0e3
HEIGHT_TROPO_EARTH = 10.0e3
P0 = 1.0e5
REL_HUM = 0.7
TEMP_TROPO = 200.0
THETA_REF = 290.0

# Mars.  https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html
GRAV_MARS = 3.721
RAD_MARS = 3.3895e6
LEN_DAY_MARS = 24.6597 * 3600
ROT_RATE_MARS = 2 * math.pi / LEN_DAY_MARS
HEIGHT_TROPO_MARS = 40.0e3
OBLIQ_MARS = 25.2

# Saturn
OBLIQ_SATURN = 26.73
ORB_PERIOD_SATURN = 10759.22 * 86400

# Titan
GRAV_TITAN = 1.35
HEIGHT_TROPO_TITAN = 40.0e3
ORB_PERIOD_TITAN = 15 * 86400. + 22 * 3600.
RAD_TITAN = 2.575e6
ROT_RATE_TITAN = 2 * math.pi / ORB_PERIOD_TITAN
