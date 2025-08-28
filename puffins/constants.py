"""Physical constants."""

import math

# Earth's dry atmosphere.
C_PD = 1003.5  # Specific heat of dry air at constant pressure.
C_P = C_PD  # Shorthand for c_pd.
R_D = 287.06  # Gas constant of dry air.
C_VD = C_PD - R_D  # Specific heat of dry air at constant volume.
C_V = C_VD
KAPPA = R_D / C_PD
MEAN_SLP_EARTH = 101325.  # Earth's mean sea level pressure (Pa)

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
# Difference in specific internal energy between water vapor and liquid at the
# triple point; Romps 2017 JAS.
E_0V = 2.374e6
# Difference in specific internal energy between water vapor and solid ice at
# the triple point; Romps 2017 JAS.
E_0S = 0.3337e6

# Radiation
STEF_BOLTZ_CONST = 5.67e-8

# Time
SECONDS_PER_MIN = 60.
SECONDS_PER_HR = 3600.
HR_PER_DAY = 24.
SECONDS_PER_EARTH_DAY = SECONDS_PER_HR * HR_PER_DAY

# Earth's solid body and orbit.
ECCENTRICITY_EARTH = 0.0167
GRAV_EARTH = 9.81
LON_PERIHELION_EARTH = 283.25
RAD_EARTH = 6.371e6
ROT_RATE_EARTH = 2 * math.pi / SECONDS_PER_EARTH_DAY
OBLIQ_EARTH = 23.437
ORB_PERIOD_EARTH = 365.25 * SECONDS_PER_EARTH_DAY
ORB_FREQ_EARTH = 2. * math.pi / ORB_PERIOD_EARTH
SOLAR_CONST_EARTH = 1365.2

# Mars.  https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html
# Except note that the perihelion longitude given on the NASA fact sheet site
# refers to some other coordinate system, giving a value of 336 degrees, while
# the correct value in terms of solar longitude, i.e. degrees relative to NH
# spring equinox, is around 250.66.
# E.g. http://www-mars.lmd.jussieu.fr/mars/time/solar_longitude.html
ECCENTRICITY_MARS = 0.0935
GRAV_MARS = 3.721
RAD_MARS = 3.3895e6
LEN_DAY_MARS = 24.6597 * SECONDS_PER_HR
LON_PERIHELION_MARS = 250.66
ROT_RATE_MARS = 2 * math.pi / LEN_DAY_MARS
HEIGHT_TROPO_MARS = 40.0e3
OBLIQ_MARS = 25.2
ORB_PERIOD_MARS = 686.98 * SECONDS_PER_EARTH_DAY
ORB_FREQ_MARS = 2. * math.pi / ORB_PERIOD_MARS
SOLAR_CONST_MARS = 586.2

# Saturn.  https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturnfact.html
ECCENTRICITY_SATURN = 0.0520
LON_PERIHELION_SATURN = 92.43194
OBLIQ_SATURN = 26.73
ORB_PERIOD_SATURN = 10759.22 * SECONDS_PER_EARTH_DAY
SOLAR_CONST_SATURN = 14.82

# Titan.
GRAV_TITAN = 1.35
HEIGHT_TROPO_TITAN = 40.0e3
ORB_PERIOD_TITAN = 15 * SECONDS_PER_EARTH_DAY + 22 * SECONDS_PER_HR
RAD_TITAN = 2.575e6
ROT_RATE_TITAN = 2 * math.pi / ORB_PERIOD_TITAN

# Venus.
OBLIQ_VENUS = 177.36
LEN_DAY_VENUS = 5832.6 * SECONDS_PER_HR

# Default values for parameters that appear in many functions.
DELTA_H = 1./6.
DELTA_V = 1./8.
HEIGHT_TROPO = 10.0e3
HEIGHT_TROPO_EARTH = 10.0e3
P0 = 1.0e5
REL_HUM = 0.7
TEMP_TROPO = 200.0
THETA_REF = 290.0
