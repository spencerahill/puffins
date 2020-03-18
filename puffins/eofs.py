"""Functions for Empirical Orthogonal Functions (EOF) analyses."""
from eofs.xarray import Eof
import numpy as np

from .names import LAT_STR, YEAR_STR
from .nb_utils import cosdeg


def eof_solver_lat(arr, lat_str=LAT_STR, time_str=YEAR_STR):
    """Generate an EOF solver for latitude-defined (such as lat-lon) data."""
    coslat = cosdeg(arr[lat_str])
    weights = np.sqrt(coslat).values[..., np.newaxis]
    return Eof(arr.rename({time_str: "time"}), weights=weights)
