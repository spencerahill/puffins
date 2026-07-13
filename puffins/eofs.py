"""Functions for Empirical Orthogonal Functions (EOF) analyses."""

from __future__ import annotations

import numpy as np
import xarray as xr
from eofs.xarray import Eof

from .names import LAT_STR, YEAR_STR
from .nb_utils import cosdeg


def eof_solver_lat(
    arr: xr.DataArray, lat_str: str = LAT_STR, time_str: str = YEAR_STR
) -> Eof:
    """Generate an EOF solver for latitude-defined (such as lat-lon) data."""
    coslat = cosdeg(arr[lat_str])
    weights = np.sqrt(coslat).values[..., np.newaxis]
    return Eof(arr.rename({time_str: "time"}), weights=weights)
