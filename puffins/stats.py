"""Functionality relating to statistics, timeseries, etc."""
import numpy as np
import xarray as xr


def trend(arr, dim, order=1, ret_slope_y0=False):
    """Compute linear or higher-order polynomial fit."""
    coord = arr.coords[dim]
    slope, y0 = np.polyfit(coord, arr, order)
    if ret_slope_y0:
        return slope, y0
    return xr.ones_like(arr)*np.polyval([slope, y0], coord)


def detrend(arr, dim, order=1):
    """Subtract off the linear or higher order polynomial fit."""
    return arr - trend(arr, dim, order)
