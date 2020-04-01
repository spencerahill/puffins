"""Functionality relating to statistics, timeseries, etc."""
import numpy as np
import xarray as xr


def trend(arr, dim, order=1):
    """Compute linear or higher-order polynomial fit."""
    coord = arr.coords[dim]
    slope, y0 = np.polyfit(coord, arr, order)
    return xr.ones_like(arr)*np.polyval([slope, y0], coord)


def detrend(arr, dim, order=1):
    """Subtract off the linear or higher order polynomial fit."""
    return arr - trend(arr, dim, order)
