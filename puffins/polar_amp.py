"""Functionality relating to polar amplification."""
import numpy as np
import xarray as xr

from .calculus import merid_avg_grid_data
from .names import LAT_STR


def polar_amp_index(arr, include_sh=True, include_nh=True,
                    denom_bounds=(-90, 90), sh_bound=-60,
                    nh_bound=60, lat_str=LAT_STR):
    """Compute ratio of value averaged over pole(s) to global average."""
    if not (include_sh or include_nh):
        raise ValueError("One or both of SH or NH must be selected.")
    denom = merid_avg_grid_data(arr, min_lat=min(denom_bounds),
                                max_lat=max(denom_bounds), lat_str=lat_str)
    if include_sh:
        sh_avg = merid_avg_grid_data(arr, max_lat=sh_bound, lat_str=lat_str)
        sh_weight = abs(-90 - sh_bound)
    else:
        sh_avg = 0
        sh_weight = 0
    if include_nh:
        nh_avg = merid_avg_grid_data(arr, min_lat=nh_bound, lat_str=lat_str)
        nh_weight = abs(90 - nh_bound)
    else:
        nh_avg = 0
        nh_weight = 0
    numer = (sh_weight * sh_avg + nh_weight * nh_avg) / (sh_weight + nh_weight)
    return numer / denom


def arctic_amp(arr, min_lat=60, denom_bounds=(-90, 90),
               lat_str=LAT_STR):
    """Ratio of NH high-latitude average to global average."""
    return polar_amp_index(arr, include_sh=False, nh_bound=min_lat,
                           denom_bounds=denom_bounds,
                           lat_str=lat_str)


def antarctic_amp(arr, max_lat=-60, denom_bounds=(-90, 90),
                  lat_str=LAT_STR):
    """Ratio of SH high-latitude average to global average."""
    return polar_amp_index(arr, include_nh=False, sh_bound=max_lat,
                           denom_bounds=denom_bounds,
                           lat_str=lat_str)


def print_polar_amp(arr, denom_bounds=(-90, 90)):
    print("Polar amplification indices")
    print("Arctic (60N-90N): {:0.2f}".format(
        float(arctic_amp(arr, denom_bounds=denom_bounds))))
    print("Antarctic (90S-60S): {:0.2f}".format(
        float(antarctic_amp(arr, denom_bounds=denom_bounds))))
    print("Both hemispheres: {:0.2f}".format(
        float(polar_amp_index(arr, denom_bounds=denom_bounds))))
    print()
