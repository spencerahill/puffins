"""Polar amplification diagnostics.

Functions for computing polar amplification indices, which quantify the
ratio of high-latitude change to global-mean change. Supports Arctic,
Antarctic, and combined polar amplification calculations.
"""

from .calculus import merid_avg_grid_data, merid_avg_sinlat_data
from .names import LAT_STR


def polar_amp_index(
    arr,
    include_sh=True,
    include_nh=True,
    denom_bounds=(-90, 90),
    sh_bound=-60,
    nh_bound=60,
    lat_str=LAT_STR,
):
    """Compute ratio of value averaged over pole(s) to global average.

    Computes a weighted average over one or both polar regions and
    divides by the average over a specified latitude range (typically
    global). Automatically detects whether the input uses a regular
    latitude grid or sine-latitude grid for the meridional averaging.

    Parameters
    ----------
    arr : xarray.DataArray
        Field to compute the polar amplification index for.
    include_sh : bool, optional
        Include the Southern Hemisphere polar region. Default: True.
    include_nh : bool, optional
        Include the Northern Hemisphere polar region. Default: True.
    denom_bounds : tuple of float, optional
        Latitude bounds (min, max) for the denominator average.
        Default: (-90, 90).
    sh_bound : float, optional
        Equatorward boundary of the SH polar region (degrees).
        Default: -60.
    nh_bound : float, optional
        Equatorward boundary of the NH polar region (degrees).
        Default: 60.
    lat_str : str, optional
        Name of the latitude dimension. Default: 'lat'.

    Returns
    -------
    xarray.DataArray
        Polar amplification index (dimensionless ratio).

    Raises
    ------
    ValueError
        If both ``include_sh`` and ``include_nh`` are False.

    See Also
    --------
    arctic_amp : Arctic-only amplification index.
    antarctic_amp : Antarctic-only amplification index.
    """
    if not (include_sh or include_nh):
        raise ValueError("One or both of SH or NH must be selected.")

    try:
        denom = merid_avg_grid_data(
            arr, min_lat=min(denom_bounds), max_lat=max(denom_bounds), lat_str=lat_str
        )
        func_avg = merid_avg_grid_data
    except ValueError:
        denom = merid_avg_sinlat_data(
            arr, min_lat=min(denom_bounds), max_lat=max(denom_bounds), lat_str=lat_str
        )
        func_avg = merid_avg_sinlat_data

    if include_sh:
        sh_avg = func_avg(arr, max_lat=sh_bound, lat_str=lat_str)
        sh_weight = abs(-90 - sh_bound)
    else:
        sh_avg = 0
        sh_weight = 0
    if include_nh:
        nh_avg = func_avg(arr, min_lat=nh_bound, lat_str=lat_str)
        nh_weight = abs(90 - nh_bound)
    else:
        nh_avg = 0
        nh_weight = 0
    numer = (sh_weight * sh_avg + nh_weight * nh_avg) / (sh_weight + nh_weight)
    return numer / denom


def arctic_amp(arr, min_lat=60, denom_bounds=(-90, 90), lat_str=LAT_STR):
    """Ratio of NH high-latitude average to global average.

    Parameters
    ----------
    arr : xarray.DataArray
        Field to compute Arctic amplification for.
    min_lat : float, optional
        Equatorward boundary of the Arctic region (degrees N).
        Default: 60.
    denom_bounds : tuple of float, optional
        Latitude bounds for the denominator average. Default: (-90, 90).
    lat_str : str, optional
        Name of the latitude dimension. Default: 'lat'.

    Returns
    -------
    xarray.DataArray
        Arctic amplification index (dimensionless ratio).

    See Also
    --------
    antarctic_amp : Antarctic-only amplification index.
    polar_amp_index : General polar amplification index.
    """
    return polar_amp_index(
        arr,
        include_sh=False,
        nh_bound=min_lat,
        denom_bounds=denom_bounds,
        lat_str=lat_str,
    )


def antarctic_amp(arr, max_lat=-60, denom_bounds=(-90, 90), lat_str=LAT_STR):
    """Ratio of SH high-latitude average to global average.

    Parameters
    ----------
    arr : xarray.DataArray
        Field to compute Antarctic amplification for.
    max_lat : float, optional
        Equatorward boundary of the Antarctic region (degrees S).
        Default: -60.
    denom_bounds : tuple of float, optional
        Latitude bounds for the denominator average. Default: (-90, 90).
    lat_str : str, optional
        Name of the latitude dimension. Default: 'lat'.

    Returns
    -------
    xarray.DataArray
        Antarctic amplification index (dimensionless ratio).

    See Also
    --------
    arctic_amp : Arctic-only amplification index.
    polar_amp_index : General polar amplification index.
    """
    return polar_amp_index(
        arr,
        include_nh=False,
        sh_bound=max_lat,
        denom_bounds=denom_bounds,
        lat_str=lat_str,
    )


def print_polar_amp(arr, denom_bounds=(-90, 90)):
    """Compute and print Arctic, Antarctic, and combined polar amplification indices.

    Prints the Arctic (60N-90N), Antarctic (90S-60S), and combined polar
    amplification indices to stdout.

    Parameters
    ----------
    arr : xarray.DataArray
        Field to compute polar amplification for.
    denom_bounds : tuple of float, optional
        Latitude bounds for the denominator average. Default: (-90, 90).
    """
    print("Polar amplification indices")
    print(f"Arctic (60N-90N): {float(arctic_amp(arr, denom_bounds=denom_bounds)):0.2f}")
    print(
        f"Antarctic (90S-60S): {float(antarctic_amp(arr, denom_bounds=denom_bounds)):0.2f}"
    )
    print(
        f"Both hemispheres: {float(polar_amp_index(arr, denom_bounds=denom_bounds)):0.2f}"
    )
    print()
