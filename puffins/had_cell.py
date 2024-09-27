"""Hadley cells and other meridional overturning circulation things."""
import numpy as np
import scipy.ndimage
import xarray as xr

from .constants import (
    DELTA_V,
    GRAV_EARTH,
    HEIGHT_TROPO,
    RAD_EARTH,
    ROT_RATE_EARTH,
)
from .dynamics import plan_burg_num
from .names import LAT_STR, LEV_STR, LON_STR, SIGMA_STR
from .interp import interpolate, zero_cross_bounds, zero_cross_interp
from .nb_utils import (
    cosdeg,
    sindeg,
    lat_area_weight,
    max_and_argmax,
    max_and_argmax_along_dim,
)
from .num_solver import brentq_solver_sweep_param
from .vert_coords import subtract_col_avg


def merid_streamfunc(v, dp, grav=GRAV_EARTH, radius=RAD_EARTH,
                     impose_zero_col_flux=True, lat_str=LAT_STR,
                     lon_str=LON_STR, lev_str=LEV_STR):
    """Meridional mass streamfunction.

    Parameters
    ----------
    v : xarray.DataArray
        Meridional wind field.
    dp : xarray.DataArray
        Pressure thickness of each gridbox, in Pascal

    Returns
    -------
    xarray.DataArray
        The meridional mass streamfunction, signed such that
        counter-clockwise circulation in lat-height plane
        (i.e. Southern Hemisphere Hadley Cell) is positive.

    """
    # Zonally average v and dp.
    if lon_str in v.dims:
        v_znl_mean = v.mean(dim=lon_str)
    else:
        v_znl_mean = v
    if lon_str in dp.dims:
        dp_znl_mean = dp.mean(dim=lon_str)
    else:
        dp_znl_mean = dp
    # If desired, impose zero net mass flux at each level.
    if impose_zero_col_flux:
        v_znl_mean = subtract_col_avg(v_znl_mean, dp_znl_mean, dim=lev_str)
    # Ensure the sum is from TOA down, since doing from surface up can
    # generate spurious errors due to larger spatiotemporal variations
    # in pressure at surface than at TOA.  C.f. Adam et al 2018.
    levels = v_znl_mean[lev_str]
    if np.all(np.diff(levels) > 0):
        levslice = slice(None, None, 1)
    elif np.all(np.diff(levels) < 0):
        levslice = slice(None, None, -1)
    else:
        raise ValueError(f"Levels are not monotonic: {levels}")
    # At each vertical level, integrate from TOA to that level.
    streamfunc = (v_znl_mean * dp_znl_mean).sel(
        **{lev_str: levslice}).cumsum(dim=lev_str) / grav
    # Weight by surface area to get a mass overturning rate.
    # Leading minus sign results in positive southern hemisphere cell
    return -(lat_area_weight(v[lat_str], radius=radius) * streamfunc).transpose(
        *v_znl_mean.dims).rename("streamfunc").where(np.isfinite(v))


def had_cell_strength(streamfunc, dim=None, min_plev=None, max_plev=None,
                      do_avg_vert=False, do_interp_lat=False, lev_str=LEV_STR,
                      lat_str=LAT_STR):
    """Hadley cell strength, as maximum of streamfunction.

    If a dimension is given, compute the strength separately at each value of
    that dimension.  Otherwise find the global extremum of the whole given
    array.

    """
    if min_plev is None and max_plev is None:
        sf_valid = streamfunc
    else:
        sf_valid = streamfunc.copy(deep=True)
        lev = streamfunc[lev_str]

    if min_plev is not None:
        sf_valid = sf_valid.where(lev >= min_plev, drop=True)
    if max_plev is not None:
        sf_valid = sf_valid.where(lev <= max_plev, drop=True)

    if do_avg_vert:
        avg_lev = sf_valid[lev_str].mean(lev_str).values
        sf_valid = sf_valid.mean(lev_str)
        sf_valid.coords[lev_str] = avg_lev

    # WIP: interpolate in latitude to the center.  I've got something that runs
    # but I don't trust it yet, and it feels clunky.  What I need is to find
    # the center point, grab it and two point on either side of it, compute the
    # derivative using 2nd order centered differences for the three points, and
    # on one side or the other there will be a sign change, and interpolate there.
    # Or is there a way to do that even more directly with xarray's interpolation?
    if do_interp_lat:
        cell_strength = max_and_argmax(sf_valid)
        lev_center = cell_strength[lev_str].values
        lat_center_raw = cell_strength[lat_str].values
        dlat_mean = sf_valid[lat_str].diff(lat_str).mean()
        dsf_dlat = sf_valid.differentiate(lat_str).sel(
            **{lev_str: lev_center, "method": "nearest"}).sel(
            **{lat_str: slice(lat_center_raw - 1.1 * dlat_mean,
                              lat_center_raw + 1.1 * dlat_mean)})
        lat_center = zero_cross_interp(dsf_dlat, lat_str).values
        sf_at_center = sf_valid.interp(
            **{lat_str: lat_center, "method": "cubic"})
        return max_and_argmax(sf_at_center)

    if dim is None:
        return max_and_argmax(sf_valid)
    return max_and_argmax_along_dim(sf_valid, dim)


def had_cells_strength(strmfunc, min_plev=None, max_plev=None,
                       do_avg_vert=False, min_lat=None, max_lat=None,
                       do_interp_lat=False, lat_str=LAT_STR, lev_str=LEV_STR):
    """Location and signed magnitude of both Hadley cell centers."""
    lat = strmfunc[lat_str]

    if min_lat is not None:
        strmfunc = strmfunc.where(lat > min_lat, drop=True)
    if max_lat is not None:
        strmfunc = strmfunc.where(lat < max_lat, drop=True)

    # Sometimes the winter Ferrel cell is stronger than the summer Hadley cell.
    # So find the global extremal negative and positive values as well as the
    # opposite-signed cell on either side.  The Hadley cells will be the two of
    # these whose centers are nearest the equator.
    cell_strength_vals = []

    kwargs_hc_strength = dict(
        min_plev=min_plev,
        max_plev=max_plev,
        do_avg_vert=do_avg_vert,
        do_interp_lat=do_interp_lat,
        lat_str=lat_str,
        lev_str=lev_str,
    )

    cell_pos_max_strength = had_cell_strength(strmfunc, **kwargs_hc_strength)
    cell_neg_max_strength = -1 * had_cell_strength(
        -1 * strmfunc, **kwargs_hc_strength
    )
    cell_strength_vals.append(cell_pos_max_strength)
    cell_strength_vals.append(cell_neg_max_strength)

    lat_pos_max = cell_pos_max_strength.coords[lat_str]
    lat_neg_max = cell_neg_max_strength.coords[lat_str]

    try:
        cell_south_of_pos_strength = -1 * had_cell_strength(
            -1 * strmfunc.where(lat < lat_pos_max), **kwargs_hc_strength,
        )
    except ValueError:
        pass
    else:
        cell_strength_vals.append(cell_south_of_pos_strength)
    try:
        cell_north_of_pos_strength = -1 * had_cell_strength(
            -1 * strmfunc.where(lat > lat_pos_max), **kwargs_hc_strength,
        )
    except ValueError:
        pass
    else:
        cell_strength_vals.append(cell_north_of_pos_strength)
    try:
        cell_south_of_neg_strength = had_cell_strength(
            strmfunc.where(lat < lat_neg_max), **kwargs_hc_strength,
        )
    except ValueError:
        pass
    else:
        cell_strength_vals.append(cell_south_of_neg_strength)

    try:
        cell_north_of_neg_strength = had_cell_strength(
            strmfunc.where(lat > lat_neg_max), **kwargs_hc_strength,
        )
    except ValueError:
        pass
    else:
        cell_strength_vals.append(cell_north_of_neg_strength)

    # The above procedure generates up to 6 cells, of which up to 2 are
    # duplicates.  Now, get rid of the duplicates.
    cell_strengths = xr.concat(cell_strength_vals, dim=lat_str,
                               coords=[lev_str])
    dupes = cell_strengths.get_index(lat_str).duplicated()
    cell_strengths = cell_strengths[~dupes]

    # Pick the two cells closest to the equator.
    # But sometimes two cells can be at identical but mirror latitudes,
    # which the logic would otherwise always select the NH one.  So
    # in that case pick the one with the larger strength.
    # For now, the logic only checks this for the first and last values, but
    # in principle this could occur for other indices and thus this logic would
    # need to be generalized.
    center_lats = cell_strengths[lat_str]
    if np.isclose(np.abs(center_lats[0]), np.abs(center_lats[-1])):
        if np.abs(cell_strengths)[0] > np.abs(cell_strengths)[-1]:
            center_lats = center_lats[:-1]
            cell_strengths = cell_strengths[:-1]
        else:
            center_lats = center_lats[1:]
            cell_strengths = cell_strengths[1:]
    hc_strengths = cell_strengths.sortby(np.abs(center_lats))[:2]

    # Order the cells from south to north.
    hc_strengths = hc_strengths.sortby(hc_strengths[lat_str])

    # Create DataArray with one label for each cell, the cell strengths
    # as the values, and the cell center latitudes and levels as coords.
    coords_out = {"cell": ["had_cell_sh", "had_cell_nh"]}
    ds_strengths = xr.Dataset(coords=coords_out)
    arr_lat_center = xr.DataArray(hc_strengths[lat_str].values,
                                  dims=["cell"], coords=coords_out)
    arr_lev_center = xr.DataArray(hc_strengths[lev_str].values,
                                  dims=["cell"], coords=coords_out)
    arr_strength = xr.DataArray(hc_strengths.values,
                                dims=["cell"], coords=coords_out)
    ds_strengths.coords[lat_str] = arr_lat_center
    ds_strengths.coords[lev_str] = arr_lev_center
    ds_strengths["cell_strength"] = arr_strength
    return ds_strengths["cell_strength"]


def _streamfunc_at_avg_lev_max(strmfunc, hc_strengths, lev_str=LEV_STR):
    """Streamfunction at the average level of the two Hadley cell centers."""
    lev_sh_max = hc_strengths[lev_str][0]
    lev_nh_max = hc_strengths[lev_str][1]
    lev = strmfunc[lev_str]
    lev_avg = lev.sel(**{lev_str: 0.5*(lev_sh_max + lev_nh_max),
                         "method": "nearest"})
    return strmfunc.sel(**{lev_str: lev_avg})


def had_cells_shared_edge(strmfunc, fixed_plev=None,
                          min_plev=None, max_plev=None, do_avg_vert=False,
                          min_lat=None, max_lat=None, cos_factor=False,
                          lat_str=LAT_STR, lev_str=LEV_STR):
    """Latitude of shared inner edge of Hadley cells."""
    lat = strmfunc[lat_str]
    hc_strengths = had_cells_strength(
        strmfunc,
        min_plev=min_plev,
        max_plev=max_plev,
        do_avg_vert=do_avg_vert,
        min_lat=min_lat,
        max_lat=max_lat,
        lat_str=lat_str,
        lev_str=lev_str,
    )
    lat_sh_max = hc_strengths[lat_str][0]
    lat_nh_max = hc_strengths[lat_str][1]

    if fixed_plev is None:
        sf_at_max = _streamfunc_at_avg_lev_max(strmfunc, hc_strengths, lev_str)
    else:
        sf_at_max = strmfunc.sel(**{lev_str: fixed_plev, "method": "nearest"})
    sf_max2max = sf_at_max.where((lat >= lat_sh_max) & (lat <= lat_nh_max),
                                 drop=True)
    return zero_cross_interp(sf_max2max, lat_str)[lat_str]


def had_cell_edge(strmfunc, cell="north", edge="north", frac_thresh=0.1,
                  fixed_plev=None, min_plev=None, max_plev=None,
                  do_avg_vert=False, min_lat=None, max_lat=None,
                  cos_factor=False, lat_str=LAT_STR, lev_str=LEV_STR):
    """Latitude of poleward edge of either the NH or SH Hadley cell."""
    hc_strengths = had_cells_strength(
        strmfunc,
        min_plev=min_plev,
        max_plev=max_plev,
        do_avg_vert=do_avg_vert,
        min_lat=min_lat,
        max_lat=max_lat,
        lat_str=lat_str,
        lev_str=lev_str,
    )
    if cell == "north":
        label = "had_cell_nh"
    elif cell == "south":
        label = "had_cell_sh"
    else:
        raise ValueError("`cell` must be either 'north' or 'south'; "
                         f"got {cell}.")

    # Restrict to streamfunction at level of the specified cell's maximum.
    cell_max = hc_strengths.sel(cell=label)
    lat_of_max = cell_max[lat_str]
    lev_of_max = cell_max[lev_str]
    if fixed_plev is None:
        lev_for_edges = float(lev_of_max)
    else:
        lev_for_edges = fixed_plev
    sf_at_max = strmfunc.sel(**{lev_str: lev_for_edges, "method": "nearest"})

    # Restrict to the latitudes north or south of the max, as specified.
    lat = strmfunc[lat_str]
    if edge == "north":
        which_zero = 0
        lat_compar = lat >= lat_of_max
    elif edge == "south":
        which_zero = -1
        lat_compar = lat <= lat_of_max
    else:
        raise ValueError("`edge` must be either 'north' or 'south'; "
                         f"got {cell}.")
    sf_one_side = sf_at_max.where(lat_compar, drop=True)

    # Restrict to the latitudes from the max to the nearest point with
    # opposite-signed value.  Also apply cubic interpolation in latitude to a
    # refined mesh.  Otherwise, the cell edge can (unphysically) vary
    # non-monotonically with `frac_thresh`.
    dlat_avg = float(sf_one_side[lat_str].diff(lat_str).mean(lat_str).values)
    # If there's only one point, assume it's one of the poles.
    if np.isnan(dlat_avg):
        if edge == "north":
            return 90
        return -90
    lats_interp = np.arange(sf_one_side[lat_str].min(),
                            sf_one_side[lat_str].max() - 0.2 * dlat_avg,
                            0.1 * dlat_avg)
    sf_one_side_interp = sf_one_side.interp(**{lat_str: lats_interp})
    # Explicitly make the last value equal to the original, as otherwise the
    # interp step can overwrite it with nan for some reason.
    sf_one_side_interp = xr.concat([sf_one_side_interp, sf_one_side[-1]],
                                   dim=lat_str)

    # Find where the streamfunction crosses the specified fractional threshold,
    # using the Singh 2019 cosine weighting if specified.
    if cos_factor:
        sf_norm = ((sf_one_side_interp / cosdeg(sf_one_side_interp[lat_str])) /
                   (cell_max / cosdeg(lat_of_max)))
    else:
        sf_norm = sf_one_side_interp / cell_max

    sf_thresh_diff = sf_norm - frac_thresh
    # If there is no zero crossing, then the cell extends to the pole.
    try:
        sf_edge_bounds = zero_cross_bounds(sf_thresh_diff, lat_str, which_zero)
    except ValueError:
        if edge == "north":
            return 90.
        return -90.
    # Interpolate between the bounding points to the crossing.
    return interpolate(sf_edge_bounds, sf_edge_bounds[lat_str], 0,
                       lat_str)[lat_str]


def had_cells_south_edge(strmfunc, frac_thresh=0.1, fixed_plev=None,
                         min_plev=None, max_plev=None, do_avg_vert=False,
                         min_lat=None, max_lat=None, cos_factor=False,
                         lat_str=LAT_STR, lev_str=LEV_STR):
    """Latitude of southern edge of southern Hadley cell."""
    return had_cell_edge(
        strmfunc,
        cell="south",
        edge="south",
        frac_thresh=frac_thresh,
        fixed_plev=fixed_plev,
        min_plev=min_plev,
        max_plev=max_plev,
        do_avg_vert=do_avg_vert,
        min_lat=min_lat,
        max_lat=max_lat,
        cos_factor=cos_factor,
        lat_str=lat_str,
        lev_str=lev_str,
    )


def had_cells_north_edge(strmfunc, frac_thresh=0.1, fixed_plev=None,
                         min_plev=None, max_plev=None, do_avg_vert=False,
                         min_lat=None, max_lat=None, cos_factor=False,
                         lat_str=LAT_STR, lev_str=LEV_STR):
    """Latitude of northern edge of northern Hadley cell."""
    return had_cell_edge(
        strmfunc,
        cell="north",
        edge="north",
        frac_thresh=frac_thresh,
        fixed_plev=fixed_plev,
        min_plev=min_plev,
        max_plev=max_plev,
        do_avg_vert=do_avg_vert,
        min_lat=min_lat,
        max_lat=max_lat,
        cos_factor=cos_factor,
        lat_str=lat_str,
        lev_str=lev_str,
    )


def had_cells_edges(strmfunc, frac_thresh=0.1, min_lev=None, max_lev=None,
                    do_avg_vert=False, min_lat=None, max_lat=None,
                    cos_factor=False, lat_str=LAT_STR, lev_str=LEV_STR):
    """Southern, shared inner, and northern edge of the Hadley cells."""
    shared_kwargs = dict(
        frac_thresh=frac_thresh,
        min_lev=min_lev,
        max_lev=max_lev,
        do_avg_vert=do_avg_vert,
        min_lat=min_lat,
        max_lat=max_lat,
        cos_factor=cos_factor,
        lat_str=lat_str,
        lev_str=lev_str,
    )
    func_and_kwargs = [
        (had_cell_edge, dict(cell="south", edge="south")),
        (had_cells_shared_edge, {}),
        (had_cell_edge, dict(cell="north", edge="north")),
         ]
    return [f_kw[0](
        strmfunc,
        lat_str=lat_str,
        lev_str=lev_str,
        **f_kw[1],
        **shared_kwargs,
    ) for f_kw in func_and_kwargs]


def cell_edges_sigma(streamfunc, frac_thresh=0.1, cos_factor=False,
                     center_min_sigma=0.1, center_max_sigma=1,
                     cell_min_sigma_depth=0.3, lat_str=LAT_STR,
                     sigma_str=SIGMA_STR):
    """Compute edges of all contiguous streamfunction overturning cells."""
    # Discard values in the boundary layer and stratosphere (if specified).
    sigma = streamfunc[sigma_str]
    if center_min_sigma > 0 or center_max_sigma < 1:
        streamfunc = streamfunc.where((sigma >= center_min_sigma) &
                                      (sigma <= center_max_sigma), drop=True)

    # Identify each cell candidate and loop over them.
    labels, indices = _split_streamfunc_into_cells(streamfunc)
    cells = []
    for n in indices:
        try:
            candidate = _find_cell_and_edges(
                streamfunc,
                labels,
                n,
                frac_thresh=frac_thresh,
                cos_factor=cos_factor,
                cell_min_sigma_depth=cell_min_sigma_depth,
                lat_str=lat_str,
                sigma_str=sigma_str,
            )
        except ValueError as error:
            if str(error) == "Cell is insufficiently deep.":
                pass
            else:
                raise error
        else:
            cells.append(candidate)

    # Convert to an xarray.DataArray
    cells_arr = xr.concat(cells, dim='cell')
    # Restrict to Hadley and Ferrel cells.
    return _hadley_ferrel_cells(cells_arr)


def _find_cell_and_edges(streamfunc, labels, n, cell_min_sigma_depth=0.3,
                         frac_thresh=0.1, cos_factor=False, lat_str=LAT_STR,
                         sigma_str=SIGMA_STR):
    """Find the cell with a given label and determine its edges."""
    # Isolate the cell labeled by 'n' and make it positive.
    cell_sign = np.sign(n)
    sf_one_cell = _single_cell(streamfunc, labels, n)*cell_sign
    # Discard cells that aren't sufficiently deep.
    sigma_cell = sf_one_cell.dropna(SIGMA_STR, how='all')[SIGMA_STR]
    cell_sigma_depth = sigma_cell.max() - sigma_cell.min()
    if cell_sigma_depth < cell_min_sigma_depth:
        raise ValueError("Cell is insufficiently deep.")
    # Restrict to the level of this cell's center.
    sf_max = had_cell_strength(sf_one_cell)
    assert sf_max > 0
    sf_at_max = _cell_at_max(streamfunc*cell_sign, sf_max, sigma_str)
    # Retain this cell and those with opposite sign.
    sf_and_opp_sign = _cell_and_opp_sign_cells(sf_at_max, labels, n,
                                               sigma_str)
    # If identified cell is physically meaningful, find its edges.
    if _cell_is_bad(sf_max, streamfunc[lat_str], sf_and_opp_sign, min_width=1,
                    lat_str=lat_str):
        raise ValueError("Cell is not physically meaningful.")
    edges = _one_cell_edges(sf_and_opp_sign, cell_sign, frac_thresh,
                            cos_factor=cos_factor, lat_str=lat_str)
    # Convert to a single DataArray with the cell center included.
    edges.insert(-1, sf_max[lat_str])
    edges_arr = xr.concat(edges, dim=lat_str)
    edges_arr[lat_str] = ['edge_south', 'center', 'edge_north']
    edges_arr.coords['cell_strength'] = float(sf_max*cell_sign)
    return edges_arr


def _split_streamfunc_into_cells(streamfunc):
    """Identify all coherent same-signed structures in a streamfunction."""
    # Algorithm works by distinguishing between zero and nonzero values.  So
    # first set negative values to zero to identify cells with positive
    # streamfunctions, and then do the opposite.
    arr_pos = xr.where(streamfunc > 0, 1, 0)
    arr_neg = xr.where(streamfunc < 0, 1, 0)
    labels_pos, num_pos = scipy.ndimage.label(arr_pos)
    labels_neg, num_neg = scipy.ndimage.label(arr_neg)
    # Subtracting labels_neg from labels_pos tags the cells with negative
    # streamfunctions with a negative value.
    labels_vals = labels_pos - labels_neg
    labels = xr.ones_like(streamfunc)*labels_vals
    indices = list(range(-num_neg, 0)) + list(range(1, num_pos + 1))
    return labels, indices


def _single_cell(streamfunc, labels, n):
    """Restrict streamfunction to single previously identified cell."""
    return streamfunc.where(labels == n)


def _cell_at_max(sf_one_cell, sf_max, sigma_str=SIGMA_STR):
    """Restrict the streamfunction to the level of the cell's maximum."""
    sigma_max = float(sf_max[sigma_str].values)
    return sf_one_cell.sel(**{sigma_str: sigma_max})


def _cell_and_opp_sign_cells(sf_at_max, labels, n, sigma_str=SIGMA_STR):
    """Restrict to one positive overturning cell and negative cells."""
    labels_at_max = labels.sel(**{sigma_str: sf_at_max[sigma_str]})
    sign_at_max = np.sign(sf_at_max)
    return sf_at_max.where((labels_at_max == n) | (sign_at_max < 0))


def _cell_is_bad(sf_max, lats, sf_and_opp_sign, min_width=3, lat_str=LAT_STR):
    """Determine if identified cell is physically meaningful or not."""
    lat_max = float(sf_max[lat_str].values)
    center_at_pole = np.abs(lat_max) == float(np.abs(lats.values[0]))
    cell_too_narrow = len(sf_and_opp_sign.where(sf_and_opp_sign > 0,
                                                drop=True)) < min_width
    return center_at_pole or cell_too_narrow


def _one_cell_edges(sf_and_opp_sign, cell_sign, frac_thresh, cos_factor=False,
                    lat_str=LAT_STR):
    """Compute northern and southern edges of a cell."""
    edges = []
    # Look for the northern edge and then, by reversing the sign of the
    # latitudes array, the southern edge.
    for sign_factor in [1, -1]:
        sf_cell = sf_and_opp_sign.copy()
        lats = sf_cell[lat_str]
        if sf_cell.max(lat_str) < np.abs(sf_cell.min(lat_str)):
            sf_max = sf_cell.min(lat_str)
            lat_max = lats[sf_cell.argmin(lat_str)]
        else:
            sf_max = sf_cell.max(lat_str)
            lat_max = lats[sf_cell.argmax(lat_str)]

        if cos_factor:
            sf_norm = ((sf_cell / cosdeg(lats)) /
                       (sf_max / cosdeg(lat_max)))
        else:
            sf_norm = sf_cell / sf_max

        # threshold = float(sf_max*frac_thresh)
        # assert threshold > 0

        # Flip the sign and the indexing if looking for southern edge.
        lat_cell = sf_norm[lat_str] * sign_factor
        sf_norm[lat_str] = lat_cell
        if sign_factor == -1:
            sf_norm = sf_norm.isel(**{lat_str: slice(None, None, -1)})
            lat_cell = sf_norm[lat_str]

        # Restrict to latitudes greater than that at the maximum.
        sf_one_side = sf_norm.where(lat_cell >= sign_factor * lat_max,
                                    drop=True)
        lat_one_side = sf_one_side[lat_str]

        # Find where the streamfunction drops below the threshold.  This
        # includes points with opposite-signed streamfunction.
        sf_below = sf_one_side.where(sf_one_side < frac_thresh, drop=True)

        # If the threshold is never crossed, assume that the cell goes to
        # zero at that pole.
        if len(sf_below) == 0:
            first_lat_below = 90. * np.sign(lat_one_side.max(lat_str))
            first_below = xr.zeros_like(sf_norm.isel(**{lat_str: -1}))
            first_below[lat_str].values = first_lat_below

        # Otherwise, keep the adjacent below-threshold point.
        else:
            first_lat_below = sf_below[lat_str].min()
            first_below = sf_below.sel(**{lat_str: first_lat_below})

        # Get the adjacent above-threshold value towards the cell center.
        last_lat_above = lat_one_side.where(lat_one_side <
                                            first_lat_below,
                                            drop=True).max()
        last_above = sf_one_side.sel(**{lat_str: last_lat_above}).copy()

        # Revert the streamfunctions and latitudes to their original sign.
        first_below *= cell_sign
        last_above *= cell_sign
        first_below[lat_str] *= sign_factor
        last_above[lat_str] *= sign_factor

        # Interpolate between the two gridpoints to the actual crossing.
        thresh_bounds = xr.concat([last_above, first_below], dim=lat_str)
        edge = interpolate(thresh_bounds, thresh_bounds[lat_str],
                           frac_thresh * cell_sign, lat_str)

        edges.append(edge)
    # Put southern edge first.
    edges.sort()
    return edges


def _hadley_ferrel_cells(cells, min_frac_strength=0.05, max_num_cells=4,
                         max_lat_gap=10, lat_str=LAT_STR):
    """From identified cell edges, keep only the Hadley and Ferrel cells.

    TODO: In a very few cases, the polar cell in one hemisphere is stronger
    than the ferrel cell in the opposite hemisphere.  As such, it gets kept
    rather than this weaker Ferrel cell.  This needs to be fixed.

    """
    # Discard cells weaker than a given fraction of the strongest cell.
    max_strength = np.abs(cells['cell_strength']).max()
    cells = cells.where(np.abs(cells['cell_strength']) >
                        min_frac_strength * max_strength, drop=True)
    # Discard cells that are entirely within the edges of a larger cell.
    for i, cell in enumerate(cells):
        bad = ((cell.sel(lat='edge_south') > cells.sel(lat='edge_south')) &
               (cell.sel(lat='edge_north') < cells.sel(lat='edge_north')))
        if np.any(bad):
            cells = cells.where(cells['cell'] != i)
    cells = cells.dropna('cell')
    # Restrict candidates to the strongest few of each sign.
    if len(cells['cell']) > max_num_cells:
        cells_pos = cells.where(cells['cell_strength'] > 0, drop=True)
        cells_pos = cells_pos.sortby(cells_pos['cell_strength'],
                                     ascending=False)
        cells_pos = cells_pos.isel(cell=slice(0, max_num_cells))
        cells_neg = cells.where(cells['cell_strength'] < 0, drop=True)
        cells_neg = cells_neg.sortby(cells_neg['cell_strength'],
                                     ascending=True)
        cells_neg = cells_neg.isel(cell=slice(0, max_num_cells))
        cells = xr.concat([cells_pos, cells_neg], dim='cell')
    # Sort from south to north.
    cells = cells.sortby(cells.sel(lat='center'))
    # Label each Hadley cell that can be identified.
    center_nearest_eq = np.abs(cells.sel(lat='center')).min()
    labels = list(range(len(cells)))
    for i, cell in enumerate(cells):
        # Assume that the cell nearest the equator is one of the Hadley cells.
        center = cell.sel(**{lat_str: 'center'})
        if np.abs(center) == center_nearest_eq:
            if cell.sel(**{lat_str: 'edge_south'}) < 0:
                labels[i] = 'sh_hadley'
                # Once one Hadley cell has been found, assume the adjacent cell
                # is the other one, provided it is physicaly close enough
                if i != len(cells) - 1:
                    lat_gap = (cells[i+1].sel(lat='edge_south') -
                               cell.sel(lat='edge_north'))
                    if lat_gap < max_lat_gap:
                        labels[i+1] = 'nh_hadley'
                break
            else:
                labels[i] = 'nh_hadley'
                if i != 0:
                    labels[i-1] = 'sh_hadley'
                break
    cells['cell'] = labels
    return cells


# Descending edge predictions based on baroclinic instability onset."""
def _fixed_ro_bci_edge(lat, ascentlat, h00lat):
    """For numerical solution of fixed-Ro, 2-layer BCI model of HC edge."""
    sinlat = sindeg(lat)
    coslat = cosdeg(lat)
    term1 = sinlat ** 4
    term2 = -sindeg(ascentlat) ** 2 * sinlat ** 2
    term3 = -np.deg2rad(h00lat) ** 4 * coslat ** 2
    return np.rad2deg(term1 + term2 + term3)


def fixed_ro_bci_edge(ascentlat, lat_fixed_ro_ann=None,
                      burg_num=None, ross_num=1., delta_v=DELTA_V,
                      height=HEIGHT_TROPO, grav=GRAV_EARTH,
                      rot_rate=ROT_RATE_EARTH, radius=RAD_EARTH,
                      zero_bounds_guess_range=np.arange(0.1, 90, 5)):
    """Numerically solve fixed-Ro, 2-layer BCI model of HC edge."""
    if lat_fixed_ro_ann is None:
        if burg_num is None:
            burg_num = plan_burg_num(height, grav=grav, rot_rate=rot_rate,
                                     radius=radius)
        lat_fixed_ro_ann = np.rad2deg((burg_num * delta_v / ross_num) ** 0.25)

    def _solver(lat_a, lat_h):
        # Start guess at the average of the two given latitudes.
        init_guess = 0.5 * (lat_a + lat_h)
        return brentq_solver_sweep_param(
            _fixed_ro_bci_edge,
            lat_a,
            init_guess,
            zero_bounds_guess_range,
            funcargs=(lat_h,),
        )
    return xr.apply_ufunc(_solver, ascentlat, lat_fixed_ro_ann, vectorize=True,
                          dask="parallelized")


def fixed_ro_bci_edge_small_angle(ascentlat, lat_fixed_ro_ann=None,
                                  burg_num=None, ross_num=1, delta_v=DELTA_V,
                                  c_descent=1., height=HEIGHT_TROPO,
                                  grav=GRAV_EARTH, rot_rate=ROT_RATE_EARTH,
                                  radius=RAD_EARTH):
    """Small-angle solution for fixed-Ro, BCI model for HC descending edge.

    Both ascentlat and lat_fixed_ro_ann should be in degrees, not radians.
    And the return value is in degrees, not radians.

    """
    if lat_fixed_ro_ann is not None:
        lat_ro_ann = np.deg2rad(lat_fixed_ro_ann)
    else:
        if burg_num is None:
            burg_num = plan_burg_num(height, grav=grav, rot_rate=rot_rate,
                                     radius=radius)
        lat_ro_ann4 = (burg_num * delta_v / ross_num)

    lat_a = np.deg2rad(ascentlat)
    return c_descent * np.rad2deg(np.sqrt(
        0.5 * lat_a + np.sqrt(0.25 * lat_a ** 4 + lat_ro_ann4)))


def lin_ro_bci_edge_small_angle_lata0(
        burg_num,
        ross_ascent,
        ross_descent,
        delta_v=DELTA_V,
):
    """Baroclinic instab. est. for cell edge, linear Ro profile."""
    return np.rad2deg((3 * burg_num * delta_v /
                       (ross_ascent + 2 * ross_descent)) ** 0.25)


# Edge solutions using supercriticality to predict ascent location.
def lat_ascent_eta0_approx(therm_ro, c_ascent=1.):
    """Approx. zero cross of abs. vort. of Lindzen-Hou solstice forcing."""
    if np.all(therm_ro == 0):
        return 0.
    return np.sign(therm_ro) * np.rad2deg(c_ascent * (0.5 * np.abs(therm_ro))
                                          ** (1. / 3.))


def fixed_ro_bci_edge_supercrit_ascent(therm_ross, max_lat=90, c_ascent=1,
                                       c_descent=1, delta_v=DELTA_V,
                                       delta_h=1. / 15., ross_num=1):
    """Descending edge approx including theory for ascent lat.

    From Hill, Bordoni, and Mitchell, 2022, JAS, "A Theory for the Hadley Cell
    Descending and Ascending Edges throughout the Annual Cycle."

    The ascending latitude is assumed equal to the c_ascent times the thermal
    Rossby number to the one-third power.  Then the descending edge assumes a
    uniform Rossby number and the 2-layer BCI condition.

    Note that this is the seasonally varying thermal Rossby number that
    includes `max_lat`: therm_ross = burg_num * delta_h * sin(max_lat), where
    burg_num is the planetary Burger number.  `max_lat` also appears separately
    in the expression, which is why it also must be provided separately.

    """
    lat_ascent = lat_ascent_eta0_approx(therm_ross, c_ascent=c_ascent)
    term1 = (2 ** (4. / 3.)) / (c_ascent ** 4)
    term2 = delta_v / (delta_h * sindeg(max_lat))
    term3 = 1. / (ross_num * (therm_ross ** (1. / 3.)))
    return c_descent * np.rad2deg(
        np.deg2rad(lat_ascent) *
        np.sqrt(0.5 + np.sqrt(0.25 + term1 * term2 * term3))
    )


if __name__ == "__main__":
    pass
