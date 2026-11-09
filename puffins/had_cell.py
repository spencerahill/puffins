"""Hadley cells and other meridional overturning circulation things."""

from __future__ import annotations

import contextvars
import dataclasses
import functools
import inspect
import warnings
from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

import numpy as np
import scipy.ndimage
import xarray as xr

from ._typing import ArrayLike
from .constants import (
    DELTA_V,
    GRAV_EARTH,
    HEIGHT_TROPO,
    RAD_EARTH,
    ROT_RATE_EARTH,
)
from .dynamics import plan_burg_num
from .interp import interpolate, zero_cross_bounds, zero_cross_interp
from .names import LAT_STR, LEV_STR, LON_STR, SIGMA_STR
from .nb_utils import (
    cosdeg,
    lat_area_weight,
    max_and_argmax,
    sindeg,
)
from .vert_coords import subtract_col_avg

if TYPE_CHECKING:
    from xarray.core.types import CompatOptions

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _each_element(
    *core_dim_params: str,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Apply a streamfunction diagnostic separately at each element.

    The decorated function's first argument is the streamfunction, and
    `core_dim_params` name its arguments that give the dimensions it works
    across: the latitude first, then the level. Every other dimension of the
    streamfunction, such as time, model or ensemble member, is looped over:
    the function is applied to each slice, and the results are concatenated
    along that dimension, which keeps the streamfunction's coordinates along
    it. With no such dimension, the function is called unchanged.

    The latitude must be a dimension of the streamfunction, since a
    misnamed one would otherwise be looped over.

    """

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        signature = inspect.signature(func)
        field_param = next(iter(signature.parameters))

        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                bound = signature.bind(*args, **kwargs)
            except TypeError as err:
                raise TypeError(f"{func.__name__}() {err}") from None
            bound.apply_defaults()
            field = bound.arguments[field_param]
            lat_param = core_dim_params[0]
            if bound.arguments[lat_param] not in field.dims:
                raise ValueError(
                    f"{func.__name__}: the streamfunction has no latitude "
                    f"dimension {bound.arguments[lat_param]!r}; its dimensions "
                    f"are {field.dims}. Give its name with {lat_param}."
                )
            core_dims = {bound.arguments[name] for name in core_dim_params}
            extra_dims = [d for d in field.dims if d not in core_dims]
            if not extra_dims:
                return func(*args, **kwargs)
            dim = extra_dims[0]
            if field.sizes[dim] == 0:
                raise ValueError(
                    f"{func.__name__}: the dimension {dim!r} has length 0, so "
                    "there is nothing to compute."
                )
            results = []
            for i in range(field.sizes[dim]):
                bound.arguments[field_param] = field.isel({dim: i})
                # Recursing handles any further extra dimensions.
                results.append(
                    cast(Callable[..., _R], wrapper)(*bound.args, **bound.kwargs)
                )
            return cast(_R, _concat_elements(results, field, dim))

        return wrapper

    return decorator


def _concat_elements(
    results: list[Any], field: xr.DataArray, dim: Hashable
) -> xr.DataArray | list[Any]:
    """Concatenate one result per element of `field` along `dim`.

    A result that is a list is combined item by item. A result that is not a
    DataArray, such as an edge latitude returned as a float at a pole, is made
    one, and only the coordinates every result has are kept. Each kept
    coordinate gains `dim`, and `field`'s coordinates along `dim` are
    attached. A dimension of the results whose labels differ between
    elements, such as the cells `cell_edges_sigma` finds, is numbered by
    position, padded with NaN, and its labels are kept in the coordinate
    named after it with the suffix "_label".

    """
    if isinstance(results[0], list):
        return [
            _concat_elements([result[k] for result in results], field, dim)
            for k in range(len(results[0]))
        ]
    arrays = [r if isinstance(r, xr.DataArray) else xr.DataArray(r) for r in results]
    shared = set.intersection(*(set(arr.coords) for arr in arrays)) - {dim}
    arrays = [
        arr.drop_vars([c for c in arr.coords if c not in shared]) for arr in arrays
    ]
    for res_dim in arrays[0].dims:
        if res_dim in arrays[0].indexes and any(
            not arr.indexes[res_dim].equals(arrays[0].indexes[res_dim])
            for arr in arrays[1:]
        ):
            arrays = [
                arr.assign_coords(
                    {f"{res_dim}_label": (res_dim, arr[res_dim].values)}
                ).assign_coords({res_dim: np.arange(arr.sizes[res_dim])})
                for arr in arrays
            ]
    combined = xr.concat(arrays, dim=dim, coords="all", compat="equals", join="outer")
    return combined.assign_coords(
        {name: coord for name, coord in field.coords.items() if coord.dims == (dim,)}
    )


def _stacklevel_outside_module() -> int:
    """`warnings.warn` stacklevel of the first caller outside this module.

    Counted from the function that calls this one and then warns. The
    per-element decorator puts a varying number of this module's frames
    between a diagnostic and its caller, so no fixed stacklevel reaches
    the caller.

    """
    frame = inspect.currentframe()
    frame = frame.f_back if frame is not None else None
    level = 1
    while frame is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back
        level += 1
    return level


def merid_streamfunc(
    v: xr.DataArray,
    dp: xr.DataArray,
    grav: float = GRAV_EARTH,
    radius: float = RAD_EARTH,
    impose_zero_col_flux: bool = True,
    lat_str: str = LAT_STR,
    lon_str: str = LON_STR,
    lev_str: str = LEV_STR,
) -> xr.DataArray:
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
        The meridional mass streamfunction, signed such that clockwise
        circulation in the latitude-height plane, drawn with latitude
        increasing to the right and height upward, is positive. So the
        Northern Hemisphere Hadley cell, with northward flow aloft, is
        positive, and the Southern Hemisphere Hadley cell negative.

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
    streamfunc = (v_znl_mean * dp_znl_mean).sel({lev_str: levslice}).cumsum(
        dim=lev_str
    ) / grav
    # Weight by surface area to get a mass overturning rate.
    return cast(
        xr.DataArray,
        (lat_area_weight(v[lat_str], radius=radius) * streamfunc)
        .transpose(*v_znl_mean.dims)
        .rename("streamfunc")
        .where(np.isfinite(v)),
    )


@_each_element("lat_str", "lev_str")
def had_cell_strength(
    streamfunc: xr.DataArray,
    *,
    min_plev: float | None = None,
    max_plev: float | None = None,
    do_avg_vert: bool = False,
    do_interp_lat: bool = False,
    lev_str: str = LEV_STR,
    lat_str: str = LAT_STR,
) -> xr.DataArray:
    """Hadley cell strength, as maximum of streamfunction.

    The maximum is taken over latitude and level. Every other dimension,
    such as time, model or ensemble member, is handled separately at each of
    its values and kept in the result.

    With `do_interp_lat`, the maximum's latitude is interpolated between
    gridpoints, to where the latitude derivative of the streamfunction crosses
    zero along the level of the gridpoint maximum. The strength is then the
    maximum, over the levels, of the streamfunction interpolated cubically to
    that latitude.

    A ValueError is raised in two cases. One is when the derivative has no
    zero crossing within 1.1 times the mean latitude spacing of the gridpoint
    maximum, as when that maximum lies at the edge of the latitudes searched.
    The other is when the streamfunction contains NaN anywhere, since the
    cubic interpolation then returns NaN everywhere. Where the NaN is
    confined to some levels, as below ground, excluding those levels with
    `min_plev` or `max_plev`, or averaging over the levels with
    `do_avg_vert`, which skips NaN, avoids the second; a latitude that is NaN
    at every level still raises. Either case at any one value of another
    dimension raises for the whole call.

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
        return _max_interp_lat(sf_valid, lat_str)
    return cast(xr.DataArray, max_and_argmax(sf_valid))


def _max_interp_lat(sf: xr.DataArray, lat_str: str = LAT_STR) -> xr.DataArray:
    """Maximum of `sf`, with its latitude interpolated between gridpoints.

    The latitude is where the latitude derivative of `sf` crosses zero within
    1.1 times the mean latitude spacing of the gridpoint maximum, along the
    line of latitudes through that maximum, with every other dimension held
    at the maximum's position. The strength is the maximum, over every other
    dimension, of `sf` interpolated cubically to that latitude.

    """
    i_max = np.unravel_index(np.nanargmax(sf.data), sf.shape)
    lat_line = sf.isel(
        {d: i for d, i in zip(sf.dims, i_max, strict=True) if d != lat_str}
    )
    lat_center_raw = sf[lat_str].values[i_max[sf.get_axis_num(lat_str)]]
    dlat_mean = sf[lat_str].diff(lat_str).mean()
    dsf_dlat = lat_line.differentiate(lat_str).sel(
        {
            lat_str: slice(
                lat_center_raw - 1.1 * dlat_mean,
                lat_center_raw + 1.1 * dlat_mean,
            )
        }
    )
    lat_center = zero_cross_interp(dsf_dlat, lat_str).values
    sf_at_center = sf.interp({lat_str: lat_center}, method="cubic")
    return cast(xr.DataArray, max_and_argmax(sf_at_center))


@_each_element("lat_str", "lev_str")
def had_cells_strength(
    strmfunc: xr.DataArray,
    min_plev: float | None = None,
    max_plev: float | None = None,
    do_avg_vert: bool = False,
    min_lat: float | None = None,
    max_lat: float | None = None,
    do_interp_lat: bool = False,
    compat: CompatOptions = "override",
    lat_str: str = LAT_STR,
    lev_str: str = LEV_STR,
) -> xr.DataArray:
    """Location and signed magnitude of both Hadley cell centers.

    Every dimension besides latitude and level, such as time, model or
    ensemble member, is handled separately at each of its values and kept in
    the result.

    Each cell is identified by its sense of circulation, which requires the
    sign that `merid_streamfunc` returns: positive for the northern Hadley
    cell, whose flow aloft is northward, and negative for the southern cell.
    Negate a streamfunction of the opposite sign before passing it.

    The southern cell is a negative extremum lying south of the northern
    cell's positive one. The candidates are two such pairs: the global
    maximum with the most negative value south of it, and the global minimum
    with the most positive value north of it. When the minimum lies south of
    the maximum, the two pairs are the same. Otherwise, in a field of the
    expected sign with both Hadley cells present, at least one of the two
    global extrema is a Ferrel cell, which can be stronger than the summer
    Hadley cell, and the pair containing the center nearest the equator is
    kept as the Hadley cells.

    Each partner is the most negative or most positive value anywhere on its
    side of the global extremum, which need not be the adjacent cell. So a
    summer Hadley cell that is missing, or weaker than the polar cell beyond
    it, is usually replaced by that polar cell, without a warning.

    A `UserWarning` is issued when the chosen southern cell is not negative
    or the chosen northern cell not positive, that is, when no value of the
    needed sign exists where the partner is sought. A streamfunction of the
    opposite sign gives wrong cells on any field with Ferrel cells, usually
    without the warning, because a Hadley cell and a Ferrel cell then make up
    a pair of the expected senses.

    With `do_interp_lat`, each center's latitude is interpolated between
    gridpoints. Where that fails, the center is placed at its gridpoint
    instead, with a separate `UserWarning` that gives the interpolation's
    error. Interpolation fails, for example, when the extremum lies at the
    edge of the latitudes searched, such as at `min_lat` or `max_lat`, or
    where the streamfunction is flat in latitude, as along a level where it
    is zero, or when the streamfunction contains NaN.

    """
    lat = strmfunc[lat_str]

    if min_lat is not None:
        strmfunc = strmfunc.where(lat > min_lat, drop=True)
    if max_lat is not None:
        strmfunc = strmfunc.where(lat < max_lat, drop=True)

    kwargs_hc_strength: dict[str, Any] = dict(
        min_plev=min_plev,
        max_plev=max_plev,
        do_avg_vert=do_avg_vert,
        do_interp_lat=do_interp_lat,
        lat_str=lat_str,
        lev_str=lev_str,
    )

    def signed_cell_strength(field: xr.DataArray, sign: int) -> xr.DataArray:
        """The maximum of `field` if `sign` is 1, or its minimum if -1.

        Raises ValueError when `field` has no values to search.
        """
        try:
            return sign * had_cell_strength(sign * field, **kwargs_hc_strength)
        except ValueError as err:
            if not do_interp_lat:
                raise
            interp_error = err
        cell = sign * had_cell_strength(
            sign * field, **{**kwargs_hc_strength, "do_interp_lat": False}
        )
        warnings.warn(
            f"Interpolating the latitude of a cell center failed ({interp_error}), "
            f"so the center is placed at the gridpoint {cell[lat_str].item():.4g}. "
            "Interpolation fails, for example, when the extremum lies at the "
            "edge of the latitudes searched or where the streamfunction is flat "
            "in latitude, or when the streamfunction contains NaN.",
            stacklevel=_stacklevel_outside_module(),
        )
        return cell

    cell_pos_max_strength = signed_cell_strength(strmfunc, 1)
    cell_neg_max_strength = signed_cell_strength(strmfunc, -1)

    # As plain numbers: each latitude coordinate also has its cell's level
    # attached as a scalar coordinate, which conflicts with the field's level
    # dimension when `where` drops the masked latitudes below.
    lat_pos_max = cell_pos_max_strength.coords[lat_str].item()
    lat_neg_max = cell_neg_max_strength.coords[lat_str].item()

    # The two candidate (southern cell, northern cell) pairs. The masked
    # latitudes are dropped: kept as NaN, they make the cubic interpolation
    # under do_interp_lat return NaN at every latitude.
    pairs: list[tuple[xr.DataArray, xr.DataArray]] = []
    try:
        cell_south_of_pos_strength = signed_cell_strength(
            strmfunc.where(lat < lat_pos_max, drop=True), -1
        )
    except ValueError:
        pass  # No values south of the global maximum.
    else:
        pairs.append((cell_south_of_pos_strength, cell_pos_max_strength))
    try:
        cell_north_of_neg_strength = signed_cell_strength(
            strmfunc.where(lat > lat_neg_max, drop=True), 1
        )
    except ValueError:
        pass  # No values north of the global minimum.
    else:
        pairs.append((cell_neg_max_strength, cell_north_of_neg_strength))
    if not pairs:
        raise ValueError(
            "No data south of the streamfunction's global maximum and none "
            "north of its global minimum, so no pair of cells can be identified."
        )

    cell_sh, cell_nh = min(
        pairs, key=lambda pair: min(abs(cell[lat_str].item()) for cell in pair)
    )
    if not cell_sh.item() < 0 < cell_nh.item():
        warnings.warn(
            "The cells chosen as the Hadley cells lack the expected sense of "
            f"circulation: the southern cell's strength is {cell_sh.item():.3g}, "
            "where negative is expected, and the northern cell's is "
            f"{cell_nh.item():.3g}, where positive is expected. Either one "
            "Hadley cell is missing, or the streamfunction is signed opposite "
            "to merid_streamfunc's, with the northern cell negative.",
            stacklevel=_stacklevel_outside_module(),
        )

    hc_strengths = xr.concat(
        [cell_sh, cell_nh], dim=lat_str, coords=[lev_str], compat=compat
    )

    # Create DataArray with one label for each cell, the cell strengths
    # as the values, and the cell center latitudes and levels as coords.
    coords_out = {"cell": ["had_cell_sh", "had_cell_nh"]}
    ds_strengths = xr.Dataset(coords=coords_out)
    arr_lat_center = xr.DataArray(
        hc_strengths[lat_str].values, dims=["cell"], coords=coords_out
    )
    arr_lev_center = xr.DataArray(
        hc_strengths[lev_str].values, dims=["cell"], coords=coords_out
    )
    arr_strength = xr.DataArray(hc_strengths.values, dims=["cell"], coords=coords_out)
    ds_strengths.coords[lat_str] = arr_lat_center
    ds_strengths.coords[lev_str] = arr_lev_center
    ds_strengths["cell_strength"] = arr_strength
    return ds_strengths["cell_strength"]


def _streamfunc_at_avg_lev_max(
    strmfunc: xr.DataArray, hc_strengths: xr.DataArray, lev_str: str = LEV_STR
) -> xr.DataArray:
    """Streamfunction at the average level of the two Hadley cell centers."""
    lev_sh_max = hc_strengths[lev_str][0]
    lev_nh_max = hc_strengths[lev_str][1]
    lev = strmfunc[lev_str]
    lev_avg = lev.sel({lev_str: 0.5 * (lev_sh_max + lev_nh_max)}, method="nearest")
    return cast(xr.DataArray, strmfunc.sel({lev_str: lev_avg}))


@_each_element("lat_str", "lev_str")
def had_cells_shared_edge(
    strmfunc: xr.DataArray,
    fixed_plev: float | None = None,
    min_plev: float | None = None,
    max_plev: float | None = None,
    do_avg_vert: bool = False,
    min_lat: float | None = None,
    max_lat: float | None = None,
    lat_str: str = LAT_STR,
    lev_str: str = LEV_STR,
) -> xr.DataArray:
    """Latitude of shared inner edge of Hadley cells.

    Every dimension besides latitude and level, such as time, model or
    ensemble member, is handled separately at each of its values and kept in
    the result.

    """
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
        sf_at_max = strmfunc.sel({lev_str: fixed_plev}, method="nearest")
    sf_max2max = sf_at_max.where((lat >= lat_sh_max) & (lat <= lat_nh_max), drop=True)
    edge_lat = zero_cross_interp(sf_max2max, lat_str)[lat_str]
    # Drop scalar coords inherited from intermediates (e.g. the level of the
    # cell max); the edge is just a latitude.
    return cast(
        xr.DataArray, edge_lat.drop_vars([c for c in edge_lat.coords if c != lat_str])
    )


@_each_element("lat_str", "lev_str")
def had_cell_edge(
    strmfunc: xr.DataArray,
    cell: str = "north",
    edge: str = "north",
    frac_thresh: float = 0.1,
    fixed_plev: float | None = None,
    min_plev: float | None = None,
    max_plev: float | None = None,
    do_avg_vert: bool = False,
    min_lat: float | None = None,
    max_lat: float | None = None,
    cos_factor: bool = False,
    do_interp: bool = False,
    lat_str: str = LAT_STR,
    lev_str: str = LEV_STR,
) -> xr.DataArray | float:
    """Latitude of poleward edge of either the NH or SH Hadley cell.

    Every dimension besides latitude and level, such as time, model or
    ensemble member, is handled separately at each of its values and kept in
    the result. Where the edge is placed at a pole, a float is returned for a
    field of latitude and level alone.

    """
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
        raise ValueError(f"`cell` must be either 'north' or 'south'; got {cell}.")

    # Restrict to streamfunction at specified level or average of levels
    cell_max = hc_strengths.sel(cell=label)
    lat_of_max = cell_max[lat_str]
    lev_of_max = cell_max[lev_str]

    if do_avg_vert:
        sf_at_max = strmfunc.sel({lev_str: slice(min_plev, max_plev)}).mean(lev_str)
    else:
        if fixed_plev is None:
            lev_for_edges = float(lev_of_max)
        else:
            lev_for_edges = fixed_plev
        sf_at_max = strmfunc.sel({lev_str: lev_for_edges}, method="nearest")

    # Restrict to the latitudes north or south of the max, as specified.
    lat = strmfunc[lat_str]
    if edge == "north":
        which_zero = 0
        lat_compar = lat >= lat_of_max
    elif edge == "south":
        which_zero = -1
        lat_compar = lat <= lat_of_max
    else:
        raise ValueError(f"`edge` must be either 'north' or 'south'; got {cell}.")
    sf_one_side = sf_at_max.where(lat_compar, drop=True)

    # Restrict to the latitudes from the max to the nearest point with
    # opposite-signed value.

    if do_interp:
        # Apply cubic interpolation in latitude to a refined mesh.
        # Otherwise, the cell edge can unphysically vary non-monotonically
        # with `frac_thresh`.
        dlat_avg = float(sf_one_side[lat_str].diff(lat_str).mean(lat_str).values)
        # If there's only one point, assume it's one of the poles.
        if np.isnan(dlat_avg):
            if edge == "north":
                return 90.0
            return -90.0
        lats_interp = np.arange(
            float(sf_one_side[lat_str].min()),
            float(sf_one_side[lat_str].max()) - 0.2 * dlat_avg,
            0.1 * dlat_avg,
        )
        sf_one_side_interp = sf_one_side.interp({lat_str: lats_interp})
        # Explicitly make the last value equal to the original, as otherwise the
        # interp step can overwrite it with nan for some reason.
        sf_one_side_interp = xr.concat(
            [sf_one_side_interp, sf_one_side[-1]], dim=lat_str
        )
    else:
        sf_one_side_interp = sf_one_side
    # Find where the streamfunction crosses the specified fractional threshold,
    # using the Singh 2019 cosine weighting if specified.
    if cos_factor:
        sf_norm = (sf_one_side_interp / cosdeg(sf_one_side_interp[lat_str])) / (
            cell_max / cosdeg(lat_of_max)
        )
    else:
        sf_norm = sf_one_side_interp / cell_max

    sf_thresh_diff = sf_norm - frac_thresh
    # If there is no zero crossing, then the cell extends to the pole.
    try:
        sf_edge_bounds = zero_cross_bounds(sf_thresh_diff, lat_str, which_zero)
    except ValueError:
        if edge == "north":
            return 90.0
        return -90.0
    # Interpolate between the bounding points to the crossing.
    edge_lat = interpolate(sf_edge_bounds, sf_edge_bounds[lat_str], 0, lat_str)[lat_str]
    # Drop scalar coords inherited from intermediates (the 'cell' selection
    # label and the level of the cell max); the edge is just a latitude.
    return cast(
        xr.DataArray, edge_lat.drop_vars([c for c in edge_lat.coords if c != lat_str])
    )


@_each_element("lat_str", "lev_str")
def had_cells_south_edge(
    strmfunc: xr.DataArray,
    frac_thresh: float = 0.1,
    fixed_plev: float | None = None,
    min_plev: float | None = None,
    max_plev: float | None = None,
    do_avg_vert: bool = False,
    min_lat: float | None = None,
    max_lat: float | None = None,
    cos_factor: bool = False,
    lat_str: str = LAT_STR,
    lev_str: str = LEV_STR,
) -> xr.DataArray | float:
    """Latitude of southern edge of southern Hadley cell.

    Every dimension besides latitude and level, such as time, model or
    ensemble member, is handled separately at each of its values and kept in
    the result.

    """
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


@_each_element("lat_str", "lev_str")
def had_cells_north_edge(
    strmfunc: xr.DataArray,
    frac_thresh: float = 0.1,
    fixed_plev: float | None = None,
    min_plev: float | None = None,
    max_plev: float | None = None,
    do_avg_vert: bool = False,
    min_lat: float | None = None,
    max_lat: float | None = None,
    cos_factor: bool = False,
    lat_str: str = LAT_STR,
    lev_str: str = LEV_STR,
) -> xr.DataArray | float:
    """Latitude of northern edge of northern Hadley cell.

    Every dimension besides latitude and level, such as time, model or
    ensemble member, is handled separately at each of its values and kept in
    the result.

    """
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


@_each_element("lat_str", "lev_str")
def had_cells_edges(
    strmfunc: xr.DataArray,
    frac_thresh: float = 0.1,
    min_plev: float | None = None,
    max_plev: float | None = None,
    do_avg_vert: bool = False,
    min_lat: float | None = None,
    max_lat: float | None = None,
    cos_factor: bool = False,
    lat_str: str = LAT_STR,
    lev_str: str = LEV_STR,
) -> list[xr.DataArray | float]:
    """Southern, shared inner, and northern edge of the Hadley cells.

    Every dimension besides latitude and level, such as time, model or
    ensemble member, is handled separately at each of its values and kept in
    each of the three results.

    """
    # Kwargs common to all three edge functions. `frac_thresh` and
    # `cos_factor` are intentionally excluded: they apply to the outer-edge
    # threshold detection in `had_cell_edge`, and the shared inner edge is a
    # zero crossing, so neither is a parameter of `had_cells_shared_edge`.
    shared_kwargs: dict[str, Any] = dict(
        min_plev=min_plev,
        max_plev=max_plev,
        do_avg_vert=do_avg_vert,
        min_lat=min_lat,
        max_lat=max_lat,
        lat_str=lat_str,
        lev_str=lev_str,
    )
    outer_kwargs = dict(frac_thresh=frac_thresh, cos_factor=cos_factor)
    func_and_kwargs: list[
        tuple[Callable[..., xr.DataArray | float], dict[str, Any]]
    ] = [
        (had_cell_edge, dict(cell="south", edge="south", **outer_kwargs)),
        (had_cells_shared_edge, {}),
        (had_cell_edge, dict(cell="north", edge="north", **outer_kwargs)),
    ]
    return [f_kw[0](strmfunc, **f_kw[1], **shared_kwargs) for f_kw in func_and_kwargs]


class _NoCellError(ValueError):
    """No cell remains in one slice of a `cell_edges_sigma` call."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"cell_edges_sigma: {reason}.")
        self.reason = reason


_DROPPED = "dropped"
_KEPT_CUT = "kept, with NaN for an edge,"
_NO_CELL = "no cell"


@dataclasses.dataclass
class _CellReport:
    """The cells one `cell_edges_sigma` call dropped or could not fully measure.

    A call covers one slice for each value of the dimensions besides latitude
    and sigma. The notes are gathered across every slice, and `warn` issues
    one warning per action and reason.
    """

    n_slices: int
    extra_dims: tuple[Hashable, ...]
    slice_index: int = -1
    slice_desc: str = ""
    notes: dict[tuple[str, str], list[tuple[int, str, str]]] = dataclasses.field(
        default_factory=dict
    )

    def start_slice(self, streamfunc: xr.DataArray) -> None:
        """Begin the notes for the next slice, `streamfunc`."""
        self.slice_index += 1
        parts = [
            f"{dim}={streamfunc[dim].values}"
            for dim in self.extra_dims
            if dim in streamfunc.coords and streamfunc[dim].ndim == 0
        ]
        if len(parts) == len(self.extra_dims):
            self.slice_desc = ", ".join(parts)
        else:
            self.slice_desc = f"slice {self.slice_index + 1} of {self.n_slices}"

    def add(
        self,
        action: str,
        reason: str,
        cell: xr.DataArray | None = None,
        lat_str: str = LAT_STR,
        sigma_str: str = SIGMA_STR,
    ) -> None:
        """Note that `cell` was dropped or kept with a NaN edge, or none remains."""
        cell_desc = ""
        if cell is not None:
            cell_desc = (
                f"centered at latitude {cell.sel({lat_str: 'center'}).item():g} "
                f"and sigma {cell[sigma_str].item():g}, of strength "
                f"{cell['cell_strength'].item():.3g}"
            )
        self.notes.setdefault((action, reason), []).append(
            (self.slice_index, self.slice_desc, cell_desc)
        )

    def warn(self) -> None:
        """Warn once per action and reason, in the order first noted."""
        for (action, reason), records in self.notes.items():
            warnings.warn(
                self._message(action, reason, records),
                UserWarning,
                stacklevel=_stacklevel_outside_module(),
            )

    def _message(
        self, action: str, reason: str, records: list[tuple[int, str, str]]
    ) -> str:
        count = len(records)
        _, first_slice, first_cell = records[0]
        along = ", ".join(str(dim) for dim in self.extra_dims)
        if action == _NO_CELL:
            return (
                f"cell_edges_sigma returned NaN for {count} of the "
                f"{self.n_slices} slices along {along}: {reason}. The first is "
                f"at {first_slice}."
            )
        if self.n_slices == 1:
            if count == 1:
                return f"cell_edges_sigma {action} the cell {first_cell}: {reason}."
            return (
                f"cell_edges_sigma {action} {count} cells: {reason}. The first "
                f"is the cell {first_cell}."
            )
        n_hit = len({index for index, _, _ in records})
        return (
            f"cell_edges_sigma {action} {count} {'cell' if count == 1 else 'cells'} "
            f"in {n_hit} of the {self.n_slices} slices along {along}: {reason}. "
            f"The first is at {first_slice}, the cell {first_cell}."
        )


_CELL_REPORT: contextvars.ContextVar[_CellReport | None] = contextvars.ContextVar(
    "_cell_report", default=None
)


def _report_cells(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Gather `cell_edges_sigma`'s notes across its slices, and warn once per reason.

    It wraps the per-element decorator, so it runs once per call, around the
    loop over the slices.
    """
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            # The per-element decorator raises the error, naming the function.
            return func(*args, **kwargs)
        bound.apply_defaults()
        field = bound.arguments["streamfunc"]
        core_dims = {bound.arguments["lat_str"], bound.arguments["sigma_str"]}
        extra_dims = tuple(dim for dim in field.dims if dim not in core_dims)
        report = _CellReport(
            n_slices=int(np.prod([field.sizes[dim] for dim in extra_dims])),
            extra_dims=extra_dims,
        )
        token = _CELL_REPORT.set(report)
        try:
            return func(*args, **kwargs)
        finally:
            _CELL_REPORT.reset(token)
            report.warn()

    return wrapper


def _no_cell_edges(name: Hashable, lat_str: str, sigma_str: str) -> xr.DataArray:
    """A `cell_edges_sigma` result with no cells, for a slice in which none remains."""
    return xr.DataArray(
        np.empty((0, 3)),
        dims=("cell", lat_str),
        coords={
            "cell": np.array([], dtype=object),
            lat_str: ["edge_south", "center", "edge_north"],
            "cell_strength": ("cell", np.empty(0)),
            sigma_str: ("cell", np.empty(0)),
        },
        name=name,
    )


@_report_cells
@_each_element("lat_str", "sigma_str")
def cell_edges_sigma(
    streamfunc: xr.DataArray,
    frac_thresh: float = 0.1,
    cos_factor: bool = False,
    center_min_sigma: float = 0.1,
    center_max_sigma: float = 1,
    cell_min_sigma_depth: float = 0.3,
    max_lat_gap: float = 10,
    min_frac_strength: float = 0.05,
    max_cells_per_sign: int = 4,
    lat_str: str = LAT_STR,
    sigma_str: str = SIGMA_STR,
) -> xr.DataArray:
    """Compute edges of all contiguous streamfunction overturning cells.

    Each cell is a connected region of latitude and sigma where the
    streamfunction is positive, or one where it is negative, between sigma
    `center_min_sigma` and `center_max_sigma`. A NaN is neither, so a column
    of NaN splits a cell in two. A cell's center is where its magnitude is
    largest, and its edges are where, at the level of its center, it falls
    to `frac_thresh` of that maximum; with `cos_factor`, where the
    streamfunction divided by cos(latitude) falls to `frac_thresh` of its
    value at the center. Where a cell is still above that fraction at the
    end of the grid, and the grid's last latitude on that side lies within
    one grid spacing of the pole, the edge is interpolated between that
    latitude and a value of zero at the pole.

    A cell whose edge on one side lies beyond the end of the data is kept,
    with NaN for that edge and with a `UserWarning`. On that side, the data
    end either at the end of a grid whose last latitude lies more than one
    grid spacing from the pole, as for a band of latitudes or a single
    hemisphere, or where a run of NaN begins that reaches the end of the
    grid, as for data masked poleward of some latitude.

    Cells are dropped, in this order:

    - a cell spanning less than `cell_min_sigma_depth` in sigma;
    - a cell no stronger than `min_frac_strength` times the strongest cell
      whose edges can be measured, a cell kept with a NaN edge counting as
      measured;
    - a cell lying entirely between the edges of another cell; a NaN edge
      compares false, so a cell with one neither contains another cell nor
      lies within one;
    - with a `UserWarning`, a cell whose edges cannot be measured: one whose
      maximum lies at the first or last latitude of the grid, beyond which
      it may peak, and one with an edge that cannot be found because the
      latitude beside the edge is NaN or lies in another cell of the same
      sign, where the streamfunction beyond is not NaN all the way to the
      end of the grid, as when a column of NaN splits a cell;
    - with a `UserWarning`, the cells of either sign beyond the
      `max_cells_per_sign` strongest of that sign.

    Each warning covers every cell dropped, or kept with a NaN edge, for one
    reason in the whole call. It gives how many such cells there were, in
    how many of the slices described below, and the first of them.

    A `ValueError` is raised when no cell remains.

    The two Hadley cells are labeled "sh_hadley" and "nh_hadley" by their
    sense of circulation, which requires the sign that `merid_streamfunc`
    returns: the southern cell negative and the northern positive. The cell
    whose maximum lies nearest the equator is taken as one Hadley cell, the
    southern one of two at the same distance. The other is its neighbor on
    the side of its ascending branch, if that neighbor has the opposite sign
    and their facing edges are less than `max_lat_gap` degrees apart, 10 by
    default. Those edges are where each cell falls to `frac_thresh` of its
    own maximum, so their gap grows with `frac_thresh` and with the cells'
    widths. Two half-sine cells meeting at a node, w_s and w_n degrees wide,
    have facing edges (w_s + w_n) arcsin(frac_thresh) / pi degrees apart:
    2.0 degrees for cells 64 degrees wide in all at the default
    `frac_thresh`, and 10.7 at 0.5, which the default `max_lat_gap` does
    not pair. Raise `max_lat_gap` for a high `frac_thresh` or wide cells.
    Every other cell is labeled by its
    position among the cells returned, as a string counting from "0" in the
    south. A streamfunction of the opposite sign gets wrong labels, without
    a warning. Where other cells lie between the Hadley cells, as in a
    double ITCZ, the cell whose maximum lies nearest the equator can be one
    of them. It then takes a Hadley label, and one of the Hadley cells goes
    unlabeled.

    Every dimension besides latitude and sigma, such as time, model or
    ensemble member, is handled separately at each of its values and kept in
    the result. Each such value, or combination of values, is a slice. Where
    the cells found differ between slices, they are numbered by position
    from south to north, padded with NaN where a slice has fewer cells, and
    their labels, such as "sh_hadley" and "nh_hadley", are kept in the
    coordinate "cell_label". A slice in which no cell remains is all NaN,
    with one `UserWarning` giving how many such slices there were and the
    first of them. The `ValueError` is raised only for a streamfunction of
    one slice.

    """
    report = _CELL_REPORT.get()
    assert report is not None, "_report_cells sets the report for each call"
    report.start_slice(streamfunc)
    try:
        return _slice_cell_edges(
            streamfunc,
            report,
            frac_thresh=frac_thresh,
            cos_factor=cos_factor,
            center_min_sigma=center_min_sigma,
            center_max_sigma=center_max_sigma,
            cell_min_sigma_depth=cell_min_sigma_depth,
            max_lat_gap=max_lat_gap,
            min_frac_strength=min_frac_strength,
            max_cells_per_sign=max_cells_per_sign,
            lat_str=lat_str,
            sigma_str=sigma_str,
        )
    except _NoCellError as error:
        if report.n_slices == 1:
            raise ValueError(str(error)) from None
        report.add(_NO_CELL, error.reason)
        return _no_cell_edges(streamfunc.name, lat_str, sigma_str)


def _slice_cell_edges(
    streamfunc: xr.DataArray,
    report: _CellReport,
    *,
    frac_thresh: float,
    cos_factor: bool,
    center_min_sigma: float,
    center_max_sigma: float,
    cell_min_sigma_depth: float,
    max_lat_gap: float,
    min_frac_strength: float,
    max_cells_per_sign: int,
    lat_str: str,
    sigma_str: str,
) -> xr.DataArray:
    """`cell_edges_sigma` of one slice, raising `_NoCellError` if no cell remains."""
    # Discard values in the boundary layer and stratosphere (if specified).
    sigma = streamfunc[sigma_str]
    if center_min_sigma > 0 or center_max_sigma < 1:
        streamfunc = streamfunc.where(
            (sigma >= center_min_sigma) & (sigma <= center_max_sigma), drop=True
        )

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

    if not cells:
        raise _NoCellError(
            "no cell found that spans at least "
            f"cell_min_sigma_depth={cell_min_sigma_depth} in sigma"
        )
    # Convert to an xarray.DataArray
    cells_arr = xr.concat(cells, dim="cell")
    # Restrict to Hadley and Ferrel cells.
    return _hadley_ferrel_cells(
        cells_arr,
        report,
        min_frac_strength=min_frac_strength,
        max_cells_per_sign=max_cells_per_sign,
        max_lat_gap=max_lat_gap,
        lat_str=lat_str,
        sigma_str=sigma_str,
    )


def _find_cell_and_edges(
    streamfunc: xr.DataArray,
    labels: xr.DataArray,
    n: int,
    cell_min_sigma_depth: float = 0.3,
    frac_thresh: float = 0.1,
    cos_factor: bool = False,
    lat_str: str = LAT_STR,
    sigma_str: str = SIGMA_STR,
) -> xr.DataArray:
    """Find the cell with a given label and determine its edges."""
    # Isolate the cell labeled by 'n' and make it positive.
    cell_sign = np.sign(n)
    sf_one_cell = _single_cell(streamfunc, labels, n) * cell_sign
    # Discard cells that aren't sufficiently deep.
    sigma_cell = sf_one_cell.dropna(sigma_str, how="all")[sigma_str]
    cell_sigma_depth = sigma_cell.max() - sigma_cell.min()
    if cell_sigma_depth < cell_min_sigma_depth:
        raise ValueError("Cell is insufficiently deep.")
    # Restrict to the level of this cell's center.
    sf_max = had_cell_strength(sf_one_cell, lev_str=sigma_str, lat_str=lat_str)
    assert sf_max > 0
    sf_at_max = _cell_at_max(streamfunc * cell_sign, sf_max, sigma_str)
    # Retain this cell and those with opposite sign.
    sf_and_opp_sign = _cell_and_opp_sign_cells(sf_at_max, labels, n, sigma_str)
    # A cell whose edges cannot be measured gets NaN edges and the reason,
    # and `_hadley_ferrel_cells` drops it with a warning. A cell with an edge
    # beyond the end of the data gets NaN for that edge and the reason, and
    # is kept with a warning.
    skip_reason = _cell_skip_reason(
        sf_max, streamfunc[lat_str], sf_and_opp_sign, min_width=1, lat_str=lat_str
    )
    cut_reason = ""
    if skip_reason:
        no_edge = sf_max[lat_str] * np.nan
        edges = [no_edge, no_edge]
    else:
        edges = _one_cell_edges(
            sf_and_opp_sign,
            cell_sign,
            frac_thresh,
            cos_factor=cos_factor,
            lat_str=lat_str,
        )
        nan_sides = [
            side
            for side, edge in zip(("southern", "northern"), edges, strict=True)
            if np.isnan(edge.item())
        ]
        own_at_max = labels.sel({sigma_str: sf_at_max[sigma_str]}) == n
        cut = {
            side: _cut_reason(sf_at_max, own_at_max, side, lat_str)
            for side in nan_sides
        }
        split = [side for side in nan_sides if not cut[side]]
        if split:
            skip_reason = (
                f"its {' and '.join(split)} "
                f"{'edge' if len(split) == 1 else 'edges'} could not be "
                "found: the latitude beside the edge is NaN, or lies in another "
                "cell of the same sign, and the streamfunction beyond is not NaN "
                "all the way to the end of the grid, as when a column of NaN "
                "splits a cell"
            )
        else:
            cut_reason = "; ".join(cut[side] for side in nan_sides)
    # Convert to a single DataArray with the cell center included.
    edges.insert(-1, sf_max[lat_str])
    edges_arr = xr.concat(edges, dim=lat_str)
    edges_arr[lat_str] = ["edge_south", "center", "edge_north"]
    edges_arr.coords["cell_strength"] = float(sf_max * cell_sign)
    edges_arr.coords["skip_reason"] = skip_reason
    edges_arr.coords["cut_reason"] = cut_reason
    return edges_arr


def _split_streamfunc_into_cells(
    streamfunc: xr.DataArray,
) -> tuple[xr.DataArray, list[int]]:
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
    labels = xr.ones_like(streamfunc) * labels_vals
    indices = list(range(-num_neg, 0)) + list(range(1, num_pos + 1))
    return labels, indices


def _single_cell(
    streamfunc: xr.DataArray, labels: xr.DataArray, n: int
) -> xr.DataArray:
    """Restrict streamfunction to single previously identified cell."""
    return cast(xr.DataArray, streamfunc.where(labels == n))


def _cell_at_max(
    sf_one_cell: xr.DataArray, sf_max: xr.DataArray, sigma_str: str = SIGMA_STR
) -> xr.DataArray:
    """Restrict the streamfunction to the level of the cell's maximum."""
    sigma_max = float(sf_max[sigma_str].values)
    return cast(xr.DataArray, sf_one_cell.sel({sigma_str: sigma_max}))


def _cell_and_opp_sign_cells(
    sf_at_max: xr.DataArray, labels: xr.DataArray, n: int, sigma_str: str = SIGMA_STR
) -> xr.DataArray:
    """Restrict to one positive overturning cell, negative cells and zeros.

    A zero streamfunction is outside every cell, so it is kept as a value below
    any edge threshold; masked out, it left the edge search no crossing beside
    a cell bounded by zeros, and so a NaN edge.
    """
    labels_at_max = labels.sel({sigma_str: sf_at_max[sigma_str]})
    sign_at_max = np.sign(sf_at_max)
    return cast(
        xr.DataArray, sf_at_max.where((labels_at_max == n) | (sign_at_max <= 0))
    )


def _cell_skip_reason(
    sf_max: xr.DataArray,
    lats: xr.DataArray,
    sf_and_opp_sign: xr.DataArray,
    min_width: int = 3,
    lat_str: str = LAT_STR,
) -> str:
    """Why an identified cell's edges cannot be measured, or "" if they can."""
    lat_max = float(sf_max[lat_str].values)
    if lat_max in (float(lats.values[0]), float(lats.values[-1])):
        return (
            "its maximum lies at the first or last latitude of the grid, beyond "
            "which it may peak"
        )
    # Never true at min_width=1, as `_find_cell_and_edges` calls it: the row
    # counted includes the cell's maximum, which is positive.
    if len(sf_and_opp_sign.where(sf_and_opp_sign > 0, drop=True)) < min_width:
        return f"it spans fewer than {min_width} latitudes at the level of its maximum"
    return ""


def _cut_reason(
    profile: xr.DataArray, own: xr.DataArray, side: str, lat_str: str = LAT_STR
) -> str:
    """Why a cell's NaN edge on `side` lies beyond the end of the data, or "".

    `profile` is the streamfunction along latitude at the level of the cell's
    center, and `own` is True at the cell's own latitudes there. The edge lies
    beyond the end of the data when the cell reaches the end of the grid on
    that side, or when every latitude beyond the cell on that side is NaN.
    Otherwise data lie beyond a NaN, as when a column of NaN splits a cell.
    """
    lats = profile[lat_str]
    own_lats = lats.where(own, drop=True)
    if side == "northern":
        beyond = profile.where(lats > own_lats.max(), drop=True)
        end = "last"
    else:
        beyond = profile.where(lats < own_lats.min(), drop=True)
        end = "first"
    if beyond.size == 0:
        return (
            f"its {side} edge lies beyond the {end} latitude of the grid, which "
            "is more than one grid spacing from the pole"
        )
    if bool(beyond.isnull().all()):
        return (
            f"its {side} edge lies where the streamfunction is NaN, from beside "
            f"the cell to the {end} latitude of the grid"
        )
    return ""


def _grid_end_near_pole(lats: xr.DataArray, sign_factor: int) -> bool:
    """Whether the grid's last latitude toward a pole lies within one spacing of it.

    `sign_factor` is 1 for the North Pole and -1 for the South Pole.
    """
    ordered = np.sort(sign_factor * lats.values)
    distance, spacing = 90.0 - ordered[-1], ordered[-1] - ordered[-2]
    return bool(distance <= spacing or np.isclose(distance, spacing))


def _one_cell_edges(
    sf_and_opp_sign: xr.DataArray,
    cell_sign: int,
    frac_thresh: float,
    cos_factor: bool = False,
    lat_str: str = LAT_STR,
) -> list[xr.DataArray]:
    """Compute northern and southern edges of a cell.

    The cell is the positive part of ``sf_and_opp_sign``. Its negative values
    are the opposite-signed cells beside it, which the edge search may cross
    into, so the edges are measured from the cell's own maximum, never from a
    stronger neighbor's.
    """
    edges = []
    # Look for the northern edge and then, by reversing the sign of the
    # latitudes array, the southern edge.
    for sign_factor in [1, -1]:
        sf_cell = sf_and_opp_sign.copy()
        lats = sf_cell[lat_str]
        sf_max = sf_cell.max(lat_str)
        lat_max = lats[sf_cell.argmax(lat_str)]

        if cos_factor:
            sf_norm = (sf_cell / cosdeg(lats)) / (sf_max / cosdeg(lat_max))
        else:
            sf_norm = sf_cell / sf_max

        # threshold = float(sf_max*frac_thresh)
        # assert threshold > 0

        # Flip the sign and the indexing if looking for southern edge.
        lat_cell = sf_norm[lat_str] * sign_factor
        sf_norm[lat_str] = lat_cell
        if sign_factor == -1:
            sf_norm = sf_norm.isel({lat_str: slice(None, None, -1)})
            lat_cell = sf_norm[lat_str]

        # Restrict to latitudes greater than that at the maximum.
        sf_one_side = sf_norm.where(lat_cell >= sign_factor * lat_max, drop=True)
        lat_one_side = sf_one_side[lat_str]

        # Find where the streamfunction drops below the threshold.  This
        # includes points with opposite-signed streamfunction.
        sf_below = sf_one_side.where(sf_one_side < frac_thresh, drop=True)

        # If the threshold is never crossed, assume that the cell goes to
        # zero at that pole, provided the grid ends within one spacing of it.
        # Otherwise the edge lies beyond the end of the grid, unmeasured.
        if len(sf_below) == 0 and not _grid_end_near_pole(lats, sign_factor):
            edges.append(lats.isel({lat_str: 0}) * np.nan)
            continue
        if len(sf_below) == 0:
            first_lat_below = 90.0 * np.sign(lat_one_side.max(lat_str))
            first_below = xr.zeros_like(sf_norm.isel({lat_str: -1}))
            first_below[lat_str].values = first_lat_below

        # Otherwise, keep the adjacent below-threshold point.
        else:
            first_lat_below = sf_below[lat_str].min()
            first_below = sf_below.sel({lat_str: first_lat_below})

        # Get the adjacent above-threshold value towards the cell center.
        last_lat_above = lat_one_side.where(
            lat_one_side < first_lat_below, drop=True
        ).max()
        last_above = sf_one_side.sel({lat_str: last_lat_above}).copy()

        # Revert the streamfunctions and latitudes to their original sign.
        first_below *= cell_sign
        last_above *= cell_sign
        first_below[lat_str] *= sign_factor
        last_above[lat_str] *= sign_factor

        # Interpolate between the two gridpoints to the actual crossing.
        thresh_bounds = xr.concat([last_above, first_below], dim=lat_str)
        edge = interpolate(
            thresh_bounds, thresh_bounds[lat_str], frac_thresh * cell_sign, lat_str
        )

        edges.append(edge)
    # The northern edge was found first. Put each edge in the slot of the side
    # it was searched on, which a sort cannot do for a NaN edge.
    return edges[::-1]


def _hadley_ferrel_cells(
    cells: xr.DataArray,
    report: _CellReport,
    min_frac_strength: float = 0.05,
    max_cells_per_sign: int = 4,
    max_lat_gap: float = 10,
    lat_str: str = LAT_STR,
    sigma_str: str = SIGMA_STR,
) -> xr.DataArray:
    """From identified cell edges, keep only the Hadley and Ferrel cells.

    TODO: In a very few cases, the polar cell in one hemisphere is stronger
    than the ferrel cell in the opposite hemisphere.  As such, it gets kept
    rather than this weaker Ferrel cell.  This needs to be fixed.

    """
    # Discard cells weaker than a given fraction of the strongest cell whose
    # edges could be measured, or of the strongest cell if none could.
    strengths = np.abs(np.broadcast_to(cells["cell_strength"].values, len(cells)))
    measurable = np.broadcast_to(cells["skip_reason"].values, len(cells)) == ""
    reference = strengths[measurable] if measurable.any() else strengths
    cells = cells.isel(cell=strengths > min_frac_strength * reference.max())
    # Discard cells that are entirely within the edges of another cell. A NaN
    # edge compares False, so a cell with one neither contains nor lies in
    # another.
    edges_south = cells.sel({lat_str: "edge_south"}).values
    edges_north = cells.sel({lat_str: "edge_north"}).values
    nested = (
        (edges_south[:, np.newaxis] > edges_south)
        & (edges_north[:, np.newaxis] < edges_north)
    ).any(axis=1)
    cells = cells.isel(cell=~nested)
    # Drop, with a warning, each cell whose edges could not be measured.
    skip_reasons = np.broadcast_to(cells["skip_reason"].values, cells.sizes["cell"])
    for i in np.flatnonzero(skip_reasons != ""):
        report.add(_DROPPED, skip_reasons[i], cells.isel(cell=i), lat_str, sigma_str)
    cells = cells.isel(cell=skip_reasons == "").drop_vars("skip_reason")
    if cells.sizes["cell"] == 0:
        raise _NoCellError("no cell remains whose edges could be measured")
    # Restrict candidates to the strongest few of each sign.
    if len(cells["cell"]) > max_cells_per_sign:
        cells_pos = cells.where(cells["cell_strength"] > 0, drop=True)
        cells_pos = cells_pos.sortby(cells_pos["cell_strength"], ascending=False)
        cells_neg = cells.where(cells["cell_strength"] < 0, drop=True)
        cells_neg = cells_neg.sortby(cells_neg["cell_strength"], ascending=True)
        for weaker in (cells_pos, cells_neg):
            for k in range(max_cells_per_sign, weaker.sizes["cell"]):
                report.add(
                    _DROPPED,
                    f"more than {max_cells_per_sign} cells of its sign remain, and "
                    f"it is not among the {max_cells_per_sign} strongest; raise "
                    "max_cells_per_sign to keep it",
                    weaker.isel(cell=k),
                    lat_str,
                    sigma_str,
                )
        cells_pos = cells_pos.isel(cell=slice(0, max_cells_per_sign))
        cells_neg = cells_neg.isel(cell=slice(0, max_cells_per_sign))
        cells = xr.concat([cells_pos, cells_neg], dim="cell")
    # Note each cell kept with a NaN edge beyond the end of the data.
    cut_reasons = np.broadcast_to(cells["cut_reason"].values, cells.sizes["cell"])
    for i in np.flatnonzero(cut_reasons != ""):
        report.add(_KEPT_CUT, cut_reasons[i], cells.isel(cell=i), lat_str, sigma_str)
    cells = cells.drop_vars("cut_reason")
    # Sort from south to north.
    cells = cells.sortby(cells.sel({lat_str: "center"}))
    # Label the Hadley cells by their sense of circulation, the southern cell
    # negative and the northern positive. The cell whose maximum lies nearest
    # the equator, the southern one of a tie, is taken as one of them. The
    # other is its neighbor on the side of its ascending branch, north of a
    # negative cell and south of a positive one, if that neighbor has the
    # opposite sign and their facing edges are less than max_lat_gap apart.
    #
    # The strength is a scalar coordinate when every cell's is the same, as
    # with only one cell.
    strengths = np.broadcast_to(cells["cell_strength"].values, cells.sizes["cell"])
    edges_south = cells.sel({lat_str: "edge_south"}).values
    edges_north = cells.sel({lat_str: "edge_north"}).values
    i_eq = int(np.argmin(np.abs(cells.sel({lat_str: "center"}).values)))
    if strengths[i_eq] < 0:
        label_eq, i_other, label_other = "sh_hadley", i_eq + 1, "nh_hadley"
    else:
        label_eq, i_other, label_other = "nh_hadley", i_eq - 1, "sh_hadley"
    labels: list[int | str] = list(range(len(cells)))
    labels[i_eq] = label_eq
    if 0 <= i_other < len(cells):
        i_south, i_north = sorted([i_eq, i_other])
        lat_gap = edges_south[i_north] - edges_north[i_south]
        if strengths[i_other] * strengths[i_eq] < 0 and lat_gap < max_lat_gap:
            labels[i_other] = label_other
    cells["cell"] = labels
    return cells


# Descending edge predictions based on baroclinic instability onset."""
def fixed_ro_bci_edge_small_angle_lata0(
    burg_num: ArrayLike,
    ross_num: float = 1,
    delta_v: float = DELTA_V,
    c_descent: float = 1,
) -> ArrayLike:
    """HC edge for fixed-Ro, BCI model with equatorial ascent.

    Eq. 6 of Hill et al. (2025), c_descent (Bu delta_v / (2 Ro))^(1/4), in
    degrees. Only its value at ``c_descent = 1`` is the ``lat_fixed_ro_ann``
    that ``fixed_ro_bci_edge`` and ``fixed_ro_bci_edge_small_angle`` take,
    since they apply their own ``c_descent``.
    """
    return cast(
        ArrayLike,
        c_descent * np.rad2deg(((burg_num * delta_v) / (2.0 * ross_num)) ** 0.25),
    )


def fixed_ro_bci_edge(
    ascentlat: ArrayLike,
    lat_fixed_ro_ann: ArrayLike | None = None,
    burg_num: ArrayLike | None = None,
    ross_num: float = 1.0,
    c_descent: float = 1.0,
    delta_v: float = DELTA_V,
    height: float = HEIGHT_TROPO,
    grav: float = GRAV_EARTH,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
) -> ArrayLike:
    """Fixed-Ro, 2-layer BCI model of HC descending edge, at full angle.

    Solves u_Ro = u_BCI, Eqs. 2 and 3 of Hill et al. (2025), without the
    small-angle approximation, and multiplies the solution by the fitting
    constant c_descent, as they define it on p. 5506 (phi_d = c_d phi_BCI).
    In s = sin^2(lat) the condition is the quadratic

        s^2 + (X - sin^2(ascentlat)) s - X = 0,  X = Bu delta_v / (2 Ro),

    which is -X at s = 0 and cos^2(ascentlat) at s = 1, so exactly one root
    lies in (0, 1]. The function computes that root in closed form as

        1 - s = 2 cos^2(ascentlat) / (1 + cos^2(ascentlat) + X
                + ((sin^2(ascentlat) - X)^2 + 4 X)^(1/2)),

    whose denominator is at least 1 and a sum of non-negative terms, so the
    result keeps full precision for any X.

    Parameters
    ----------
    ascentlat : float or array-like
        Latitude of the ascending edge (degrees). Only its magnitude enters.
    lat_fixed_ro_ann : float or array-like, optional
        X^(1/4) in degrees: the small-angle edge for equatorial ascent at
        ``c_descent = 1``, which ``fixed_ro_bci_edge_small_angle_lata0``
        returns with its default ``c_descent``. The full-angle edge for
        equatorial ascent is smaller. If given, ``burg_num``, ``ross_num``,
        ``delta_v`` and the planetary arguments are unused.
    burg_num : float or array-like, optional
        Planetary Burger number. Default: computed from ``height``, ``grav``,
        ``rot_rate`` and ``radius``.
    ross_num : float, optional
        Cell-mean Rossby number. Default: 1.
    c_descent : float, optional
        Fitting constant multiplying the solution. Default: 1.
    delta_v : float, optional
        Bulk static stability. Default: DELTA_V.
    height, grav, rot_rate, radius : float, optional
        Tropopause height (m), gravity (m/s^2), rotation rate (1/s) and
        radius (m) of the planet. Default: Earth.

    Returns
    -------
    float or array-like
        Descending edge latitude (degrees), non-negative. NaN where an input
        is NaN.
    """
    lat_fixed_ro_ann4: ArrayLike
    if lat_fixed_ro_ann is None:
        if burg_num is None:
            burg_num = plan_burg_num(
                height, grav=grav, rot_rate=rot_rate, radius=radius
            )
        lat_fixed_ro_ann4 = burg_num * delta_v / (2.0 * ross_num)
    else:
        lat_fixed_ro_ann4 = np.deg2rad(lat_fixed_ro_ann) ** 4
    # The root as 1 - s, the squared cosine of the edge. Every term of the
    # denominator is non-negative and the first is 1, so for no X does it
    # cancel or divide by zero.
    sin2_ascent = sindeg(ascentlat) ** 2
    cos2_ascent = cosdeg(ascentlat) ** 2
    cos2_edge = (
        2
        * cos2_ascent
        / (
            1
            + cos2_ascent
            + lat_fixed_ro_ann4
            + np.sqrt((sin2_ascent - lat_fixed_ro_ann4) ** 2 + 4 * lat_fixed_ro_ann4)
        )
    )
    return cast(ArrayLike, c_descent * np.rad2deg(np.arccos(np.sqrt(cos2_edge))))


def fixed_ro_bci_edge_small_angle(
    ascentlat: ArrayLike,
    lat_fixed_ro_ann: ArrayLike | None = None,
    burg_num: ArrayLike | None = None,
    ross_num: float = 1,
    delta_v: float = DELTA_V,
    c_descent: float = 1.0,
    height: float = HEIGHT_TROPO,
    grav: float = GRAV_EARTH,
    rot_rate: float = ROT_RATE_EARTH,
    radius: float = RAD_EARTH,
) -> ArrayLike:
    """Small-angle solution for fixed-Ro, BCI model for HC descending edge.

    Eq. 4 of Hill et al. (2025).

    Both ascentlat and lat_fixed_ro_ann should be in degrees, not radians.
    And the return value is in degrees, not radians. ``lat_fixed_ro_ann`` is
    the small-angle edge for equatorial ascent at ``c_descent = 1``, as in
    ``fixed_ro_bci_edge``; ``c_descent`` multiplies the solution in either
    case.

    """
    lat_fixed_ro_ann4: ArrayLike
    if lat_fixed_ro_ann is None:
        if burg_num is None:
            burg_num = plan_burg_num(
                height, grav=grav, rot_rate=rot_rate, radius=radius
            )
        lat_fixed_ro_ann4 = burg_num * delta_v / (2.0 * ross_num)
    else:
        lat_fixed_ro_ann4 = np.deg2rad(lat_fixed_ro_ann) ** 4

    lat_a2 = np.deg2rad(ascentlat) ** 2
    return cast(
        ArrayLike,
        c_descent
        * np.rad2deg(
            np.sqrt(0.5 * lat_a2 + np.sqrt(0.25 * lat_a2**2 + lat_fixed_ro_ann4))
        ),
    )


def lin_ro_bci_edge_small_angle_lata0(
    burg_num: ArrayLike,
    ross_ascent: float,
    ross_descent: float,
    delta_v: float = DELTA_V,
) -> ArrayLike:
    """Baroclinic instab. est. for cell edge, linear Ro profile.

    Small-angle, with ascent at the equator, and the Rossby number varying
    linearly in latitude from ``ross_ascent`` there to ``ross_descent`` at the
    edge. The wind of ``grad_bal.u_lin_ro_small_ang`` is then
    Omega a lat_d^2 (Ro_a + 2 Ro_d) / 3 at the edge lat_d. Setting it equal to
    the small-angle u_BCI = g H delta_v / (2 Omega a lat_d^2), Eq. 3 of Hill et
    al. (2025), gives lat_d^4 = 3 Bu delta_v / (2 (Ro_a + 2 Ro_d)). With
    Ro_a = Ro_d this is ``fixed_ro_bci_edge_small_angle_lata0`` at its default
    ``c_descent``.
    """
    return cast(
        ArrayLike,
        np.rad2deg(
            (3 * burg_num * delta_v / (2 * (ross_ascent + 2 * ross_descent))) ** 0.25
        ),
    )


# Edge solutions using supercriticality to predict ascent location.
def lat_ascent_eta0_approx(
    burg_delh_sin_latmax: ArrayLike, c_ascent: float = 1.0
) -> ArrayLike:
    """Approx. zero cross of abs. vort. of Lindzen-Hou solstice forcing.

    Returns c_ascent (Ro_th / 2)^(1/3) in degrees: Eq. 13 of Hill et al.
    (2022), which is Eq. 12 of Hill, Bordoni and Mitchell (2021), the
    approximate small-angle solution of their Eq. 10, times the fitting
    constant c_ascent.

    Parameters
    ----------
    burg_delh_sin_latmax : float or array-like
        Planetary Burger number times delta_h times sin(lat_max), with lat_max
        the latitude of maximum forcing: the thermal Rossby number Ro_th of
        Hill, Bordoni and Mitchell (2021, Eq. 7) and Hill et al. (2022,
        Eq. 14). A negative value gives the latitude mirrored about the
        equator.
    c_ascent : float, optional
        Fitting constant multiplying the solution. Default: 1.

    Returns
    -------
    float or array-like
        Latitude (degrees).
    """
    if np.all(burg_delh_sin_latmax == 0):
        return 0.0
    return cast(
        ArrayLike,
        np.sign(burg_delh_sin_latmax)
        * np.rad2deg(c_ascent * (0.5 * np.abs(burg_delh_sin_latmax)) ** (1.0 / 3.0)),
    )


def fixed_ro_bci_edge_supercrit_ascent(
    burg_delh_sin_latmax: ArrayLike,
    max_lat: float = 90,
    c_ascent: float = 1,
    c_descent: float = 1,
    delta_v: float = DELTA_V,
    delta_h: float = 1.0 / 15.0,
    ross_num: float = 1,
) -> ArrayLike:
    """Descending edge approx including theory for ascent lat.

    From Hill, Bordoni, and Mitchell, 2022, JAS, "A Theory for the Hadley Cell
    Descending and Ascending Edges throughout the Annual Cycle."

    The ascending latitude is ``lat_ascent_eta0_approx``, c_ascent times
    (burg_delh_sin_latmax / 2) to the one-third power (2022 Eq. 13). The
    descending edge is then ``fixed_ro_bci_edge_small_angle``, Eq. 4 of Hill
    et al. (2025), with the planetary Burger number
    Bu = |burg_delh_sin_latmax| / (delta_h |sin(max_lat)|). This is 2022
    Eq. 15 with its coefficient 2^(4/3) / c_ascent^4 replaced by
    2^(1/3) / c_ascent^4: that paper's critical wind for baroclinic
    instability omitted a factor of 2 (Hill et al. 2025, footnote 1).

    ``burg_delh_sin_latmax`` is the planetary Burger number times delta_h
    times sin(max_lat): the seasonally varying thermal Rossby number Ro_th
    of Hill et al. (2022, Eq. 14).  `max_lat` also appears separately in the
    expression, which is why it also must be provided separately. A negative
    ``burg_delh_sin_latmax``, for forcing peaked in the Southern Hemisphere,
    gives the edge mirrored about the equator, negative; the sign of
    ``max_lat`` does not matter.

    The returned edge has the sign of ``burg_delh_sin_latmax``, so it lies in
    the hemisphere of the forcing maximum, as the summer cell's descending
    edge does. The edge condition is symmetric about the equator, and Hill et
    al. (2022, p. 2518) apply it equally to the cross-equatorial winter cell,
    given a Rossby number representative of that cell; that cell's
    descending edge is then the negative of the returned value. At
    ``max_lat = 0`` the Burger number cannot be recovered from
    ``burg_delh_sin_latmax``, and the function divides by zero.

    """
    lat_ascent = lat_ascent_eta0_approx(burg_delh_sin_latmax, c_ascent=c_ascent)
    burg_num = np.abs(burg_delh_sin_latmax) / (delta_h * np.abs(sindeg(max_lat)))
    edge = fixed_ro_bci_edge_small_angle(
        lat_ascent,
        burg_num=burg_num,
        ross_num=ross_num,
        delta_v=delta_v,
        c_descent=c_descent,
    )
    return cast(ArrayLike, np.sign(burg_delh_sin_latmax) * edge)


if __name__ == "__main__":
    pass
