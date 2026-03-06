"""Column budget adjustment quantities."""
import numpy as np
import windspharm.xarray
import xarray as xr

from .names import LAT_STR, LON_STR, TIME_STR


def uv_col_budg_adj(
    u_col_int: xr.DataArray,
    v_col_int: xr.DataArray,
    tendency: xr.DataArray,
    source: xr.DataArray,
    lat_str: str = LAT_STR,
    lon_str: str = LON_STR,
    time_str: str = TIME_STR,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Apply the column tracer budget adjustment method to enforce closure.

    For tracers other than dry mass, the `u_col_int` and `v_col_int` fields
    must be the column integral of the *product* of the given tracer and the
    given wind component.  So for MSE (denoted h), these would be
    int_0^{p_s} h*u dp/g and int_0^{p_s} h*v dp/g.

    `tendency` is the d/dt term appearing in the column budget to be closed.
    For MSE, this would be d<E>/dt where E = c_v T + gz + L_v q.
    For mass, this is d p_s / dt.

    `source` is the RHS term appearing in the column budget to be closed.
    For MSE, this is F_net (TOA radiative + SFC radiative + turbulent net
    fluxes directed into the atmosphere).
    For mass, this is g(E-P).

    Returns the adjusted (u_col, v_col) that close the budget.
    """
    vecwind_col = windspharm.xarray.VectorWind(u_col_int, v_col_int)
    div_col = vecwind_col.divergence()
    resid = tendency + div_col - source
    # Spectral transform of residual; assumed purely divergent.
    div_adj_spec = vecwind_col._api.s.grdtospec(
        resid.transpose(lat_str, lon_str, time_str).values
    )
    vort_adj_spec = np.zeros_like(div_adj_spec)
    # Invert to get gridded wind adjustment.
    u_adj_vals, v_adj_vals = vecwind_col._api.s.getuv(
        vort_adj_spec, div_adj_spec
    )
    # Wrap back into DataArrays with original dimension order.
    u_col_adj = (
        xr.ones_like(u_col_int.transpose(lat_str, lon_str, time_str))
        * u_adj_vals
    ).transpose(*u_col_int.dims)
    v_col_adj = (
        xr.ones_like(v_col_int.transpose(lat_str, lon_str, time_str))
        * v_adj_vals
    ).transpose(*v_col_int.dims)
    return u_col_int - u_col_adj, v_col_int - v_col_adj


def resid_after_col_adj(
    u_col_adjusted: xr.DataArray,
    v_col_adjusted: xr.DataArray,
    tendency: xr.DataArray,
    source: xr.DataArray,
) -> xr.DataArray:
    """Compute the column budget residual after adjustment.

    Should be near machine zero if adjustment was applied correctly.
    """
    vecwind = windspharm.xarray.VectorWind(u_col_adjusted, v_col_adjusted)
    div_col_adjusted = vecwind.divergence()
    return tendency + div_col_adjusted - source
