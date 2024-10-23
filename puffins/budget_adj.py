"""Mass budget-related quantities."""
# from animal_spharm import SpharmInterface
# import numpy as np

# from .constants import RAD_EARTH
# from .vertcoord import (
#     d_deta_from_pfull,
#     d_deta_from_phalf,
#     to_pfull_from_phalf,
#     dp_from_ps,
#     int_dp_g,
#     integrate,
# )
# from .names import PFULL_STR, PHALF_STR, PLEVEL_STR, TIME_STR
# from .numerics import (d_dx_from_latlon, d_dy_from_lat, d_dp_from_p,
#                        d_dx_at_const_p_from_eta, d_dy_at_const_p_from_eta)
# from .advection import horiz_advec, horiz_advec_spharm
# from .tendencies import (time_tendency_first_to_last,
#                          time_tendency_each_timestep)


# def horiz_divg(u, v, radius):
#     """Mass horizontal divergence."""
#     du_dx = d_dx_from_latlon(u, radius)
#     dv_dy = d_dy_from_lat(v, radius, vec_field=True)
#     return du_dx + dv_dy


# def horiz_divg_spharm(u, v, radius=RAD_EARTH):
#     sph_int = SpharmInterface(u, v, rsphere=radius, make_vectorwind=True)
#     del u, v
#     divg = sph_int.vectorwind.divergence()
#     return sph_int.to_xarray(divg)


# def horiz_divg_from_eta(u, v, ps, radius, bk, pk):
#     return (d_dx_at_const_p_from_eta(u, ps, radius, bk, pk) +
#             d_dy_at_const_p_from_eta(v, ps, radius, bk, pk, vec_field=True))


# def vert_divg(omega, p):
#     """Mass vertical divergence."""
#     return d_dp_from_p(omega, p)


# def divg_3d(u, v, omega, radius, p):
#     """Total (3-D) divergence.  Should nearly equal 0 by continuity."""
#     return horiz_divg(u, v, radius) + vert_divg(omega, p)


# def dp(ps, bk, pk, arr):
#     """Pressure thickness of hybrid coordinate levels from surface pressure."""
#     return dp_from_ps(bk, pk, ps, arr[PFULL_STR])


# def mass_column(ps):
#     """Total mass per square meter of atmospheric column."""
#     return ps / GRAV_EARTH


# def mass_column_integral(bk, pk, ps):
#     """
#     Total mass per square meter of atmospheric column.

#     Explicitly computed by integrating over pressure, rather than implicitly
#     using surface pressure.  Useful for checking if model data conserves mass.
#     """
#     dp = dp_from_ps(bk, pk, ps)
#     return dp.sum(dim=PFULL_STR)


# def mass_column_source(evap, precip, grav=GRAV_EARTH):
#     """Source term of column mass budget."""
#     return grav * (evap - precip)


# def mass_column_divg(u, v, radius, dp):
#     """Horizontal divergence of vertically integrated flow."""
#     u_int = integrate(u, dp, is_pressure=True)
#     v_int = integrate(v, dp, is_pressure=True)
#     return horiz_divg(u_int, v_int, radius)


# def mass_column_divg_spharm(u, v, radius, dp):
#     """Horizontal divergence of vertically integrated flow."""
#     u_int = integrate(u, dp, is_pressure=True)
#     v_int = integrate(v, dp, is_pressure=True)
#     return horiz_divg_spharm(u_int, v_int, radius)


# def budget_residual(tendency, transport, source=None, freq='1M'):
#     """Compute residual between tendency and transport terms.

#     Resamples transport and source terms to specified frequency, since often
#     tendencies are computed at monthly intervals while the transport is much
#     higher frequencies (e.g. 3- or 6-hourly).
#     """
#     resid = (tendency +
#              transport.resample(freq, TIME_STR, how='mean').dropna(TIME_STR))
#     if source is not None:
#         resid -= source.resample(freq, TIME_STR, how='mean').dropna(TIME_STR)
#     return resid


# def mass_column_budget_lhs(ps, u, v, radius, dp, freq='1M'):
#     """Tendency plus flux terms in the column-integrated mass budget.

#     Theoretically the sum of the tendency and transport terms exactly equals
#     the source term, however artifacts introduced by numerics and other things
#     yield a residual.
#     """
#     # tendency = time_tendency_first_to_last(ps, freq=freq)
#     tendency = time_tendency_each_timestep(ps)
#     transport = mass_column_divg_spharm(u, v, radius, dp)
#     return budget_residual(tendency, transport, freq=freq)


# def mass_column_budget_with_adj_lhs(ps, u, v, q, radius, dp, freq='1M'):
#     """Tendency plus flux terms in the column-integrated mass budget.

#     Theoretically the sum of the tendency and transport terms exactly equals
#     the source term, however artifacts introduced by numerics and other things
#     yield a residual.
#     """
#     tendency = time_tendency_first_to_last(ps, freq=freq)
#     transport = mass_column_divg_adj(u, v, q, ps, radius, dp)
#     return budget_residual(tendency, transport, freq=freq)


# def mass_column_budget_residual(ps, u, v, evap, precip, radius, dp, freq='1M'):
#     """Residual in the mass budget.

#     Theoretically the sum of the tendency and transport terms exactly equals
#     the source term, however artifacts introduced by numerics and other things
#     yield a residual.
#     """
#     # tendency = time_tendency_first_to_last(ps, freq=freq)
#     tendency = time_tendency_each_timestep(ps)
#     transport = mass_column_divg_spharm(u, v, radius, dp)
#     source = mass_column_source(evap, precip)
#     return tendency + transport - source
#     # return budget_residual(tendency, transport, source, freq=freq)


# def uv_column_budget_adjustment(u, v, residual, col_integral, radius):
#     """Generic column budget conservation adjustment to apply to horiz wind."""
#     for p_str in [PFULL_STR, PHALF_STR, PLEVEL_STR]:
#         if hasattr(u, p_str):
#             dim = p_str
#             break
#     else:
#         raise AttributeError("Couldn't find vertical dimension "
#                              "of {}".format(u))
#     sph_int = SpharmInterface(u.isel(**{dim: 0}), v.isel(**{dim: 0}),
#                               rsphere=radius, make_spharmt=True, squeeze=True)
#     # Assume residual stems entirely from divergent flow.
#     resid_spectral = SpharmInterface.prep_for_spharm(residual)
#     resid_spectral = sph_int.spharmt.grdtospec(resid_spectral)
#     vort_spectral = np.zeros_like(resid_spectral)

#     u_adj, v_adj = sph_int.spharmt.getuv(vort_spectral, resid_spectral)
#     u_arr, v_arr = sph_int.to_xarray(u_adj), sph_int.to_xarray(v_adj)
#     return u_arr / col_integral, v_arr / col_integral


# def uv_mass_adjustment(ps, u, v, evap, precip, radius, dp, freq='1M'):
#     """Adjustment to horizontal winds to enforce column mass budget closure."""
#     residual = mass_column_budget_residual(ps, u, v, evap, precip, radius, dp,
#                                            freq=freq)
#     return uv_column_budget_adjustment(u, v, residual, ps, radius)


# def uv_mass_adjusted(ps, u, v, evap, precip, radius, dp, freq='1M'):
#     """Horizontal winds adjusted to impose column mass budget closure."""
#     u_adj, v_adj = uv_mass_adjustment(ps, u, v, evap, precip, radius, dp,
#                                       freq='1M')
#     return u - u_adj, v - v_adj


# def u_mass_adjustment(ps, u, v, evap, precip, radius, dp, freq='1M'):
#     """Adjustment to zonal wind to enforce column mass budget closure."""
#     u_adj, _ = uv_mass_adjustment(ps, u, v, evap, precip, radius, dp,
#                                   freq='1M')
#     return u_adj


# def v_mass_adjustment(ps, u, v, evap, precip, radius, dp, freq='1M'):
#     """Adjustment to meridional wind to enforce column mass budget closure."""
#     _, v_adj = uv_mass_adjustment(ps, u, v, evap, precip, radius, dp,
#                                   freq='1M')
#     return v_adj


# def u_mass_adjusted(ps, u, v, evap, precip, radius, dp, freq='1M'):
#     """Zonal wind adjusted to impose column mass budget closure."""
#     u_adj, _ = uv_mass_adjusted(ps, u, v, evap, precip, radius, dp, freq='1M')
#     return u_adj


# def v_mass_adjusted(ps, u, v, evap, precip, radius, dp, freq='1M'):
#     """Meridional wind adjusted to impose column mass budget closure."""
#     _, v_adj = uv_mass_adjusted(ps, u, v, evap, precip, radius, dp, freq='1M')
#     return v_adj


# def mass_column_divg_adj(ps, u, v, evap, precip, radius, dp, freq='1M'):
#     u_adj, v_adj = uv_mass_adjusted(ps, u, v, evap, precip, radius, dp,
#                                     freq=freq)
#     return mass_column_divg_spharm(u_adj, v_adj, radius, dp)


# def mass_column_budget_adj_residual(ps, u, v, evap, precip, radius, dp,
#                                     freq='1M'):
#     tendency = time_tendency_each_timestep(ps)
#     u_adj, v_adj = uv_mass_adjusted(ps, u, v, evap, precip, radius, dp,
#                                     freq=freq)
#     transport = mass_column_divg_spharm(u_adj, v_adj, radius, dp)
#     source = mass_column_source(evap, precip)
#     return tendency + transport - source


# def column_flux_divg(arr, u, v, radius, dp):
#     """Column flux divergence, with the field defined per unit mass of air."""
#     return horiz_divg_spharm(int_dp_g(arr*u, dp), int_dp_g(arr*v, dp), radius)


# def column_flux_divg_adj(arr, ps, u, v, evap, precip, radius, dp, freq='1M'):
#     """Column flux divergence, with the field defined per unit mass of air."""
#     u_adj, v_adj = uv_mass_adjusted(ps, u, v, evap, precip, radius, dp,
#                                     freq=freq)
#     return horiz_divg_spharm(int_dp_g(arr*u_adj, dp), int_dp_g(arr*v_adj, dp),
#                              radius)


# def horiz_divg_mass_adj(u, v, evap, precip, ps, radius, dp):
#     u_adj, v_adj = uv_mass_adjusted(ps, u, v, evap, precip, radius, dp)
#     return horiz_divg(u_adj, v_adj, radius)


# def horiz_divg_mass_adj_spharm(u, v, evap, precip, ps, radius, dp):
#     u_adj, v_adj = uv_mass_adjusted(ps, u, v, evap, precip, radius, dp)
#     return horiz_divg_spharm(u_adj, v_adj, radius)


# def horiz_divg_mass_adj_from_eta(u, v, evap, precip, ps, radius, dp, bk, pk):
#     """Mass-balance adjusted horizontal divergence from model coordinates."""
#     u_adj, v_adj = uv_mass_adjusted(ps, u, v, evap, precip, radius, dp)
#     divg_eta = horiz_divg_spharm(u_adj, v_adj, radius)
#     du_deta, dv_deta = d_deta_from_pfull(u_adj), d_deta_from_pfull(v_adj)
#     pfull_coord = u[PFULL_STR]
#     bk_at_pfull = to_pfull_from_phalf(bk, pfull_coord)
#     da_deta = d_deta_from_phalf(pk, pfull_coord)
#     db_deta = d_deta_from_phalf(bk, pfull_coord)
#     return (divg_eta - (bk_at_pfull / (da_deta + db_deta*ps)) *
#             horiz_advec_spharm(ps, du_deta, dv_deta, radius))


# def horiz_advec_mass_adj(arr, u, v, evap, precip, ps, radius, dp):
#     u_adj, v_adj = uv_mass_adjusted(ps, u, v, evap, precip, radius, dp)
#     return horiz_advec(arr, u_adj, v_adj, radius)


# def horiz_advec_mass_adj_spharm(arr, u, v, evap, precip, ps, radius, dp):
#     u_adj, v_adj = uv_mass_adjusted(ps, u, v, evap, precip, radius, dp)
#     return horiz_advec_spharm(arr, u_adj, v_adj, radius)


# def ps_horiz_advec(ps, u, v, evap, precip, radius, dp):
#     """Horizontal advection of surface pressure."""
#     u_adj, v_adj = uv_mass_adjusted(ps, u, v, evap, precip, radius, dp)
#     sfc_sel = {PFULL_STR: u_adj[PFULL_STR].max()}

#     def sel(arr):
#         """Grab the value at the level nearest the surface."""
#         return arr.sel(**sfc_sel).drop(PFULL_STR)

#     u_adj = sel(u_adj)
#     v_adj = sel(v_adj)
#     return horiz_advec_spharm(ps, u_adj, v_adj, radius)


# def column_dry_air_mass(ps, wvp):
#     """Total mass of dry air in an atmospheric column (from Trenberth 1991)"""
#     return ps / grav - wvp


# def dry_mass_column_tendency(ps, q, dp, freq='1M'):
#     """Combined time-tendency term in column mass budget equation.

#     See e.g. Trenberth 1991, Eq. 9.
#     """
#     return (time_tendency_first_to_last(ps, freq=freq) -
#             GRAV_EARTH * time_tendency_first_to_last(int_dp_g(q, dp),
#                                                      freq=freq))


# def dry_mass_column_divg(u, v, q, radius, dp):
#     """Transport term of atmospheric column mass budget.

#     E.g. Trenberth 1991, Eq. 9
#     """
#     u_int = integrate((1. - q)*u, dp, is_pressure=True)
#     v_int = integrate((1. - q)*v, dp, is_pressure=True)
#     return horiz_divg(u_int, v_int, radius)


# def dry_mass_column_budget_residual(ps, u, v, q, radius, dp, freq='1M'):
#     """Residual in the dry mass budget.

#     Theoretically the sum of the tendency and transport terms is exactly zero,
#     however artifacts introduced by numerics and other things yield a
#     residual.
#     """
#     tendency = dry_mass_column_tendency(ps, q, dp, freq=freq)
#     transport = dry_mass_column_divg(u, v, q, radius, dp)
#     return budget_residual(tendency, transport, freq=freq)


# def uv_dry_mass_adjustment(ps, u, v, q, radius, dp, freq='1M'):
#     """Adjustment to horiz. winds to enforce column dry mass budget closure."""
#     residual = dry_mass_column_budget_residual(ps, u, v, q, radius,
#                                                dp, freq=freq)
#     return uv_column_budget_adjustment(u, v, residual, ps, radius)


# def uv_dry_mass_adjusted(ps, u, v, q, radius, dp, freq='1M'):
#     """Horizontal winds adjusted to impose column dry mass budget closure."""
#     u_adj, v_adj = uv_dry_mass_adjustment(ps, u, v, q, radius, dp, freq=freq)
#     return u - u_adj, v - v_adj


# def dry_mass_column_divg_adj(ps, u, v, q, radius, dp, freq='1M'):
#     """Column divergence of dry mass with budget correction applied."""
#     u_adj, v_adj = uv_dry_mass_adjusted(ps, u, v, q, radius, dp, freq=freq)
#     return column_flux_divg(1 - q, u_adj, v_adj, radius, dp)


# def dry_mass_column_budget_adj_residual(ps, u, v, q, radius, dp, freq='1M'):
#     """Residual in column mass budget when flow is adjusted for balance."""
#     tendency = dry_mass_column_tendency(ps, q, dp, freq=freq)
#     transport = dry_mass_column_divg_adj(ps, u, v, q, radius, dp)
#     return budget_residual(tendency, transport, freq=freq)
