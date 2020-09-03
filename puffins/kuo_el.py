"""Kuo-Eliassen equation solver and related functionality."""
import numpy as np
import xarray as xr


from .calculus import lat_deriv
from .constants import C_P, GRAV_EARTH, P0, R_D, RAD_EARTH, ROT_RATE_EARTH
from .dynamics import coriolis_param
from .names import LAT_STR, LEV_STR, SIGMA_STR
from .num_solver import kj_from_n, setup_bc_row, sor_solver
from .nb_utils import coord_arr_1d, cosdeg, sindeg


def kuo_el_eddy_mom_term(u_merid_flux_eddy, p_sfc=None, is_sigma=True,
                         radius=RAD_EARTH, rot_rate=ROT_RATE_EARTH,
                         lat_str=LAT_STR, lev_str=LEV_STR):
    """Eddy momentum term on the RHS of the Kuo-Eliassen equation.

    Note that the `u_merid_flux_eddy` should include a cos(lat)
    factor, i.e. it should be u'v'cos(lat).

    """
    lat = u_merid_flux_eddy[lat_str]
    coslat = cosdeg(lat)
    uv_eddy_cos2lat = u_merid_flux_eddy*coslat
    duv_eddy_cos2lat_dlat = lat_deriv(uv_eddy_cos2lat, lat_str)
    d2uv_eddy_cos2lat_dp_dlat = duv_eddy_cos2lat_dlat.differentiate(lev_str)
    if is_sigma:
        d2uv_eddy_cos2lat_dp_dlat /= p_sfc
    return (2*rot_rate*sindeg(lat) * d2uv_eddy_cos2lat_dp_dlat /
            (radius * coslat**2)).transpose(*u_merid_flux_eddy.dims)


def kuo_el_eddy_temp_term(temp, temp_eddy_merid_flux, pressure, r_d=R_D,
                          radius=RAD_EARTH, lat_str=LAT_STR):
    """Eddy heat flux term on the RHS of the Kuo-Eliassen equation.

    Note that `temp_merid_flux_eddy` should include a cos(lat)
    factor, i.e. it should be v'T'cos(lat).

    """
    if isinstance(pressure, str):
        pressure = temp[pressure]
    dtemp_eddy_merid_flux_dlat = lat_deriv(temp_eddy_merid_flux, lat_str)
    coslat = cosdeg(temp[lat_str])
    d2temp_eddy_flux_dlat2_term = lat_deriv(dtemp_eddy_merid_flux_dlat /
                                            coslat, lat_str)
    return -r_d / (pressure*radius**2)*d2temp_eddy_flux_dlat2_term


def kuo_el_fric_term(zonal_friction, pressure, rot_rate=ROT_RATE_EARTH,
                     vert_str=SIGMA_STR, lat_str=LAT_STR):
    """Zonal friction term on the RHS of the Kuo-Eliassen equation."""
    zonal_friction["p_for_deriv"] = pressure
    d_zonal_fric_dp = zonal_friction.differentiate("p_for_deriv")
    return -d_zonal_fric_dp*coriolis_param(zonal_friction[lat_str],
                                           rot_rate=rot_rate)


def kuo_el_diab_term(diab_heat, pressure, r_d=R_D, c_p=C_P,
                     radius=RAD_EARTH, p0=P0, lat_str=LAT_STR):
    """Diabatic heating term on the RHS of the Kuo-Eliassen equation.

    diab_heat : (p_0/p)^\kappa*\bar Q / c_p

    """
    kappa = r_d / c_p
    return (r_d*pressure**(kappa-1)*lat_deriv(diab_heat, lat_str) /
            (radius*p0**kappa))


def _kuo_el_matrix(pot_temp, spec_vol, grav=GRAV_EARTH, radius=RAD_EARTH,
                   rot_rate=ROT_RATE_EARTH, lat_str=LAT_STR, lev_str=LEV_STR):
    """Generate the matrix used in the Kuo-Eliassen equation solver.

    Warning: assumes that pressure and latitude spacing are both uniform but
    doesn't warn or raise an error if they aren't.

    """
    lats = pot_temp[lat_str]
    plevs = pot_temp[lev_str]
    coslat = cosdeg(lats)
    f = coriolis_param(lats, rot_rate=rot_rate)

    # TODO: insert check that lat and pressure spacings are uniform.
    dp = float(plevs.diff(lev_str).mean())
    dlat = float(np.deg2rad(lats.diff(lat_str).mean()))

    num_k = len(plevs)
    num_j = len(lats)
    num_points = num_k*num_j
    chi = coord_arr_1d(0, num_points-1, 1, "chi", dtype=np.int)

    dln_pot_temp_dp = np.log(pot_temp).differentiate(lev_str).mean(lat_str)

    nu = spec_vol.mean(LAT_STR) * dln_pot_temp_dp / (radius*dlat)**2
    xi = grav / (2*np.pi*radius)
    alpha = (f / dp)**2 / coslat

    al_vals = alpha.values
    nu_vals = nu.values

    coslathalf_north = cosdeg(lats+0.5*np.rad2deg(dlat))
    coslathalf_south = cosdeg(lats-0.5*np.rad2deg(dlat))
    inv_coshalf_north = 1. / coslathalf_north
    inv_coshalf_south = 1. / coslathalf_south

    # Define the matrix A.
    A_arr = xr.DataArray(
        np.zeros((num_points, num_points)),
        dims=["chi1", "chi2"],
        coords={"chi1": chi.values, "chi2": chi.values},
        name="A",
    )

    # Populate its interior.
    for n in range(num_j, num_points-num_j):
        row = A_arr[n]
        k, j = kj_from_n(n, num_j)

        # For gridoint (k, j).
        row[n] = nu_vals[k] * (inv_coshalf_south[j] +
                               inv_coshalf_north[j]) - 2*al_vals[j]
        # For gridpoint (k-1, j).
        row[n-num_j] = al_vals[j]
        # For gridpoint (k+1, j).
        row[n+num_j] = al_vals[j]
        # For gridpoint (k, j-1).
        row[n-1] = -nu_vals[k] * inv_coshalf_south[j]
        # For gridpoint (k, j+1).
        row[n+1] = -nu_vals[k] * inv_coshalf_north[j]

    # Apply the leading factor "xi".
    A_arr *= xi

    # Apply boundary conditions.
    bc_index_ranges = (
        range(0, num_j, 1),  # k=0
        range(num_points-num_j, num_points, 1),  # k=K-1
        range(0, num_points-num_j+1, num_j),  # j=0
        range(num_j - 1, num_points, num_j),  # j=J-1
    )
    for bc_ind_range in bc_index_ranges:
        for n in bc_ind_range:
            A_arr[n] = setup_bc_row(A_arr, n)

    return A_arr


def _kuo_el_apply_bc(arr, lat_str=LAT_STR, lev_str=LEV_STR, bc=0.):
    """Apply Kuo-Eliassen boundary conditions."""
    lats = arr[lat_str]
    plevs = arr[lev_str]
    # Warning: assumes both pressure and latitude values are monotonically
    # increasing.
    p_interior = (plevs > plevs[0]) & (plevs < plevs[-1])
    lat_interior = (lats > lats[0]) & (lats < lats[-1])
    return arr.where(p_interior).where(lat_interior).fillna(bc)


def kuo_el_solver(pot_temp, spec_vol, forcing, init_guess=None, omega=1.2,
                  conv_crit=1e-13, grav=GRAV_EARTH, radius=RAD_EARTH,
                  rot_rate=ROT_RATE_EARTH, lat_str=LAT_STR, lev_str=LEV_STR,
                  verbose=True):
    """Numerical solver of Kuo-Eliassen equation.

    Note that numerics assume data on uniformly spaced pressure levels and
    uniformly spaced latitudes.  Answer will be incorrect (but no warning or
    error will be raised) if these conditions aren't met.

    """
    kuo_el_matrix = _kuo_el_matrix(
        pot_temp,
        spec_vol,
        grav=grav,
        radius=radius,
        rot_rate=rot_rate,
        lat_str=lat_str,
        lev_str=lev_str,
    )

    if init_guess is None:
        init_guess = np.zeros_like(pot_temp)
    elif isinstance(init_guess, xr.DataArray):
        init_guess = init_guess.values

    ke_sol_vals = sor_solver(
        kuo_el_matrix.values,
        _kuo_el_apply_bc(forcing, lat_str=lat_str,
                         lev_str=lev_str).values.flatten(),
        init_guess.flatten(),
        omega=omega,
        conv_crit=conv_crit,
        verbose=verbose,
    )
    return -1*xr.ones_like(pot_temp)*ke_sol_vals.reshape(pot_temp.shape)


if __name__ == '__main__':
    pass
