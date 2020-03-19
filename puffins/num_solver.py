#! /usr/bin/env python
"""Numerical solvers."""


import numpy as np
import scipy
import xarray as xr


from .calculus import z_deriv
from .constants import GRAV_EARTH, RAD_EARTH, ROT_RATE_EARTH
from .dynamics import coriolis_param
from .names import LAT_STR, LEV_STR
from .nb_utils import coord_arr_1d, cosdeg


def brentq_solver_sweep_param(func, param_range, init_guess, bound_guess_range,
                              funcargs=None):
    """Numerical solutions to a given function over a given parameter range.

    Uses the Brent (1973) root finding algorithm, as implemented
    in scipy's ``scipy.optimize.brentq`` function.

    Parameters
    ----------

    func : function
        Call signature must be `func(x, param_val, *funcargs)`, where `x` is
        the dependent variable, `param_val` is the parameter being swept over,
        and `funcargs` is a sequence (possibly empty) of all remaining
        arguments to `func`.
    param_range : sequence or scalar
        Range of values of the given parameter over which to sweep
    init_guess : scalar
        Initial guess for the solution corresponding to the first value of
        `param_range` (a.k.a. `func(guess, param_range[0], *funcargs`)
    bound_guess_range : sequence
        Range of guesses in which sign of solution should change sign.  The
        root-finder algorithm requires an interval in which the function
        changes sign.  We don't have such a range a priori, so find one by
        marching through guesses until we find two whose solutions have
        opposite sign.
    funcargs : tuple or None, default None
        Remaining arguments to `func`

    Returns
    -------

    solutions : xarray.DataArray
        Array of the numerical solution for each parameter value in
        ``param_range``

    """
    if funcargs is None:
        funcargs = tuple()
    old_guess = init_guess
    solutions = []

    if isinstance(param_range, (int, float)):
        param_range = [param_range]
    for val in param_range:
        args = (val,) + funcargs
        old_bound = func(old_guess, *args)
        for guess in bound_guess_range:
            bound = func(guess, *args)
            # Find two values between which the function changes sign.
            if np.sign(bound) - np.sign(old_bound) != 0:
                solution = scipy.optimize.brentq(func, old_guess,
                                                 guess, args=args)
                # Solution found, so move on to next value.
                break
            else:
                old_guess = guess
                old_bound = bound
        else:
            # No solution found, so mask it.
            solution = np.nan

        solutions.append(solution)
    try:
        dims = param_range.dims
        coords = param_range.coords
    except AttributeError:
        dims = None
        coords = None
    return xr.DataArray(solutions, dims=dims, coords=coords)


def sor_solver(A, b, initial_guess, omega=1.2, conv_crit=1e-6, verbose=True):
    """
    Successive over-relaxation numerical solver.

    Parameters
    ----------

    A : nxn matrix
        Matrix in the equation A*phi=b to be solved
    b : length-n vector
        Range of values of the given parameter over which to sweep
    inititial_guess : length-n vector
        Initial guess for the solution
    omega : scalar
        Value of `omega` parameter in the SOR solver.  If value is one,
        algorithm is identical to Gauss-Seidel
    conv_crit : float
        Maximum acceptable error in numerical solution; algorithm stops
        once this threshold is passed
    verbose : bool
        Whether or not to print the residual after each iteration

    Returns
    -------

    phi : length-n vector
        The numerical solution to the given equation A*phi=b

    Notes
    -----

    Adapted from
    https://en.wikipedia.org/wiki/Successive_over-relaxation#Example

    """
    phi = initial_guess.copy()[:]
    initial_residual = np.linalg.norm(np.matmul(A, phi) - b)
    residual = initial_residual
    if verbose:
        print('Initial residual: {0:10.6g}'.format(residual))
    while residual > conv_crit:
        for i in range(A.shape[0]):
            j_neq_i = np.ones(A.shape[0])
            j_neq_i[i] = 0
            sigma = np.sum(A[i]*phi*j_neq_i)
            phi[i] = (1 - omega) * phi[i] + (omega / A[i, i]) * (b[i] - sigma)
        residual = np.linalg.norm(np.matmul(A, phi) - b)
        if verbose:
            print('Residual: {0:10.6g}'.format(residual))
    return phi


def n_from_kj(k, j, num_y):
    """Convert 2D coordinates to scalar for converting to matrix."""
    return np.rint(k*num_y + j)


def kj_from_n(n, num_y):
    """Given scalar form of 2D coordinates, get the 2D pair."""
    return n // num_y, n % num_y


def _setup_bc_row(matrix, n, bc=0.):
    """Setup row of matrix corresponding to boundary condition."""
    bc_row = matrix[n]
    bc_row[:] = bc
    bc_row[n] = 1.
    return bc_row


# Kuo-Eliassen equation.
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

    dln_pot_temp_dp = z_deriv(np.log(pot_temp), plevs,
                              z_str=lev_str).mean(lat_str)

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
            A_arr[n] = _setup_bc_row(A_arr, n)

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
                  conv_crit=1e-14, grav=GRAV_EARTH, radius=RAD_EARTH,
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
