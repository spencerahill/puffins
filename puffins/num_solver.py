#! /usr/bin/env python
"""Numerical solvers."""


import numpy as np
import scipy
import xarray as xr


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


def setup_bc_row(matrix, n, bc=0.):
    """Setup row of matrix corresponding to boundary condition."""
    bc_row = matrix[n]
    bc_row[:] = bc
    bc_row[n] = 1.
    return bc_row


if __name__ == '__main__':
    pass
