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


if __name__ == '__main__':
    pass
