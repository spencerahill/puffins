"""Internal type aliases for puffins."""

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import xarray as xr

Scalar: TypeAlias = float | int | np.floating | np.integer
ArrayLike: TypeAlias = xr.DataArray | np.ndarray | Scalar
XarrayObj: TypeAlias = xr.DataArray | xr.Dataset

# Argument types of `num_solver.brentq_solver_sweep_param`, shared so that the
# solver and the model functions wrapping it cannot drift apart. Unlike
# `ArrayLike` these admit plain sequences, since both are iterated rather than
# broadcast over.
SolverParamRange: TypeAlias = Scalar | Sequence[float] | np.ndarray | xr.DataArray
SolverGuessRange: TypeAlias = Sequence[float] | np.ndarray
