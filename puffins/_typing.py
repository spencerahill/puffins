"""Internal type aliases for puffins."""

from typing import TypeAlias

import numpy as np
import xarray as xr

Scalar: TypeAlias = float | int | np.floating | np.integer
ArrayLike: TypeAlias = xr.DataArray | np.ndarray | Scalar
XarrayObj: TypeAlias = xr.DataArray | xr.Dataset
