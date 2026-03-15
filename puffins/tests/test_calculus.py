"""Tests for puffins.calculus module."""

import numpy as np
import pytest
import xarray as xr

from puffins.calculus import _bounds_from_array


class TestBoundsFromArray:
    """Tests for _bounds_from_array."""

    def test_1d_uniform_spacing(self):
        """Basic 1D case with uniform spacing."""
        arr = xr.DataArray([1.0, 2.0, 3.0], dims=["x"], coords={"x": [1.0, 2.0, 3.0]})
        bounds = _bounds_from_array(arr, "x")
        # Spacing is 1.0 everywhere, so bounds should be center +/- 0.5
        lower = bounds.isel(bounds=0)
        upper = bounds.isel(bounds=1)
        np.testing.assert_allclose(lower.values, [0.5, 1.5, 2.5])
        np.testing.assert_allclose(upper.values, [1.5, 2.5, 3.5])

    def test_1d_nonuniform_spacing(self):
        """1D case with non-uniform spacing."""
        arr = xr.DataArray([0.0, 1.0, 3.0], dims=["x"], coords={"x": [0.0, 1.0, 3.0]})
        bounds = _bounds_from_array(arr, "x")
        lower = bounds.isel(bounds=0)
        upper = bounds.isel(bounds=1)
        # spacing = [1.0, 2.0]; last element reuses spacing[-1]=2.0
        np.testing.assert_allclose(lower.values, [-0.5, 0.0, 2.0])
        np.testing.assert_allclose(upper.values, [0.5, 2.0, 4.0])

    def test_2d_dim_not_axis0(self):
        """Ensure it works when target dim is not axis 0.

        This is the bug the TODO was about: the old implementation
        used raw numpy [:-1]/[-1] indexing which assumes axis=0.
        """
        # Create a 2D array where the target dim 'x' is axis=1
        data = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        arr = xr.DataArray(
            data,
            dims=["y", "x"],
            coords={"y": [0, 1], "x": [10.0, 20.0, 30.0]},
        )
        bounds = _bounds_from_array(arr, "x")
        lower = bounds.isel(bounds=0)
        upper = bounds.isel(bounds=1)
        # .T transposes, so result dims are (x, y) with shape (3, 2)
        np.testing.assert_allclose(
            lower.values,
            [[5.0, 35.0], [15.0, 45.0], [25.0, 55.0]],
        )
        np.testing.assert_allclose(
            upper.values,
            [[15.0, 45.0], [25.0, 55.0], [35.0, 65.0]],
        )

    def test_2d_dim_is_axis0(self):
        """When target dim is axis 0, should also work."""
        data = np.array([[10.0, 40.0], [20.0, 50.0], [30.0, 60.0]])
        arr = xr.DataArray(
            data,
            dims=["x", "y"],
            coords={"x": [10.0, 20.0, 30.0], "y": [0, 1]},
        )
        bounds = _bounds_from_array(arr, "x")
        lower = bounds.isel(bounds=0)
        upper = bounds.isel(bounds=1)
        # .T transposes, so result dims are (y, x) with shape (2, 3)
        np.testing.assert_allclose(
            lower.values,
            [[5.0, 15.0, 25.0], [35.0, 45.0, 55.0]],
        )
        np.testing.assert_allclose(
            upper.values,
            [[15.0, 25.0, 35.0], [45.0, 55.0, 65.0]],
        )
