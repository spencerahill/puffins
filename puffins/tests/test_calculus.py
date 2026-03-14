"""Tests for calculus module."""
import numpy as np
import pytest
import xarray as xr

from puffins.calculus import infer_bounds
from puffins.names import BOUNDS_STR, LAT_STR


class TestInferBounds:
    """Tests for the infer_bounds function."""

    def test_uniform_spacing(self) -> None:
        """Uniform spacing should succeed and return correct bounds."""
        vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        arr = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        result = infer_bounds(arr, "x")

        assert result.dims == ("x", BOUNDS_STR)
        assert result.shape == (5, 2)
        # For uniform spacing of 1.0, bounds should be [-0.5, 0.5], [0.5, 1.5], ...
        np.testing.assert_allclose(result.values[0], [-0.5, 0.5])
        np.testing.assert_allclose(result.values[2], [1.5, 2.5])
        np.testing.assert_allclose(result.values[4], [3.5, 4.5])

    def test_nonuniform_spacing_raises(self) -> None:
        """Non-uniform spacing should raise ValueError."""
        vals = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
        arr = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        with pytest.raises(ValueError, match="Uniform spacing required"):
            infer_bounds(arr, "x")

    def test_nearly_uniform_within_tolerance(self) -> None:
        """Nearly uniform spacing within tolerance should succeed."""
        vals = np.array([0.0, 1.0, 2.001, 3.001, 4.001])
        arr = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        # Default tol=0.01 should accept this
        result = infer_bounds(arr, "x")
        assert result.shape == (5, 2)

    def test_custom_tolerance(self) -> None:
        """Custom tolerance should be respected."""
        vals = np.array([0.0, 1.0, 2.1, 3.1, 4.1])
        arr = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        # Should fail with tight tolerance
        with pytest.raises(ValueError, match="Uniform spacing required"):
            infer_bounds(arr, "x", spacing_tol=0.01)
        # Should pass with loose tolerance
        result = infer_bounds(arr, "x", spacing_tol=0.15)
        assert result.shape == (5, 2)

    def test_identical_values_raises(self) -> None:
        """All-identical values should raise ValueError."""
        vals = np.array([5.0, 5.0, 5.0])
        arr = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        with pytest.raises(ValueError, match="all identical"):
            infer_bounds(arr, "x")

    def test_custom_dim_bounds_name(self) -> None:
        """Custom dim_bounds name should be used."""
        vals = np.linspace(0, 10, 5)
        arr = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        result = infer_bounds(arr, "x", dim_bounds="my_bounds")
        assert result.name == "my_bounds"

    def test_latitude_like_values(self) -> None:
        """Typical latitude-like values should work."""
        lats = np.arange(-90, 91, 2.0)
        arr = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
        result = infer_bounds(arr, LAT_STR)
        assert result.shape == (len(lats), 2)
        # First lower bound should be -91, last upper bound should be 91
        np.testing.assert_allclose(result.values[0, 0], -91.0)
        np.testing.assert_allclose(result.values[-1, 1], 91.0)

    def test_wrong_arr_type_raises(self) -> None:
        """Passing a non-DataArray for arr should raise."""
        with pytest.raises(AttributeError):
            infer_bounds(np.array([1.0, 2.0, 3.0]), "x")

    def test_wrong_dim_type_raises(self) -> None:
        """Passing a non-string for dim should raise."""
        vals = np.array([0.0, 1.0, 2.0])
        arr = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        with pytest.raises(TypeError):
            infer_bounds(arr, 123)

    def test_wrong_spacing_tol_type_raises(self) -> None:
        """Passing a non-numeric spacing_tol should raise."""
        vals = np.array([0.0, 1.0, 2.0])
        arr = xr.DataArray(vals, dims=["x"], coords={"x": vals})
        with pytest.raises(TypeError):
            infer_bounds(arr, "x", spacing_tol="strict")
