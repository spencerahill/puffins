"""Tests for kuo_el module."""
import numpy as np
import pytest
import xarray as xr

from puffins.kuo_el import _check_uniform_spacing


class TestCheckUniformSpacing:
    """Tests for _check_uniform_spacing."""

    def test_uniform_spacing_passes(self) -> None:
        """Uniform spacing should not raise."""
        coord = xr.DataArray(
            np.linspace(0, 10, 11), dims=["x"]
        )
        _check_uniform_spacing(coord, "x", "test")

    def test_nonuniform_spacing_raises(self) -> None:
        """Non-uniform spacing should raise ValueError."""
        coord = xr.DataArray(
            np.array([0.0, 1.0, 2.0, 5.0, 10.0]), dims=["x"]
        )
        with pytest.raises(ValueError, match="Uniform test spacing required"):
            _check_uniform_spacing(coord, "x", "test")

    def test_nearly_uniform_within_tolerance(self) -> None:
        """Spacing within tolerance should not raise."""
        vals = np.linspace(0, 10, 11)
        # Add small perturbation within default tol=0.01
        vals[5] += 0.005
        coord = xr.DataArray(vals, dims=["x"])
        _check_uniform_spacing(coord, "x", "test")

    def test_custom_tolerance(self) -> None:
        """Custom tol should be respected."""
        coord = xr.DataArray(
            np.array([0.0, 1.0, 2.0, 2.5]), dims=["x"]
        )
        # Should raise with tight tolerance
        with pytest.raises(ValueError):
            _check_uniform_spacing(coord, "x", "test", tol=0.01)
        # Should pass with loose tolerance
        _check_uniform_spacing(coord, "x", "test", tol=1.0)

    def test_error_message_contains_max_deviation(self) -> None:
        """Error message should report actual max fractional deviation."""
        coord = xr.DataArray(
            np.array([0.0, 1.0, 2.0, 5.0]), dims=["x"]
        )
        with pytest.raises(ValueError, match="Actual max fractional deviation"):
            _check_uniform_spacing(coord, "x", "test")
