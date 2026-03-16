"""Tests for budget_adj module."""

import numpy as np
import pytest
import xarray as xr

spharm = pytest.importorskip("spharm")
pytest.importorskip("windspharm")

from windspharm.xarray import VectorWind  # noqa: E402

from puffins.budget_adj import resid_after_col_adj, uv_col_budg_adj  # noqa: E402


def _make_gaussian_grid(
    nlat: int = 32, nlon: int = 64, ntime: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create Gaussian lat/lon/time coordinates for testing."""
    lat, _ = spharm.gaussian_lats_wts(nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    time = np.arange(ntime, dtype=float)
    return lat, lon, time


def _make_field(
    lat: np.ndarray,
    lon: np.ndarray,
    time: np.ndarray,
    values: np.ndarray | float,
) -> xr.DataArray:
    """Create a DataArray on the Gaussian grid."""
    nlat, nlon, ntime = len(lat), len(lon), len(time)
    if isinstance(values, (int, float)):
        data = np.full((nlat, nlon, ntime), values)
    else:
        data = values
    return xr.DataArray(
        data,
        dims=["lat", "lon", "time"],
        coords={"lat": lat, "lon": lon, "time": time},
    )


class TestUvColBudgAdjReturnTypes:
    """Verify returned types and dimensions."""

    def test_returns_dataarrays(self) -> None:
        lat, lon, time = _make_gaussian_grid()
        u = _make_field(lat, lon, time, 1.0)
        v = _make_field(lat, lon, time, 0.5)
        tendency = _make_field(lat, lon, time, 0.0)
        source = _make_field(lat, lon, time, 0.0)

        u_adj, v_adj = uv_col_budg_adj(u, v, tendency, source)

        assert isinstance(u_adj, xr.DataArray)
        assert isinstance(v_adj, xr.DataArray)

    def test_preserves_dims(self) -> None:
        lat, lon, time = _make_gaussian_grid()
        u = _make_field(lat, lon, time, 1.0)
        v = _make_field(lat, lon, time, 0.5)
        tendency = _make_field(lat, lon, time, 0.0)
        source = _make_field(lat, lon, time, 0.0)

        u_adj, v_adj = uv_col_budg_adj(u, v, tendency, source)

        assert u_adj.dims == u.dims
        assert v_adj.dims == v.dims
        np.testing.assert_array_equal(u_adj.lat.values, u.lat.values)
        np.testing.assert_array_equal(u_adj.lon.values, u.lon.values)


class TestResidualNearZeroAfterAdjustment:
    """The whole point: after adjustment, the budget should close."""

    def test_residual_near_zero(self) -> None:
        lat, lon, time = _make_gaussian_grid()
        nlat, nlon, ntime = len(lat), len(lon), len(time)
        # Create non-trivial wind fields (spherical harmonic pattern).
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        u_vals = (
            np.cos(lat_rad)[:, None, None]
            * np.sin(lon_rad)[None, :, None]
            * np.ones(ntime)
        )
        v_vals = (
            np.sin(lat_rad)[:, None, None]
            * np.cos(lon_rad)[None, :, None]
            * np.ones(ntime)
        )

        u = _make_field(lat, lon, time, u_vals)
        v = _make_field(lat, lon, time, v_vals)
        # Use zero tendency and source so the residual is just div(u,v),
        # which has zero global mean (required for correction on a sphere).
        tendency = _make_field(lat, lon, time, 0.0)
        source = _make_field(lat, lon, time, 0.0)

        u_adj, v_adj = uv_col_budg_adj(u, v, tendency, source)
        resid = resid_after_col_adj(u_adj, v_adj, tendency, source)

        np.testing.assert_allclose(resid.values, 0.0, atol=1e-10)


class TestAdjustmentIsPurelyDivergent:
    """The adjustment should not change the rotational component."""

    def test_vorticity_of_adjustment_near_zero(self) -> None:
        lat, lon, time = _make_gaussian_grid()
        nlat, nlon, ntime = len(lat), len(lon), len(time)
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        u_vals = np.cos(lat_rad)[:, None, None] * np.ones((1, nlon, ntime))
        v_vals = np.zeros((nlat, nlon, ntime))

        u = _make_field(lat, lon, time, u_vals)
        v = _make_field(lat, lon, time, v_vals)
        tendency = _make_field(lat, lon, time, 1e-5)
        source = _make_field(lat, lon, time, 0.0)

        u_adj, v_adj = uv_col_budg_adj(u, v, tendency, source)
        du = u_adj - u
        dv = v_adj - v

        # Compute vorticity of the difference (adjustment) for each time.
        for t in range(len(time)):
            du_t = du.isel(time=t)
            dv_t = dv.isel(time=t)
            w = VectorWind(du_t, dv_t)
            vort = w.vorticity()
            np.testing.assert_allclose(vort.values, 0.0, atol=1e-10)


class TestNoAdjustmentWhenBudgetClosed:
    """When the budget is already closed, adjustment should be negligible."""

    def test_no_adjustment_needed(self) -> None:
        lat, lon, time = _make_gaussian_grid()
        nlat, nlon, ntime = len(lat), len(lon), len(time)
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        u_vals = (
            np.cos(lat_rad)[:, None, None]
            * np.sin(lon_rad)[None, :, None]
            * np.ones(ntime)
        )
        v_vals = (
            np.sin(lat_rad)[:, None, None]
            * np.cos(lon_rad)[None, :, None]
            * np.ones(ntime)
        )
        u = _make_field(lat, lon, time, u_vals)
        v = _make_field(lat, lon, time, v_vals)

        # Compute div(u, v) so that tendency + div - source = 0.
        vecwind = VectorWind(u.isel(time=0), v.isel(time=0))
        div_2d = vecwind.divergence()
        div_vals = div_2d.values[:, :, None] * np.ones(ntime)
        # Set source = tendency + div, so residual = 0.
        tendency = _make_field(lat, lon, time, 1e-5)
        source_vals = tendency.values + div_vals
        source = _make_field(lat, lon, time, source_vals)

        u_adj, v_adj = uv_col_budg_adj(u, v, tendency, source)

        np.testing.assert_allclose(u_adj.values, u.values, atol=1e-10)
        np.testing.assert_allclose(v_adj.values, v.values, atol=1e-10)
