"""Tests for tropopause module."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from puffins.names import LAT_STR, LEV_STR
from puffins.tropopause import (
    _tropo_cold_point,
    _tropo_fixed_height,
    _tropo_fixed_temp,
    _tropo_max_vert_curv,
    _tropo_wmo,
    tropo_wmo,
    tropopause_cold_point,
    tropopause_fixed_height,
    tropopause_fixed_temp,
    tropopause_max_vert_curv,
    tropopause_wmo,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _standard_profile(
    p_hpa: np.ndarray,
    temp_surf: float = 288.15,
    lapse_rate: float = 6.5e-3,
    z_tropo: float = 11000.0,
    strato_lapse_rate: float = 1.0e-3,
    z_strato_start: float = 20000.0,
    scale_height: float = 8000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a crude US-standard-atmosphere-like (T, z) profile.

    Troposphere has constant lapse rate; an isothermal layer sits above
    the tropopause up to ``z_strato_start``; above that, a positive lapse
    rate is imposed.
    """
    z = scale_height * np.log(1000.0 / p_hpa)
    temp_tropo = temp_surf - lapse_rate * z_tropo
    temp = np.where(z < z_tropo, temp_surf - lapse_rate * z, temp_tropo)
    temp = np.where(
        z > z_strato_start, temp_tropo + strato_lapse_rate * (z - z_strato_start), temp
    )
    return temp, z


def _profile_dataarrays(
    p_hpa: np.ndarray,
    p_str: str = LEV_STR,
    **kwargs: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return (temp, height) DataArrays along a single pressure dimension."""
    temp_vals, z_vals = _standard_profile(p_hpa, **kwargs)
    temp = xr.DataArray(temp_vals, coords=[(p_str, p_hpa)], name="temp")
    height = xr.DataArray(z_vals, coords=[(p_str, p_hpa)], name="height")
    return temp, height


def _two_d_profiles(
    lats: np.ndarray,
    p_hpa: np.ndarray,
    p_str: str = LEV_STR,
    lat_str: str = LAT_STR,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Stack single-column idealized profiles across latitudes."""
    temps = np.zeros((lats.size, p_hpa.size))
    heights = np.zeros_like(temps)
    for i, lat in enumerate(lats):
        # Vary surface temperature with latitude for realism.
        t_surf = 288.0 - 20.0 * np.sin(np.deg2rad(lat)) ** 2
        t_col, z_col = _standard_profile(p_hpa, temp_surf=t_surf)
        temps[i] = t_col
        heights[i] = z_col
    temp = xr.DataArray(temps, coords=[(lat_str, lats), (p_str, p_hpa)], name="temp")
    height = xr.DataArray(
        heights, coords=[(lat_str, lats), (p_str, p_hpa)], name="height"
    )
    return temp, height


# ---------------------------------------------------------------------------
# TestTropoWmo
# ---------------------------------------------------------------------------


class TestTropoWmo:
    """Tests for tropo_wmo (the maintained WMO tropopause implementation)."""

    def test_idealized_profile(self) -> None:
        """Recovers the prescribed tropopause height on a standard profile."""
        p = np.arange(1000.0, 20.0, -10.0)
        temp, height = _profile_dataarrays(p, z_tropo=11000.0)
        result = tropo_wmo(temp, height)
        # The lapse-rate definition plus cubic interpolation will place the
        # tropopause close to (but not exactly at) the prescribed height.
        assert abs(float(result) - 11000.0) < 500.0

    def test_returns_dataarray(self) -> None:
        """Output is an xarray.DataArray with the expected name."""
        p = np.arange(1000.0, 20.0, -10.0)
        temp, height = _profile_dataarrays(p)
        result = tropo_wmo(temp, height)
        assert isinstance(result, xr.DataArray)
        assert result.name == "tropopause_wmo"

    def test_custom_p_str(self) -> None:
        """Works when the pressure dimension has a non-default name."""
        p = np.arange(1000.0, 20.0, -10.0)
        temp, height = _profile_dataarrays(p, p_str="mylev")
        result = tropo_wmo(temp, height, p_str="mylev")
        assert abs(float(result) - 11000.0) < 500.0

    def test_no_interp(self) -> None:
        """``do_interp=False`` returns a tropopause on the native grid."""
        # Restrict to pressures below ``max_pressure`` so the un-interpolated
        # code path has values to work with.
        p = np.arange(500.0, 20.0, -10.0)
        temp, height = _profile_dataarrays(p)
        result = tropo_wmo(temp, height, do_interp=False)
        assert isinstance(result, xr.DataArray)
        # Tropopause is somewhere in the stratosphere transition region.
        assert 8000.0 < float(result) < 20000.0

    def test_custom_interp_vals(self) -> None:
        """User-supplied ``interp_vals`` override the default grid."""
        p = np.arange(1000.0, 20.0, -10.0)
        temp, height = _profile_dataarrays(p)
        # Coarser interp grid than the default; tropopause should still lie
        # within a bounded window of the prescribed value.
        interp_vals = np.arange(500.0, 50.0, -1.0)
        result = tropo_wmo(temp, height, interp_vals=interp_vals)
        assert abs(float(result) - 11000.0) < 1000.0

    def test_fills_nans(self) -> None:
        """NaNs in the input are filled with zeros before interpolation."""
        p = np.arange(1000.0, 20.0, -10.0)
        temp, height = _profile_dataarrays(p)
        # Inject a NaN below the interpolation region; it should be tolerated.
        temp_nan = temp.copy()
        temp_nan.values[0] = np.nan
        result = tropo_wmo(temp_nan, height)
        assert np.isfinite(float(result))

    def test_max_pressure_gates_result(self) -> None:
        """Lowering ``max_pressure`` restricts the tropopause to lower pressure."""
        p = np.arange(1000.0, 20.0, -5.0)
        temp, height = _profile_dataarrays(p)
        strict = tropo_wmo(temp, height, max_pressure=200)
        default = tropo_wmo(temp, height)
        # Stricter cutoff forces the tropopause above where the default lies;
        # both should be finite and in the stratospheric range.
        assert float(strict) >= float(default) - 100.0
        assert np.isfinite(float(strict))

    def test_threshold_positive(self) -> None:
        """Very permissive threshold still returns a finite tropopause."""
        p = np.arange(1000.0, 20.0, -10.0)
        temp, height = _profile_dataarrays(p)
        result = tropo_wmo(temp, height, threshold=-1e-1)
        assert np.isfinite(float(result))


# ---------------------------------------------------------------------------
# Legacy tropopause implementations
# ---------------------------------------------------------------------------
#
# The SAH-2024-02-23 source comment notes that the six functions below have
# been broken since ~2020.  The smoke tests below confirm the failure modes
# are still present so that when the implementations are revisited, the
# expected-fail markers can be removed and the tests flipped to positive
# assertions.  Each is marked ``xfail(strict=True)`` so that an accidental
# fix is surfaced rather than silently masked.


_LEGACY_FAIL_REASON = (
    "Legacy tropopause implementation has been broken since ~2020; see "
    "SAH note in puffins/tropopause.py."
)


@pytest.fixture
def _two_d_data() -> tuple[xr.DataArray, xr.DataArray]:
    lats = np.linspace(-80.0, 80.0, 9)
    p = np.arange(1000.0, 20.0, -20.0)
    return _two_d_profiles(lats, p)


class TestLegacyTropoWmo:
    """Smoke tests for the broken ``_tropo_wmo`` / ``tropopause_wmo``."""

    @pytest.mark.xfail(strict=True, reason=_LEGACY_FAIL_REASON)
    def test_tropopause_wmo_smoke(
        self, _two_d_data: tuple[xr.DataArray, xr.DataArray]
    ) -> None:
        temp, height = _two_d_data
        result = tropopause_wmo(temp, height)
        assert isinstance(result, xr.DataArray)

    @pytest.mark.xfail(strict=True, reason=_LEGACY_FAIL_REASON)
    def test_private_tropo_wmo_smoke(
        self, _two_d_data: tuple[xr.DataArray, xr.DataArray]
    ) -> None:
        temp, height = _two_d_data
        result = _tropo_wmo(temp, height)
        assert isinstance(result, xr.DataArray)


class TestLegacyColdPoint:
    """Smoke tests for the broken cold-point tropopause implementation."""

    @pytest.mark.xfail(strict=True, reason=_LEGACY_FAIL_REASON)
    def test_tropopause_cold_point_smoke(
        self, _two_d_data: tuple[xr.DataArray, xr.DataArray]
    ) -> None:
        temp, _ = _two_d_data
        result = tropopause_cold_point(temp)
        assert isinstance(result, xr.DataArray)

    @pytest.mark.xfail(strict=True, reason=_LEGACY_FAIL_REASON)
    def test_private_cold_point_smoke(
        self, _two_d_data: tuple[xr.DataArray, xr.DataArray]
    ) -> None:
        temp, _ = _two_d_data
        result = _tropo_cold_point(temp)
        assert isinstance(result, xr.DataArray)


class TestLegacyMaxVertCurv:
    """Smoke tests for the broken max-curvature tropopause implementation."""

    @pytest.mark.xfail(strict=True, reason=_LEGACY_FAIL_REASON)
    def test_tropopause_max_vert_curv_smoke(
        self, _two_d_data: tuple[xr.DataArray, xr.DataArray]
    ) -> None:
        temp, height = _two_d_data
        result = tropopause_max_vert_curv(temp, height)
        assert isinstance(result, xr.DataArray)

    @pytest.mark.xfail(strict=True, reason=_LEGACY_FAIL_REASON)
    def test_private_max_vert_curv_smoke(
        self, _two_d_data: tuple[xr.DataArray, xr.DataArray]
    ) -> None:
        temp, height = _two_d_data
        result = _tropo_max_vert_curv(temp, height)
        assert isinstance(result, xr.DataArray)


class TestLegacyFixedTemp:
    """Smoke tests for the broken fixed-temperature tropopause implementation."""

    @pytest.mark.xfail(strict=True, reason=_LEGACY_FAIL_REASON)
    def test_tropopause_fixed_temp_smoke(
        self, _two_d_data: tuple[xr.DataArray, xr.DataArray]
    ) -> None:
        temp, _ = _two_d_data
        result = tropopause_fixed_temp(temp)
        assert isinstance(result, xr.DataArray)

    @pytest.mark.xfail(strict=True, reason=_LEGACY_FAIL_REASON)
    def test_private_fixed_temp_smoke(
        self, _two_d_data: tuple[xr.DataArray, xr.DataArray]
    ) -> None:
        temp, _ = _two_d_data
        result = _tropo_fixed_temp(temp)
        assert isinstance(result, xr.DataArray)


class TestLegacyFixedHeight:
    """Smoke tests for the broken fixed-height tropopause implementation."""

    @pytest.mark.xfail(strict=True, reason=_LEGACY_FAIL_REASON)
    def test_tropopause_fixed_height_smoke(
        self, _two_d_data: tuple[xr.DataArray, xr.DataArray]
    ) -> None:
        temp, height = _two_d_data
        result = tropopause_fixed_height(temp, height)
        assert isinstance(result, xr.DataArray)

    @pytest.mark.xfail(strict=True, reason=_LEGACY_FAIL_REASON)
    def test_private_fixed_height_smoke(
        self, _two_d_data: tuple[xr.DataArray, xr.DataArray]
    ) -> None:
        temp, height = _two_d_data
        result = _tropo_fixed_height(temp, height)
        assert isinstance(result, xr.DataArray)
