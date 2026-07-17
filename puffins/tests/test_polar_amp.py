"""Tests for polar_amp module."""

import numpy as np
import pytest
import xarray as xr

from puffins.calculus import merid_avg_grid_data
from puffins.names import LAT_STR
from puffins.polar_amp import (
    antarctic_amp,
    arctic_amp,
    polar_amp_index,
    print_polar_amp,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_sinlat_grid(n: int = 100) -> np.ndarray:
    """Latitudes (degrees) uniformly spaced in sin(lat), spanning the poles.

    A field on this grid is non-uniform in latitude, so ``polar_amp_index``
    falls back to the sin(lat) averaging path, where each region average is
    a plain unweighted mean over the retained points. That makes the whole
    index reconstructable from raw numpy.
    """
    sinlats = np.linspace(-1, 1, n)
    return np.rad2deg(np.arcsin(sinlats))


def _sinlat_field(data: np.ndarray, lats: np.ndarray) -> xr.DataArray:
    return xr.DataArray(data, dims=[LAT_STR], coords={LAT_STR: lats}, name="field")


def _reconstruct_region_mean(
    data: np.ndarray, lats: np.ndarray, min_lat: float, max_lat: float
) -> float:
    """Plain mean over lats strictly between min_lat and max_lat.

    Mirrors ``merid_avg_sinlat_data``'s masking (strict inequalities, no
    weighting) so tests can rebuild the index without the module's helpers.
    """
    mask = (lats > min_lat) & (lats < max_lat)
    return float(data[mask].mean())


# ---------------------------------------------------------------------------
# TestPolarAmpIndex
# ---------------------------------------------------------------------------


class TestPolarAmpIndex:
    """Tests for polar_amp_index."""

    def test_constant_field_is_one(self) -> None:
        """A spatially constant field has polar amplification index 1."""
        lats = np.linspace(-89, 89, 180)
        arr = _sinlat_field(np.full(lats.size, 7.0), lats)
        result = polar_amp_index(arr)
        np.testing.assert_allclose(float(result), 1.0, rtol=1e-12)

    def test_known_value_sinlat_reconstruction(self) -> None:
        """Full index rebuilt from raw numpy on the sin(lat) path.

        Uses an asymmetric field and asymmetric polar bounds so the two
        span weights (abs(-90 - sh_bound) and abs(90 - nh_bound)) and the
        SH/NH region assignment are all pinned: mutating any of them in the
        source changes the expected value.
        """
        lats = _uniform_sinlat_grid(100)
        data = lats + 100.0  # monotonic, asymmetric about the equator
        arr = _sinlat_field(data, lats)

        sh_bound, nh_bound = -50.0, 70.0
        denom_bounds = (-90.0, 90.0)

        result = polar_amp_index(
            arr, sh_bound=sh_bound, nh_bound=nh_bound, denom_bounds=denom_bounds
        )

        denom = _reconstruct_region_mean(data, lats, *denom_bounds)
        sh_avg = _reconstruct_region_mean(data, lats, -90.0, sh_bound)
        nh_avg = _reconstruct_region_mean(data, lats, nh_bound, 90.0)
        sh_weight = abs(-90.0 - sh_bound)
        nh_weight = abs(90.0 - nh_bound)
        numer = (sh_weight * sh_avg + nh_weight * nh_avg) / (sh_weight + nh_weight)
        expected = numer / denom

        # Guard against a degenerate (symmetric) setup with no teeth.
        assert not np.isclose(sh_avg, nh_avg)
        assert sh_weight != nh_weight
        np.testing.assert_allclose(float(result), expected, rtol=1e-10)

    def test_grid_path_matches_calculus(self) -> None:
        """On a uniform-lat grid the area-weighted (grid) path is used.

        A uniform-latitude grid is non-uniform in sin(lat), so the sin(lat)
        path would raise; a finite result therefore proves the grid path
        ran. The index is checked against an independent assembly built from
        the (separately tested) grid averaging helper.
        """
        lats = np.linspace(-89, 89, 180)
        data = np.cos(np.deg2rad(lats)) + 0.5 * np.sin(np.deg2rad(lats))
        arr = _sinlat_field(data, lats)

        result = polar_amp_index(arr)  # defaults: sh_bound=-60, nh_bound=60

        denom = merid_avg_grid_data(arr, min_lat=-90, max_lat=90)
        sh_avg = merid_avg_grid_data(arr, max_lat=-60)
        nh_avg = merid_avg_grid_data(arr, min_lat=60)
        numer = (30.0 * sh_avg + 30.0 * nh_avg) / 60.0
        expected = float(numer / denom)

        np.testing.assert_allclose(float(result), expected, rtol=1e-10)

    def test_poleward_increasing_field_index_above_one(self) -> None:
        """A field peaked at the poles amplifies: index exceeds 1."""
        lats = np.linspace(-89, 89, 180)
        data = np.sin(np.deg2rad(lats)) ** 2  # 1 at poles, 0 at equator
        arr = _sinlat_field(data, lats)
        result = polar_amp_index(arr)
        assert float(result) > 1.0

    def test_raises_when_neither_hemisphere(self) -> None:
        """Excluding both hemispheres raises ValueError."""
        lats = np.linspace(-89, 89, 180)
        arr = _sinlat_field(np.ones(lats.size), lats)
        with pytest.raises(ValueError, match="One or both"):
            polar_amp_index(arr, include_sh=False, include_nh=False)

    def test_custom_lat_str(self) -> None:
        """Works with a non-default latitude dimension name."""
        lats = np.linspace(-89, 89, 180)
        arr = xr.DataArray(
            np.full(lats.size, 3.0), dims=["latitude"], coords={"latitude": lats}
        )
        result = polar_amp_index(arr, lat_str="latitude")
        np.testing.assert_allclose(float(result), 1.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# TestArcticAmp
# ---------------------------------------------------------------------------


class TestArcticAmp:
    """Tests for arctic_amp."""

    def test_known_value_sinlat_reconstruction(self) -> None:
        """Arctic index equals the NH-cap mean over the global mean."""
        lats = _uniform_sinlat_grid(100)
        data = lats + 100.0
        arr = _sinlat_field(data, lats)

        min_lat = 55.0
        result = arctic_amp(arr, min_lat=min_lat)

        denom = _reconstruct_region_mean(data, lats, -90.0, 90.0)
        nh_avg = _reconstruct_region_mean(data, lats, min_lat, 90.0)
        expected = nh_avg / denom

        np.testing.assert_allclose(float(result), expected, rtol=1e-10)

    def test_matches_polar_amp_index_nh_only(self) -> None:
        """arctic_amp is polar_amp_index restricted to the NH."""
        lats = _uniform_sinlat_grid(100)
        arr = _sinlat_field(lats + 100.0, lats)
        via_arctic = float(arctic_amp(arr, min_lat=60))
        via_index = float(polar_amp_index(arr, include_sh=False, nh_bound=60))
        np.testing.assert_allclose(via_arctic, via_index, rtol=1e-12)

    def test_min_lat_shifts_region(self) -> None:
        """A different min_lat selects a different NH cap, changing the value."""
        lats = _uniform_sinlat_grid(100)
        arr = _sinlat_field(lats + 100.0, lats)
        assert not np.isclose(
            float(arctic_amp(arr, min_lat=60)), float(arctic_amp(arr, min_lat=75))
        )


# ---------------------------------------------------------------------------
# TestAntarcticAmp
# ---------------------------------------------------------------------------


class TestAntarcticAmp:
    """Tests for antarctic_amp."""

    def test_known_value_sinlat_reconstruction(self) -> None:
        """Antarctic index equals the SH-cap mean over the global mean."""
        lats = _uniform_sinlat_grid(100)
        data = lats + 100.0
        arr = _sinlat_field(data, lats)

        max_lat = -55.0
        result = antarctic_amp(arr, max_lat=max_lat)

        denom = _reconstruct_region_mean(data, lats, -90.0, 90.0)
        sh_avg = _reconstruct_region_mean(data, lats, -90.0, max_lat)
        expected = sh_avg / denom

        np.testing.assert_allclose(float(result), expected, rtol=1e-10)

    def test_symmetric_field_equals_arctic(self) -> None:
        """For a hemispherically symmetric field the two caps match."""
        lats = _uniform_sinlat_grid(100)
        data = lats**2  # symmetric about the equator
        arr = _sinlat_field(data, lats)
        np.testing.assert_allclose(
            float(antarctic_amp(arr)), float(arctic_amp(arr)), rtol=1e-10
        )


# ---------------------------------------------------------------------------
# TestPrintPolarAmp
# ---------------------------------------------------------------------------


class TestPrintPolarAmp:
    """Tests for print_polar_amp."""

    def test_prints_expected_labels(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Prints the three indices, each labeled, to stdout."""
        lats = _uniform_sinlat_grid(100)
        arr = _sinlat_field(lats + 100.0, lats)
        print_polar_amp(arr)
        captured = capsys.readouterr().out
        assert "Polar amplification" in captured
        assert "Arctic" in captured
        assert "Antarctic" in captured
