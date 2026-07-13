"""Tests for the eofs module."""

import numpy as np
import xarray as xr
from eofs.xarray import Eof

from puffins.eofs import eof_solver_lat
from puffins.names import LAT_STR, LON_STR, YEAR_STR


def _lat_lon_coords(nlat: int = 13, nlon: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Latitudes bounded away from the poles (where cos(lat) weights vanish)."""
    lats = np.linspace(-80.0, 80.0, nlat)
    lons = np.linspace(0.0, 360.0, nlon, endpoint=False)
    return lats, lons


def _rank_one_field() -> tuple[xr.DataArray, np.ndarray]:
    """A separable field ``amp(year) * pattern(lat, lon)`` and its amplitude.

    Being rank one, its leading EOF captures 100% of the variance and its
    leading PC is proportional to ``amp`` regardless of the area weighting.
    """
    lats, lons = _lat_lon_coords()
    years = np.arange(24.0)
    amp = np.sin(2.0 * np.pi * years / years.size)
    pattern = np.cos(np.deg2rad(lats))[:, None] * np.cos(np.deg2rad(lons))[None, :]
    data = amp[:, None, None] * pattern[None, :, :]
    arr = xr.DataArray(
        data,
        dims=[YEAR_STR, LAT_STR, LON_STR],
        coords={YEAR_STR: years, LAT_STR: lats, LON_STR: lons},
        name="field",
    )
    return arr, amp


class TestEofSolverLat:
    """Tests for eof_solver_lat."""

    def test_returns_eof_solver(self) -> None:
        """The wrapper returns an eofs solver object."""
        arr, _ = _rank_one_field()
        assert isinstance(eof_solver_lat(arr), Eof)

    def test_rank_one_variance_fraction(self) -> None:
        """A separable (rank-one) field puts all variance in the first mode."""
        arr, _ = _rank_one_field()
        var_frac = eof_solver_lat(arr).varianceFraction()
        np.testing.assert_allclose(var_frac.values[0], 1.0, atol=1e-10)

    def test_leading_pc_tracks_amplitude(self) -> None:
        """The leading PC is perfectly correlated with the input amplitude."""
        arr, amp = _rank_one_field()
        pc0 = eof_solver_lat(arr).pcs(npcs=1).isel(mode=0).values
        corr = np.corrcoef(pc0, amp)[0, 1]
        np.testing.assert_allclose(abs(corr), 1.0, atol=1e-10)

    def test_reconstructs_rank_one_field(self) -> None:
        """The leading mode alone reconstructs the original rank-one field."""
        arr, _ = _rank_one_field()
        recon = eof_solver_lat(arr).reconstructedField(1)
        np.testing.assert_allclose(recon.values, arr.values, atol=1e-10)

    def test_sqrt_coslat_weighting(self) -> None:
        """The area weighting is exactly ``sqrt(cos(lat))``.

        A rank-one field would give the same EOFs under any weighting, so this
        uses a full-rank random field, where the weighting genuinely changes the
        decomposition, and compares against a reference solver built with
        ``sqrt(cos(lat))`` weights computed from raw numpy (independently of the
        module's ``cosdeg`` helper). Dropping the square root, or the degree-to-
        radian conversion, changes the weights and fails this test.
        """
        lats, lons = _lat_lon_coords(nlat=8, nlon=6)
        rng = np.random.default_rng(0)
        data = rng.standard_normal((20, lats.size, lons.size))
        arr = xr.DataArray(
            data,
            dims=[YEAR_STR, LAT_STR, LON_STR],
            coords={YEAR_STR: np.arange(20.0), LAT_STR: lats, LON_STR: lons},
            name="field",
        )
        ref_weights = np.sqrt(np.cos(np.deg2rad(lats)))[:, np.newaxis]
        ref = Eof(arr.rename({YEAR_STR: "time"}), weights=ref_weights)
        np.testing.assert_allclose(
            eof_solver_lat(arr).eofs(neofs=3).values, ref.eofs(neofs=3).values
        )

    def test_custom_time_str(self) -> None:
        """A non-default sampling-dimension name is handled via time_str."""
        lats, lons = _lat_lon_coords()
        data = np.random.default_rng(1).standard_normal((10, lats.size, lons.size))
        arr = xr.DataArray(
            data,
            dims=["month", LAT_STR, LON_STR],
            coords={"month": np.arange(10.0), LAT_STR: lats, LON_STR: lons},
            name="field",
        )
        solver = eof_solver_lat(arr, time_str="month")
        assert isinstance(solver, Eof)
        assert solver.pcs(npcs=1).sizes["time"] == 10
