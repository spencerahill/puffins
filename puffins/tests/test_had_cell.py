"""Tests for had_cell module."""

import inspect
import warnings
from collections.abc import Callable
from decimal import Decimal, localcontext
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr
from scipy.optimize import brentq

from puffins.constants import (
    DELTA_V,
    GRAV_EARTH,
    HEIGHT_TROPO,
    RAD_EARTH,
    ROT_RATE_EARTH,
)
from puffins.dynamics import u_bci_2layer_qg
from puffins.grad_bal import u_lin_ro_small_ang, u_unif_ro, u_unif_ro_small_ang
from puffins.had_cell import (
    cell_edges_sigma,
    fixed_ro_bci_edge,
    fixed_ro_bci_edge_small_angle,
    fixed_ro_bci_edge_small_angle_lata0,
    fixed_ro_bci_edge_supercrit_ascent,
    had_cell_edge,
    had_cell_strength,
    had_cells_edges,
    had_cells_north_edge,
    had_cells_shared_edge,
    had_cells_south_edge,
    had_cells_strength,
    lat_ascent_eta0_approx,
    lin_ro_bci_edge_small_angle_lata0,
    merid_streamfunc,
)
from puffins.names import LAT_STR, LEV_STR, LON_STR, SIGMA_STR, TIME_STR

EDGE_KWARGS: dict[str, Any] = dict(
    min_lat=-50,
    max_lat=50,
    min_plev=500,
    max_plev=800,
    do_avg_vert=True,
    frac_thresh=0.05,
)


def _make_streamfunc(edge_lat: float = 30.0) -> xr.DataArray:
    """Idealized two-cell streamfunction: NH cell positive, SH negative.

    Cells span [-edge_lat, edge_lat] with extrema at +/- edge_lat / 2 and
    zeros at 0 and +/- edge_lat; zero poleward of the cells.
    """
    lats = np.arange(-89.5, 90, 1.0)
    plevs = np.arange(100.0, 1001.0, 50.0)
    lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
    plev = xr.DataArray(plevs, dims=[LEV_STR], coords={LEV_STR: plevs})
    merid = xr.where(np.abs(lat) <= edge_lat, np.sin(np.pi * lat / edge_lat), 0.0)
    vert = np.sin(np.pi * (plev - 100.0) / 900.0)
    return cast(xr.DataArray, (1e11 * vert * merid).rename("streamfunc"))


def _two_cell_streamfunc(amp: float = 1e11) -> xr.DataArray:
    """Two-cell streamfunction on an integer-degree grid.

    Extrema of exactly -amp at (lat=-15, plev=550) and +amp at
    (lat=15, plev=550); zero poleward of +/- 30.
    """
    lats = np.arange(-85.0, 86.0, 1.0)
    plevs = np.arange(100.0, 1001.0, 50.0)
    lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
    plev = xr.DataArray(plevs, dims=[LEV_STR], coords={LEV_STR: plevs})
    merid = xr.where(np.abs(lat) <= 30.0, np.sin(np.pi * lat / 30.0), 0.0)
    vert = np.sin(np.pi * (plev - 100.0) / 900.0)
    return cast(xr.DataArray, (amp * vert * merid).rename("streamfunc"))


def _tilted_streamfunc() -> xr.DataArray:
    """Two-cell streamfunction whose zero crossings tilt with pressure.

    At plev=550 the inner (shared) zero crossing is at lat=0; at each
    other level it shifts by (plev - 550) / 50 degrees.
    """
    lats = np.arange(-89.5, 90, 1.0)
    plevs = np.arange(100.0, 1001.0, 50.0)
    lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
    plev = xr.DataArray(plevs, dims=[LEV_STR], coords={LEV_STR: plevs})
    shift = (plev - 550.0) / 50.0
    merid = xr.where(
        np.abs(lat - shift) <= 30.0, np.sin(np.pi * (lat - shift) / 30.0), 0.0
    )
    vert = np.sin(np.pi * (plev - 100.0) / 900.0)
    return cast(xr.DataArray, (1e11 * vert * merid).rename("streamfunc"))


def _bump(lat: xr.DataArray, center: float, halfwidth: float) -> xr.DataArray:
    """cos**2 bump in latitude: 1 at `center`, 0 beyond `halfwidth` from it."""
    x = (lat - center) / halfwidth
    return cast(
        xr.DataArray, xr.where(np.abs(x) < 1, np.cos(0.5 * np.pi * x) ** 2, 0.0)
    )


def _bump_cells_streamfunc(cells: list[tuple[float, float, float]]) -> xr.DataArray:
    """Sum of cos**2 cells on the grid of `_two_cell_streamfunc`.

    Each cell is (signed peak, center latitude, half-width in latitude). The
    vertical structure is that of `_two_cell_streamfunc`, peaking at
    plev=550, so each cell's extremum is its signed peak at (center, 550)
    wherever no other cell overlaps it.
    """
    lats = np.arange(-85.0, 86.0, 1.0)
    plevs = np.arange(100.0, 1001.0, 50.0)
    lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
    plev = xr.DataArray(plevs, dims=[LEV_STR], coords={LEV_STR: plevs})
    merid = xr.zeros_like(lat)
    for peak, center, halfwidth in cells:
        merid = merid + peak * _bump(lat, center, halfwidth)
    vert = cast(xr.DataArray, np.sin(np.pi * (plev - 100.0) / 900.0))
    return (vert * merid).rename("streamfunc")


# A southern-winter field: a strong southern Hadley cell centered at 5S, a
# weak northern (summer) Hadley cell at 25N, and a southern Ferrel cell at
# 45S stronger than the northern Hadley cell, so the global maximum is the
# Ferrel cell. The field also has polar and northern Ferrel cells, so that
# the southern polar and Ferrel cells are also a pair of the Hadley cells'
# senses (negative south of positive).
_SH_WINTER_CELLS = [
    (-2e11, -5.0, 25.0),  # southern Hadley cell
    (0.3e11, 25.0, 10.0),  # northern Hadley cell
    (0.6e11, -45.0, 12.0),  # southern Ferrel cell
    (-0.2e11, -72.0, 9.0),  # southern polar cell
    (-0.1e11, 50.0, 12.0),  # northern Ferrel cell
]
# Its mirror image, a northern-winter field: each cell moved to the mirrored
# latitude, with its sign reversed, as the sense of circulation reverses.
_NH_WINTER_CELLS = [(-peak, -center, width) for peak, center, width in _SH_WINTER_CELLS]


def _issue_112_streamfunc(i_north: int = 25) -> xr.DataArray:
    """The reproducer of issue #112.

    A dominant positive cell centered at 2.57S, plus weak negative cells at
    43.71S and at latitude index `i_north`, the southern one stronger. The
    grid has EMARS's 180/35-degree spacing, built with np.arange. At the
    default index the northern cell is at 43.71N, and master's identification
    fails there only because the mirrored latitudes differ in magnitude in
    their last bits. At index 24 (38.57N) it fails whatever the rounding.
    The vertical structure peaks at plev=4.
    """
    lats = np.arange(-90 + 180 / 35, 90 - 90 / 35, 180 / 35)
    plevs = np.linspace(0.5, 7.5, 15)
    lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
    plev = xr.DataArray(plevs, dims=[LEV_STR], coords={LEV_STR: plevs})
    merid = (
        16.6e9 * _bump(lat, lats[16], 40.0)
        - 0.566e9 * _bump(lat, lats[8], 10.0)
        - 0.131e9 * _bump(lat, lats[i_north], 10.0)
    )
    vert = cast(xr.DataArray, np.sin(np.pi * plev / 8.0))
    return (vert * merid).rename("streamfunc")


# The grid of issue #117's reproducers: the odd-degree latitudes, so an even
# latitude lies midway between two gridpoints.
_GAUSS_LATS = np.arange(-89.0, 90.0, 2.0)
_GAUSS_PLEVS = np.array([100.0, 200.0, 300.0, 500.0, 700.0, 850.0, 1000.0])
# The peak of the vertical structure sin(pi * (plev - 100) / 900) on that grid,
# at plev=500.
_GAUSS_VERT_PEAK = np.sin(4.0 * np.pi / 9.0)


def _gauss_cell(center: float, amp: float = 1e11, tilt: float = 0.0) -> xr.DataArray:
    """One Gaussian cell of e-folding half-width 15 degrees, from issue #117.

    The cell's center at each level is `center` plus `tilt` times
    (plev - 500). So at plev=500, where the vertical structure peaks, the
    latitude profile is symmetric about `center`, and at every other level
    it is offset from it.
    """
    lat, plev = np.meshgrid(_GAUSS_LATS, _GAUSS_PLEVS)
    merid = np.exp(-(((lat - center - tilt * (plev - 500.0)) / 15.0) ** 2))
    return xr.DataArray(
        amp * np.sin(np.pi * (plev - 100.0) / 900.0) * merid,
        dims=(LEV_STR, LAT_STR),
        coords={LEV_STR: _GAUSS_PLEVS, LAT_STR: _GAUSS_LATS},
        name="streamfunc",
    )


class TestMeridStreamfunc:
    """merid_streamfunc reconstructed from raw numpy."""

    def _v_dp(self) -> tuple[xr.DataArray, xr.DataArray, np.ndarray, np.ndarray]:
        lats = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
        plevs = np.array([200e2, 500e2, 800e2])
        v_vals = np.arange(15, dtype=float).reshape(5, 3) - 5.0
        dp_vals = np.array([250e2, 300e2, 350e2])
        v = xr.DataArray(
            v_vals,
            dims=[LAT_STR, LEV_STR],
            coords={LAT_STR: lats, LEV_STR: plevs},
            name="v",
        )
        dp = xr.DataArray(dp_vals, dims=[LEV_STR], coords={LEV_STR: plevs}, name="dp")
        return v, dp, v_vals, dp_vals

    def test_known_value(self) -> None:
        grav, radius = 10.0, 2e6
        v, dp, v_vals, dp_vals = self._v_dp()
        result = merid_streamfunc(
            v, dp, grav=grav, radius=radius, impose_zero_col_flux=False
        )
        lats = v[LAT_STR].values
        expected = (
            2.0
            * np.pi
            * radius
            * np.cos(np.deg2rad(lats))[:, np.newaxis]
            * np.cumsum(v_vals * dp_vals[np.newaxis, :], axis=1)
            / grav
        )
        np.testing.assert_allclose(result.transpose(LAT_STR, LEV_STR).values, expected)
        assert result.name == "streamfunc"

    def test_level_order_invariance(self) -> None:
        v, dp, _, _ = self._v_dp()
        result_inc = merid_streamfunc(v, dp, impose_zero_col_flux=False)
        v_dec = v.isel({LEV_STR: slice(None, None, -1)})
        dp_dec = dp.isel({LEV_STR: slice(None, None, -1)})
        result_dec = merid_streamfunc(v_dec, dp_dec, impose_zero_col_flux=False)
        np.testing.assert_allclose(
            result_inc.sortby(LEV_STR).transpose(LAT_STR, LEV_STR).values,
            result_dec.sortby(LEV_STR).transpose(LAT_STR, LEV_STR).values,
        )

    def test_nonmonotonic_levels_raise(self) -> None:
        v, dp, _, _ = self._v_dp()
        shuffled = [1, 0, 2]
        v_bad = v.isel({LEV_STR: shuffled})
        dp_bad = dp.isel({LEV_STR: shuffled})
        with pytest.raises(ValueError, match="not monotonic"):
            merid_streamfunc(v_bad, dp_bad)

    def test_impose_zero_col_flux(self) -> None:
        # With zero net column mass flux imposed, the streamfunction must
        # return to ~zero at the bottom of the column at every latitude.
        v, dp, _, _ = self._v_dp()
        result = merid_streamfunc(v, dp, impose_zero_col_flux=True)
        result_free = merid_streamfunc(v, dp, impose_zero_col_flux=False)
        scale = float(np.abs(result_free).max())
        surface = result.isel({LEV_STR: -1})
        assert float(np.abs(surface).max()) < 1e-10 * scale
        # And the free version must not satisfy this, else no teeth.
        assert float(np.abs(result_free.isel({LEV_STR: -1})).max()) > 1e-3 * scale

    def test_zonal_mean_matches_preaveraged(self) -> None:
        v, dp, _, _ = self._v_dp()
        v_lon = xr.concat([v, v + 2.0], dim=xr.DataArray([0.0, 180.0], dims=[LON_STR]))
        result_lon = merid_streamfunc(v_lon, dp, impose_zero_col_flux=False)
        result_mean = merid_streamfunc(
            v_lon.mean(LON_STR), dp, impose_zero_col_flux=False
        )
        assert bool((result_lon == result_mean).all())

    def test_nan_masked(self) -> None:
        v, dp, _, _ = self._v_dp()
        v_nan = v.copy(deep=True)
        v_nan.loc[{LAT_STR: 0.0, LEV_STR: 500e2}] = np.nan
        result = merid_streamfunc(v_nan, dp, impose_zero_col_flux=False)
        assert np.isnan(result.sel({LAT_STR: 0.0, LEV_STR: 500e2}))
        # Other latitudes are unaffected.
        clean = merid_streamfunc(v, dp, impose_zero_col_flux=False)
        np.testing.assert_allclose(
            result.sel({LAT_STR: 30.0}).values, clean.sel({LAT_STR: 30.0}).values
        )

    def test_custom_dim_names(self) -> None:
        v, dp, _, _ = self._v_dp()
        default = merid_streamfunc(v, dp)
        renames = {LAT_STR: "latitude", LEV_STR: "pressure"}
        result = merid_streamfunc(
            v.rename(renames),
            dp.rename({LEV_STR: "pressure"}),
            lat_str="latitude",
            lon_str="longitude",
            lev_str="pressure",
        )
        np.testing.assert_allclose(result.values, default.values)
        assert set(result.dims) == {"latitude", "pressure"}


class TestHadCellStrength:
    """had_cell_strength extremum values and locations."""

    def test_global_max(self) -> None:
        sf = _two_cell_streamfunc()
        strength = had_cell_strength(sf)
        assert strength.item() == pytest.approx(1e11)
        assert strength[LAT_STR].item() == 15.0
        assert strength[LEV_STR].item() == 550.0

    def test_min_plev(self) -> None:
        sf = _two_cell_streamfunc()
        sf.loc[{LAT_STR: 40.0, LEV_STR: 100.0}] = 5e11
        assert had_cell_strength(sf).item() == pytest.approx(5e11)
        result = had_cell_strength(sf, min_plev=300.0)
        assert result.item() == pytest.approx(1e11)
        assert result[LAT_STR].item() == 15.0

    def test_max_plev(self) -> None:
        sf = _two_cell_streamfunc()
        sf.loc[{LAT_STR: 50.0, LEV_STR: 1000.0}] = 7e11
        assert had_cell_strength(sf).item() == pytest.approx(7e11)
        result = had_cell_strength(sf, max_plev=900.0)
        assert result.item() == pytest.approx(1e11)
        assert result[LAT_STR].item() == 15.0

    def test_do_avg_vert(self) -> None:
        lats = np.array([0.0, 10.0])
        plevs = np.array([200.0, 400.0])
        sf = xr.DataArray(
            [[10.0, -2.0], [5.0, 5.0]],
            dims=[LAT_STR, LEV_STR],
            coords={LAT_STR: lats, LEV_STR: plevs},
        )
        pointwise = had_cell_strength(sf)
        assert pointwise.item() == pytest.approx(10.0)
        assert pointwise[LAT_STR].item() == 0.0
        vert_avg = had_cell_strength(sf, do_avg_vert=True)
        assert vert_avg.item() == pytest.approx(5.0)
        assert vert_avg[LAT_STR].item() == 10.0
        assert vert_avg[LEV_STR].item() == pytest.approx(300.0)

    def test_each_time(self) -> None:
        # The maximum is taken separately at each time, not over all times.
        lats = np.array([0.0, 10.0])
        plevs = np.array([200.0, 400.0])
        times = np.array([0, 1])
        sf = xr.DataArray(
            [[[8.0, 1.0], [2.0, 3.0]], [[0.0, 1.0], [2.0, 3.0]]],
            dims=[TIME_STR, LAT_STR, LEV_STR],
            coords={TIME_STR: times, LAT_STR: lats, LEV_STR: plevs},
        )
        result = had_cell_strength(sf)
        assert result.dims == (TIME_STR,)
        assert result.sel({TIME_STR: 0}).item() == pytest.approx(8.0)
        assert result.sel({TIME_STR: 1}).item() == pytest.approx(3.0)
        assert result[LAT_STR].sel({TIME_STR: 0}).item() == 0.0
        assert result[LAT_STR].sel({TIME_STR: 1}).item() == 10.0

    def test_do_interp_lat_with_do_avg_vert(self) -> None:
        # https://github.com/spencerahill/puffins/issues/117: the two options
        # together raised. The level-averaged cell is symmetric about 10N,
        # midway between the gridpoints 9N and 11N, so its interpolated
        # center is 10N, and its strength is the level mean of the vertical
        # structure times the amplitude.
        sf = _gauss_cell(10.0)
        expected = 1e11 * np.mean(np.sin(np.pi * (_GAUSS_PLEVS - 100.0) / 900.0))
        gridpoint = had_cell_strength(sf, do_avg_vert=True)
        assert gridpoint[LAT_STR].item() == 9.0
        result = had_cell_strength(sf, do_avg_vert=True, do_interp_lat=True)
        assert result[LAT_STR].item() == pytest.approx(10.0, abs=1e-9)
        assert result.item() == pytest.approx(expected, rel=1e-4)
        assert result[LEV_STR].item() == pytest.approx(np.mean(_GAUSS_PLEVS))

    @pytest.mark.parametrize(
        ("lev_wide", "lev_narrow"), [(400.0, 600.0), (600.0, 400.0)]
    )
    def test_do_interp_lat_strength_is_max_over_levels(
        self, lev_wide: float, lev_narrow: float
    ) -> None:
        # A wide cell of peak 1e11 on one level and a narrow cell of peak
        # 1.02e11 on the other, both centered at 10N, midway between
        # gridpoints. The gridpoint maximum lies on the wide cell, since the
        # narrow one is lower at 9N and 11N. At the interpolated center, 10N,
        # the narrow cell is higher, so the strength, the maximum over the
        # levels there, is the narrow cell's, close to its peak.
        sf = xr.DataArray(
            np.stack(
                [
                    1e11 * np.exp(-(((_GAUSS_LATS - 10.0) / 15.0) ** 2)),
                    1.02e11 * np.exp(-(((_GAUSS_LATS - 10.0) / 5.0) ** 2)),
                ]
            ),
            dims=(LEV_STR, LAT_STR),
            coords={LEV_STR: [lev_wide, lev_narrow], LAT_STR: _GAUSS_LATS},
        ).sortby(LEV_STR)
        assert had_cell_strength(sf)[LEV_STR].item() == lev_wide
        result = had_cell_strength(sf, do_interp_lat=True)
        assert result[LAT_STR].item() == pytest.approx(10.0, abs=1e-9)
        assert result[LEV_STR].item() == lev_narrow
        assert result.item() == pytest.approx(1.02e11, rel=2e-3)

    def test_do_interp_lat_extra_dim_of_length_one(self) -> None:
        # https://github.com/spencerahill/puffins/issues/117: with
        # do_interp_lat, a dimension besides latitude and level raised, even
        # one of length 1. The cell is symmetric about 10N at plev=500, where
        # its maximum is.
        sf = _gauss_cell(10.0, tilt=0.004).expand_dims({TIME_STR: [0]})
        result = had_cell_strength(sf, do_interp_lat=True)
        assert result.dims == (TIME_STR,)
        assert result[LAT_STR].values.ravel().tolist() == pytest.approx(
            [10.0], abs=1e-9
        )
        assert result.values.ravel().tolist() == pytest.approx(
            [1e11 * _GAUSS_VERT_PEAK], rel=1e-4
        )
        assert result[LEV_STR].values.ravel().tolist() == [500.0]

    @pytest.mark.parametrize(
        "order",
        [(TIME_STR, LEV_STR, LAT_STR), (LAT_STR, TIME_STR, LEV_STR)],
        ids=["lat_last", "lat_first"],
    )
    def test_do_interp_lat_each_time(self, order: tuple[str, str, str]) -> None:
        # Two times whose cells differ in center, tilt and amplitude. Each
        # time's center is interpolated along the line of latitudes through
        # its own gridpoint maximum, at plev=500, where the cell is
        # symmetric about its center; along any other level's line the
        # tilt would move the center. The dimension order must not matter.
        sf = (
            xr.concat(
                [_gauss_cell(10.0, 1e11, 0.004), _gauss_cell(-6.0, 1.5e11, -0.004)],
                dim=TIME_STR,
            )
            .assign_coords({TIME_STR: [0, 1]})
            .transpose(*order)
        )
        result = had_cell_strength(sf, do_interp_lat=True)
        assert result.dims == (TIME_STR,)
        assert result[TIME_STR].values.tolist() == [0, 1]
        assert result[LAT_STR].values.tolist() == pytest.approx([10.0, -6.0], abs=1e-9)
        assert result.values.tolist() == pytest.approx(
            [1e11 * _GAUSS_VERT_PEAK, 1.5e11 * _GAUSS_VERT_PEAK], rel=1e-4
        )

    def test_do_interp_lat_raises_if_one_time_fails(self) -> None:
        # At the second time the cell is centered beyond the northernmost
        # latitude, 89N, so the maximum is there, and the latitude derivative
        # has no zero crossing to interpolate. That raises for the whole call.
        sf = xr.concat([_gauss_cell(10.0), _gauss_cell(95.0)], dim=TIME_STR)
        sf = sf.assign_coords({TIME_STR: [0, 1]})
        assert had_cell_strength(sf)[LAT_STR].values.tolist() == [9.0, 89.0]
        with pytest.raises(ValueError, match="zero crossings"):
            had_cell_strength(sf, do_interp_lat=True)

    @pytest.mark.parametrize("descending", [False, True], ids=["s_to_n", "n_to_s"])
    @pytest.mark.parametrize(
        ("center", "lat_gridpoint"),
        [(9.5, 9.0), (10.5, 11.0)],
        ids=["north_of_gridpoint", "south_of_gridpoint"],
    )
    def test_do_interp_lat_center_off_midpoint(
        self, center: float, lat_gridpoint: float, descending: bool
    ) -> None:
        # A cell centered a quarter of the grid spacing from a gridpoint,
        # north of its gridpoint maximum or south of it, so either half of the
        # search window is needed. The center is not midway between
        # gridpoints, so it is recovered only to the accuracy of the method,
        # measured at 3.3e-3 degrees on this grid. Latitude stored north to
        # south, as in ERA5, gives the same center.
        sf = _gauss_cell(center)
        if descending:
            sf = sf.isel({LAT_STR: slice(None, None, -1)})
        assert had_cell_strength(sf)[LAT_STR].item() == lat_gridpoint
        result = had_cell_strength(sf, do_interp_lat=True)
        assert result[LAT_STR].item() == pytest.approx(center, abs=5e-3)
        assert result.item() == pytest.approx(1e11 * _GAUSS_VERT_PEAK, rel=1e-4)

    def test_do_interp_lat_ignores_crossings_beyond_one_spacing(self) -> None:
        # The gridpoint maximum is at the northernmost latitude, 89N, where
        # the latitude derivative does not change sign. A narrow second cell
        # at 83N gives the derivative zero crossings between 83N and 85N and
        # between 85N and 87N. The search window around 89N holds only the
        # gridpoints 87N and 89N, so no crossing is found, and the call
        # raises.
        merid = np.exp(-(((_GAUSS_LATS - 95.0) / 15.0) ** 2)) + 0.3 * np.exp(
            -(((_GAUSS_LATS - 83.0) / 3.0) ** 2)
        )
        sf = xr.DataArray(
            1e11 * np.outer(np.sin(np.pi * (_GAUSS_PLEVS - 100.0) / 900.0), merid),
            dims=(LEV_STR, LAT_STR),
            coords={LEV_STR: _GAUSS_PLEVS, LAT_STR: _GAUSS_LATS},
        )
        assert had_cell_strength(sf)[LAT_STR].item() == 89.0
        with pytest.raises(ValueError, match="zero crossings"):
            had_cell_strength(sf, do_interp_lat=True)

    @pytest.mark.parametrize("extra_dim", [False, True], ids=["lat_lev", "with_time"])
    def test_do_interp_lat_custom_lat_name(self, extra_dim: bool) -> None:
        sf = _gauss_cell(10.0).rename({LAT_STR: "latitude"})
        if extra_dim:
            sf = sf.expand_dims({TIME_STR: [0]})
        result = had_cell_strength(sf, do_interp_lat=True, lat_str="latitude")
        assert result["latitude"].values.ravel().tolist() == pytest.approx(
            [10.0], abs=1e-9
        )
        assert result.values.ravel().tolist() == pytest.approx(
            [1e11 * _GAUSS_VERT_PEAK], rel=1e-4
        )

    def test_do_interp_lat_nan_anywhere_raises(self) -> None:
        # One NaN, at 79S and 700 hPa, far from the maximum at 10N and
        # 500 hPa, makes the cubic interpolation NaN at every level, so the
        # call raises, and with a time dimension it raises for both times
        # though only one holds the NaN. Averaging over the levels skips the
        # NaN, and excluding its level with max_plev or min_plev removes it;
        # each strength is then that of the field without it.
        sf = _gauss_cell(10.0).copy(deep=True)
        sf.loc[{LAT_STR: -79.0, LEV_STR: 700.0}] = np.nan
        with pytest.raises(ValueError, match="All-NaN"):
            had_cell_strength(sf, do_interp_lat=True)
        with pytest.raises(ValueError, match="All-NaN"):
            had_cell_strength(
                xr.concat([sf, _gauss_cell(10.0)], dim=TIME_STR), do_interp_lat=True
            )
        vert = np.sin(np.pi * (_GAUSS_PLEVS - 100.0) / 900.0)
        cases: list[tuple[dict[str, Any], float, float]] = [
            (
                {"do_avg_vert": True},
                float(np.mean(_GAUSS_PLEVS)),
                float(1e11 * np.mean(vert)),
            ),
            ({"max_plev": 600.0}, 500.0, float(1e11 * vert[3])),
            ({"min_plev": 800.0}, 850.0, float(1e11 * vert[5])),
        ]
        for kwargs, lev, strength in cases:
            result = had_cell_strength(sf, do_interp_lat=True, **kwargs)
            assert result[LAT_STR].item() == pytest.approx(10.0, abs=1e-9)
            assert result[LEV_STR].item() == pytest.approx(lev)
            assert result.item() == pytest.approx(strength, rel=1e-4)

    def test_do_interp_lat_follows_the_maximums_level(self) -> None:
        # The gridpoint maximum, 1e11, is at 9N on the 500 hPa cell, so the
        # center is found along 500 hPa, where the latitude derivative is zero
        # at 9N itself. A cell at 300 hPa, 0.99e11 at 11.4N, makes the upper
        # envelope of the levels rise from 9N to 11N, so a center found on
        # that envelope would lie north of 9N.
        lat = _GAUSS_LATS
        sf = xr.DataArray(
            np.stack(
                [
                    0.99e11 * np.exp(-(((lat - 11.4) / 4.0) ** 2)),
                    1e11 * np.exp(-(((lat - 9.0) / 4.0) ** 2)),
                ]
            ),
            dims=(LEV_STR, LAT_STR),
            coords={LEV_STR: [300.0, 500.0], LAT_STR: lat},
        )
        result = had_cell_strength(sf, do_interp_lat=True)
        assert result[LAT_STR].item() == pytest.approx(9.0, abs=1e-9)
        assert result[LEV_STR].item() == 500.0
        assert result.item() == pytest.approx(1e11, rel=1e-12)


class TestHadCellsStrength:
    """Both-cell finder: locations, signed magnitudes, cell selection."""

    def test_two_cells_known_values(self) -> None:
        sf = _two_cell_streamfunc()
        result = had_cells_strength(sf)
        assert list(result["cell"].values) == ["had_cell_sh", "had_cell_nh"]
        sh = result.sel(cell="had_cell_sh")
        nh = result.sel(cell="had_cell_nh")
        assert sh.item() == pytest.approx(-1e11)
        assert nh.item() == pytest.approx(1e11)
        assert sh[LAT_STR].item() == -15.0
        assert nh[LAT_STR].item() == 15.0
        assert sh[LEV_STR].item() == 550.0
        assert nh[LEV_STR].item() == 550.0

    def test_stronger_ferrel_cell_not_selected(self) -> None:
        # A "Ferrel" cell stronger than either Hadley cell must lose to
        # the two cells nearest the equator.
        sf = _two_cell_streamfunc()
        lat = sf[LAT_STR]
        plev = sf[LEV_STR]
        merid_ferrel = xr.where(
            (lat >= 30.0) & (lat <= 60.0),
            np.sin(np.pi * (lat - 30.0) / 30.0),
            0.0,
        )
        vert = np.sin(np.pi * (plev - 100.0) / 900.0)
        sf_with_ferrel = sf - 2e11 * vert * merid_ferrel
        result = had_cells_strength(sf_with_ferrel)
        assert result.sel(cell="had_cell_sh")[LAT_STR].item() == -15.0
        assert result.sel(cell="had_cell_nh")[LAT_STR].item() == 15.0
        assert result.sel(cell="had_cell_sh").item() == pytest.approx(-1e11)
        assert result.sel(cell="had_cell_nh").item() == pytest.approx(1e11)

    def test_min_max_lat(self) -> None:
        # Asymmetric bounds; test_symmetric_min_max_lat covers symmetric ones.
        sf = _two_cell_streamfunc()
        result = had_cells_strength(sf, min_lat=-12.0, max_lat=10.0)
        sh = result.sel(cell="had_cell_sh")
        nh = result.sel(cell="had_cell_nh")
        assert sh[LAT_STR].item() == -11.0
        assert nh[LAT_STR].item() == 9.0
        assert sh.item() == pytest.approx(-1e11 * np.sin(np.pi * 11.0 / 30.0))
        assert nh.item() == pytest.approx(1e11 * np.sin(np.pi * 9.0 / 30.0))

    def test_custom_dim_names(self) -> None:
        sf = _two_cell_streamfunc().rename({LAT_STR: "latitude", LEV_STR: "pressure"})
        result = had_cells_strength(sf, lat_str="latitude", lev_str="pressure")
        assert result.sel(cell="had_cell_sh")["latitude"].item() == -15.0
        assert result.sel(cell="had_cell_nh")["latitude"].item() == 15.0
        assert result.sel(cell="had_cell_nh")["pressure"].item() == 550.0

    def test_symmetric_min_max_lat(self) -> None:
        # Symmetric bounds leave exactly two candidates, at mirrored
        # latitudes, which raised before:
        # https://github.com/spencerahill/puffins/issues/55
        sf = _two_cell_streamfunc()
        result = had_cells_strength(sf, min_lat=-10.0, max_lat=10.0)
        sh = result.sel(cell="had_cell_sh")
        nh = result.sel(cell="had_cell_nh")
        assert sh[LAT_STR].item() == -9.0
        assert nh[LAT_STR].item() == 9.0
        assert sh.item() == pytest.approx(-1e11 * np.sin(np.pi * 9.0 / 30.0))
        assert nh.item() == pytest.approx(1e11 * np.sin(np.pi * 9.0 / 30.0))

    @pytest.mark.parametrize(
        ("cells", "lat_sh", "lat_nh"),
        [(_SH_WINTER_CELLS, -5.0, 25.0), (_NH_WINTER_CELLS, -25.0, 5.0)],
        ids=["sh_winter", "nh_winter"],
    )
    def test_winter_ferrel_stronger_than_summer_hadley(
        self, cells: list[tuple[float, float, float]], lat_sh: float, lat_nh: float
    ) -> None:
        # The winter Ferrel cell is a global extremum, and it and the polar
        # cell beside it are a pair of the Hadley cells' senses. The Hadley
        # cells are the pair containing the cell nearest the equator.
        sf = _bump_cells_streamfunc(cells)
        peaks = {center: peak for peak, center, _ in cells}
        result = had_cells_strength(sf)
        sh = result.sel(cell="had_cell_sh")
        nh = result.sel(cell="had_cell_nh")
        assert sh[LAT_STR].item() == lat_sh
        assert nh[LAT_STR].item() == lat_nh
        assert sh.item() == pytest.approx(peaks[lat_sh])
        assert nh.item() == pytest.approx(peaks[lat_nh])
        assert sh[LEV_STR].item() == 550.0
        assert nh[LEV_STR].item() == 550.0

    @pytest.mark.parametrize("i_north", [25, 24], ids=["mirrored", "nearer"])
    def test_cross_equatorial_cell_beside_weak_cells(self, i_north: int) -> None:
        # https://github.com/spencerahill/puffins/issues/112: the dominant
        # positive cell is the northern Hadley cell though its center lies
        # south of the equator. The southern Hadley cell is the negative
        # cell south of it, not the weaker negative cell in the north, at
        # the mirrored latitude or nearer the equator, which is a northern
        # Ferrel cell.
        sf = _issue_112_streamfunc(i_north)
        lats = sf[LAT_STR].values
        result = had_cells_strength(sf)
        sh = result.sel(cell="had_cell_sh")
        nh = result.sel(cell="had_cell_nh")
        assert nh[LAT_STR].item() == lats[16]
        assert sh[LAT_STR].item() == lats[8]
        assert nh.item() == pytest.approx(16.6e9)
        assert sh.item() == pytest.approx(-0.566e9)
        assert nh[LEV_STR].item() == 4.0
        assert sh[LEV_STR].item() == 4.0
        # The north edge is the dominant cell's own: the first crossing of
        # 0.1 of its peak north of its center, at the level of its peak,
        # interpolated linearly between gridpoints.
        prof = sf.sel({LEV_STR: 4.0}).values / 16.6e9 - 0.1
        i = 16 + int(np.argmax(prof[16:] < 0))
        expected = np.interp(0.0, [prof[i], prof[i - 1]], [lats[i], lats[i - 1]])
        north = _edge_arr(had_cells_north_edge(sf))
        assert north.item() == pytest.approx(expected)
        assert north.item() < lats[i_north]

    def test_negated_issue_112_field(self) -> None:
        # Negating the field reverses every cell's sense of circulation. The
        # dominant cell is then the southern Hadley cell, the northern
        # Hadley cell is the weak positive cell north of it, and the
        # stronger weak positive cell, south of it, is a southern Ferrel
        # cell.
        sf = -_issue_112_streamfunc()
        lats = sf[LAT_STR].values
        result = had_cells_strength(sf)
        sh = result.sel(cell="had_cell_sh")
        nh = result.sel(cell="had_cell_nh")
        assert sh[LAT_STR].item() == lats[16]
        assert nh[LAT_STR].item() == lats[25]
        assert sh.item() == pytest.approx(-16.6e9)
        assert nh.item() == pytest.approx(0.131e9)

    def test_one_cell_warns(self) -> None:
        # A lone positive cell is the northern Hadley cell. With no southern
        # cell, the southern entry is not negative, and a warning is issued.
        sf = _bump_cells_streamfunc([(1e11, 10.0, 30.0)])
        with pytest.warns(UserWarning, match="sense of circulation"):
            result = had_cells_strength(sf)
        nh = result.sel(cell="had_cell_nh")
        assert nh[LAT_STR].item() == 10.0
        assert nh[LEV_STR].item() == 550.0
        assert nh.item() == pytest.approx(1e11)
        assert result.sel(cell="had_cell_sh").item() >= 0.0

    def test_missing_summer_cell(self) -> None:
        # A northern-winter field with no southern Hadley cell, and northern
        # Ferrel and polar cells close together. Those two are the only pair
        # of the Hadley cells' senses, but the pair containing the center
        # nearest the equator is the northern Hadley cell with no partner,
        # so the northern cell is identified, the southern entry is not
        # negative, and a warning is issued.
        sf = _bump_cells_streamfunc(
            [(2e11, 5.0, 25.0), (-0.6e11, 36.0, 6.0), (0.2e11, 48.0, 6.0)]
        )
        with pytest.warns(UserWarning, match="sense of circulation"):
            result = had_cells_strength(sf)
        nh = result.sel(cell="had_cell_nh")
        assert nh[LAT_STR].item() == 5.0
        assert nh.item() == pytest.approx(2e11)
        assert result.sel(cell="had_cell_sh").item() >= 0.0

    @pytest.mark.parametrize(
        ("sf", "expected"),
        [
            (_two_cell_streamfunc(), [(-15.0, -1e11), (15.0, 1e11)]),
            (
                _bump_cells_streamfunc(
                    [
                        (-8e10, -12.0, 12.0),
                        (7.5e10, 14.0, 14.0),
                        (2e10, -36.0, 12.0),
                        (-2e10, 42.0, 14.0),
                    ]
                ),
                [(-12.0, -8e10), (14.0, 7.5e10)],
            ),
            (_bump_cells_streamfunc(_SH_WINTER_CELLS), [(-5.0, -2e11), (25.0, 3e10)]),
            (_bump_cells_streamfunc(_NH_WINTER_CELLS), [(-25.0, -3e10), (5.0, 2e11)]),
            (
                _bump_cells_streamfunc(
                    [
                        (2e11, 5.0, 25.0),
                        (-0.3e11, -25.5, 10.0),
                        (-0.6e11, 45.0, 12.0),
                        (0.2e11, 72.0, 9.0),
                        (0.1e11, -50.0, 12.0),
                    ]
                ),
                [(-25.5, -3e10), (5.0, 2e11)],
            ),
        ],
        ids=[
            "equinox",
            "asymmetric_equinox",
            "sh_winter",
            "nh_winter",
            "nh_winter_off_grid",
        ],
    )
    def test_do_interp_lat(
        self, sf: xr.DataArray, expected: list[tuple[float, float]]
    ) -> None:
        # With do_interp_lat, each candidate's latitude is interpolated, and
        # the searches for each extremum's partner must still find the cells
        # on either side. Each cell is symmetric about its center, so the
        # interpolated center is the cell's center. In the last field the
        # southern cell is centered between gridpoints, and is the partner
        # found south of the global maximum, so its interpolated center
        # requires the search for that partner to interpolate. Every search
        # interpolates successfully, so none falls back on its gridpoint.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message="Interpolating the latitude", category=UserWarning
            )
            result = had_cells_strength(sf, do_interp_lat=True)
        for cell, (lat_center, strength) in zip(
            ("had_cell_sh", "had_cell_nh"), expected, strict=True
        ):
            assert result.sel(cell=cell)[LAT_STR].item() == pytest.approx(
                lat_center, abs=1e-6
            )
            assert result.sel(cell=cell).item() == pytest.approx(strength, rel=1e-3)

    @pytest.mark.parametrize(
        ("cells", "min_lat", "max_lat", "expected_lats"),
        [
            (_NH_WINTER_CELLS, -25.5, None, [-25.0, 5.0]),
            (_NH_WINTER_CELLS, -24.5, None, [-24.0, 5.0]),
            (_SH_WINTER_CELLS, None, 25.5, [-5.0, 25.0]),
        ],
        ids=["nh_winter_center_at_min_lat", "nh_winter_cut_by_min_lat", "sh_winter"],
    )
    def test_do_interp_lat_partner_at_domain_edge(
        self,
        cells: list[tuple[float, float, float]],
        min_lat: float | None,
        max_lat: float | None,
        expected_lats: list[float],
    ) -> None:
        # The summer cell's gridpoint extremum is the last latitude inside
        # the bound, where the latitude derivative has no zero crossing to
        # interpolate. The summer cell's center is placed at that gridpoint,
        # with a warning, and the cells are those found without interpolation.
        sf = _bump_cells_streamfunc(cells)
        expected = had_cells_strength(sf, min_lat=min_lat, max_lat=max_lat)
        with pytest.warns(UserWarning, match="Interpolating the latitude") as record:
            result = had_cells_strength(
                sf, min_lat=min_lat, max_lat=max_lat, do_interp_lat=True
            )
        assert not any("sense of circulation" in str(w.message) for w in record)
        assert result[LAT_STR].values.tolist() == pytest.approx(expected_lats, abs=1e-6)
        assert expected[LAT_STR].values.tolist() == expected_lats
        assert result.values.tolist() == pytest.approx(
            expected.values.tolist(), rel=1e-3
        )

    @pytest.mark.parametrize("max_lat", [None, 50.0], ids=["min_lat", "both_bounds"])
    def test_do_interp_lat_bounds_exclude_summer_cell(
        self, max_lat: float | None
    ) -> None:
        # min_lat excludes the northern-winter field's summer cell at 25S, so
        # no negative value lies south of the winter cell. With interpolation,
        # as without it, the southern entry is not negative and the warning
        # for a missing cell is issued. The southern entry is zero along the
        # top level, where interpolation fails, so it is placed at its
        # gridpoint, with the other warning.
        sf = _bump_cells_streamfunc(_NH_WINTER_CELLS)
        with pytest.warns(UserWarning) as record:
            result = had_cells_strength(
                sf, min_lat=-12.0, max_lat=max_lat, do_interp_lat=True
            )
        messages = [str(w.message) for w in record]
        assert any("sense of circulation" in m for m in messages)
        assert any("Interpolating the latitude" in m for m in messages)
        nh = result.sel(cell="had_cell_nh")
        assert nh[LAT_STR].item() == pytest.approx(5.0, abs=1e-6)
        assert nh.item() == pytest.approx(2e11, rel=1e-3)
        assert result.sel(cell="had_cell_sh").item() >= 0.0

    @pytest.mark.parametrize(
        ("peak", "center", "present", "missing"),
        [
            (1e11, 10.0, "had_cell_nh", "had_cell_sh"),
            (-1e11, -10.0, "had_cell_sh", "had_cell_nh"),
        ],
        ids=["lone_northern_cell", "lone_southern_cell"],
    )
    def test_do_interp_lat_one_cell_warns(
        self, peak: float, center: float, present: str, missing: str
    ) -> None:
        # A lone cell's field is zero along the top level, where the global
        # extremum of the missing cell's sign lies, so the latitude
        # derivative there has no zero crossing to interpolate. That
        # extremum is placed at its gridpoint, and the missing cell is
        # flagged as it is without interpolation.
        sf = _bump_cells_streamfunc([(peak, center, 30.0)])
        with pytest.warns(UserWarning) as record:
            result = had_cells_strength(sf, do_interp_lat=True)
        messages = [str(w.message) for w in record]
        assert any("sense of circulation" in m for m in messages)
        assert any("Interpolating the latitude" in m for m in messages)
        cell = result.sel(cell=present)
        assert cell[LAT_STR].item() == pytest.approx(center, abs=1e-6)
        assert cell.item() == pytest.approx(peak, rel=1e-3)
        assert result.sel(cell=missing).item() * peak <= 0.0

    @pytest.mark.parametrize("do_avg_vert", [True, False], ids=["avg_vert", "levels"])
    @pytest.mark.parametrize("extra_dim", [False, True], ids=["lat_lev", "with_time"])
    def test_do_interp_lat_vertical_average_and_extra_dim(
        self, do_avg_vert: bool, extra_dim: bool
    ) -> None:
        # https://github.com/spencerahill/puffins/issues/117: under
        # do_interp_lat, do_avg_vert and a dimension besides latitude and
        # level each made every search fall back on its gridpoint. Each cell
        # is centered midway between gridpoints and is symmetric about its
        # center, so its interpolated center is its center. Its strength is
        # its peak times the mean of the vertical structure over the levels
        # up to max_plev, or times that structure's peak, 1 at plev=550.
        # max_plev=900 puts the level mean at 500, apart from the peak's 550.
        sf = _bump_cells_streamfunc([(-1e11, -12.5, 12.0), (0.8e11, 14.5, 14.0)])
        if extra_dim:
            sf = sf.expand_dims({TIME_STR: [0]})
        plevs = np.arange(100.0, 901.0, 50.0)
        vert = np.mean(np.sin(np.pi * (plevs - 100.0) / 900.0)) if do_avg_vert else 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = had_cells_strength(
                sf, max_plev=900.0, do_avg_vert=do_avg_vert, do_interp_lat=True
            )
        if extra_dim:
            assert result.dims == (TIME_STR, "cell")
            result = result.isel({TIME_STR: 0})
        assert result[LAT_STR].values.tolist() == pytest.approx([-12.5, 14.5], abs=1e-9)
        assert result.values.tolist() == pytest.approx(
            [-1e11 * vert, 0.8e11 * vert], rel=1e-4
        )
        expected_lev = 500.0 if do_avg_vert else 550.0
        assert result[LEV_STR].values.tolist() == pytest.approx([expected_lev] * 2)

    @pytest.mark.parametrize(
        ("plev_spike", "kwargs", "n_fallbacks"),
        [
            (150.0, {"min_plev": 300.0}, 1),
            (950.0, {"max_plev": 900.0}, 1),
            (150.0, {"do_avg_vert": True}, 2),
        ],
        ids=["min_plev", "max_plev", "do_avg_vert"],
    )
    def test_vertical_options_reach_gridpoint_fallback(
        self, plev_spike: float, kwargs: dict[str, Any], n_fallbacks: int
    ) -> None:
        # A spike of -5e11 at 25S, the summer cell's center and the last
        # latitude inside min_lat, at a level the option excludes or averages
        # over. Under interpolation that cell's center falls back on its
        # gridpoint, and the fallback search must receive the option too.
        # Only the searches that find that cell fall back: the search south
        # of the global maximum, and with do_avg_vert, which makes that cell
        # the global minimum, the search for the global minimum too.
        sf = _bump_cells_streamfunc(_NH_WINTER_CELLS).copy(deep=True)
        sf.loc[{LAT_STR: -25.0, LEV_STR: plev_spike}] = -5e11
        spike = had_cells_strength(sf, min_lat=-25.5).sel(cell="had_cell_sh")
        assert spike.item() == -5e11
        expected = had_cells_strength(sf, min_lat=-25.5, **kwargs)
        with pytest.warns(UserWarning, match="Interpolating the latitude") as record:
            result = had_cells_strength(sf, min_lat=-25.5, do_interp_lat=True, **kwargs)
        assert len(record) == n_fallbacks
        assert all("gridpoint -25." in str(w.message) for w in record)
        sh, sh_expected = (arr.sel(cell="had_cell_sh") for arr in (result, expected))
        assert sh_expected.item() != spike.item()
        assert sh.item() == sh_expected.item()
        assert sh[LAT_STR].item() == sh_expected[LAT_STR].item() == -25.0
        assert sh[LEV_STR].item() == sh_expected[LEV_STR].item()

    def test_no_pair_raises(self) -> None:
        # A field decreasing northward at every level has its maximum at the
        # southernmost latitude and its minimum at the northernmost, so no
        # cell lies south of the one or north of the other.
        two_cell = _two_cell_streamfunc()
        sf = -two_cell[LAT_STR] * xr.ones_like(two_cell)
        with pytest.raises(ValueError, match="no pair of cells"):
            had_cells_strength(sf)

    def test_opposite_sign_two_cell_field_warns(self) -> None:
        # A two-cell streamfunction signed opposite to merid_streamfunc's,
        # with the northern cell negative, has no pair of the expected
        # senses. A field with Ferrel cells has one, and is not flagged.
        with pytest.warns(UserWarning, match="sense of circulation"):
            had_cells_strength(-_two_cell_streamfunc())

    @pytest.mark.parametrize(
        "sf",
        [
            _two_cell_streamfunc(),
            _issue_112_streamfunc(),
            _bump_cells_streamfunc(_SH_WINTER_CELLS),
            _bump_cells_streamfunc(_NH_WINTER_CELLS),
        ],
        ids=["equinox", "issue_112", "sh_winter", "nh_winter"],
    )
    def test_no_warning_when_both_cells_exist(self, sf: xr.DataArray) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message="The cells chosen as the Hadley cells",
                category=UserWarning,
            )
            had_cells_strength(sf)

    def test_lone_southern_cell_warns(self) -> None:
        # The mirror image of test_one_cell_warns. A lone negative cell is
        # the southern Hadley cell. The global maximum is zero at the
        # southernmost latitude, so only the pair built on the global minimum
        # exists, and its northern entry is not positive.
        sf = _bump_cells_streamfunc([(-1e11, -10.0, 30.0)])
        with pytest.warns(UserWarning, match="sense of circulation"):
            result = had_cells_strength(sf)
        sh = result.sel(cell="had_cell_sh")
        assert sh[LAT_STR].item() == -10.0
        assert sh[LEV_STR].item() == 550.0
        assert sh.item() == pytest.approx(-1e11)
        assert result.sel(cell="had_cell_nh").item() <= 0.0

    def test_pair_measured_by_its_center_nearest_the_equator(self) -> None:
        # A southern-winter field whose southern Hadley cell is centered at
        # 2S and whose northern Hadley cell lies far north, at 35N. The other
        # candidate pair, the southern polar and Ferrel cells, has its
        # northern member nearer the equator (30S) than the Hadley pair's
        # northern member, but its center nearest the equator is farther.
        sf = _bump_cells_streamfunc(
            [
                (-2e11, -2.0, 20.0),
                (0.3e11, 35.0, 15.0),
                (0.6e11, -30.0, 8.0),
                (-0.2e11, -50.0, 10.0),
            ]
        )
        result = had_cells_strength(sf)
        assert result[LAT_STR].values.tolist() == [-2.0, 35.0]
        assert result.values.tolist() == pytest.approx([-2e11, 0.3e11])

    def test_cells_at_different_levels(self) -> None:
        # The southern cell peaks at plev=550 and the northern at plev=400,
        # and each cell's level must be its own.
        lats = np.arange(-85.0, 86.0, 1.0)
        plevs = np.arange(100.0, 1001.0, 50.0)
        lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
        plev = xr.DataArray(plevs, dims=[LEV_STR], coords={LEV_STR: plevs})
        vert_sh = cast(xr.DataArray, np.sin(np.pi * (plev - 100.0) / 900.0))
        vert_nh = xr.where(plev <= 700.0, np.sin(np.pi * (plev - 100.0) / 600.0), 0.0)
        sf = -1e11 * vert_sh * _bump(lat, -15.0, 15.0) + 1e11 * vert_nh * _bump(
            lat, 15.0, 15.0
        )
        result = had_cells_strength(sf)
        assert result.sel(cell="had_cell_sh")[LEV_STR].item() == 550.0
        assert result.sel(cell="had_cell_nh")[LEV_STR].item() == 400.0
        assert result.sel(cell="had_cell_sh").item() == pytest.approx(-1e11)
        assert result.sel(cell="had_cell_nh").item() == pytest.approx(1e11)

    @pytest.mark.parametrize(
        ("plev_spike", "kwargs", "strength", "level"),
        [
            (150.0, {"min_plev": 300.0}, 1e11, 550.0),
            (950.0, {"max_plev": 900.0}, 1e11, 550.0),
            (
                150.0,
                {"do_avg_vert": True},
                1e11 * np.mean(np.sin(np.pi * np.arange(0.0, 901.0, 50.0) / 900.0)),
                550.0,
            ),
        ],
        ids=["min_plev", "max_plev", "do_avg_vert"],
    )
    def test_vertical_options_reach_every_search(
        self, plev_spike: float, kwargs: dict[str, Any], strength: float, level: float
    ) -> None:
        # Single-gridpoint spikes at 2S (negative) and 2N (positive), at a
        # level the option excludes or averages away, each five times the
        # Hadley cells' peak. Without the option they are the global extrema.
        # With it, any of the four extremum searches that did not receive the
        # option would return a spike, nearer the equator than the Hadley
        # cells, and the pair containing it would be chosen.
        sf = _two_cell_streamfunc().copy(deep=True)
        sf.loc[{LAT_STR: -2.0, LEV_STR: plev_spike}] = -5e11
        sf.loc[{LAT_STR: 2.0, LEV_STR: plev_spike}] = 5e11
        assert had_cells_strength(sf)[LAT_STR].values.tolist() == [-2.0, 2.0]
        result = had_cells_strength(sf, **kwargs)
        assert result[LAT_STR].values.tolist() == [-15.0, 15.0]
        assert result.values.tolist() == pytest.approx([-strength, strength])
        assert result[LEV_STR].values.tolist() == pytest.approx([level, level])

    def test_stronger_ferrel_polar_pair_not_selected(self) -> None:
        # The northern Ferrel and polar cells (2e11 and 0.5e11) together
        # outweigh the two Hadley cells (1e11 each), and both pairs have the
        # expected senses, but the Hadley cells are the pair containing the
        # center nearest the equator.
        sf = _bump_cells_streamfunc(
            [
                (-1e11, -15.0, 15.0),
                (1e11, 15.0, 15.0),
                (-2e11, 45.0, 15.0),
                (0.5e11, 72.0, 9.0),
            ]
        )
        result = had_cells_strength(sf)
        assert result[LAT_STR].values.tolist() == [-15.0, 15.0]
        assert result.values.tolist() == pytest.approx([-1e11, 1e11])

    def test_compat_forwarded(self) -> None:
        # An auxiliary coordinate along latitude takes a different value at
        # each cell center. The default compat="override" keeps the first,
        # and a strict compat refuses to combine them.
        sf = _two_cell_streamfunc()
        sf = sf.assign_coords(sinlat=(LAT_STR, np.sin(np.deg2rad(sf[LAT_STR].values))))
        assert had_cells_strength(sf)[LAT_STR].values.tolist() == [-15.0, 15.0]
        with pytest.raises(xr.MergeError):
            had_cells_strength(sf, compat="equals")


class TestHadCellEdgeCoords:
    """Edge functions must not leak the internal 'cell' selection coord."""

    def test_had_cell_edge_no_cell_coord(self) -> None:
        sf = _make_streamfunc()
        for cell, edge in (
            ("north", "north"),
            ("south", "south"),
            ("south", "north"),
        ):
            result = had_cell_edge(sf, cell=cell, edge=edge, **EDGE_KWARGS)
            assert isinstance(result, xr.DataArray)
            assert "cell" not in result.coords, (cell, edge, result.coords)

    def test_north_south_shared_no_cell_coord(self) -> None:
        sf = _make_streamfunc()
        for func in (had_cells_north_edge, had_cells_south_edge):
            result = func(sf, **EDGE_KWARGS)
            assert isinstance(result, xr.DataArray)
            assert "cell" not in result.coords, (func.__name__, result.coords)
        result = had_cells_shared_edge(
            sf, min_lat=-50, max_lat=50, min_plev=500, max_plev=800, do_avg_vert=True
        )
        # The inner edge must also shed the level-of-max 'plev' scalar coord.
        assert list(result.coords) == [LAT_STR], result.coords


def _edge_arr(edge: xr.DataArray | float) -> xr.DataArray:
    """Narrow an edge-function result, asserting it isn't a pole fallback."""
    assert isinstance(edge, xr.DataArray), edge
    return edge


class TestHadCellEdgeValues:
    """Edge latitudes recovered from the idealized streamfunction."""

    def test_edge_latitudes(self) -> None:
        sf = _make_streamfunc(edge_lat=30.0)
        north = _edge_arr(had_cells_north_edge(sf, **EDGE_KWARGS))
        south = _edge_arr(had_cells_south_edge(sf, **EDGE_KWARGS))
        shared = _edge_arr(had_cell_edge(sf, cell="south", edge="north", **EDGE_KWARGS))
        assert 29.0 < north.item() < 30.0, north.item()
        assert -30.0 < south.item() < -29.0, south.item()
        assert abs(shared.item()) < 1.0, shared.item()

    def test_known_value_do_interp(self) -> None:
        # On the offset grid the cell max is sin(pi * 14.5 / 30), and the
        # threshold crossing of sin(pi * lat / 30) is analytic.
        sf = _make_streamfunc(edge_lat=30.0)
        cell_max_rel = np.sin(np.pi * 14.5 / 30.0)
        expected = 30.0 * (1.0 - np.arcsin(0.2 * cell_max_rel) / np.pi)
        north = _edge_arr(had_cell_edge(sf, frac_thresh=0.2, do_interp=True))
        south = _edge_arr(
            had_cell_edge(
                sf, cell="south", edge="south", frac_thresh=0.2, do_interp=True
            )
        )
        assert north.item() == pytest.approx(expected, abs=0.02)
        assert south.item() == pytest.approx(-expected, abs=0.02)

    def test_fixed_plev(self) -> None:
        # The tilted field's inner zero crossing sits at lat=0 at the
        # level of the cell centers (550) but at lat=5 at plev=800.
        sf = _tilted_streamfunc()
        shared_default = had_cells_shared_edge(sf)
        shared_800 = had_cells_shared_edge(sf, fixed_plev=800.0)
        assert abs(shared_default.item()) < 0.1
        assert shared_800.item() == pytest.approx(5.0, abs=0.1)
        # Outer NH edge at plev=800: normalized threshold crossing of the
        # shifted sine with reduced amplitude, reconstructed analytically.
        cell_max_rel = np.sin(np.pi * 14.5 / 30.0)
        vert_800 = np.sin(np.pi * (800.0 - 100.0) / 900.0)
        sin_cross = 0.1 * cell_max_rel / vert_800
        expected = 5.0 + 30.0 * (1.0 - np.arcsin(sin_cross) / np.pi)
        north_800 = _edge_arr(had_cell_edge(sf, fixed_plev=800.0, do_interp=True))
        assert north_800.item() == pytest.approx(expected, abs=0.05)

    def test_cos_factor(self) -> None:
        # Singh 2019 cosine weighting: reconstruct the crossing of
        # [sf / cos(lat)] / [max / cos(lat_max)] = frac_thresh with scipy.
        sf = _make_streamfunc(edge_lat=30.0)
        cell_max_rel = np.sin(np.pi * 14.5 / 30.0)
        coslat_max = np.cos(np.deg2rad(14.5))

        def _crossing(lat: float) -> float:
            sf_norm = (np.sin(np.pi * lat / 30.0) / np.cos(np.deg2rad(lat))) / (
                cell_max_rel / coslat_max
            )
            return float(sf_norm - 0.1)

        expected = brentq(_crossing, 15.0, 29.99)
        plain = _edge_arr(had_cell_edge(sf, do_interp=True))
        weighted = _edge_arr(had_cell_edge(sf, cos_factor=True, do_interp=True))
        assert weighted.item() == pytest.approx(expected, abs=0.05)
        assert weighted.item() > plain.item()


class TestHadCellsEdges:
    """had_cells_edges must route kwargs correctly to all three edge funcs."""

    def test_returns_three_matching_edges(self) -> None:
        sf = _make_streamfunc(edge_lat=30.0)
        south, shared, north = (
            _edge_arr(edge) for edge in had_cells_edges(sf, **EDGE_KWARGS)
        )
        # South/north match had_cells_south_edge/had_cells_north_edge; the
        # shared inner edge sits at ~0 for the symmetric profile.
        assert -30.0 < south.item() < -29.0, south.item()
        assert abs(shared.item()) < 1.0, shared.item()
        assert 29.0 < north.item() < 30.0, north.item()
        # All three edges are bare latitudes: no leaked 'cell'/'plev' coords.
        for edge in (south, shared, north):
            assert list(edge.coords) == [LAT_STR], edge.coords

    def test_cos_factor_not_forwarded_to_shared_edge(self) -> None:
        # `cos_factor` is a parameter of had_cell_edge but not
        # had_cells_shared_edge; had_cells_edges must not forward it there.
        sf = _make_streamfunc(edge_lat=30.0)
        south, shared, north = (
            _edge_arr(edge)
            for edge in had_cells_edges(sf, cos_factor=True, **EDGE_KWARGS)
        )
        assert -30.0 < south.item() < -29.0, south.item()
        assert abs(shared.item()) < 1.0, shared.item()
        assert 29.0 < north.item() < 30.0, north.item()

    def test_frac_thresh_forwarded_to_outer_edges(self) -> None:
        # frac_thresh must reach the two outer-edge calls (it is not a
        # parameter of the shared-edge call, whose edge is a zero crossing).
        sf = _make_streamfunc(edge_lat=30.0)
        south, shared, north = (
            _edge_arr(edge) for edge in had_cells_edges(sf, frac_thresh=0.3)
        )
        cell_max_rel = np.sin(np.pi * 14.5 / 30.0)
        expected = 30.0 * (1.0 - np.arcsin(0.3 * cell_max_rel) / np.pi)
        assert north.item() == pytest.approx(expected, abs=0.1)
        assert south.item() == pytest.approx(-expected, abs=0.1)
        assert abs(shared.item()) < 1.0, shared.item()

    def test_custom_dim_names(self) -> None:
        # Exercises had_cell_edge and had_cells_shared_edge (via
        # had_cells_edges) with non-default lat_str/lev_str.
        sf = _make_streamfunc(edge_lat=30.0)
        default = [
            _edge_arr(edge).item() for edge in had_cells_edges(sf, **EDGE_KWARGS)
        ]
        renamed_edges = [
            _edge_arr(edge)
            for edge in had_cells_edges(
                sf.rename({LAT_STR: "latitude", LEV_STR: "pressure"}),
                lat_str="latitude",
                lev_str="pressure",
                **EDGE_KWARGS,
            )
        ]
        np.testing.assert_allclose([edge.item() for edge in renamed_edges], default)
        for edge in renamed_edges:
            assert list(edge.coords) == ["latitude"], edge.coords


# The meridional node latitudes are offset from the integer grid by 0.25
# deg, so each cell's true center lies 0.25 deg from its gridded maximum and
# an edge search mirrored about the gridded maximum would miss by 0.5 deg.
# Nodes lying on gridpoints are tested separately.
_SIGMA_NODE_OFFSET = 0.25


def _sigma_streamfunc(amp: float = 1e10) -> xr.DataArray:
    """Two-cell streamfunction in sigma coordinates.

    Cell nodes at lat = 0.25 and +/- 30.25 (offset from the integer grid),
    vertical maximum at sigma = 0.55, peak magnitude amp.
    """
    lats = np.arange(-85.0, 86.0, 1.0)
    sigmas = np.linspace(0.1, 1.0, 19)
    lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
    sigma = xr.DataArray(sigmas, dims=[SIGMA_STR], coords={SIGMA_STR: sigmas})
    lat_c = lat - _SIGMA_NODE_OFFSET
    merid = xr.where(np.abs(lat_c) <= 30.0, np.sin(np.pi * lat_c / 30.0), 0.0)
    vert = np.sin(np.pi * (sigma - 0.1) / 0.9)
    return cast(xr.DataArray, (amp * vert * merid).rename("streamfunc"))


def _add_shallow_blob(sf: xr.DataArray, amp: float) -> xr.DataArray:
    """Add a negative cell spanning only ~0.1 in sigma, poleward of the NH cell.

    It directly abuts the NH cell in latitude, between 30.25 and 50.25.
    """
    lat = sf[LAT_STR]
    sigma = sf[SIGMA_STR]
    lat_c = lat - _SIGMA_NODE_OFFSET
    merid = xr.where(
        (lat_c > 30.0) & (lat_c <= 50.0), np.sin(np.pi * (lat_c - 30.0) / 20.0), 0.0
    )
    vert = xr.where((sigma > 0.47) & (sigma < 0.63), 1.0, 0.0)
    return cast(xr.DataArray, sf - amp * merid * vert)


def _sigma_vert(sig: float) -> float:
    """Vertical profile of the sigma-coordinate test streamfunction."""
    return float(np.sin(np.pi * (sig - 0.1) / 0.9))


# Gridded peak of the meridional profile: the max falls at lat=15, i.e.
# 14.75 deg from the node at lat=0.25.
_SIGMA_PEAK_REL = np.sin(np.pi * 14.75 / 30.0)


def _sigma_cells(
    sh_scale: float = 1.0, nh_scale: float = 1.0, ferrel: float = 0.0, amp: float = 1e10
) -> xr.DataArray:
    """Hadley cells of ``_sigma_streamfunc`` scaled per hemisphere, plus Ferrel cells.

    The Ferrel cells have the opposite sign to their hemisphere's Hadley cell
    and amplitude ``ferrel * amp``. Each is a half sine wave 30 degrees wide
    touching the Hadley cell, between nodes at 30.25 and 60.25 or at -59.75 and
    -29.75, so every cell's gridded peak is ``_SIGMA_PEAK_REL`` of its
    amplitude.
    """
    hadley = _sigma_streamfunc(amp)
    lat_c = hadley[LAT_STR] - _SIGMA_NODE_OFFSET
    hadley = hadley * xr.where(lat_c < 0, sh_scale, nh_scale)
    x = np.abs(lat_c) - 30.0
    vert = np.sin(np.pi * (hadley[SIGMA_STR] - 0.1) / 0.9)
    merid = xr.where(
        (x > 0) & (x <= 30.0), -np.sign(lat_c) * np.sin(np.pi * x / 30.0), 0.0
    )
    return cast(
        xr.DataArray, (hadley + ferrel * amp * vert * merid).rename("streamfunc")
    )


def _sigma_edge_inset(frac_thresh: float) -> float:
    """How far inside its two nodes each half-sine cell's edges lie.

    The edge is where sin(pi * x / 30), x degrees from a node, falls to
    ``frac_thresh`` times the cell's gridded peak.
    """
    return float(30.0 * np.arcsin(frac_thresh * _SIGMA_PEAK_REL) / np.pi)


# The nodes bounding each cell of ``_sigma_cells``, from south to north.
_SIGMA_CELL_NODES = [
    (_SIGMA_NODE_OFFSET - 60.0, _SIGMA_NODE_OFFSET - 30.0),
    (_SIGMA_NODE_OFFSET - 30.0, _SIGMA_NODE_OFFSET),
    (_SIGMA_NODE_OFFSET, _SIGMA_NODE_OFFSET + 30.0),
    (_SIGMA_NODE_OFFSET + 30.0, _SIGMA_NODE_OFFSET + 60.0),
]


def _sigma_half_sine_cells(cells: list[tuple[float, float, float]]) -> xr.DataArray:
    """Half-sine cells on the grid and vertical profile of ``_sigma_streamfunc``.

    Each cell is (southern node, northern node, signed amplitude in units of
    1e10) and is that amplitude times sin(pi * (lat - south) / (north -
    south)) between its nodes, zero elsewhere.
    """
    lats = np.arange(-85.0, 86.0, 1.0)
    sigmas = np.linspace(0.1, 1.0, 19)
    lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
    sigma = xr.DataArray(sigmas, dims=[SIGMA_STR], coords={SIGMA_STR: sigmas})
    merid = xr.zeros_like(lat)
    for south, north, amp in cells:
        x = (lat - south) / (north - south)
        merid = merid + xr.where((x > 0) & (x < 1), amp * np.sin(np.pi * x), 0.0)
    return cast(xr.DataArray, 1e10 * np.sin(np.pi * (sigma - 0.1) / 0.9) * merid)


def _mirror_cells(
    cells: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """The cells reflected across the equator, which reverses each one's sign."""
    return [(-north, -south, -amp) for south, north, amp in reversed(cells)]


class TestCellEdgesSigma:
    """Edges, centers, and strengths of cells found in sigma coords."""

    def test_two_cells_known_values(self) -> None:
        amp = 1e10
        cells = cell_edges_sigma(_sigma_streamfunc(amp), frac_thresh=0.2)
        assert list(cells["cell"].values) == ["sh_hadley", "nh_hadley"]
        # sin(pi * (lat - 0.25) / 30) crosses the threshold (0.2 of the
        # gridded cell max) at these analytic latitudes.
        half_width = 30.0 * np.arcsin(0.2 * _SIGMA_PEAK_REL) / np.pi
        offset = _SIGMA_NODE_OFFSET
        nh = cells.sel(cell="nh_hadley")
        sh = cells.sel(cell="sh_hadley")
        assert nh.sel(lat="center").item() == 15.0
        assert sh.sel(lat="center").item() == -15.0
        assert nh.sel(lat="edge_south").item() == pytest.approx(
            offset + half_width, abs=0.02
        )
        assert nh.sel(lat="edge_north").item() == pytest.approx(
            offset + 30.0 - half_width, abs=0.02
        )
        assert sh.sel(lat="edge_south").item() == pytest.approx(
            offset - 30.0 + half_width, abs=0.02
        )
        assert sh.sel(lat="edge_north").item() == pytest.approx(
            offset - half_width, abs=0.02
        )
        assert nh["cell_strength"].item() == pytest.approx(amp * _SIGMA_PEAK_REL)
        assert sh["cell_strength"].item() == pytest.approx(-amp * _SIGMA_PEAK_REL)

    def test_cell_min_sigma_depth(self) -> None:
        amp = 1e10
        sf = _add_shallow_blob(_sigma_streamfunc(amp), 0.5 * amp)
        # Depth ~0.1 in sigma < default threshold 0.3: blob discarded.
        cells = cell_edges_sigma(sf)
        assert len(cells["cell"]) == 2
        # Lowering the threshold keeps it as a third (unlabeled) cell.
        cells_shallow = cell_edges_sigma(sf, cell_min_sigma_depth=0.05)
        assert len(cells_shallow["cell"]) == 3
        assert list(cells_shallow["cell"].values[:2]) == ["sh_hadley", "nh_hadley"]

    def test_center_sigma_bounds(self) -> None:
        # Restricting the searchable sigma range moves the cell max off
        # the vertical profile's peak at sigma=0.55 to the nearest
        # retained gridpoint (0.5 and 0.6, respectively).
        amp = 1e10
        sf = _sigma_streamfunc(amp)
        cells_upper = cell_edges_sigma(sf, center_max_sigma=0.52)
        assert cells_upper.sel(cell="nh_hadley")[
            "cell_strength"
        ].item() == pytest.approx(amp * _SIGMA_PEAK_REL * _sigma_vert(0.5))
        cells_lower = cell_edges_sigma(sf, center_min_sigma=0.58)
        assert cells_lower.sel(cell="nh_hadley")[
            "cell_strength"
        ].item() == pytest.approx(amp * _SIGMA_PEAK_REL * _sigma_vert(0.6))

    def test_cos_factor_moves_edge_poleward(self) -> None:
        sf = _sigma_streamfunc()
        plain = cell_edges_sigma(sf, frac_thresh=0.2)
        weighted = cell_edges_sigma(sf, frac_thresh=0.2, cos_factor=True)
        assert (
            weighted.sel(cell="nh_hadley").sel(lat="edge_north").item()
            > plain.sel(cell="nh_hadley").sel(lat="edge_north").item()
        )

    def test_cos_factor_known_values(self) -> None:
        """With cos_factor, the edges are where psi / cos(lat), scaled to 1 at the
        gridded maximum, falls to the threshold.

        The crossings of the continuous profile come from brentq. The code
        interpolates linearly between gridpoints and lands within 0.005 degrees
        of them; leaving cos(lat_max) out of the scaling moves each edge by
        0.06 to 0.07 degrees.
        """
        frac = 0.2
        cells = cell_edges_sigma(_sigma_streamfunc(), frac_thresh=frac, cos_factor=True)
        for label, sign, lat_max, (south, north) in [
            ("sh_hadley", -1.0, -15.0, _SIGMA_CELL_NODES[1]),
            ("nh_hadley", 1.0, 15.0, _SIGMA_CELL_NODES[2]),
        ]:

            def excess(
                lat: float, sign: float = sign, lat_max: float = lat_max
            ) -> float:
                weighted = sign * np.sin(np.pi * (lat - _SIGMA_NODE_OFFSET) / 30.0)
                weighted /= np.cos(np.deg2rad(lat))
                return float(
                    weighted / (_SIGMA_PEAK_REL / np.cos(np.deg2rad(lat_max))) - frac
                )

            cell = cells.sel(cell=label)
            assert cell.sel(lat="edge_south").item() == pytest.approx(
                brentq(excess, south + 0.01, lat_max), abs=0.01
            )
            assert cell.sel(lat="edge_north").item() == pytest.approx(
                brentq(excess, lat_max, north - 0.01), abs=0.01
            )

    def test_custom_dim_names(self) -> None:
        amp = 1e10
        sf = _sigma_streamfunc(amp)
        default = cell_edges_sigma(sf, frac_thresh=0.2)
        renamed = sf.rename({LAT_STR: "latitude", SIGMA_STR: "s"})
        result = cell_edges_sigma(
            renamed, frac_thresh=0.2, lat_str="latitude", sigma_str="s"
        )
        assert "latitude" in result.dims
        assert list(result["cell"].values) == ["sh_hadley", "nh_hadley"]
        np.testing.assert_allclose(result.values, default.values)
        np.testing.assert_allclose(
            result["cell_strength"].values, default["cell_strength"].values
        )

    @pytest.mark.parametrize(("sh_scale", "nh_scale"), [(0.9, 1.0), (1.0, 0.9)])
    def test_unequal_hadley_cells(self, sh_scale: float, nh_scale: float) -> None:
        """Each Hadley cell gets its own edges when the two differ in strength.

        Issue #127: the weaker cell got the stronger one's edges, and with the
        southern cell weaker the two Hadley labels went to the wrong cells.
        """
        amp = 1e10
        cells = cell_edges_sigma(_sigma_cells(sh_scale, nh_scale, amp=amp))
        assert list(cells["cell"].values) == ["sh_hadley", "nh_hadley"]
        inset = _sigma_edge_inset(0.1)
        expected = {
            "sh_hadley": (_SIGMA_CELL_NODES[1], -15.0, -sh_scale),
            "nh_hadley": (_SIGMA_CELL_NODES[2], 15.0, nh_scale),
        }
        for label, ((south, north), center, scale) in expected.items():
            cell = cells.sel(cell=label)
            assert cell.sel(lat="edge_south").item() == pytest.approx(
                south + inset, abs=0.01
            )
            assert cell.sel(lat="edge_north").item() == pytest.approx(
                north - inset, abs=0.01
            )
            assert cell.sel(lat="center").item() == center
            assert cell["cell_strength"].item() == pytest.approx(
                scale * amp * _SIGMA_PEAK_REL
            )

    @pytest.mark.parametrize("frac_thresh", [0.1, 0.3])
    def test_four_cells_known_values(self, frac_thresh: float) -> None:
        """Hadley and Ferrel cells of unequal strengths each get their own edges.

        The southern Hadley cell is 0.8 of the northern and the Ferrel cells
        0.3, so every cell but the northern Hadley cell has a stronger cell of
        the opposite sign beside it.
        """
        amp = 1e10
        cells = cell_edges_sigma(
            _sigma_cells(0.8, 1.0, ferrel=0.3, amp=amp), frac_thresh=frac_thresh
        )
        assert list(cells["cell"].values) == ["0", "sh_hadley", "nh_hadley", "3"]
        inset = _sigma_edge_inset(frac_thresh)
        centers = [-45.0, -15.0, 15.0, 45.0]
        scales = [0.3, -0.8, 1.0, -0.3]
        for i, (south, north) in enumerate(_SIGMA_CELL_NODES):
            cell = cells.isel(cell=i)
            assert cell.sel(lat="edge_south").item() == pytest.approx(
                south + inset, abs=0.01
            )
            assert cell.sel(lat="edge_north").item() == pytest.approx(
                north - inset, abs=0.01
            )
            assert cell.sel(lat="center").item() == centers[i]
            assert cell["cell_strength"].item() == pytest.approx(
                scales[i] * amp * _SIGMA_PEAK_REL
            )

    def test_cos_factor_four_unequal_cells(self) -> None:
        """With cos_factor, each of four unequal cells gets its own edges.

        Every cell is sin(pi * (lat - a) / 30) between its nodes a and a + 30,
        so its cos-weighted edges are where that over cos(lat), scaled to 1 at
        the cell's gridded maximum, falls to the threshold, here by brentq. The
        worst edge is 0.017 degrees off, the southern Ferrel cell's southern
        edge, whose bracketing gridpoint at -60 lies past the node at -59.75;
        scaling by the largest magnitude of either sign moves edges by up to
        2.7 degrees.
        """
        cells = cell_edges_sigma(_sigma_cells(0.8, 1.0, ferrel=0.3), cos_factor=True)
        assert list(cells["cell"].values) == ["0", "sh_hadley", "nh_hadley", "3"]
        for i, ((south, north), center) in enumerate(
            zip(_SIGMA_CELL_NODES, [-45.0, -15.0, 15.0, 45.0], strict=True)
        ):

            def excess(
                lat: float, south: float = south, center: float = center
            ) -> float:
                weighted = np.sin(np.pi * (lat - south) / 30.0) / np.cos(
                    np.deg2rad(lat)
                )
                scale = _SIGMA_PEAK_REL / np.cos(np.deg2rad(center))
                return float(weighted / scale - 0.1)

            cell = cells.isel(cell=i)
            assert cell.sel(lat="center").item() == center
            assert cell.sel(lat="edge_south").item() == pytest.approx(
                brentq(excess, south + 0.01, center), abs=0.02
            )
            assert cell.sel(lat="edge_north").item() == pytest.approx(
                brentq(excess, center, north - 0.01), abs=0.02
            )

    def test_edge_beside_zero_streamfunction(self) -> None:
        """An edge is found where the streamfunction falls to zero between gridpoints.

        The cell spans nodes at 0.98 and 30.98, with zero streamfunction
        outside, so its last gridpoint, 30, lies just above the threshold and
        the next, 31, is zero. Zeros used to be masked out, which left no
        crossing and gave a NaN edge, and the cell was then dropped.
        """
        lats = np.arange(-85.0, 86.0, 1.0)
        sigmas = np.linspace(0.1, 1.0, 19)
        lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
        sigma = xr.DataArray(sigmas, dims=[SIGMA_STR], coords={SIGMA_STR: sigmas})
        x = lat - 0.98
        merid = xr.where((x > 0) & (x < 30.0), np.sin(np.pi * x / 30.0), 0.0)
        sf = 1e10 * np.sin(np.pi * (sigma - 0.1) / 0.9) * merid
        cells = cell_edges_sigma(sf)
        assert list(cells["cell"].values) == ["nh_hadley"]
        # Linear interpolation of the normalized field between 30 and 31.
        peak = np.sin(np.pi * 15.02 / 30.0)
        at_30 = np.sin(np.pi * 29.02 / 30.0) / peak
        edge_north = cells.sel(cell="nh_hadley", lat="edge_north").item()
        assert edge_north == pytest.approx(30.0 + (at_30 - 0.1) / at_30, rel=1e-12)
        # Within 0.01 degrees of the closed-form crossing, 0.956 inside the node.
        inset = 30.0 * np.arcsin(0.1 * peak) / np.pi
        assert edge_north == pytest.approx(30.98 - inset, abs=0.01)

    @pytest.mark.parametrize("hemisphere", [1.0, -1.0], ids=["north", "south"])
    def test_edge_past_the_end_of_the_grid(self, hemisphere: float) -> None:
        """A cell still above the threshold at the last latitude ends at the pole.

        The cell is a half sine wave from 0.25 to 120.25 degrees poleward of
        the equator, of the sign of a Hadley cell in that hemisphere, so at
        the last latitude, 85, it is still 0.8 of its peak. The edge is then
        interpolated between 85 and a zero placed at the pole. The southern
        case runs the search with the latitudes flipped.
        """
        lats = np.arange(-85.0, 86.0, 1.0)
        sigmas = np.linspace(0.1, 1.0, 19)
        lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
        sigma = xr.DataArray(sigmas, dims=[SIGMA_STR], coords={SIGMA_STR: sigmas})
        x = hemisphere * lat - _SIGMA_NODE_OFFSET
        merid = hemisphere * xr.where(x > 0, np.sin(np.pi * x / 120.0), 0.0)
        sf = 1e10 * np.sin(np.pi * (sigma - 0.1) / 0.9) * merid
        cells = cell_edges_sigma(sf)
        label = "nh_hadley" if hemisphere > 0 else "sh_hadley"
        assert list(cells["cell"].values) == [label]
        at_85 = np.sin(np.pi * 84.75 / 120.0) / np.sin(np.pi * 59.75 / 120.0)
        poleward = "edge_north" if hemisphere > 0 else "edge_south"
        edge = cells.sel(cell=label, lat=poleward).item()
        assert edge == pytest.approx(hemisphere * (85.0 + 5.0 * (at_85 - 0.1) / at_85))

    def test_cell_one_gridpoint_wide(self) -> None:
        """A cell above the threshold at its maximum alone still gets both edges.

        The cell is 1 at 10 N and 0.05 at 9 and 11 N, so each edge is
        interpolated between the maximum and a neighbor: 10 -/+ 0.9 / 0.95.
        """
        lats = np.arange(-85.0, 86.0, 1.0)
        sigmas = np.linspace(0.1, 1.0, 19)
        lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
        sigma = xr.DataArray(sigmas, dims=[SIGMA_STR], coords={SIGMA_STR: sigmas})
        merid = xr.where(
            lat == 10.0, 1.0, xr.where(np.abs(lat - 10.0) == 1.0, 0.05, 0.0)
        )
        sf = 1e10 * np.sin(np.pi * (sigma - 0.1) / 0.9) * merid
        cell = cell_edges_sigma(sf).sel(cell="nh_hadley")
        assert cell.sel(lat="edge_south").item() == pytest.approx(10.0 - 0.9 / 0.95)
        assert cell.sel(lat="edge_north").item() == pytest.approx(10.0 + 0.9 / 0.95)

    def test_nodes_on_gridpoints(self) -> None:
        """Two cells meeting at a gridpoint where the streamfunction is zero.

        The equator and +/-30 are gridpoints with zero streamfunction, which
        used to be masked out of both cells; master raised a ValueError.
        """
        lats = np.arange(-85.0, 86.0, 1.0)
        sigmas = np.linspace(0.1, 1.0, 19)
        lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
        sigma = xr.DataArray(sigmas, dims=[SIGMA_STR], coords={SIGMA_STR: sigmas})
        merid = xr.where(np.abs(lat) <= 30.0, np.sin(np.pi * lat / 30.0), 0.0)
        sf = 1e10 * np.sin(np.pi * (sigma - 0.1) / 0.9) * merid
        cells = cell_edges_sigma(sf)
        assert list(cells["cell"].values) == ["sh_hadley", "nh_hadley"]
        # The gridded peaks, at +/-15, are the true peaks here.
        inset = 30.0 * np.arcsin(0.1) / np.pi
        for label, (south, north) in [
            ("sh_hadley", (-30.0, 0.0)),
            ("nh_hadley", (0.0, 30.0)),
        ]:
            cell = cells.sel(cell=label)
            assert cell.sel(lat="edge_south").item() == pytest.approx(
                south + inset, abs=0.01
            )
            assert cell.sel(lat="edge_north").item() == pytest.approx(
                north - inset, abs=0.01
            )

    @pytest.mark.parametrize("boundary", [-20.25, -9.75, -1.75, 0.25, 9.75, 20.25])
    def test_hadley_labels_follow_sign(self, boundary: float) -> None:
        """The negative Hadley cell is sh_hadley and the positive one nh_hadley.

        The two cells, each 30 degrees wide, meet at ``boundary``. At -20.25
        and 20.25 the cell nearest the equator is centered 5 degrees into the
        hemisphere opposite its sign, which tells the sign apart from the
        hemisphere. The labels used to follow whether the southern edge of
        the cell nearest the equator lay south of it, so with the cells
        meeting at -20.25, -9.75 or -1.75 the positive cell was labeled
        sh_hadley and the negative one 0.
        """
        cells = cell_edges_sigma(
            _sigma_half_sine_cells(
                [(boundary - 30.0, boundary, -1.0), (boundary, boundary + 30.0, 1.0)]
            )
        )
        assert list(cells["cell"].values) == ["sh_hadley", "nh_hadley"]
        assert cells.sel(cell="sh_hadley")["cell_strength"].item() < 0
        assert cells.sel(cell="nh_hadley")["cell_strength"].item() > 0
        assert cells.sel(cell="sh_hadley", lat="center").item() == np.round(
            boundary - 15.0
        )
        assert cells.sel(cell="nh_hadley", lat="center").item() == np.round(
            boundary + 15.0
        )

    @pytest.mark.parametrize("boundary", [-12.25, 12.25])
    def test_hadley_labels_among_six_cells(self, boundary: float) -> None:
        """Hadley cells meeting off the equator, with Ferrel and polar cells.

        Each hemisphere has a polar, a Ferrel and a Hadley cell, of
        alternating sign, and the Hadley cells meet at ``boundary``, so one of
        them is centered on the equator. With the cells meeting at -12.25,
        the labels used to go to the positive Hadley cell and the northern
        Ferrel cell.
        """
        sf = _sigma_half_sine_cells(
            [
                (-85.25, -60.25, -0.2),
                (-60.25, boundary - 25.0, 0.3),
                (boundary - 25.0, boundary, -0.8),
                (boundary, boundary + 25.0, 1.0),
                (boundary + 25.0, 60.25, -0.3),
                (60.25, 85.25, 0.2),
            ]
        )
        cells = cell_edges_sigma(sf)
        assert list(cells["cell"].values) == [
            "0",
            "1",
            "sh_hadley",
            "nh_hadley",
            "4",
            "5",
        ]
        np.testing.assert_array_equal(
            np.sign(cells["cell_strength"].values), [-1, 1, -1, 1, -1, 1]
        )
        assert cells.sel(cell="sh_hadley", lat="center").item() == np.round(
            boundary - 12.5
        )
        assert cells.sel(cell="nh_hadley", lat="center").item() == np.round(
            boundary + 12.5
        )

    @pytest.mark.parametrize("mirror", [False, True], ids=["no_sh_cell", "no_nh_cell"])
    def test_lone_hadley_cell(self, mirror: bool) -> None:
        """With one Hadley cell missing, only the other gets a Hadley label.

        The positive Hadley cell spans the equator, from -22.25 to 29.75,
        with the southern Ferrel cell, also positive, 4 degrees south of it.
        The mirrored case is the same field reflected across the equator,
        which reverses each cell's sign. In both cases the northern Ferrel
        cell, negative, used to be labeled nh_hadley, and with no southern
        Hadley cell the positive Hadley cell was labeled sh_hadley.
        """
        cells_in = [
            (-85.25, -60.25, -0.2),
            (-60.25, -26.25, 0.3),
            (-22.25, 29.75, 1.0),
            (29.75, 60.25, -0.3),
            (60.25, 85.25, 0.2),
        ]
        if mirror:
            cells_in = _mirror_cells(cells_in)
        cells = cell_edges_sigma(_sigma_half_sine_cells(cells_in))
        label = "sh_hadley" if mirror else "nh_hadley"
        assert list(cells["cell"].values) == ["0", "1", label, "3", "4"]
        center = cells.sel(cell=label, lat="center").item()
        assert center == (-4.0 if mirror else 4.0)
        # The Ferrel cell beside it, of the same sign, has its facing edge
        # within 10 degrees, so only its sign keeps it from being labeled.
        i_south, i_north = (2, 3) if mirror else (1, 2)
        gap = (
            cells.isel(cell=i_north).sel(lat="edge_south").item()
            - cells.isel(cell=i_south).sel(lat="edge_north").item()
        )
        assert gap < 10.0

    @pytest.mark.parametrize("mirror", [False, True], ids=["nh_nearer", "sh_nearer"])
    @pytest.mark.parametrize(("node_gap", "paired"), [(7.8, True), (8.3, False)])
    def test_second_hadley_cell_within_max_lat_gap(
        self, node_gap: float, paired: bool, mirror: bool
    ) -> None:
        """The second Hadley cell is labeled only if it lies near the first.

        The positive cell, from 0.25 to 30.25, is nearer the equator, and the
        negative cell ends ``node_gap`` degrees south of it, so their facing
        edges are about node_gap + 1.9 degrees apart, 9.7 and 10.2 degrees
        here, either side of the 10 degrees below which the second cell is
        labeled. The mirrored case puts
        the negative cell nearer the equator. With the positive cell nearer,
        the second label used to be given at any distance.
        """
        cells_in = [(0.25 - node_gap - 30.0, 0.25 - node_gap, -1.0), (0.25, 30.25, 1.0)]
        if mirror:
            cells_in = _mirror_cells(cells_in)
        cells = cell_edges_sigma(_sigma_half_sine_cells(cells_in))
        gap = (
            cells.isel(cell=1).sel(lat="edge_south").item()
            - cells.isel(cell=0).sel(lat="edge_north").item()
        )
        assert abs(gap - 10.0) < 0.3
        assert (gap < 10.0) == paired
        if paired:
            expected = ["sh_hadley", "nh_hadley"]
        else:
            expected = ["sh_hadley", "1"] if mirror else ["0", "nh_hadley"]
        assert list(cells["cell"].values) == expected

    @pytest.mark.parametrize("mirror", [False, True], ids=["first", "last"])
    def test_hadley_cell_at_either_end(self, mirror: bool) -> None:
        """A Hadley cell with no cell on its ascending side gets no partner.

        The positive Hadley cell, from -20.25 to 30.25, is the southernmost
        cell, with a negative Ferrel cell north of it, on its descending
        side. Mirrored, the negative Hadley cell is the northernmost. The
        partner sought one place past either end of the cells must not wrap
        around to the cell at the other end, here the Ferrel cell. The
        positive Hadley cell used to be labeled sh_hadley, and the Ferrel cell
        nh_hadley.
        """
        cells_in = [(-20.25, 30.25, 1.0), (30.25, 60.25, -0.3)]
        expected = ["nh_hadley", "1"]
        if mirror:
            cells_in, expected = _mirror_cells(cells_in), ["0", "sh_hadley"]
        cells = cell_edges_sigma(_sigma_half_sine_cells(cells_in))
        assert list(cells["cell"].values) == expected

    def test_ferrel_cells_stronger_than_hadley_cells(self) -> None:
        """The Hadley cells are found by position, whatever their strength.

        The southern Ferrel cell, of amplitude 1, is the strongest and the
        northern one, 0.8, the next, with Hadley cells of 0.5 and 0.6
        between them, so the labels go to the two cells nearest the equator
        and to neither of the strongest.
        """
        cells = cell_edges_sigma(
            _sigma_half_sine_cells(
                [
                    (-59.75, -29.75, 1.0),
                    (-29.75, 0.25, -0.5),
                    (0.25, 30.25, 0.6),
                    (30.25, 60.25, -0.8),
                ]
            )
        )
        assert list(cells["cell"].values) == ["0", "sh_hadley", "nh_hadley", "3"]
        np.testing.assert_array_equal(
            np.sign(cells["cell_strength"].values), [1, -1, 1, -1]
        )

    def test_cells_of_equal_strength(self) -> None:
        """Two cells of the same strength, which comes back as a scalar.

        Both cells are positive, of amplitude 1 and 30 degrees wide, centered
        at -25 and 20. The one nearer the equator is nh_hadley, and the
        other, of the same sign, is unlabeled; it used to be labeled
        sh_hadley.
        """
        cells = cell_edges_sigma(
            _sigma_half_sine_cells([(-40.25, -10.25, 1.0), (5.25, 35.25, 1.0)])
        )
        # The case under test: one strength shared by two cells, held as a
        # scalar coordinate.
        assert cells["cell_strength"].ndim == 0
        assert list(cells["cell"].values) == ["0", "nh_hadley"]

    def test_cell_nearest_the_equator_by_its_maximum(self) -> None:
        """The first Hadley cell is the one whose maximum lies nearest the equator.

        The negative cell is two half-sine cells summed, from -60.25 to 0.25
        at 0.4 and from -16.25 to 0.25 at 1, so its maximum lies at -9 and
        the midpoint of its edges at -27.4. The positive cell, from 10.25 to
        40.25, has its maximum at 25 and its edges' midpoint at 25.25. Their
        facing edges are 11.5 degrees apart, too far to pair, so only the
        negative cell is labeled.
        """
        cells = cell_edges_sigma(
            _sigma_half_sine_cells(
                [(-60.25, 0.25, -0.4), (-16.25, 0.25, -1.0), (10.25, 40.25, 1.0)]
            )
        )
        assert list(cells["cell"].values) == ["sh_hadley", "1"]
        assert cells.sel(cell="sh_hadley", lat="center").item() == -9.0

    @pytest.mark.parametrize(
        ("max_lat_gap", "paired"), [(None, False), (10.5, False), (11.0, True)]
    )
    def test_max_lat_gap(self, max_lat_gap: float | None, paired: bool) -> None:
        """max_lat_gap sets how far apart the Hadley cells' facing edges may be.

        Two cells 32 degrees wide each meet at 0.25. At frac_thresh 0.5 each
        facing edge lies 32 arcsin(0.5) / pi = 16 / 3 degrees from the node
        the cells share, so the facing edges are 32 / 3 = 10.67 degrees
        apart: beyond the default limit of 10 and beyond 10.5, within 11.
        """
        sf = _sigma_half_sine_cells([(-31.75, 0.25, -1.0), (0.25, 32.25, 1.0)])
        if max_lat_gap is None:
            cells = cell_edges_sigma(sf, frac_thresh=0.5)
        else:
            cells = cell_edges_sigma(sf, frac_thresh=0.5, max_lat_gap=max_lat_gap)
        gap = (
            cells.isel(cell=1).sel(lat="edge_south").item()
            - cells.isel(cell=0).sel(lat="edge_north").item()
        )
        assert gap == pytest.approx(32.0 / 3.0, abs=0.02)
        expected = ["sh_hadley", "nh_hadley"] if paired else ["sh_hadley", "1"]
        assert list(cells["cell"].values) == expected


def _dropped_messages(record: pytest.WarningsRecorder) -> list[str]:
    """The messages of the warnings for cells that cell_edges_sigma dropped."""
    messages = [str(w.message) for w in record]
    return [m for m in messages if m.startswith("cell_edges_sigma dropped")]


# Southern polar, Ferrel and Hadley cells and a northern Hadley cell. The
# polar cell, 50 degrees wide, peaks at 85.25S, so its gridded maximum is at
# 85S, the first latitude of the grid of ``_sigma_half_sine_cells``.
_POLAR_CELL_AT_GRID_END = [
    (-110.25, -60.25, -0.3),
    (-60.25, -30.25, 0.4),
    (-30.25, 0.25, -1.0),
    (0.25, 30.25, 1.0),
]


def _six_sigma_cells(polar: float) -> xr.DataArray:
    """Polar, Ferrel and Hadley cells in each hemisphere, each 30 degrees wide.

    The Hadley cells' amplitude is 1, the Ferrel cells' 0.4 and the polar
    cells' ``polar``. Each cell's maximum lies 0.25 degrees from a gridpoint,
    so every gridded peak is the same fraction of its amplitude, and a cell's
    strength relative to the strongest is its amplitude.
    """
    return _sigma_half_sine_cells(
        [
            (-90.25, -60.25, -polar),
            (-60.25, -30.25, 0.4),
            (-30.25, 0.25, -1.0),
            (0.25, 30.25, 1.0),
            (30.25, 60.25, -0.4),
            (60.25, 90.25, polar),
        ]
    )


class TestCellEdgesSigmaDroppedCells:
    """The cells cell_edges_sigma drops, and the warnings for those it reports."""

    @pytest.mark.parametrize(
        ("nan_lat", "kept", "dropped"),
        [
            (20.0, ["sh_hadley"], [(15.0, "northern"), (21.0, "southern")]),
            (29.0, ["sh_hadley"], [(15.0, "northern")]),
            (30.0, ["sh_hadley"], [(15.0, "northern")]),
            (-30.0, ["nh_hadley"], [(-15.0, "southern")]),
            (31.0, ["sh_hadley", "nh_hadley"], []),
            (60.0, ["sh_hadley", "nh_hadley"], []),
        ],
    )
    def test_nan_column(
        self, nan_lat: float, kept: list[str], dropped: list[tuple[float, str]]
    ) -> None:
        """A NaN column inside a cell or beside its edge drops it, with a warning.

        The field is sin(pi * (lat - 0.25) / 30) between -30.25 and 30.25,
        two Hadley cells with edges where it falls to 0.1 of its gridded
        peak. A NaN is in neither cell, so a NaN at 20N splits the northern
        cell into two, each with an edge beside the other half, and a NaN at
        29N, 30N or 30S lies beside an edge. Each such cell is dropped with a
        warning naming the side of the edge not found. NaN outside both
        cells, at 31N or 60N, changes nothing.
        """
        sf = _sigma_half_sine_cells([(-30.25, 0.25, -1.0), (0.25, 30.25, 1.0)])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            full = cell_edges_sigma(sf)
        masked = sf.where(sf[LAT_STR] != nan_lat)
        if dropped:
            with pytest.warns(UserWarning) as record:
                cells = cell_edges_sigma(masked)
            messages = _dropped_messages(record)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                cells = cell_edges_sigma(masked)
            messages = []
        assert list(cells["cell"].values) == kept
        inset = _sigma_edge_inset(0.1)
        for label in kept:
            south, north = (-30.25, 0.25) if label == "sh_hadley" else (0.25, 30.25)
            cell = cells.sel(cell=label)
            np.testing.assert_array_equal(cell.values, full.sel(cell=label).values)
            assert cell.sel(lat="edge_south").item() == pytest.approx(
                south + inset, abs=0.02
            )
            assert cell.sel(lat="edge_north").item() == pytest.approx(
                north - inset, abs=0.02
            )
        assert len(messages) == len(dropped)
        for center, side in dropped:
            strength = 1e10 * np.sin(np.pi * (center - 0.25) / 30.0)
            matching = [m for m in messages if f"at latitude {center:g} and" in m]
            assert len(matching) == 1
            assert f"of strength {strength:.3g}:" in matching[0]
            assert f"its {side} edge could not be found" in matching[0]

    @pytest.mark.parametrize(
        ("cells", "lats", "dropped_center", "kept"),
        [
            (
                _POLAR_CELL_AT_GRID_END,
                slice(None),
                -85.0,
                [("0", -45.0), ("sh_hadley", -15.0), ("nh_hadley", 15.0)],
            ),
            (
                _mirror_cells(_POLAR_CELL_AT_GRID_END),
                slice(None),
                85.0,
                [("sh_hadley", -15.0), ("nh_hadley", 15.0), ("2", 45.0)],
            ),
            (
                [
                    (-30.25, 0.25, -1.0),
                    (0.25, 30.25, 1.0),
                    (30.25, 59.75, -0.4),
                    (59.75, 100.25, 0.3),
                ],
                slice(None, 80.0),
                80.0,
                [("sh_hadley", -15.0), ("nh_hadley", 15.0), ("2", 45.0)],
            ),
        ],
        ids=["first_latitude", "last_latitude", "grid_cut_at_80N"],
    )
    def test_maximum_at_end_of_grid(
        self,
        cells: list[tuple[float, float, float]],
        lats: slice,
        dropped_center: float,
        kept: list[tuple[str, float]],
    ) -> None:
        """A cell whose maximum is at either end of the grid is dropped with a warning.

        The other cells are returned. Such a cell used to make the whole call
        raise when its maximum was at the first latitude or at the latitude of
        the same magnitude, and it was kept, with an edge placed by assuming
        the cell falls to zero at the pole, when its maximum was at the last
        latitude of a grid cut short of that magnitude.
        """
        sf = _sigma_half_sine_cells(cells).sel({LAT_STR: lats})
        with pytest.warns(UserWarning) as record:
            result = cell_edges_sigma(sf)
        assert list(result["cell"].values) == [label for label, _ in kept]
        assert result.sel(lat="center").values.tolist() == [c for _, c in kept]
        messages = _dropped_messages(record)
        assert len(messages) == 1
        assert f"at latitude {dropped_center:g} and" in messages[0]
        assert "first or last latitude of the grid" in messages[0]

    def test_maximum_at_magnitude_of_first_latitude_kept(self) -> None:
        """Only the grid's two end latitudes are tested, not their magnitudes.

        On the grid from 80S to 85N, a cell peaking at 80N lies inside the
        grid. It used to be taken as lying at the pole, because 80 is the
        magnitude of the first latitude, which made the call raise. The cell
        runs from 60.25 to 100.25, so its southern edge lies (40 / pi)
        arcsin(0.1 p) north of 60.25, where p = sin(pi * 19.75 / 40) is its
        gridded peak, and it is still 0.93 of that peak at 85N, so its
        northern edge is interpolated toward zero at the pole.
        """
        sf = _sigma_half_sine_cells(
            [
                (-30.25, 0.25, -1.0),
                (0.25, 30.25, 1.0),
                (30.25, 60.25, -0.4),
                (60.25, 100.25, 0.3),
            ]
        ).sel({LAT_STR: slice(-80.0, None)})
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            cells = cell_edges_sigma(sf)
        assert list(cells["cell"].values) == ["sh_hadley", "nh_hadley", "2", "3"]
        cell = cells.sel(cell="3")
        assert cell.sel(lat="center").item() == 80.0
        peak = np.sin(np.pi * 19.75 / 40.0)
        assert cell.sel(lat="edge_south").item() == pytest.approx(
            60.25 + 40.0 * np.arcsin(0.1 * peak) / np.pi, abs=0.01
        )
        at_85 = np.sin(np.pi * 24.75 / 40.0) / peak
        assert cell.sel(lat="edge_north").item() == pytest.approx(
            85.0 + 5.0 * (at_85 - 0.1) / at_85
        )

    def test_one_element_at_grid_end_keeps_the_others(self) -> None:
        """A cell at the grid's end in one element no longer fails the whole call.

        The two times differ in their Hadley cells' amplitudes, so each
        time's result can be told from the other's.
        """
        first = _sigma_half_sine_cells(
            [(-60.25, -30.25, 0.4), (-30.25, 0.25, -0.8), (0.25, 30.25, 0.8)]
        )
        second = _sigma_half_sine_cells(_POLAR_CELL_AT_GRID_END)
        with pytest.warns(UserWarning) as record:
            result = cell_edges_sigma(xr.concat([first, second], dim=TIME_STR))
        assert len(_dropped_messages(record)) == 1
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            alone_first = cell_edges_sigma(first)
        with pytest.warns(UserWarning):
            alone_second = cell_edges_sigma(second)
        for i_time, alone in enumerate([alone_first, alone_second]):
            got = result.isel({TIME_STR: i_time})
            np.testing.assert_array_equal(got.values, alone.values)
            np.testing.assert_array_equal(
                got["cell_strength"].values, alone["cell_strength"].values
            )
        assert result["cell"].values.tolist() == ["0", "sh_hadley", "nh_hadley"]

    def test_no_measurable_cell_raises(self) -> None:
        sf = _sigma_half_sine_cells([(-110.25, -60.25, -0.3)])
        with (
            pytest.warns(UserWarning, match="first or last latitude"),
            pytest.raises(ValueError, match="no cell remains"),
        ):
            cell_edges_sigma(sf)

    @pytest.mark.parametrize(
        ("polar", "min_frac_strength", "polar_kept"),
        [
            (0.04, None, False),
            (0.04, 0.03, True),
            (0.2, None, True),
            (0.2, 0.25, False),
        ],
    )
    def test_min_frac_strength(
        self, polar: float, min_frac_strength: float | None, polar_kept: bool
    ) -> None:
        """Cells no stronger than min_frac_strength of the strongest are dropped.

        Each polar cell's strength is ``polar`` times the Hadley cells', so
        polar cells of 0.04 fall below the default of 0.05 and pass 0.03, and
        polar cells of 0.2 pass the default and fall below 0.25. Dropping a
        weak cell gives no warning.
        """
        sf = _six_sigma_cells(polar)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            if min_frac_strength is None:
                cells = cell_edges_sigma(sf)
            else:
                cells = cell_edges_sigma(sf, min_frac_strength=min_frac_strength)
        centers = [-75.0, -45.0, -15.0, 15.0, 45.0, 75.0]
        expected = centers if polar_kept else centers[1:-1]
        assert cells.sel(lat="center").values.tolist() == expected
        assert "sh_hadley" in cells["cell"].values
        assert "nh_hadley" in cells["cell"].values

    @pytest.mark.parametrize("max_num_cells", [None, 5, 3])
    def test_max_num_cells(self, max_num_cells: int | None) -> None:
        """Beyond max_num_cells cells, only the strongest of each sign are kept.

        Ten alternating half-sine cells, each 17 degrees wide, with every
        maximum 0.25 degrees from a gridpoint, so their strengths rank as
        their amplitudes. The cells beyond the strongest ``max_num_cells`` of
        each sign are dropped, each with a warning.
        """
        nodes = [-85.25 + 17.0 * k for k in range(11)]
        amps = [-0.2, 0.3, -0.4, 0.5, -1.0, 1.0, -0.5, 0.4, -0.3, 0.2]
        sf = _sigma_half_sine_cells(
            [(nodes[k], nodes[k + 1], amps[k]) for k in range(10)]
        )
        limit = 4 if max_num_cells is None else max_num_cells
        centers = np.round(np.array(nodes[:-1]) + 8.5)
        kept = set()
        for sign in (1.0, -1.0):
            same_sign = [k for k in range(10) if np.sign(amps[k]) == sign]
            kept |= set(sorted(same_sign, key=lambda k: -abs(amps[k]))[:limit])
        dropped = [centers[k] for k in range(10) if k not in kept]
        if dropped:
            with pytest.warns(UserWarning) as record:
                if max_num_cells is None:
                    cells = cell_edges_sigma(sf)
                else:
                    cells = cell_edges_sigma(sf, max_num_cells=max_num_cells)
            messages = _dropped_messages(record)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                cells = cell_edges_sigma(sf, max_num_cells=limit)
            messages = []
        assert cells.sel(lat="center").values.tolist() == [
            centers[k] for k in sorted(kept)
        ]
        assert len(dropped) == {None: 2, 5: 0, 3: 4}[max_num_cells]
        assert len(messages) == len(dropped)
        for center in dropped:
            matching = [m for m in messages if f"at latitude {center:g} and" in m]
            assert len(matching) == 1
            assert f"not among the {limit} strongest of its sign" in matching[0]

    @pytest.mark.parametrize(
        ("lower_cell", "expected"),
        [
            ((10.25, 30.25), [("sh_hadley", -15.0), ("nh_hadley", 20.0)]),
            (
                (30.25, 50.25),
                [("sh_hadley", -15.0), ("nh_hadley", 20.0), ("2", 40.0)],
            ),
        ],
        ids=["within", "partly_outside"],
    )
    def test_cell_within_another_cells_edges(
        self, lower_cell: tuple[float, float], expected: list[tuple[str, float]]
    ) -> None:
        """A cell lying entirely between another cell's edges is dropped, silently.

        A positive cell from 0.25 to 40.25 fills sigma 0.1 to 0.5, and a
        negative cell of half its amplitude fills sigma 0.6 to 1.0, at the
        latitudes given; a southern Hadley cell fills the whole column. The
        upper cell's edges lie near 1.5N and 39N, so the lower cell is
        dropped when it spans 10.25 to 30.25 and kept when it spans 30.25 to
        50.25.
        """
        lats = np.arange(-85.0, 86.0, 1.0)
        sigmas = np.linspace(0.1, 1.0, 19)
        lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
        sigma = xr.DataArray(sigmas, dims=[SIGMA_STR], coords={SIGMA_STR: sigmas})

        def half_sine(south: float, north: float) -> xr.DataArray:
            x = (lat - south) / (north - south)
            return cast(
                xr.DataArray, xr.where((x > 0) & (x < 1), np.sin(np.pi * x), 0.0)
            )

        upper = xr.where(sigma < 0.53, np.sin(np.pi * (sigma - 0.05) / 0.5), 0.0)
        lower = xr.where(sigma > 0.57, np.sin(np.pi * (sigma - 0.55) / 0.5), 0.0)
        full = np.sin(np.pi * (sigma - 0.1) / 0.9)
        sf = 1e10 * (
            -full * half_sine(-30.25, 0.25)
            + upper * half_sine(0.25, 40.25)
            - 0.5 * lower * half_sine(*lower_cell)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            cells = cell_edges_sigma(sf.transpose(SIGMA_STR, LAT_STR))
        assert list(cells["cell"].values) == [label for label, _ in expected]
        assert cells.sel(lat="center").values.tolist() == [c for _, c in expected]

    def test_cell_beneath_cell_at_grid_end_kept(self) -> None:
        """A cell whose maximum is at the grid's end has no edges to contain another.

        A negative cell fills sigma 0.1 to 0.5 from beyond the grid's end to
        55.25S and peaks at 85S, the first latitude. A positive cell of
        amplitude 0.5 fills sigma 0.6 to 1.0 from 75.25S to 60.25S, beneath
        it. The upper cell is dropped with a warning. The lower one is kept:
        edges searched from the upper cell's maximum would place the lower
        cell between them, and it would be dropped too.
        """
        lats = np.arange(-85.0, 86.0, 1.0)
        sigmas = np.linspace(0.1, 1.0, 19)
        lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
        sigma = xr.DataArray(sigmas, dims=[SIGMA_STR], coords={SIGMA_STR: sigmas})

        def half_sine(south: float, north: float) -> xr.DataArray:
            x = (lat - south) / (north - south)
            return cast(
                xr.DataArray, xr.where((x > 0) & (x < 1), np.sin(np.pi * x), 0.0)
            )

        upper = xr.where(sigma < 0.53, np.sin(np.pi * (sigma - 0.05) / 0.5), 0.0)
        lower = xr.where(sigma > 0.57, np.sin(np.pi * (sigma - 0.55) / 0.5), 0.0)
        full = np.sin(np.pi * (sigma - 0.1) / 0.9)
        sf = 1e10 * (
            -0.6 * upper * half_sine(-115.25, -55.25)
            + 0.5 * lower * half_sine(-75.25, -60.25)
            - full * half_sine(-30.25, 0.25)
            + full * half_sine(0.25, 30.25)
        )
        with pytest.warns(UserWarning) as record:
            cells = cell_edges_sigma(sf.transpose(SIGMA_STR, LAT_STR))
        messages = _dropped_messages(record)
        assert len(messages) == 1
        assert "at latitude -85 and" in messages[0]
        assert list(cells["cell"].values) == ["0", "sh_hadley", "nh_hadley"]
        lower_cell = cells.sel(cell="0")
        assert lower_cell.sel(lat="center").item() == -68.0
        # The half-sine cell 15 degrees wide peaks 0.25 degrees from -68, so
        # each edge lies (15 / pi) arcsin(0.1 p) inside its node, p being the
        # gridded peak; linear interpolation between gridpoints moves the
        # northern edge by 0.09 degrees.
        inset = 15.0 * np.arcsin(0.1 * np.sin(np.pi * 7.25 / 15.0)) / np.pi
        assert lower_cell.sel(lat="edge_south").item() == pytest.approx(
            -75.25 + inset, abs=0.1
        )
        assert lower_cell.sel(lat="edge_north").item() == pytest.approx(
            -60.25 - inset, abs=0.1
        )

    @pytest.mark.parametrize("extra_dim", [False, True], ids=["lat_sigma", "with_time"])
    def test_warnings_point_at_the_caller(self, extra_dim: bool) -> None:
        sf = _sigma_half_sine_cells(_POLAR_CELL_AT_GRID_END)
        if extra_dim:
            sf = sf.expand_dims({TIME_STR: [0, 1]})
        with pytest.warns(UserWarning) as record:
            cell_edges_sigma(sf)
        assert len(_dropped_messages(record)) == (2 if extra_dim else 1)
        assert {w.filename for w in record} == {__file__}


def _time_model_streamfunc() -> xr.DataArray:
    """Two-cell fields of 3 times by 2 models, each with its own edge latitude.

    The field also has a non-index coordinate, season, along time.
    """
    edge_lats = [[30.0, 26.0], [22.0, 34.0], [28.0, 18.0]]
    sf = xr.concat(
        [
            xr.concat([_make_streamfunc(edge) for edge in row], dim="model")
            for row in edge_lats
        ],
        dim=TIME_STR,
    )
    return sf.assign_coords(
        {
            TIME_STR: [10, 20, 30],
            "model": ["a", "b"],
            "season": (TIME_STR, ["x", "y", "z"]),
        }
    )


_EACH_ELEMENT_CALLS: dict[str, tuple[Callable[..., Any], dict[str, Any]]] = {
    "had_cell_strength": (had_cell_strength, {}),
    "had_cell_strength_interp": (had_cell_strength, {"do_interp_lat": True}),
    "had_cells_strength": (had_cells_strength, {}),
    "had_cells_strength_interp_avg": (
        had_cells_strength,
        {"do_interp_lat": True, "do_avg_vert": True},
    ),
    "had_cells_shared_edge": (had_cells_shared_edge, {}),
    "had_cell_edge": (had_cell_edge, EDGE_KWARGS),
    "had_cells_south_edge": (had_cells_south_edge, EDGE_KWARGS),
    "had_cells_north_edge": (had_cells_north_edge, EDGE_KWARGS),
    "had_cells_edges": (had_cells_edges, EDGE_KWARGS),
}


class TestEachElement:
    """Every dimension besides latitude and level is handled per element."""

    @pytest.mark.parametrize("name", list(_EACH_ELEMENT_CALLS))
    def test_matches_each_slice(self, name: str) -> None:
        # The result at each time and model equals the function applied to
        # that slice alone, and keeps the field's time, model and season
        # coordinates.
        func, kwargs = _EACH_ELEMENT_CALLS[name]
        sf = _time_model_streamfunc()
        whole = func(sf, **kwargs)
        results = whole if isinstance(whole, list) else [whole]
        for res in results:
            assert res.dims[:2] == (TIME_STR, "model")
            assert res[TIME_STR].values.tolist() == [10, 20, 30]
            assert res["model"].values.tolist() == ["a", "b"]
            assert res["season"].values.tolist() == ["x", "y", "z"]
        for i_time in range(3):
            for i_model in range(2):
                sel = {TIME_STR: i_time, "model": i_model}
                alone = func(sf.isel(sel), **kwargs)
                alone_list = alone if isinstance(alone, list) else [alone]
                for res, one in zip(results, alone_list, strict=True):
                    np.testing.assert_array_equal(res.isel(sel).values, np.asarray(one))
                    if isinstance(one, xr.DataArray) and LAT_STR in one.coords:
                        np.testing.assert_array_equal(
                            res.isel(sel)[LAT_STR].values, one[LAT_STR].values
                        )

    def test_strengths_known_values(self) -> None:
        # Each slice is a pair of opposite cells, so its southern cell is the
        # slice's minimum and its northern cell its maximum, found here with
        # raw numpy over the flattened level and latitude.
        sf = _time_model_streamfunc().transpose(TIME_STR, "model", LEV_STR, LAT_STR)
        flat = sf.values.reshape(3, 2, -1)
        i_lev, i_lat = np.unravel_index(flat.argmax(axis=-1), sf.shape[2:])
        strength = had_cell_strength(sf)
        np.testing.assert_array_equal(strength.values, flat.max(axis=-1))
        np.testing.assert_array_equal(
            strength[LAT_STR].values, sf[LAT_STR].values[i_lat]
        )
        np.testing.assert_array_equal(
            strength[LEV_STR].values, sf[LEV_STR].values[i_lev]
        )
        cells = had_cells_strength(sf)
        assert cells.dims == (TIME_STR, "model", "cell")
        np.testing.assert_array_equal(
            cells.sel(cell="had_cell_sh").values, flat.min(axis=-1)
        )
        np.testing.assert_array_equal(
            cells.sel(cell="had_cell_nh").values, flat.max(axis=-1)
        )

    def test_dimension_without_coordinate(self) -> None:
        sf = _time_model_streamfunc().isel(model=0).drop_vars([TIME_STR, "season"])
        result = had_cell_strength(sf)
        assert result.dims == (TIME_STR,)
        assert TIME_STR not in result.coords
        expected = [had_cell_strength(sf.isel({TIME_STR: i})).item() for i in range(3)]
        assert result.values.tolist() == expected

    def test_core_dimensions_first_and_extra_of_length_one(self) -> None:
        # The extra dimensions come first in the result, in the field's
        # order, wherever latitude and level sit in the field.
        sf = (
            _time_model_streamfunc()
            .isel({TIME_STR: [0]})
            .transpose(LAT_STR, "model", TIME_STR, LEV_STR)
        )
        result = had_cells_strength(sf)
        assert result.dims == ("model", TIME_STR, "cell")
        assert result.shape == (2, 1, 2)

    def test_edge_at_pole_in_one_element(self) -> None:
        # At the second time the northern cell stays at its peak all the way
        # to the pole, so the northern edge is placed there and returned as a
        # float, while the first time's edge is a DataArray.
        sf_poleward = _make_streamfunc(30.0)
        lat = sf_poleward[LAT_STR]
        peak = sf_poleward.max(LAT_STR)
        sf_poleward = sf_poleward.where(lat < 15.0, peak)
        alone = had_cells_north_edge(sf_poleward)
        assert alone == 90.0 and not isinstance(alone, xr.DataArray)
        first = had_cells_north_edge(_make_streamfunc(30.0))
        sf = xr.concat([_make_streamfunc(30.0), sf_poleward], dim=TIME_STR)
        result = had_cells_north_edge(sf)
        assert isinstance(result, xr.DataArray)
        assert result.values.tolist() == [float(first), 90.0]
        # Only the first time's edge carries a latitude coordinate, so none
        # may be kept for the whole result, which would label the pole edge
        # with the first time's latitude.
        assert LAT_STR not in result.coords

    def test_cell_edges_sigma_cells_by_position(self) -> None:
        # The second time has a southern polar cell that the first lacks, so
        # every other cell sits one position further north there, and a label
        # such as "3" names different cells at the two times. So the cells are
        # numbered by position at each time, as found alone, with their labels
        # in cell_label and NaN where a time has fewer cells.
        lats = np.arange(-85.0, 86.0, 1.0) + _SIGMA_NODE_OFFSET
        sigmas = np.linspace(0.1, 1.0, 19)
        lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
        sigma = xr.DataArray(sigmas, dims=[SIGMA_STR], coords={SIGMA_STR: sigmas})
        vert = cast(xr.DataArray, np.sin(np.pi * (sigma - 0.1) / 0.9))
        cells = [
            (-1e10, -11.0, 10.0),
            (1.1e10, 11.0, 10.0),
            (0.3e10, -40.0, 12.0),
            (-0.35e10, 40.0, 12.0),
        ]
        polar = [(-0.2e10, -70.0, 10.0)]
        fields = [
            sum(peak * _bump(lat, center, width) for peak, center, width in cs) * vert
            for cs in (cells, cells + polar)
        ]
        result = cell_edges_sigma(xr.concat(fields, dim=TIME_STR))
        assert result.sizes["cell"] == 5
        for i_time, field in enumerate(fields):
            alone = cell_edges_sigma(field)
            n_cells = alone.sizes["cell"]
            got = result.isel({TIME_STR: i_time})
            np.testing.assert_array_equal(
                got.isel(cell=slice(0, n_cells)).values, alone.values
            )
            np.testing.assert_array_equal(
                got["cell_strength"].values[:n_cells], alone["cell_strength"].values
            )
            assert got["cell_label"].values[:n_cells].tolist() == (
                alone["cell"].values.tolist()
            )
            assert bool(got.isel(cell=slice(n_cells, None)).isnull().all())

    def test_coordinates_gain_the_extra_dimension(self) -> None:
        # Two times whose maxima share a latitude and level: the center's
        # latitude and level still come back along time.
        sf = xr.concat([_gauss_cell(10.0), _gauss_cell(10.0, 2e11)], dim=TIME_STR)
        result = had_cell_strength(sf)
        assert result[LAT_STR].dims == (TIME_STR,)
        assert result[LEV_STR].dims == (TIME_STR,)
        assert result[LAT_STR].values.tolist() == [9.0, 9.0]

    def test_coordinate_along_time_and_latitude(self) -> None:
        # A coordinate spanning time and latitude is taken at each time's
        # maximum, like the latitude itself: here it is 100 * time + lat.
        sf = xr.concat([_gauss_cell(9.0), _gauss_cell(-31.0)], dim=TIME_STR)
        sf = sf.assign_coords(
            {TIME_STR: [0, 1], "aux": 100.0 * sf[TIME_STR] + sf[LAT_STR]}
        )
        result = had_cell_strength(sf)
        assert result["aux"].dims == (TIME_STR,)
        assert result["aux"].values.tolist() == [9.0, 69.0]

    def test_decorated_functions_keep_name_doc_and_signature(self) -> None:
        for func in (
            had_cell_strength,
            had_cells_strength,
            had_cells_shared_edge,
            had_cell_edge,
            had_cells_south_edge,
            had_cells_north_edge,
            had_cells_edges,
            cell_edges_sigma,
        ):
            params = list(inspect.signature(func).parameters)
            assert params[0] in ("streamfunc", "strmfunc"), func
            assert "lat_str" in params, func
            assert func.__doc__ and "Every" in func.__doc__, func
        assert had_cell_strength.__name__ == "had_cell_strength"

    def test_had_cell_strength_options_are_keyword_only(self) -> None:
        # Dropping dim shifted the positions of the later arguments, so a
        # positional call written for the old order is refused rather than
        # silently given a new meaning.
        old_order: tuple[Any, ...] = (None, 850.0)
        with pytest.raises(TypeError, match="had_cell_strength"):
            had_cell_strength(_gauss_cell(10.0), *old_order)

    def test_latitude_must_be_a_dimension(self) -> None:
        # A latitude not named by lat_str would otherwise be looped over.
        sf = _gauss_cell(10.0).rename({LAT_STR: "latitude"})
        with pytest.raises(ValueError, match="no latitude dimension 'lat'"):
            had_cell_strength(sf)
        with pytest.raises(ValueError, match="no latitude dimension 'lat'"):
            cell_edges_sigma(_sigma_streamfunc().rename({LAT_STR: "latitude"}))

    def test_dim_argument_removed(self) -> None:
        removed: dict[str, Any] = {"dim": TIME_STR}
        with pytest.raises(TypeError, match=r"had_cell_strength\(\).*'dim'"):
            had_cell_strength(_gauss_cell(10.0), **removed)

    def test_extra_dimension_of_length_zero_raises(self) -> None:
        sf = _gauss_cell(10.0).expand_dims({TIME_STR: [0]}).isel({TIME_STR: []})
        with pytest.raises(ValueError, match="'time' has length 0"):
            had_cell_strength(sf)

    @pytest.mark.parametrize("extra_dim", [False, True], ids=["lat_lev", "with_time"])
    def test_warnings_point_at_the_caller(self, extra_dim: bool) -> None:
        # A lone cell gives both of had_cells_strength's warnings under
        # do_interp_lat. Each is attributed to this file, the caller, not to
        # had_cell.py, however many of its frames the decorator adds.
        sf = _bump_cells_streamfunc([(1e11, 10.0, 30.0)])
        if extra_dim:
            sf = sf.expand_dims({TIME_STR: [0, 1]})
        with pytest.warns(UserWarning) as record:
            had_cells_strength(sf, do_interp_lat=True)
        messages = [str(w.message) for w in record]
        assert any("sense of circulation" in m for m in messages)
        assert any("Interpolating the latitude" in m for m in messages)
        assert {w.filename for w in record} == {__file__}

    def test_cell_edges_sigma_same_cells_keep_labels(self) -> None:
        # Where every time finds the same cells, the labels stay the index.
        sf = xr.concat([_sigma_streamfunc(amp) for amp in (1e10, 2e10)], dim=TIME_STR)
        result = cell_edges_sigma(sf, frac_thresh=0.2)
        assert result["cell"].values.tolist() == ["sh_hadley", "nh_hadley"]
        assert "cell_label" not in result.coords


class TestBciEdgeFormulas:
    """Known-value reconstructions of the BCI edge theory functions."""

    def test_fixed_ro_small_angle_lata0_known_value(self) -> None:
        result = fixed_ro_bci_edge_small_angle_lata0(
            0.3, ross_num=0.7, delta_v=0.2, c_descent=1.3
        )
        expected = 1.3 * np.rad2deg((0.3 * 0.2 / (2.0 * 0.7)) ** 0.25)
        assert result == pytest.approx(expected)

    def test_lin_ro_small_angle_lata0_known_value(self) -> None:
        result = lin_ro_bci_edge_small_angle_lata0(
            0.3, ross_ascent=0.4, ross_descent=0.8, delta_v=0.2
        )
        expected = np.rad2deg((3.0 * 0.3 * 0.2 / (2.0 * (0.4 + 2.0 * 0.8))) ** 0.25)
        assert result == pytest.approx(expected)

    def test_lin_ro_equal_rossby_numbers_give_uniform_edge(self) -> None:
        """With Ro_a = Ro_d the linear profile is the uniform one (#113)."""
        assert lin_ro_bci_edge_small_angle_lata0(
            0.46, ross_ascent=0.3, ross_descent=0.3, delta_v=0.125
        ) == pytest.approx(
            fixed_ro_bci_edge_small_angle_lata0(0.46, ross_num=0.3, delta_v=0.125),
            rel=1e-12,
        )

    def test_lin_ro_edge_balances_bci_wind(self) -> None:
        """At the edge, grad_bal's small-angle linear-Ro wind equals u_BCI.

        u_BCI is the small-angle form of Eq. 3 of Hill et al. (2025),
        g H delta_v / (2 Omega a lat^2), with lat in radians.
        """
        grav, height, rot_rate, radius = 3.71, 1.1e4, 7.088e-5, 3.39e6
        ross_ascent, ross_descent, delta_v = 0.9, 0.2, 0.15
        burg = grav * height / (rot_rate * radius) ** 2
        lat_d = lin_ro_bci_edge_small_angle_lata0(
            burg, ross_ascent=ross_ascent, ross_descent=ross_descent, delta_v=delta_v
        )
        u = u_lin_ro_small_ang(
            lat_d, 0.0, lat_d, ross_ascent, ross_descent, rot_rate, radius
        )
        u_bci = (
            grav * height * delta_v / (2 * rot_rate * radius * np.deg2rad(lat_d) ** 2)
        )
        assert u == pytest.approx(u_bci, rel=1e-12)

    def test_lat_ascent_eta0_approx(self) -> None:
        assert lat_ascent_eta0_approx(0.0) == 0.0
        result = lat_ascent_eta0_approx(0.5, c_ascent=1.2)
        expected = np.rad2deg(1.2 * (0.5 * 0.5) ** (1.0 / 3.0))
        assert result == pytest.approx(expected)
        # Odd in the thermal Rossby number.
        assert lat_ascent_eta0_approx(-0.5, c_ascent=1.2) == pytest.approx(-expected)

    def test_small_angle_known_value_explicit_ann_lat(self) -> None:
        result = fixed_ro_bci_edge_small_angle(
            10.0, lat_fixed_ro_ann=20.0, c_descent=1.1
        )
        lat_a2 = np.deg2rad(10.0) ** 2
        ann4 = np.deg2rad(20.0) ** 4
        expected = 1.1 * np.rad2deg(
            np.sqrt(0.5 * lat_a2 + np.sqrt(0.25 * lat_a2**2 + ann4))
        )
        assert result == pytest.approx(expected)

    def test_small_angle_equatorial_ascent_recovers_ann_lat(self) -> None:
        # With equatorial ascent, the formula collapses to
        # c_descent * lat_fixed_ro_ann exactly.
        assert fixed_ro_bci_edge_small_angle(
            0.0, lat_fixed_ro_ann=25.0
        ) == pytest.approx(25.0)
        assert fixed_ro_bci_edge_small_angle(
            0.0, lat_fixed_ro_ann=25.0, c_descent=1.2
        ) == pytest.approx(30.0)

    def test_small_angle_known_value_burger_path(self) -> None:
        # Planetary Burger number path with all non-default planet params.
        height, grav, rot_rate, radius = 1.5e4, 10.0, 8e-5, 5e6
        result = fixed_ro_bci_edge_small_angle(
            8.0,
            ross_num=0.9,
            delta_v=0.15,
            c_descent=1.05,
            height=height,
            grav=grav,
            rot_rate=rot_rate,
            radius=radius,
        )
        burg = height * grav / (rot_rate * radius) ** 2
        ann4 = burg * 0.15 / (2.0 * 0.9)
        lat_a2 = np.deg2rad(8.0) ** 2
        expected = 1.05 * np.rad2deg(
            np.sqrt(0.5 * lat_a2 + np.sqrt(0.25 * lat_a2**2 + ann4))
        )
        assert result == pytest.approx(expected)

    _SUPERCRIT = dict(
        max_lat=60.0,
        c_ascent=1.1,
        c_descent=0.9,
        delta_v=0.3,
        delta_h=0.1,
        ross_num=0.8,
    )

    def test_supercrit_ascent_known_value(self) -> None:
        """Eq. 15 of Hill et al. (2022) with the factor of 2 restored (#113).

        Restoring it in the critical wind turns the coefficient 2^(4/3) / c_a^4
        into 2^(1/3) / c_a^4.
        """
        burg_delh_sin_latmax = 0.4
        kw = self._SUPERCRIT
        result = fixed_ro_bci_edge_supercrit_ascent(burg_delh_sin_latmax, **kw)
        lat_ascent = kw["c_ascent"] * (0.5 * burg_delh_sin_latmax) ** (1.0 / 3.0)
        term1 = 2.0 ** (1.0 / 3.0) / kw["c_ascent"] ** 4
        term2 = kw["delta_v"] / (kw["delta_h"] * np.sin(np.deg2rad(kw["max_lat"])))
        term3 = 1.0 / (kw["ross_num"] * burg_delh_sin_latmax ** (1.0 / 3.0))
        expected = kw["c_descent"] * np.rad2deg(
            lat_ascent * np.sqrt(0.5 + np.sqrt(0.25 + term1 * term2 * term3))
        )
        assert result == pytest.approx(expected, rel=1e-12)

    def test_supercrit_ascent_is_composition(self) -> None:
        """The ascent latitude fed to the small-angle edge, Bu from Ro_th."""
        kw = self._SUPERCRIT
        burg_delh_sin_latmax = 0.6
        lat_ascent = lat_ascent_eta0_approx(
            burg_delh_sin_latmax, c_ascent=kw["c_ascent"]
        )
        burg = burg_delh_sin_latmax / (
            kw["delta_h"] * np.sin(np.deg2rad(kw["max_lat"]))
        )
        expected = fixed_ro_bci_edge_small_angle(
            lat_ascent,
            burg_num=burg,
            ross_num=kw["ross_num"],
            delta_v=kw["delta_v"],
            c_descent=kw["c_descent"],
        )
        result = fixed_ro_bci_edge_supercrit_ascent(burg_delh_sin_latmax, **kw)
        assert result == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("max_lat_sign", [1.0, -1.0])
    def test_supercrit_ascent_southern_forcing_mirrors(
        self, max_lat_sign: float
    ) -> None:
        """Negative Ro_th gives the mirrored edge, whatever the sign of max_lat."""
        kw = dict(self._SUPERCRIT, max_lat=max_lat_sign * self._SUPERCRIT["max_lat"])
        north = fixed_ro_bci_edge_supercrit_ascent(0.4, **self._SUPERCRIT)
        south = fixed_ro_bci_edge_supercrit_ascent(-0.4, **kw)
        assert north > 0
        assert south == pytest.approx(-north, rel=1e-12)

    def test_supercrit_ascent_zero_forcing(self) -> None:
        assert fixed_ro_bci_edge_supercrit_ascent(0.0, **self._SUPERCRIT) == 0.0

    def test_supercrit_ascent_array(self) -> None:
        ro_th = np.array([-0.6, 0.2, 0.6])
        result = np.asarray(
            fixed_ro_bci_edge_supercrit_ascent(ro_th, **self._SUPERCRIT)
        )
        expected = [
            float(np.asarray(fixed_ro_bci_edge_supercrit_ascent(r, **self._SUPERCRIT)))
            for r in ro_th
        ]
        # assert_allclose counts NaN as equal to NaN, so check finiteness too.
        assert np.isfinite(result).all()
        assert result[0] == pytest.approx(-result[2], rel=1e-12)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_supercrit_ascent_defaults_known_value(self) -> None:
        """At the defaults: max_lat = 90, delta_h = 1/15, c_a = c_d = Ro = 1."""
        burg_delh_sin_latmax = 0.4
        result = fixed_ro_bci_edge_supercrit_ascent(burg_delh_sin_latmax)
        lat_ascent = (0.5 * burg_delh_sin_latmax) ** (1.0 / 3.0)
        term = (
            2.0 ** (1.0 / 3.0)
            * DELTA_V
            / ((1.0 / 15.0) * burg_delh_sin_latmax ** (1.0 / 3.0))
        )
        expected = np.rad2deg(lat_ascent * np.sqrt(0.5 + np.sqrt(0.25 + term)))
        assert result == pytest.approx(expected, rel=1e-12)

    def test_supercrit_edge_balances_small_angle_bci_wind(self) -> None:
        """At c_descent = 1, small-angle u_Ro equals small-angle u_BCI at the edge.

        u_BCI is the small-angle Eq. 3 of Hill et al. (2025), Bu Omega a
        delta_v / (2 lat^2), with Bu = Ro_th / (delta_h sin(max_lat)).
        """
        kw = dict(self._SUPERCRIT, c_descent=1.0)
        burg_delh_sin_latmax = 0.4
        edge = float(
            np.asarray(fixed_ro_bci_edge_supercrit_ascent(burg_delh_sin_latmax, **kw))
        )
        lat_ascent = np.rad2deg(
            kw["c_ascent"] * (0.5 * burg_delh_sin_latmax) ** (1.0 / 3.0)
        )
        burg = burg_delh_sin_latmax / (
            kw["delta_h"] * np.sin(np.deg2rad(kw["max_lat"]))
        )
        u_ro = u_unif_ro_small_ang(edge, lat_ascent, kw["ross_num"])
        u_bci = (
            burg
            * ROT_RATE_EARTH
            * RAD_EARTH
            * kw["delta_v"]
            / (2 * np.deg2rad(edge) ** 2)
        )
        assert u_ro == pytest.approx(u_bci, rel=1e-12)


def _bci_polynomial_residual(lat: float, ascentlat: float, h00lat: float) -> float:
    """Raw-numpy version of the fixed-Ro BCI edge condition."""
    sinlat = np.sin(np.deg2rad(lat))
    coslat = np.cos(np.deg2rad(lat))
    return float(
        sinlat**4
        - np.sin(np.deg2rad(ascentlat)) ** 2 * sinlat**2
        - np.deg2rad(h00lat) ** 4 * coslat**2
    )


def _bci_root(ascentlat: float, h00lat: float) -> float:
    """Root in (0, 90) degrees of the raw edge condition, by scipy's brentq."""
    root = brentq(
        _bci_polynomial_residual, 1e-9, 90.0, args=(ascentlat, h00lat), xtol=1e-14
    )
    return float(root)


def _bci_root_60_digits(ascentlat: float, x: float) -> float:
    """Edge from s^2 + (X - sin^2 lat_a) s - X = 0, s = sin^2, at 60 digits.

    Converted to degrees by whichever of arcsin(sqrt(s)) and
    arccos(sqrt(1 - s)) is well conditioned, so that large X, with s near 1,
    keeps full precision.
    """
    with localcontext() as ctx:
        ctx.prec = 60
        sin2_a = Decimal(float(np.sin(np.deg2rad(ascentlat)) ** 2))
        x_dec = Decimal(x)
        s = ((sin2_a - x_dec) + ((sin2_a - x_dec) ** 2 + 4 * x_dec).sqrt()) / 2
        if s < Decimal("0.5"):
            return float(np.rad2deg(np.arcsin(np.sqrt(float(s)))))
        return float(np.rad2deg(np.arccos(np.sqrt(float(1 - s)))))


class TestFixedRoBciEdge:
    """The closed-form edge solves the full-angle BCI edge condition (#111)."""

    @pytest.mark.parametrize("ascentlat", [0.0, 5.0, 20.0, 35.0])
    @pytest.mark.parametrize("h00lat", [2.0, 20.0, 45.0, 80.0])
    def test_matches_brentq_root(self, ascentlat: float, h00lat: float) -> None:
        edge = fixed_ro_bci_edge(ascentlat, lat_fixed_ro_ann=h00lat)
        assert edge == pytest.approx(_bci_root(ascentlat, h00lat), abs=1e-9)

    def test_dataarray_input_keeps_dims(self) -> None:
        ascentlats = xr.DataArray(
            [5.0, 15.0], dims=["case"], coords={"case": [0, 1]}, name="lat"
        )
        edges = fixed_ro_bci_edge(ascentlats, lat_fixed_ro_ann=20.0)
        assert isinstance(edges, xr.DataArray)
        assert edges.dims == ("case",)
        for lat_a, lat_edge in zip(ascentlats.values, edges.values):
            resid = _bci_polynomial_residual(float(lat_edge), float(lat_a), 20.0)
            assert abs(resid) < 1e-12, (lat_a, lat_edge, resid)
        # Edge moves poleward with the ascent latitude.
        assert edges.values[1] > edges.values[0]

    def test_default_burger_number_path(self) -> None:
        edge = float(np.asarray(fixed_ro_bci_edge(0.0)))
        burg = HEIGHT_TROPO * GRAV_EARTH / (ROT_RATE_EARTH * RAD_EARTH) ** 2
        h00lat = np.rad2deg((burg * DELTA_V / 2.0) ** 0.25)
        assert edge == pytest.approx(_bci_root(0.0, float(h00lat)), abs=1e-9)

    def test_burger_path_nondefault_params(self) -> None:
        """c_descent multiplies the solution of the c_descent = 1 condition.

        Hill et al. (2025, p. 5506) define phi_d = c_d phi_BCI. With the
        defaults, ross_num and c_descent are multiplicative identities and
        dropping their forwarding would go unnoticed.
        """
        height, grav, rot_rate, radius = 1.5e4, 10.0, 8e-5, 5e6
        ross_num, c_descent, delta_v = 0.8, 1.2, 0.15
        ascentlat = 5.0
        edge = float(
            np.asarray(
                fixed_ro_bci_edge(
                    ascentlat,
                    ross_num=ross_num,
                    c_descent=c_descent,
                    delta_v=delta_v,
                    height=height,
                    grav=grav,
                    rot_rate=rot_rate,
                    radius=radius,
                )
            )
        )
        burg = height * grav / (rot_rate * radius) ** 2
        h00lat = np.rad2deg((burg * delta_v / (2.0 * ross_num)) ** 0.25)
        assert edge == pytest.approx(
            c_descent * _bci_root(ascentlat, float(h00lat)), abs=1e-9
        )

    def test_explicit_burg_num_value(self) -> None:
        """A given burg_num, at non-default ross_num, delta_v and c_descent."""
        burg, ross_num, delta_v, c_descent = 0.37, 0.6, 0.2, 1.1
        edge = float(
            np.asarray(
                fixed_ro_bci_edge(
                    12.0,
                    burg_num=burg,
                    ross_num=ross_num,
                    delta_v=delta_v,
                    c_descent=c_descent,
                )
            )
        )
        h00lat = float(np.rad2deg((burg * delta_v / (2.0 * ross_num)) ** 0.25))
        assert edge == pytest.approx(c_descent * _bci_root(12.0, h00lat), abs=1e-9)

    def test_lat_fixed_ro_ann_overrides_other_args(self) -> None:
        """Given lat_fixed_ro_ann, burg_num, ross_num, delta_v and the planet
        are unused, as the docstring says."""
        ref = fixed_ro_bci_edge(12.0, lat_fixed_ro_ann=25.0)
        other = fixed_ro_bci_edge(
            12.0,
            lat_fixed_ro_ann=25.0,
            burg_num=5.0,
            ross_num=0.3,
            delta_v=0.5,
            height=1e3,
            grav=3.0,
            rot_rate=1e-5,
            radius=1e6,
        )
        assert other == pytest.approx(ref, rel=1e-14)

    @pytest.mark.parametrize("ascentlat", [0.0, 8.0, 25.0])
    def test_edge_balances_bci_wind(self, ascentlat: float) -> None:
        """At the edge, grad_bal's u_Ro equals dynamics' u_BCI.

        These are Eqs. 2 and 3 of Hill et al. (2025), on a planet other than
        Earth, so the test also checks how X is built from the winds.
        """
        height, grav, rot_rate, radius = 1.2e4, 9.0, 6e-5, 5.5e6
        ross_num, delta_v = 0.6, 0.2
        edge = fixed_ro_bci_edge(
            ascentlat,
            ross_num=ross_num,
            delta_v=delta_v,
            height=height,
            grav=grav,
            rot_rate=rot_rate,
            radius=radius,
        )
        u_ro = u_unif_ro(edge, ascentlat, ross_num, rot_rate=rot_rate, radius=radius)
        u_bci = u_bci_2layer_qg(
            edge,
            height=height,
            delta_v=delta_v,
            grav=grav,
            rot_rate=rot_rate,
            radius=radius,
        )
        assert u_ro == pytest.approx(u_bci, rel=1e-12)

    @pytest.mark.parametrize("ascentlat", [0.0, 30.0])
    @pytest.mark.parametrize("x", [1e6, 1e12, 1e18])
    def test_large_x_keeps_precision(self, ascentlat: float, x: float) -> None:
        """Near the pole the edge keeps full precision, for any X.

        The textbook root formula for s = sin^2(lat) cancels for large X; at
        X = 1e18 it returns 0 degrees in place of 89.99999994.
        """
        edge = fixed_ro_bci_edge(ascentlat, lat_fixed_ro_ann=np.rad2deg(x**0.25))
        assert edge == pytest.approx(_bci_root_60_digits(ascentlat, x), abs=1e-10)

    def test_c_descent_multiplies_explicit_ann_lat_solution(self) -> None:
        base = fixed_ro_bci_edge(20.0, lat_fixed_ro_ann=30.0)
        scaled = fixed_ro_bci_edge(20.0, lat_fixed_ro_ann=30.0, c_descent=1.2)
        assert scaled == pytest.approx(1.2 * base, rel=1e-12)
        assert base == pytest.approx(_bci_root(20.0, 30.0), abs=1e-9)

    def test_only_magnitude_of_ascent_enters(self) -> None:
        assert fixed_ro_bci_edge(-15.0, lat_fixed_ro_ann=25.0) == pytest.approx(
            fixed_ro_bci_edge(15.0, lat_fixed_ro_ann=25.0), rel=1e-12
        )

    def test_small_angle_limit(self) -> None:
        """Near the equator the full-angle edge approaches Eq. 4 of Hill et al.

        The relative gap between the two is of second order in the angles, so
        doubling both latitudes multiplies it by 4.
        """

        def rel_gap(scale: float) -> float:
            full = fixed_ro_bci_edge(0.5 * scale, lat_fixed_ro_ann=0.6 * scale)
            small = fixed_ro_bci_edge_small_angle(
                0.5 * scale, lat_fixed_ro_ann=0.6 * scale
            )
            return float(np.asarray((small - full) / full))

        assert 0 < rel_gap(1.0) < 1e-5
        assert rel_gap(2.0) / rel_gap(1.0) == pytest.approx(4.0, rel=1e-2)

    def test_nan_input_gives_nan(self) -> None:
        """A NaN ascent latitude or Rossby number gives NaN, not an error (#108)."""
        edges = np.asarray(
            fixed_ro_bci_edge(
                np.array([5.0, np.nan, 15.0]), burg_num=2.586, ross_num=0.3
            )
        )
        assert np.isnan(edges[1])
        h00lat = float(np.rad2deg((2.586 * DELTA_V / (2.0 * 0.3)) ** 0.25))
        assert edges[0] == pytest.approx(_bci_root(5.0, h00lat), abs=1e-9)
        assert edges[2] == pytest.approx(_bci_root(15.0, h00lat), abs=1e-9)
        assert np.isnan(fixed_ro_bci_edge(10.0, burg_num=2.586, ross_num=np.nan))

    def test_takes_no_solver_bracket(self) -> None:
        """The closed form needs no bracketing guesses (#111)."""
        params = inspect.signature(fixed_ro_bci_edge).parameters
        assert "zero_bounds_guess_range" not in params


@pytest.mark.parametrize(
    "func", [lat_ascent_eta0_approx, fixed_ro_bci_edge_supercrit_ascent]
)
def test_takes_burg_delh_sin_latmax(func: Any) -> None:
    """Callers pass Bu * delta_h * sin(lat_max) by this keyword (#115)."""
    params = inspect.signature(func).parameters
    assert "burg_delh_sin_latmax" in params
    assert not {"therm_ro", "therm_ross"} & set(params)
