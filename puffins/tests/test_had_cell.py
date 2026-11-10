"""Tests for had_cell module."""

import inspect
import warnings
from collections.abc import Callable, Iterable
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
    @pytest.mark.parametrize(
        ("south_end", "north_end"),
        [(-89.5, 89.5), (-85.0, 85.0), (-89.5, 84.5)],
        ids=["near_pole", "far", "near_south_pole_only"],
    )
    def test_edge_past_the_end_of_the_grid(
        self, hemisphere: float, south_end: float, north_end: float
    ) -> None:
        """A cell above the threshold at the grid's end ends at the pole only near it.

        The cell is a half sine wave from 0.25 to 120.25 degrees poleward of
        the equator, of the sign of a Hadley cell in that hemisphere, so at
        the grid's end it is still 0.72 or more of its gridded peak. On a 1-degree
        grid ending at 89.5, within one spacing of the pole, the edge is
        interpolated between 89.5 and a zero placed at the pole. On one
        ending at 85 or 84.5, five or more spacings from the pole, the edge
        is NaN, and the cell is kept with a warning. The third grid ends at
        89.5S and 84.5N, so each end is judged by its own pole. The southern
        case runs the search with the latitudes flipped.
        """
        lats = np.arange(south_end, north_end + 0.5, 1.0)
        last_lat = abs(south_end if hemisphere < 0 else north_end)
        sigmas = np.linspace(0.1, 1.0, 19)
        lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
        sigma = xr.DataArray(sigmas, dims=[SIGMA_STR], coords={SIGMA_STR: sigmas})
        x = hemisphere * lat - _SIGMA_NODE_OFFSET
        merid = hemisphere * xr.where(x > 0, np.sin(np.pi * x / 120.0), 0.0)
        sf = 1e10 * np.sin(np.pi * (sigma - 0.1) / 0.9) * merid
        label = "nh_hadley" if hemisphere > 0 else "sh_hadley"
        poleward = "edge_north" if hemisphere > 0 else "edge_south"
        if last_lat == 89.5:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                cells = cell_edges_sigma(sf)
            # The gridded peak is at 60.5, 60.25 from the node.
            at_end = np.sin(np.pi * 89.25 / 120.0) / np.sin(np.pi * 60.25 / 120.0)
            expected = hemisphere * (89.5 + 0.5 * (at_end - 0.1) / at_end)
            edge = cells.sel(cell=label, lat=poleward).item()
            assert edge == pytest.approx(expected, rel=1e-12)
        else:
            with pytest.warns(UserWarning) as record:
                cells = cell_edges_sigma(sf)
            assert np.isnan(cells.sel(cell=label, lat=poleward).item())
            assert _dropped_messages(record) == []
            messages = _cut_messages(record)
            assert len(messages) == 1
            side, end = (
                ("northern", "last") if hemisphere > 0 else ("southern", "first")
            )
            assert (
                f"its {side} edge lies beyond the {end} latitude of the grid"
                in (messages[0])
            )
        assert list(cells["cell"].values) == [label]
        # The gridded peak is at 60.5 on the grids of half-degree latitudes.
        assert cells.sel(cell=label, lat="center").item() == hemisphere * (
            60.5 if south_end == -89.5 else 60.0
        )

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


def _dropped_messages(record: Iterable[warnings.WarningMessage]) -> list[str]:
    """The messages of the warnings for cells that cell_edges_sigma dropped."""
    messages = [str(w.message) for w in record]
    return [m for m in messages if m.startswith("cell_edges_sigma dropped")]


def _cut_messages(record: Iterable[warnings.WarningMessage]) -> list[str]:
    """The messages of the warnings for cells kept with a NaN edge."""
    messages = [str(w.message) for w in record]
    return [m for m in messages if m.startswith("cell_edges_sigma kept")]


# Southern polar, Ferrel and Hadley cells and a northern Hadley cell. The
# polar cell, 50.5 degrees wide, peaks at 85S, the first latitude of the grid
# of ``_sigma_half_sine_cells``; the other cells are 30 degrees wide.
_POLAR_CELL_AT_GRID_END = [
    (-110.25, -59.75, -0.3),
    (-59.75, -29.75, 0.4),
    (-29.75, 0.25, -1.0),
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
            (-89.75, -59.75, -polar),
            (-59.75, -29.75, 0.4),
            (-29.75, 0.25, -1.0),
            (0.25, 30.25, 1.0),
            (30.25, 60.25, -0.4),
            (60.25, 90.25, polar),
        ]
    )


class TestCellEdgesSigmaDroppedCells:
    """The cells cell_edges_sigma drops, and the warnings for those it reports."""

    @pytest.mark.parametrize(
        ("nan_lats", "kept", "dropped"),
        [
            (
                [20.0],
                ["sh_hadley"],
                [(15.0, "its northern edge"), (21.0, "its southern edge")],
            ),
            (
                [10.0, 20.0],
                ["sh_hadley"],
                [
                    (9.0, "its northern edge"),
                    (15.0, "its southern and northern edges"),
                    (21.0, "its southern edge"),
                ],
            ),
            ([31.0], ["sh_hadley", "nh_hadley"], []),
            ([60.0], ["sh_hadley", "nh_hadley"], []),
        ],
    )
    def test_nan_column(
        self,
        nan_lats: list[float],
        kept: list[str],
        dropped: list[tuple[float, str]],
    ) -> None:
        """A NaN column splitting a cell drops each part, with a warning.

        The field is sin(pi * (lat - 0.25) / 30) between -29.75 and 30.25,
        two Hadley cells 30 degrees wide with edges where it falls to 0.1 of
        its gridded peak, at sigma 0.55. A NaN is in neither cell, so a NaN at
        20N splits the northern cell into two, each with an edge beside the
        other half, which beyond the NaN is still above 0.1 of its peak; NaN
        at 10N and 20N split it into three, the middle one with neither edge.
        Each such part is dropped with a warning naming the side of the edge
        not found. NaN outside both cells, at 31N or 60N, changes nothing. A
        NaN column holding an edge is tested separately.
        """
        sf = _sigma_half_sine_cells([(-29.75, 0.25, -1.0), (0.25, 30.25, 1.0)])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            full = cell_edges_sigma(sf)
        masked = sf.where(~sf[LAT_STR].isin(nan_lats))
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
        assert "skip_reason" not in cells.coords
        assert "cut_reason" not in cells.coords
        assert "cut_reason" not in full.coords
        inset = _sigma_edge_inset(0.1)
        for label in kept:
            south, north = (-29.75, 0.25) if label == "sh_hadley" else (0.25, 30.25)
            cell = cells.sel(cell=label)
            np.testing.assert_array_equal(cell.values, full.sel(cell=label).values)
            assert cell.sel(lat="edge_south").item() == pytest.approx(
                south + inset, abs=0.02
            )
            assert cell.sel(lat="edge_north").item() == pytest.approx(
                north - inset, abs=0.02
            )
        assert len(messages) == len(dropped)
        for center, phrase in dropped:
            strength = 1e10 * np.sin(np.pi * (center - 0.25) / 30.0)
            matching = [m for m in messages if f"at latitude {center:g} and" in m]
            assert len(matching) == 1
            assert f"and sigma 0.55, of strength {strength:.3g}:" in matching[0]
            assert f"{phrase} could not be found" in matching[0]

    @pytest.mark.parametrize(
        ("nan_lats", "nan_sigma", "label", "edge"),
        [
            ([29.0], None, "nh_hadley", "edge_north"),
            ([30.0], None, "nh_hadley", "edge_north"),
            ([28.0, 29.0, 30.0], None, "nh_hadley", "edge_north"),
            ([-29.0], None, "sh_hadley", "edge_south"),
            ([29.0], 0.55, "nh_hadley", "edge_north"),
        ],
        ids=["column_29N", "column_30N", "columns_28N_to_30N", "column_29S", "point"],
    )
    def test_edge_in_a_run_of_nan_kept(
        self, nan_lats: list[float], nan_sigma: float | None, label: str, edge: str
    ) -> None:
        """A cell whose edge lies in a run of NaN is kept, with NaN for that edge.

        The two Hadley cells of ``test_nan_column`` have their northern edge
        at 29.29N and southern at 28.79S. Beyond NaN at 29N, 30N or 29S, or at
        28N to 30N, the cell is below 0.1 of its gridded peak: 0.026 at 30N,
        and zero at 31N and 30S. So the edge lies within the NaN, and the
        cell is kept, with NaN for that edge, a warning, and its other values
        as without the NaN. The same holds for a NaN at 29N only at sigma
        0.55, the level of the cell's maximum. These cells used to be
        dropped. A NaN column at 29N leaves 30N as a cell of its own, of 0.026
        times the strongest cell's strength, which the strength filter drops
        without a warning.
        """
        sf = _sigma_half_sine_cells([(-29.75, 0.25, -1.0), (0.25, 30.25, 1.0)])
        full = cell_edges_sigma(sf)
        mask = sf[LAT_STR].isin(nan_lats)
        if nan_sigma is not None:
            mask = mask & (np.abs(sf[SIGMA_STR] - nan_sigma) < 1e-9)
        with pytest.warns(UserWarning) as record:
            cells = cell_edges_sigma(sf.where(~mask))
        assert cells["cell"].values.tolist() == ["sh_hadley", "nh_hadley"]
        for name in cells["cell"].values:
            for lat in ("edge_south", "center", "edge_north"):
                got = cells.sel(cell=name, lat=lat).item()
                if (name, lat) == (label, edge):
                    assert np.isnan(got)
                else:
                    assert got == full.sel(cell=name, lat=lat).item()
        side = "northern" if edge == "edge_north" else "southern"
        center = full.sel(cell=label, lat="center").item()
        messages = _cut_messages(record)
        assert len(record) == len(messages) == 1
        assert f"at latitude {center:g} and sigma 0.55" in messages[0]
        assert messages[0].endswith(
            f": its {side} edge lies in a run of NaN beside the cell, beyond which "
            "the streamfunction is below frac_thresh of the cell's maximum."
        )

    @pytest.mark.parametrize("cos_factor", [False, True])
    def test_edge_in_a_run_of_nan_with_cos_factor(self, cos_factor: bool) -> None:
        """Whether an edge lies in a run of NaN is judged as the edge is found.

        A northern cell from 0.25 to 80.25 peaks at 40N, and a NaN column at
        77N lies between 76N and 78N. Without the NaN, the cell's northern
        edge is at 77.70N, where it falls to 0.1 of its peak, and at 79.63N
        with cos_factor, where, divided by cos(latitude), it falls to 0.1 of
        its value at 40N. At 78N, beyond the NaN, it is 0.088 of its peak, and
        0.325 once divided by cos(latitude). So only without cos_factor does
        the edge lie within the NaN, and the cell is kept with NaN for it;
        with cos_factor, the cell is dropped as one split by NaN.
        """
        sf = _sigma_half_sine_cells([(-29.75, 0.25, -1.0), (0.25, 80.25, 1.0)])
        full = cell_edges_sigma(sf, cos_factor=cos_factor)
        north = full.sel(cell="nh_hadley", lat="edge_north").item()
        assert (76.0 < north < 78.0) == (not cos_factor)
        with pytest.warns(UserWarning) as record:
            cells = cell_edges_sigma(
                sf.where(sf[LAT_STR] != 77.0), cos_factor=cos_factor
            )
        if cos_factor:
            assert cells["cell"].values.tolist() == ["sh_hadley"]
            assert _cut_messages(record) == []
            dropped = _dropped_messages(record)
            matching = [m for m in dropped if "at latitude 40 and" in m]
            assert len(matching) == 1
            assert ": its northern edge could not be found:" in matching[0]
        else:
            assert cells["cell"].values.tolist() == ["sh_hadley", "nh_hadley"]
            assert np.isnan(cells.sel(cell="nh_hadley", lat="edge_north").item())
            messages = _cut_messages(record)
            assert len(messages) == 1
            assert "at latitude 40 and" in messages[0]
            assert ": its northern edge lies in a run of NaN" in messages[0]

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
                    (-29.75, 0.25, -1.0),
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
        gridded peak. It is still 0.93 of that peak at 85N, five grid
        spacings from the pole, so its northern edge is NaN, with a warning.
        """
        sf = _sigma_half_sine_cells(
            [
                (-29.75, 0.25, -1.0),
                (0.25, 30.25, 1.0),
                (30.25, 60.25, -0.4),
                (60.25, 100.25, 0.3),
            ]
        ).sel({LAT_STR: slice(-80.0, None)})
        with pytest.warns(UserWarning) as record:
            cells = cell_edges_sigma(sf)
        assert _dropped_messages(record) == []
        messages = _cut_messages(record)
        assert len(messages) == 1
        assert "the cell centered at latitude 80 and" in messages[0]
        assert list(cells["cell"].values) == ["sh_hadley", "nh_hadley", "2", "3"]
        cell = cells.sel(cell="3")
        assert cell.sel(lat="center").item() == 80.0
        peak = np.sin(np.pi * 19.75 / 40.0)
        assert cell.sel(lat="edge_south").item() == pytest.approx(
            60.25 + 40.0 * np.arcsin(0.1 * peak) / np.pi, abs=0.01
        )
        assert np.isnan(cell.sel(lat="edge_north").item())

    def test_one_element_at_grid_end_keeps_the_others(self) -> None:
        """A cell at the grid's end in one element no longer fails the whole call.

        The two times differ in their Hadley cells' amplitudes, so each
        time's result can be told from the other's.
        """
        first = _sigma_half_sine_cells(
            [(-59.75, -29.75, 0.4), (-29.75, 0.25, -0.8), (0.25, 30.25, 0.8)]
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

    def test_no_deep_enough_cell_raises(self) -> None:
        # The only cell spans sigma 0.5 to 0.6, less than the default
        # cell_min_sigma_depth of 0.3.
        lats = np.arange(-85.0, 86.0, 1.0)
        sigmas = np.linspace(0.1, 1.0, 19)
        lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
        sigma = xr.DataArray(sigmas, dims=[SIGMA_STR], coords={SIGMA_STR: sigmas})
        shallow = xr.where((sigma > 0.47) & (sigma < 0.63), 1.0, 0.0)
        merid = xr.where(
            (lat > 0.25) & (lat < 30.25), np.sin(np.pi * (lat - 0.25) / 30.0), 0.0
        )
        sf = (1e10 * shallow * merid).transpose(SIGMA_STR, LAT_STR)
        with pytest.raises(
            ValueError,
            match="no cell found that spans at least cell_min_sigma_depth=0.3 in",
        ):
            cell_edges_sigma(sf)
        with pytest.raises(ValueError, match="cell_min_sigma_depth=0.5 in sigma"):
            cell_edges_sigma(sf, cell_min_sigma_depth=0.5)
        assert cell_edges_sigma(sf, cell_min_sigma_depth=0.05).sizes["cell"] == 1
        # An all-NaN field, as for a missing month, has no cell at all.
        with pytest.raises(
            ValueError, match="is NaN or zero at every latitude and sigma searched"
        ):
            cell_edges_sigma(sf * np.nan)

    def test_no_measurable_cell_raises(self) -> None:
        sf = _sigma_half_sine_cells([(-110.25, -60.25, -0.3)])
        with (
            pytest.warns(UserWarning, match="first or last latitude"),
            pytest.raises(ValueError, match="no cell remains"),
        ):
            cell_edges_sigma(sf)

    def test_strength_threshold_with_no_measurable_cell(self) -> None:
        """With no cell measurable, the strongest of them sets the strength threshold.

        Two cells peak at the grid's two ends, the northern one 0.001 times the
        strength of the southern one. The northern one falls below the
        threshold and is dropped silently; the southern one is dropped with a
        warning, and the call raises.
        """
        sf = _sigma_half_sine_cells([(-110.25, -60.25, -0.3), (60.25, 110.25, 3e-4)])
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always", UserWarning)
            with pytest.raises(ValueError, match="no cell remains"):
                cell_edges_sigma(sf)
        messages = _dropped_messages(record)
        assert len(record) == len(messages) == 1
        assert messages[0].startswith(
            "cell_edges_sigma dropped the cell centered at latitude -85 and"
        )

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
        weak cell gives no warning. The polar cells reach past the grid's end
        at 85, five spacings from the pole, so each kept one has a NaN
        poleward edge, with a warning.
        """
        sf = _six_sigma_cells(polar)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always", UserWarning)
            if min_frac_strength is None:
                cells = cell_edges_sigma(sf)
            else:
                cells = cell_edges_sigma(sf, min_frac_strength=min_frac_strength)
        assert _dropped_messages(record) == []
        assert len(_cut_messages(record)) == (2 if polar_kept else 0)
        centers = [-75.0, -45.0, -15.0, 15.0, 45.0, 75.0]
        expected = centers if polar_kept else centers[1:-1]
        assert cells.sel(lat="center").values.tolist() == expected
        assert "sh_hadley" in cells["cell"].values
        assert "nh_hadley" in cells["cell"].values

    @pytest.mark.parametrize("max_cells_per_sign", [None, 5, 3])
    def test_max_cells_per_sign(self, max_cells_per_sign: int | None) -> None:
        """Only the max_cells_per_sign strongest cells of each sign are kept.

        Ten alternating half-sine cells, each 17 degrees wide, with every
        maximum 0.25 degrees from a gridpoint, so their strengths rank as
        their amplitudes. The cells beyond the strongest
        ``max_cells_per_sign`` of each sign are dropped, with one warning
        giving their number and the first of them. The positive cells are
        all stronger than the negative ones, so keeping the strongest of each
        sign differs from keeping the strongest twice ``max_cells_per_sign``
        overall.
        """
        nodes = [-85.25 + 17.0 * k for k in range(11)]
        amps = [-0.2, 0.6, -0.4, 0.8, -0.5, 1.0, -0.3, 0.9, -0.15, 0.7]
        sf = _sigma_half_sine_cells(
            [(nodes[k], nodes[k + 1], amps[k]) for k in range(10)]
        )
        limit = 4 if max_cells_per_sign is None else max_cells_per_sign
        centers = np.round(np.array(nodes[:-1]) + 8.5)
        kept = set()
        for sign in (1.0, -1.0):
            same_sign = [k for k in range(10) if np.sign(amps[k]) == sign]
            kept |= set(sorted(same_sign, key=lambda k: -abs(amps[k]))[:limit])
        overall = set(sorted(range(10), key=lambda k: -abs(amps[k]))[: 2 * limit])
        assert (kept != overall) == (limit < 5)
        dropped = [centers[k] for k in range(10) if k not in kept]
        if dropped:
            with pytest.warns(UserWarning) as record:
                if max_cells_per_sign is None:
                    cells = cell_edges_sigma(sf)
                else:
                    cells = cell_edges_sigma(sf, max_cells_per_sign=max_cells_per_sign)
            messages = _dropped_messages(record)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                cells = cell_edges_sigma(sf, max_cells_per_sign=limit)
            messages = []
        assert cells.sel(lat="center").values.tolist() == [
            centers[k] for k in sorted(kept)
        ]
        assert len(dropped) == {None: 2, 5: 0, 3: 4}[max_cells_per_sign]
        assert len(messages) == (1 if dropped else 0)
        if dropped:
            assert messages[0].startswith(
                f"cell_edges_sigma dropped {len(dropped)} cells: more than {limit} "
                f"cells of its sign remain, and it is not among the {limit} "
                "strongest; raise max_cells_per_sign to keep it. The first is the "
                "cell centered at latitude "
            )
            # The first named is the strongest dropped positive cell.
            first = float(messages[0].split("centered at latitude ")[1].split()[0])
            positive = [k for k in range(10) if amps[k] > 0]
            strongest_dropped = sorted(positive, key=lambda k: -amps[k])[limit]
            assert first == centers[strongest_dropped]

    def test_max_cells_per_sign_with_one_sign_over_the_limit(self) -> None:
        """The limit applies to each sign, whatever the total number of cells.

        Five positive cells, each 14 degrees wide with a gap of two zero
        gridpoints between them, and one negative cell: six cells in all,
        fewer than twice the default limit of 4, yet the weakest positive cell,
        centered at 71N, is one more than the limit and is dropped with a
        warning.
        """
        positive = [
            (16.0 * k + 0.25, 16.0 * k + 14.25, 1.0 - 0.1 * k) for k in range(5)
        ]
        sf = _sigma_half_sine_cells([(-29.75, 0.25, -1.0), *positive])
        with pytest.warns(UserWarning) as record:
            cells = cell_edges_sigma(sf)
        assert cells.sel(lat="center").values.tolist() == [-15.0, 7.0, 23.0, 39.0, 55.0]
        messages = _dropped_messages(record)
        assert len(record) == len(messages) == 1
        assert messages[0].startswith(
            "cell_edges_sigma dropped the cell centered at latitude 71 and"
        )
        assert "more than 4 cells of its sign remain" in messages[0]

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
        latitudes given; a southern Hadley cell fills sigma 0.15 to 0.95. The
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
            -full * half_sine(-29.75, 0.25)
            + upper * half_sine(0.25, 40.25)
            - 0.5 * lower * half_sine(*lower_cell)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            cells = cell_edges_sigma(sf.transpose(SIGMA_STR, LAT_STR))
        assert list(cells["cell"].values) == [label for label, _ in expected]
        assert cells.sel(lat="center").values.tolist() == [c for _, c in expected]

    @pytest.mark.parametrize(
        ("lower_cell", "nan_where", "expected", "nan_edge"),
        [
            (
                (10.25, 30.25),
                "poleward_of_37N",
                [("sh_hadley", -15.0), ("nh_hadley", 20.0)],
                ("nh_hadley", "edge_north"),
            ),
            (
                (10.25, 44.25),
                "poleward_of_37N_aloft",
                [("sh_hadley", -15.0), ("nh_hadley", 20.0), ("2", 27.0)],
                ("nh_hadley", "edge_north"),
            ),
            (
                (12.25, 30.25),
                "poleward_of_25N_beneath",
                [("sh_hadley", -15.0), ("nh_hadley", 20.0), ("2", 21.0)],
                ("2", "edge_north"),
            ),
        ],
        ids=["within_the_reach", "past_the_reach", "nan_edge_never_within"],
    )
    @pytest.mark.parametrize("mirror", [False, True])
    def test_cell_within_a_cell_with_a_nan_edge(
        self,
        lower_cell: tuple[float, float],
        nan_where: str,
        expected: list[tuple[str, float]],
        nan_edge: tuple[str, str],
        mirror: bool,
    ) -> None:
        """A NaN edge bounds the cells within its cell at the latitude it reaches.

        The fields are those of ``test_cell_within_another_cells_edges``: a
        positive cell from 0.25 to 40.25 above sigma 0.53, with edges at 1.53N
        and 38.97N, and a negative cell of half its amplitude beneath sigma
        0.57. NaN poleward of 37N at every level leaves the upper cell with
        NaN for its northern edge, and known to reach 37N. The lower cell,
        from 10.25 to 30.25 with edges at 10.9N and 29.6N, lies within that
        reach and is dropped, silently, as on the complete field. It used to
        be returned between the Hadley cells, labeled "1", and the northern
        Hadley cell was labeled "2" in place of "nh_hadley". With NaN
        poleward of 37N only above sigma 0.53, a lower cell
        from 10.25 to 44.25 has its northern edge at 43.17N, beyond the
        reach, and is kept. A cell with NaN for an edge never lies within
        another: NaN poleward of 25N beneath sigma 0.57 leaves a lower cell
        from 12.25 to 30.25 with NaN for its northern edge, and it is kept,
        though 25N, which it reaches, lies within the upper cell's edges.
        Each field is also tested mirrored across the equator, with its sign
        reversed, so that the NaN edge is a southern one.
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
            -full * half_sine(-29.75, 0.25)
            + upper * half_sine(0.25, 40.25)
            - 0.5 * lower * half_sine(*lower_cell)
        )
        masks = {
            "poleward_of_37N": lat <= 37,
            "poleward_of_37N_aloft": (lat <= 37) | (sigma > 0.53),
            "poleward_of_25N_beneath": (lat <= 25) | (sigma < 0.57),
        }

        def oriented(field: xr.DataArray) -> xr.DataArray:
            field = field.transpose(SIGMA_STR, LAT_STR)
            if not mirror:
                return field
            return -field.assign_coords({LAT_STR: -field[LAT_STR]}).sortby(LAT_STR)

        if mirror:
            flip = {"sh_hadley": "nh_hadley", "nh_hadley": "sh_hadley", "2": "0"}
            expected = [(flip[label], -c) for label, c in reversed(expected)]
            nan_edge = (
                flip[nan_edge[0]],
                "edge_south" if nan_edge[1] == "edge_north" else "edge_north",
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            complete = cell_edges_sigma(oriented(sf))
        with pytest.warns(UserWarning) as record:
            cells = cell_edges_sigma(oriented(sf.where(masks[nan_where])))
        assert list(cells["cell"].values) == [label for label, _ in expected]
        assert cells.sel(lat="center").values.tolist() == [c for _, c in expected]
        assert np.isnan(cells.sel(cell=nan_edge[0], lat=nan_edge[1]).item())
        # The reach bounding each cell in this test is internal.
        for result in (cells, complete):
            assert set(result.coords) == {"cell", LAT_STR, "cell_strength", SIGMA_STR}
        for name in set(cells["cell"].values) & set(complete["cell"].values):
            for edge in ("edge_south", "center", "edge_north"):
                if (name, edge) != nan_edge:
                    got = cells.sel(cell=name, lat=edge).item()
                    assert got == complete.sel(cell=name, lat=edge).item()
        messages = _cut_messages(record)
        assert len(record) == len(messages) == 1
        center = cells.sel(cell=nan_edge[0], lat="center").item()
        assert f"at latitude {center:g} and" in messages[0]

    def test_cell_beneath_cell_at_grid_end_kept(self) -> None:
        """A cell whose maximum is at the grid's end has no edges to contain another.

        A negative cell fills sigma 0.1 to 0.5 from beyond the grid's end to
        55.25S and peaks at 85S, the first latitude. A positive cell of
        amplitude 0.5 fills sigma 0.6 to 1.0 from 75.25S to 60.25S, beneath
        it. The upper cell is dropped with a warning, and the lower one is
        kept, with the edges it has alone. The next test covers a grid on
        which edges searched from the upper cell's maximum would contain the
        lower cell; on this one the southern edge would be NaN, since 85S is
        five spacings from the pole.
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
            - full * half_sine(-29.75, 0.25)
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

        # Each edge is interpolated linearly between the gridpoints on either
        # side of 0.1 of the gridded peak, sin(pi * 7.25 / 15) at -68: between
        # -74 and -75 to the south, and between -61 and -60, where the cell is
        # zero, to the north.
        def profile(lat: float) -> float:
            x = (lat + 75.25) / 15.0
            peak = np.sin(np.pi * 7.25 / 15.0)
            return float(np.sin(np.pi * x) / peak) if 0 < x < 1 else 0.0

        for edge, (above, below) in [
            ("edge_south", (-74.0, -75.0)),
            ("edge_north", (-61.0, -60.0)),
        ]:
            f_above, f_below = profile(above), profile(below)
            assert f_above > 0.1 > f_below
            expected = above + (f_above - 0.1) / (f_above - f_below) * (below - above)
            assert lower_cell.sel(lat=edge).item() == pytest.approx(expected, abs=1e-9)

    def test_cell_beneath_cell_at_grid_end_near_the_pole_kept(self) -> None:
        """A cell at the grid's end has no edges to contain another, near the pole too.

        On a grid from 89.5S to 89.5N, a negative cell fills sigma 0.1 to 0.5
        from 130.25S to 49.25S and peaks at 89.5S, the first latitude. Edges
        searched from that maximum would lie near the pole and near 50S, both
        real numbers, and contain the positive cell beneath it, from 75.25S to
        60.25S. The upper cell is dropped with a warning, and the lower one is
        returned as on the field without the upper cell.
        """
        lats = np.arange(-89.5, 90.0, 1.0)
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
        rest = 1e10 * (
            0.5 * lower * half_sine(-75.25, -60.25)
            - full * half_sine(-29.75, 0.25)
            + full * half_sine(0.25, 30.25)
        )
        sf = rest - 0.6e10 * upper * half_sine(-130.25, -49.25)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            alone = cell_edges_sigma(rest.transpose(SIGMA_STR, LAT_STR))
        with pytest.warns(UserWarning) as record:
            cells = cell_edges_sigma(sf.transpose(SIGMA_STR, LAT_STR))
        messages = _dropped_messages(record)
        assert len(record) == len(messages) == 1
        assert "at latitude -89.5 and" in messages[0]
        assert list(cells["cell"].values) == ["0", "sh_hadley", "nh_hadley"]
        np.testing.assert_array_equal(cells.values, alone.values)

    def test_cell_positive_at_one_gridpoint_kept(self) -> None:
        """A cell positive at a single latitude of its level is kept.

        The cell is 1 at 10N and 0 at every other latitude, at every level
        from sigma 0.15 to 0.95, so each edge is interpolated between 10N and
        a neighbor at zero:
        10 -/+ 0.9.
        """
        lats = np.arange(-85.0, 86.0, 1.0)
        sigmas = np.linspace(0.1, 1.0, 19)
        lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
        sigma = xr.DataArray(sigmas, dims=[SIGMA_STR], coords={SIGMA_STR: sigmas})
        merid = xr.where(lat == 10.0, 1.0, 0.0)
        sf = 1e10 * np.sin(np.pi * (sigma - 0.1) / 0.9) * merid
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            cell = cell_edges_sigma(sf).sel(cell="nh_hadley")
        assert cell.sel(lat="edge_south").item() == pytest.approx(9.1)
        assert cell.sel(lat="edge_north").item() == pytest.approx(10.9)

    def test_custom_dim_names_with_dropped_cell(self) -> None:
        """Custom names leave a field of one slice, its warnings and its error alone.

        The warning for a dropped cell reads the level from sigma_str, and
        latitude and sigma under their custom names are not counted as
        dimensions to slice along, so a field whose only cell is dropped still
        raises.
        """
        names = {LAT_STR: "latitude", SIGMA_STR: "s"}
        sf = _sigma_half_sine_cells(_POLAR_CELL_AT_GRID_END).rename(names)
        with pytest.warns(UserWarning) as record:
            cells = cell_edges_sigma(sf, lat_str="latitude", sigma_str="s")
        assert list(cells["cell"].values) == ["0", "sh_hadley", "nh_hadley"]
        messages = _dropped_messages(record)
        assert len(messages) == 1
        assert messages[0].startswith(
            "cell_edges_sigma dropped the cell centered at latitude -85 and sigma 0.55,"
        )
        only_end = _sigma_half_sine_cells([(-110.25, -60.25, -0.3)]).rename(names)
        with (
            pytest.warns(UserWarning, match="dropped the cell centered"),
            pytest.raises(ValueError, match="no cell remains"),
        ):
            cell_edges_sigma(only_end, lat_str="latitude", sigma_str="s")

    def test_custom_dim_names_with_a_slice_without_cells(self) -> None:
        """An empty slice's result takes the custom names of latitude and sigma."""
        names = {LAT_STR: "latitude", SIGMA_STR: "s"}
        outer = _sigma_half_sine_cells([(-29.75, 0.25, -1.0), (0.25, 30.25, 1.0)])
        sf = (
            xr.concat([outer, outer * np.nan], dim=TIME_STR)
            .assign_coords({TIME_STR: [10, 20]})
            .rename(names)
        )
        with pytest.warns(UserWarning, match="returned NaN for 1 of the 2 slices"):
            result = cell_edges_sigma(sf, lat_str="latitude", sigma_str="s")
        assert dict(result.sizes) == {TIME_STR: 2, "cell": 2, "latitude": 3}
        assert result["latitude"].values.tolist() == [
            "edge_south",
            "center",
            "edge_north",
        ]
        assert bool(result.sel({TIME_STR: 20}).isnull().all())

    @pytest.mark.parametrize(
        ("min_frac_strength", "kept"), [(0.5, False), (0.49, True)]
    )
    def test_cell_at_exactly_the_strength_threshold(
        self, min_frac_strength: float, kept: bool
    ) -> None:
        """A cell exactly min_frac_strength times the strongest is dropped.

        The northern polar cell has the shape of the northern Hadley cell and
        half its amplitude, so its strength is exactly half the strongest.
        It reaches past the grid's end at 85N, so when kept it has a NaN
        northern edge, with a warning.
        """
        sf = _sigma_half_sine_cells(
            [(-29.75, 0.25, -1.0), (0.25, 30.25, 1.0), (60.25, 90.25, 0.5)]
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always", UserWarning)
            cells = cell_edges_sigma(sf, min_frac_strength=min_frac_strength)
        assert len(_cut_messages(record)) == (1 if kept else 0)
        strengths = np.abs(cells["cell_strength"].values)
        assert (75.0 in cells.sel(lat="center").values) == kept
        if kept:
            assert strengths.min() == 0.5 * strengths.max()

    @pytest.mark.parametrize("extra_dim", [False, True], ids=["lat_sigma", "with_time"])
    def test_warnings_point_at_the_caller(self, extra_dim: bool) -> None:
        sf = _sigma_half_sine_cells(_POLAR_CELL_AT_GRID_END)
        if extra_dim:
            sf = sf.expand_dims({TIME_STR: [0, 1]})
        with pytest.warns(UserWarning) as record:
            cell_edges_sigma(sf)
        # One warning for the call, however many slices it covers.
        assert len(_dropped_messages(record)) == 1
        assert {w.filename for w in record} == {__file__}

    @pytest.mark.parametrize(
        ("extra", "first_slice"),
        [
            ({TIME_STR: [10, 20, 30]}, "time=10"),
            ({TIME_STR: [10, 20, 30], "model": ["a", "b"]}, "time=10, model=a"),
            ({TIME_STR: 3}, "slice 1 of 3"),
            # A dimension without a coordinate leaves the others unable to
            # say which slice is meant.
            ({TIME_STR: [10, 20, 30], "member": 2}, "slice 1 of 6"),
        ],
        ids=["time", "time_and_model", "no_coordinate", "one_without_coordinate"],
    )
    def test_one_warning_per_reason_across_slices(
        self, extra: dict[str, Any], first_slice: str
    ) -> None:
        """Each reason gets one warning per call, giving the count and the first cell.

        Every slice is the six-cell field cut to a band from 40S to 40N, whose
        Ferrel cells peak beyond the band and so at its ends, where they are
        dropped. This field used to make the call raise.
        """
        sf = _six_sigma_cells(0.2).sel({LAT_STR: slice(-40, 40)}).expand_dims(extra)
        n_slices = int(np.prod([sf.sizes[dim] for dim in extra]))
        with pytest.warns(UserWarning) as record:
            cell_edges_sigma(sf)
        assert len(record) == 1
        message = str(record[0].message)
        assert message.startswith(
            f"cell_edges_sigma dropped {2 * n_slices} cells in {n_slices} of the "
            f"{n_slices} slices along {', '.join(extra)}: its maximum lies at the "
            "first or last latitude of the grid, beyond which it may peak. The "
            f"first is at {first_slice}, the cell centered at latitude "
        )
        assert message.split("centered at latitude ")[1].split()[0] in ("40", "-40")
        assert record[0].filename == __file__

    @pytest.mark.parametrize(
        ("lats", "nan_edges"),
        [
            (slice(-25, 25), {"sh_hadley": "edge_south", "nh_hadley": "edge_north"}),
            (slice(2, None), {"nh_hadley": "edge_south"}),
            (slice(None, -2), {"sh_hadley": "edge_north"}),
        ],
        ids=["band_25S_25N", "northern_hemisphere", "southern_hemisphere"],
    )
    def test_cell_cut_by_the_end_of_a_band_or_hemisphere(
        self, lats: slice, nan_edges: dict[str, str]
    ) -> None:
        """A cell reaching a grid end far from the pole gets NaN for that edge.

        The four Hadley and Ferrel cells of ``_sigma_cells`` are cut to a band
        from 25S to 25N, or to one hemisphere without the equator. Each Hadley
        cell reaching the cut is kept, with NaN for its edge on that side, its
        other values as on the full grid, and a warning. Those edges used to
        be interpolated toward zero at the pole, out to 76.4S and 77.6N for
        the band, and a hemisphere grid made the call raise.
        """
        full = cell_edges_sigma(_sigma_cells(ferrel=0.4))
        with pytest.warns(UserWarning) as record:
            cells = cell_edges_sigma(_sigma_cells(ferrel=0.4).sel({LAT_STR: lats}))
        assert _dropped_messages(record) == []
        assert {"skip_reason", "cut_reason"}.isdisjoint(cells.coords)
        messages = _cut_messages(record)
        assert len(messages) == len(nan_edges)
        # A cell's positional label differs between the grids, so each cell
        # is matched to the full grid's by its center.
        full_centers = full.sel(lat="center").values.tolist()
        for label in cells["cell"].values:
            center = cells.sel(cell=label, lat="center").item()
            on_full = full.isel(cell=full_centers.index(center))
            for edge in ("edge_south", "edge_north"):
                got = cells.sel(cell=label, lat=edge).item()
                if nan_edges.get(label) == edge:
                    assert np.isnan(got)
                else:
                    assert got == on_full.sel(lat=edge).item()
        for label, edge in nan_edges.items():
            side, end = (
                ("southern", "first") if edge == "edge_south" else ("northern", "last")
            )
            center = cells.sel(cell=label, lat="center").item()
            matching = [m for m in messages if f"at latitude {center:g} and" in m]
            assert len(matching) == 1
            assert matching[0].endswith(
                f": its {side} edge lies beyond the {end} latitude of the grid, "
                "which is more than one grid spacing from the pole."
            )

    @pytest.mark.parametrize(
        ("grid", "near"),
        [
            ("spacing_2.5", True),
            ("spacing_2.4", False),
            ("spacing_3.125", True),
            ("rounded_1.2", True),
        ],
    )
    def test_grid_end_exactly_one_spacing_from_the_pole(
        self, grid: str, near: bool
    ) -> None:
        """A grid end within one spacing of the pole, inclusive, counts as near it.

        On the first three grids the northern end is 87.5, 2.5 degrees from
        the pole, and the spacing before it is 2.5, 2.4 or 3.125 degrees. A
        spacing of 3.125 puts the end 0.8 of a spacing from the pole, near
        the 0.77 of a 64-latitude Gaussian grid. The fourth,
        ``np.linspace(-88.8, 88.8, 149)``, ends 1.2 degrees from the pole
        after a spacing of 1.2, which rounding leaves 1.4e-14 short of that
        distance. A cell from 0.25 to 120.25, peaking at the gridpoint 60, is
        still more than 0.7 of that peak at the end. Near the pole, its
        northern edge is interpolated between the end and a zero at the pole;
        otherwise it is NaN, with a warning.
        """
        if grid == "rounded_1.2":
            lats = np.linspace(-88.8, 88.8, 149)
        else:
            last_spacing = float(grid.removeprefix("spacing_"))
            lats = np.concatenate(
                [np.arange(-87.5, 84.0, 2.5), [87.5 - last_spacing, 87.5]]
            )
        end = lats[-1]
        assert (90.0 - end <= end - lats[-2]) == (
            grid in ("spacing_2.5", "spacing_3.125")
        )
        sigmas = np.linspace(0.1, 1.0, 19)
        lat = xr.DataArray(lats, dims=[LAT_STR], coords={LAT_STR: lats})
        sigma = xr.DataArray(sigmas, dims=[SIGMA_STR], coords={SIGMA_STR: sigmas})
        x = lat - _SIGMA_NODE_OFFSET
        merid = xr.where(x > 0, np.sin(np.pi * x / 120.0), 0.0)
        sf = 1e10 * np.sin(np.pi * (sigma - 0.1) / 0.9) * merid
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always", UserWarning)
            cells = cell_edges_sigma(sf)
        edge = cells.sel(cell="nh_hadley", lat="edge_north").item()
        center = cells.sel(cell="nh_hadley", lat="center").item()
        assert center == pytest.approx(60.0, abs=1e-12)
        if near:
            peak = np.sin(np.pi * 59.75 / 120.0)
            at_end = np.sin(np.pi * (end - 0.25) / 120.0) / peak
            expected = end + (90.0 - end) * (at_end - 0.1) / at_end
            assert edge == pytest.approx(expected, rel=1e-12)
            assert len(record) == 0
        else:
            assert np.isnan(edge)
            assert len(_cut_messages(record)) == len(record) == 1

    @pytest.mark.parametrize(
        ("keep", "nan_edges"),
        [
            ("both", {"sh_hadley": "edge_south", "nh_hadley": "edge_north"}),
            ("north", {"nh_hadley": "edge_north"}),
        ],
    )
    def test_cell_cut_by_nan_to_the_end_of_the_grid(
        self, keep: str, nan_edges: dict[str, str]
    ) -> None:
        """A cell reaching NaN that lasts to the grid's end gets NaN for that edge.

        The four Hadley and Ferrel cells of ``_sigma_cells`` are masked
        poleward of 20 degrees in both hemispheres, or in the northern one
        only. Each Hadley cell reaching the mask is kept, with NaN for its
        edge on that side, its other values as unmasked, and a warning. Such
        a cell used to be dropped, and with both masked the call raised.
        """
        sf = _sigma_cells(ferrel=0.4)
        lat = sf[LAT_STR]
        mask = (np.abs(lat) <= 20) if keep == "both" else (lat <= 20)
        full = cell_edges_sigma(sf)
        with pytest.warns(UserWarning) as record:
            cells = cell_edges_sigma(sf.where(mask))
        assert _dropped_messages(record) == []
        messages = _cut_messages(record)
        assert len(messages) == len(nan_edges)
        expected = (
            ["sh_hadley", "nh_hadley"]
            if keep == "both"
            else ["0", "sh_hadley", "nh_hadley"]
        )
        assert cells["cell"].values.tolist() == expected
        for label in cells["cell"].values:
            for edge in ("edge_south", "center", "edge_north"):
                got = cells.sel(cell=label, lat=edge).item()
                if nan_edges.get(label) == edge:
                    assert np.isnan(got)
                else:
                    assert got == full.sel(cell=label, lat=edge).item()
        for edge in nan_edges.values():
            side, end = (
                ("southern", "first") if edge == "edge_south" else ("northern", "last")
            )
            phrase = (
                f": its {side} edge lies where the streamfunction is NaN, from "
                f"beside the cell to the {end} latitude of the grid."
            )
            assert sum(m.endswith(phrase) for m in messages) == 1

    def test_cell_split_on_one_side_and_cut_on_the_other_dropped(self) -> None:
        """A cell split by NaN on one side is dropped, whatever the other side.

        The two Hadley cells of ``_sigma_half_sine_cells`` are masked at 20N
        and poleward of 25N. The northern cell's half south of 20N has a cell
        of its own sign beyond the NaN, and is dropped. So is the half north
        of 20N, which has the other half beyond the NaN to its south, though
        to its north the NaN lasts to the grid's end.
        """
        sf = _sigma_half_sine_cells([(-29.75, 0.25, -1.0), (0.25, 30.25, 1.0)])
        lat = sf[LAT_STR]
        with pytest.warns(UserWarning) as record:
            cells = cell_edges_sigma(sf.where((lat <= 25) & (lat != 20)))
        assert cells["cell"].values.tolist() == ["sh_hadley"]
        assert _cut_messages(record) == []
        messages = _dropped_messages(record)
        assert len(messages) == 2
        for center, side in [(15.0, "northern"), (21.0, "southern")]:
            matching = [m for m in messages if f"at latitude {center:g} and" in m]
            assert len(matching) == 1
            assert f": its {side} edge could not be found:" in matching[0]

    def test_cell_cut_on_both_sides(self) -> None:
        """A cell cut by both ends of a band gets NaN for both edges, in one warning.

        A single cell from 40.25S to 40.25N, peaking at the equator, is cut to
        a band from 20S to 20N, at both ends of which it is still 0.7 of its
        peak.
        """
        sf = _sigma_half_sine_cells([(-40.25, 40.25, 1.0)])
        with pytest.warns(UserWarning) as record:
            cells = cell_edges_sigma(sf.sel({LAT_STR: slice(-20, 20)}))
        messages = _cut_messages(record)
        assert len(record) == len(messages) == 1
        assert messages[0].endswith(
            ": its southern edge lies beyond the first latitude of the grid, which "
            "is more than one grid spacing from the pole; its northern edge lies "
            "beyond the last latitude of the grid, which is more than one grid "
            "spacing from the pole."
        )
        cell = cells.isel(cell=0)
        assert cell.sel(lat="center").item() == 0.0
        assert np.isnan(cell.sel(lat="edge_south").item())
        assert np.isnan(cell.sel(lat="edge_north").item())

    def test_cut_cell_beyond_the_limit_gets_one_warning(self) -> None:
        """A cut cell dropped by the per-sign limit is reported as dropped only.

        The six cells of ``_six_sigma_cells`` include two polar cells that
        reach past the grid's end at 85 degrees, so each would have a NaN
        poleward edge. With max_cells_per_sign=2, they are the weakest of
        their signs and are dropped, with one warning, and no warning says
        they were kept.
        """
        with pytest.warns(UserWarning) as record:
            cells = cell_edges_sigma(_six_sigma_cells(0.2), max_cells_per_sign=2)
        assert cells.sel(lat="center").values.tolist() == [-45.0, -15.0, 15.0, 45.0]
        assert _cut_messages(record) == []
        messages = _dropped_messages(record)
        assert len(record) == len(messages) == 1
        assert messages[0].startswith("cell_edges_sigma dropped 2 cells: more than 2")

    @pytest.mark.parametrize(
        ("polar", "warned"), [(-25.0, True), (-0.001, False)], ids=["strong", "weak"]
    )
    def test_strength_threshold_from_measurable_cells(
        self, polar: float, warned: bool
    ) -> None:
        """The strength threshold is set by the cells whose edges can be measured.

        A polar cell peaking at 85S, the grid's first latitude, cannot be
        measured. Were the threshold taken from every cell, a polar cell of 25
        times the Hadley cells' strength would set it, at 0.05 times its own
        strength, above both Hadley cells, which would be dropped silently,
        and the call would raise. The Hadley cells set it, so the strong polar
        cell is dropped with a warning and the Hadley cells are returned as
        without it. A polar cell of 0.001 times their strength falls below the
        threshold and is dropped silently. Both fields used to make the call
        raise, for the polar cell's maximum at the grid's end.
        """
        hadley = [(-29.75, 0.25, -1.0), (0.25, 30.25, 1.0)]
        alone = cell_edges_sigma(_sigma_half_sine_cells(hadley))
        sf = _sigma_half_sine_cells([(-110.25, -59.75, polar), *hadley])
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always", UserWarning)
            cells = cell_edges_sigma(sf)
        assert cells["cell"].values.tolist() == ["sh_hadley", "nh_hadley"]
        np.testing.assert_array_equal(cells.values, alone.values)
        assert len(record) == (1 if warned else 0)
        if warned:
            message = _dropped_messages(record)[0]
            assert "at latitude -85 and" in message
            assert "first or last latitude of the grid" in message

    def test_cut_cell_counts_as_measured_for_the_strength_threshold(self) -> None:
        """A cell kept with a NaN edge sets the strength threshold like any other.

        On a grid starting at 25S, the southern Hadley cell, of strength 1, is
        cut by the grid's end. The northern Hadley cell has strength 0.5 and
        the northern Ferrel cell 0.04, which falls below 0.05 times the
        southern Hadley cell but would pass 0.05 times the northern one.
        """
        sf = _sigma_half_sine_cells(
            [(-29.75, 0.25, -1.0), (0.25, 30.25, 0.5), (30.25, 60.25, -0.04)]
        ).sel({LAT_STR: slice(-25, None)})
        with pytest.warns(UserWarning, match="its southern edge lies beyond"):
            cells = cell_edges_sigma(sf)
        assert cells["cell"].values.tolist() == ["sh_hadley", "nh_hadley"]
        # The Ferrel cell itself is measured and kept below a threshold of
        # 0.04 times the strongest cell.
        with pytest.warns(UserWarning, match="its southern edge lies beyond"):
            lower = cell_edges_sigma(sf, min_frac_strength=0.035)
        assert lower["cell"].values.tolist() == ["sh_hadley", "nh_hadley", "2"]

    @pytest.mark.parametrize(
        ("middle", "reason"),
        [
            (
                "all_nan",
                "the streamfunction is NaN or zero at every latitude and sigma "
                "searched, so it has no cell",
            ),
            (
                "too_shallow",
                "no cell found that spans at least cell_min_sigma_depth=0.3 in sigma",
            ),
            ("unmeasurable", "no cell remains whose edges could be measured"),
        ],
    )
    def test_slice_with_no_cell_is_nan(self, middle: str, reason: str) -> None:
        """A slice in which no cell remains is NaN, and the other slices are returned.

        Of three times, the middle one is all NaN, as for a missing month,
        holds only a cell spanning sigma 0.5 to 0.6, or holds only a cell
        peaking at the grid's first latitude. Each used to make the call raise
        for all three. A lone such field, or a series of one, still raises.
        """
        outer = _sigma_half_sine_cells([(-29.75, 0.25, -1.0), (0.25, 30.25, 1.0)])
        if middle == "all_nan":
            mid = outer * np.nan
        elif middle == "too_shallow":
            sigma, lat = outer[SIGMA_STR], outer[LAT_STR]
            band = xr.where((sigma > 0.47) & (sigma < 0.63), 1.0, 0.0)
            mid = 1e10 * band * xr.where((lat > 0.25) & (lat < 30.25), 1.0, 0.0)
        else:
            mid = _sigma_half_sine_cells([(-110.25, -60.25, -0.3)])
        sf = xr.concat([outer, mid, 2 * outer], dim=TIME_STR).assign_coords(
            {TIME_STR: [10, 20, 30]}
        )
        with pytest.warns(UserWarning) as record:
            result = cell_edges_sigma(sf)
        messages = [str(w.message) for w in record]
        assert len(messages) == (2 if middle == "unmeasurable" else 1)
        # The warnings come in the order of their first note: the cell dropped
        # in the middle slice, then the slice left with no cell.
        assert messages[-1] == (
            f"cell_edges_sigma returned NaN for 1 of the 3 slices along time: "
            f"{reason}. The first is at time=20."
        )
        if middle == "unmeasurable":
            assert messages[0].startswith(
                "cell_edges_sigma dropped 1 cell in 1 of the 3 slices along "
                "time: its maximum lies at the first or last latitude"
            )
        assert result[LAT_STR].values.tolist() == ["edge_south", "center", "edge_north"]
        assert set(result.dims) == {TIME_STR, "cell", LAT_STR}
        assert bool(result.sel({TIME_STR: 20}).isnull().all())
        for time, field in [(10, outer), (30, 2 * outer)]:
            alone = cell_edges_sigma(field)
            got = result.sel({TIME_STR: time})
            np.testing.assert_array_equal(got.values, alone.values)
            assert got["cell_label"].values.tolist() == alone["cell"].values.tolist()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(ValueError, match=reason):
                cell_edges_sigma(mid)
            with pytest.raises(ValueError, match=reason):
                cell_edges_sigma(mid.expand_dims({TIME_STR: [20]}))

    def test_every_slice_with_no_cell(self) -> None:
        sf = (_sigma_half_sine_cells([(0.25, 30.25, 1.0)]) * np.nan).rename("sf")
        with pytest.warns(
            UserWarning,
            match="returned NaN for 2 of the 2 slices along time: the streamfunction "
            "is NaN or zero",
        ):
            result = cell_edges_sigma(sf.expand_dims({TIME_STR: [10, 20]}))
        assert dict(result.sizes) == {TIME_STR: 2, "cell": 0, LAT_STR: 3}
        assert result[LAT_STR].values.tolist() == ["edge_south", "center", "edge_north"]
        assert result.name == "sf"

    @pytest.mark.parametrize("case", ["ferrel_poleward_of_40", "hadley_south_of_5"])
    def test_maximum_beside_nan_to_the_grid_end_as_on_a_trimmed_grid(
        self, case: str
    ) -> None:
        """A cell peaking beside NaN that lasts to the grid's end is dropped.

        In the first case the Ferrel cells of ``_sigma_cells``, peaking at 45
        degrees, are masked poleward of 40, so their maxima on the data lie at
        40, beside the NaN. In the second the southern Hadley cell reaches
        8.25N and is masked south of 5N, leaving a fragment peaking at 5N.
        Each such cell is dropped with a warning, and the result is that of
        the grid trimmed where the NaN begins. Such a cell used to be kept,
        with a center and strength describing where the data end.
        """
        if case == "ferrel_poleward_of_40":
            sf = _sigma_cells(ferrel=0.4)
            keep = abs(sf[LAT_STR]) <= 40
            n_dropped = 2
        else:
            sf = _sigma_half_sine_cells([(-29.75, 8.25, -1.0), (8.25, 30.25, 1.0)])
            keep = sf[LAT_STR] >= 5
            n_dropped = 1
        with pytest.warns(UserWarning) as trimmed_record:
            trimmed = cell_edges_sigma(sf.where(keep, drop=True))
        assert len(_dropped_messages(trimmed_record)) == 1
        with pytest.warns(UserWarning) as record:
            masked = cell_edges_sigma(sf.where(keep))
        assert _cut_messages(record) == []
        messages = _dropped_messages(record)
        assert len(record) == len(messages) == 1
        what = "the cell" if n_dropped == 1 else f"{n_dropped} cells"
        assert messages[0].startswith(f"cell_edges_sigma dropped {what}")
        assert (
            "its maximum lies beside NaN that lasts to the end of the grid, beyond "
            "which it may peak"
        ) in messages[0]
        assert masked["cell"].values.tolist() == trimmed["cell"].values.tolist()
        np.testing.assert_array_equal(masked.values, trimmed.values)

    def test_cut_messages_on_a_descending_grid(self) -> None:
        """On a grid ordered from north to south, the southern end comes last.

        The Hadley cells of ``_sigma_cells``, cut to a band from 25S to 25N,
        with the latitudes reversed. Their edges are those of the ascending
        band.
        """
        band = _sigma_cells(ferrel=0.4).sel({LAT_STR: slice(-25, 25)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ascending = cell_edges_sigma(band)
        with pytest.warns(UserWarning) as record:
            cells = cell_edges_sigma(band.isel({LAT_STR: slice(None, None, -1)}))
        np.testing.assert_array_equal(cells.values, ascending.values)
        messages = _cut_messages(record)
        assert len(messages) == 2
        for side, end in [("southern", "last"), ("northern", "first")]:
            phrase = (
                f": its {side} edge lies beyond the {end} latitude of the grid, "
                "which is more than one grid spacing from the pole."
            )
            assert sum(m.endswith(phrase) for m in messages) == 1

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"max_cells_per_sign": 0}, "max_cells_per_sign is 0, and must be"),
            ({"min_frac_strength": 1.0}, "min_frac_strength is 1.0, and must be"),
            ({"min_frac_strength": -0.1}, "min_frac_strength is -0.1, and must be"),
        ],
    )
    def test_filter_parameters_out_of_range_raise(
        self, kwargs: dict[str, Any], match: str
    ) -> None:
        """max_cells_per_sign below 1 and min_frac_strength outside [0, 1) raise.

        They used to fail in the labeling step, on an empty array, or to
        blame the edges for cells the strength filter had dropped. The
        smallest valid values run.
        """
        with pytest.raises(ValueError, match=match):
            cell_edges_sigma(_sigma_streamfunc(), **kwargs)
        assert (
            cell_edges_sigma(_sigma_streamfunc(), max_cells_per_sign=1).sizes["cell"]
            == 2
        )
        assert (
            cell_edges_sigma(_sigma_streamfunc(), min_frac_strength=0.0).sizes["cell"]
            == 2
        )

    def test_error_kept_when_warnings_are_errors(self) -> None:
        """The ValueError for no cell is raised even when warnings are errors.

        The warning for the cell dropped is attached to it as a note. The
        warning raised as an error used to replace the ValueError.
        """
        sf = _sigma_half_sine_cells([(-110.25, -60.25, -0.3)])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with pytest.raises(ValueError, match="no cell remains") as excinfo:
                cell_edges_sigma(sf)
        notes = getattr(excinfo.value, "__notes__", [])
        assert len(notes) == 1
        assert "first or last latitude of the grid" in notes[0]


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

    def test_cell_edges_sigma_unknown_argument_names_the_function(self) -> None:
        # The renamed max_num_cells, through both of cell_edges_sigma's
        # decorators, is refused with the function's name.
        old_name: dict[str, Any] = {"max_num_cells": 2}
        with pytest.raises(TypeError, match=r"cell_edges_sigma\(\).*'max_num_cells'"):
            cell_edges_sigma(_sigma_streamfunc(), **old_name)

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
