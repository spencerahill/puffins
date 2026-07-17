"""Tests for had_cell module."""

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

    def test_along_dim(self) -> None:
        lats = np.array([0.0, 10.0])
        plevs = np.array([200.0, 400.0])
        times = np.array([0, 1])
        sf = xr.DataArray(
            [[[8.0, 1.0], [2.0, 3.0]], [[0.0, 1.0], [2.0, 3.0]]],
            dims=[TIME_STR, LAT_STR, LEV_STR],
            coords={TIME_STR: times, LAT_STR: lats, LEV_STR: plevs},
        )
        result = had_cell_strength(sf, dim=TIME_STR)
        assert result.sel({TIME_STR: 0}).item() == pytest.approx(8.0)
        assert result.sel({TIME_STR: 1}).item() == pytest.approx(3.0)
        assert result[LAT_STR].sel({TIME_STR: 0}).item() == 0.0
        assert result[LAT_STR].sel({TIME_STR: 1}).item() == 10.0


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
        # Asymmetric bounds, so that the two surviving candidates aren't
        # at exactly mirrored latitudes (which trips a separate bug; see
        # the PR discussion).
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


# The meridional node latitudes are offset from the integer grid by 0.25
# deg: a gridpoint lying exactly on a streamfunction zero is excluded from
# both bounding cells, and the resulting NaN breaks the edge search.
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

    It directly abuts the NH cell in latitude so that its edge search finds
    adjacent opposite-signed values rather than a gap of NaNs.
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
        expected = np.rad2deg((3.0 * 0.3 * 0.2 / (0.4 + 2.0 * 0.8)) ** 0.25)
        assert result == pytest.approx(expected)

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

    def test_supercrit_ascent_known_value(self) -> None:
        therm_ross, max_lat = 0.4, 60.0
        c_ascent, c_descent = 1.1, 0.9
        delta_v, delta_h, ross_num = 0.3, 0.1, 0.8
        result = fixed_ro_bci_edge_supercrit_ascent(
            therm_ross,
            max_lat=max_lat,
            c_ascent=c_ascent,
            c_descent=c_descent,
            delta_v=delta_v,
            delta_h=delta_h,
            ross_num=ross_num,
        )
        lat_ascent = np.rad2deg(c_ascent * (0.5 * therm_ross) ** (1.0 / 3.0))
        term1 = 2.0 ** (4.0 / 3.0) / c_ascent**4
        term2 = delta_v / (delta_h * np.sin(np.deg2rad(max_lat)))
        term3 = 1.0 / (ross_num * therm_ross ** (1.0 / 3.0))
        expected = c_descent * np.rad2deg(
            np.deg2rad(lat_ascent)
            * np.sqrt(0.5 + np.sqrt(0.25 + term1 * term2 * term3))
        )
        assert result == pytest.approx(expected)


def _bci_polynomial_residual(lat: float, ascentlat: float, h00lat: float) -> float:
    """Raw-numpy version of the fixed-Ro BCI edge condition."""
    sinlat = np.sin(np.deg2rad(lat))
    coslat = np.cos(np.deg2rad(lat))
    return float(
        sinlat**4
        - np.sin(np.deg2rad(ascentlat)) ** 2 * sinlat**2
        - np.deg2rad(h00lat) ** 4 * coslat**2
    )


class TestFixedRoBciEdgeSolver:
    """Numerical solver satisfies the BCI edge condition it solves."""

    def test_roots_satisfy_edge_condition(self) -> None:
        ascentlats = xr.DataArray([5.0, 15.0], dims=["case"], coords={"case": [0, 1]})
        edges = fixed_ro_bci_edge(
            ascentlats,
            lat_fixed_ro_ann=20.0,
            zero_bounds_guess_range=np.arange(0.5, 80.0, 2.0),
        )
        for lat_a, lat_edge in zip(ascentlats.values, np.asarray(edges).ravel()):
            resid = _bci_polynomial_residual(float(lat_edge), float(lat_a), 20.0)
            assert abs(resid) < 1e-6, (lat_a, lat_edge, resid)
        # Edge moves poleward with the ascent latitude.
        edge_vals = np.asarray(edges).ravel()
        assert edge_vals[1] > edge_vals[0]

    def test_default_burger_number_path(self) -> None:
        edge = float(np.asarray(fixed_ro_bci_edge(0.0)))
        burg = HEIGHT_TROPO * GRAV_EARTH / (ROT_RATE_EARTH * RAD_EARTH) ** 2
        h00lat = np.rad2deg((burg * DELTA_V / 2.0) ** 0.25)
        resid = _bci_polynomial_residual(edge, 0.0, float(h00lat))
        assert abs(resid) < 1e-6, (edge, resid)
