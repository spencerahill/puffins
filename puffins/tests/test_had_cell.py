"""Tests for had_cell module."""

import numpy as np
import xarray as xr

from puffins.had_cell import (
    had_cell_edge,
    had_cells_north_edge,
    had_cells_shared_edge,
    had_cells_south_edge,
)
from puffins.names import LAT_STR, LEV_STR

EDGE_KWARGS = dict(
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
    return (1e11 * vert * merid).rename("streamfunc")


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
            assert "cell" not in result.coords, (cell, edge, result.coords)

    def test_north_south_shared_no_cell_coord(self) -> None:
        sf = _make_streamfunc()
        for func in (had_cells_north_edge, had_cells_south_edge):
            result = func(sf, **EDGE_KWARGS)
            assert "cell" not in result.coords, (func.__name__, result.coords)
        result = had_cells_shared_edge(
            sf, min_lat=-50, max_lat=50, min_plev=500, max_plev=800, do_avg_vert=True
        )
        assert "cell" not in result.coords, result.coords


class TestHadCellEdgeValues:
    """Edge latitudes recovered from the idealized streamfunction."""

    def test_edge_latitudes(self) -> None:
        sf = _make_streamfunc(edge_lat=30.0)
        north = had_cells_north_edge(sf, **EDGE_KWARGS)
        south = had_cells_south_edge(sf, **EDGE_KWARGS)
        shared = had_cell_edge(sf, cell="south", edge="north", **EDGE_KWARGS)
        assert 29.0 < north.item() < 30.0, north.item()
        assert -30.0 < south.item() < -29.0, south.item()
        assert abs(shared.item()) < 1.0, shared.item()
