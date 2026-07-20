"""Tests for the plumb_hou_1992 module."""

import numpy as np
import xarray as xr

from puffins.grad_bal import grad_wind_bouss
from puffins.names import LAT_STR
from puffins.plumb_hou_1992 import dtheta_rce_ph92_dlat, u_ph92_rce

ROT_RATE = 7.292e-5
RADIUS = 6.371e6
GRAV = 9.81


def _lats() -> xr.DataArray:
    return xr.DataArray(np.arange(2.0, 41.0, 2.0), dims=[LAT_STR], name=LAT_STR)


class TestUPh92Rce:
    """Regression: this previously always raised ``TypeError`` because it
    forwarded an unsupported ``plus_solution`` kwarg to ``grad_wind_bouss``."""

    def test_runs_and_returns_finite_dataarray(self) -> None:
        lats = _lats()
        out = u_ph92_rce(
            lats,
            theta_max=20.0,
            lat_max=25.0,
            nonzero_width=15.0,
            height=10e3,
            temp_ref=300.0,
        )
        assert isinstance(out, xr.DataArray)
        assert out.dims == (LAT_STR,)
        assert np.isfinite(out.values).all()

    def test_equals_grad_wind_bouss_of_rce_gradient(self) -> None:
        """u_ph92_rce is grad_wind_bouss applied to the Eq. 9 theta gradient."""
        lats = _lats()
        theta_max, lat_max, width, height, temp_ref = 20.0, 25.0, 15.0, 10e3, 300.0
        dtheta_dlat = dtheta_rce_ph92_dlat(lats, theta_max, lat_max, width)
        expected = grad_wind_bouss(
            lats,
            height,
            temp_ref,
            dtheta_dlat,
            grav=GRAV,
            rot_rate=ROT_RATE,
            radius=RADIUS,
        )
        actual = u_ph92_rce(
            lats,
            theta_max,
            lat_max,
            width,
            height,
            temp_ref,
            grav=GRAV,
            rot_rate=ROT_RATE,
            radius=RADIUS,
        )
        assert isinstance(actual, xr.DataArray)
        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-13)
