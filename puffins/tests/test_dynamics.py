"""Tests for dynamics module."""

import numpy as np
import xarray as xr

from puffins.constants import (
    GRAV_EARTH,
    HEIGHT_TROPO,
    RAD_EARTH,
    ROT_RATE_EARTH,
)
from puffins.dynamics import (
    abs_ang_mom,
    abs_vort_from_u,
    abs_vort_vert_comp,
    brunt_vaisala_freq,
    bulk_stat_stab,
    coriolis_param,
    plan_burg_num,
    rel_vort_from_u,
    ross_num_from_uwind,
    ross_num_gen,
    rossby_radius,
    therm_ross_num,
    u_bci_2layer_qg,
    z_from_hypso,
    zonal_fric_inferred_steady,
)
from puffins.names import LAT_STR, LEV_STR


def _lat_dataarray(values: list[float] | np.ndarray) -> xr.DataArray:
    """Create a DataArray with a latitude coordinate."""
    vals = np.array(values, dtype=float)
    return xr.DataArray(vals, coords={LAT_STR: vals}, dims=[LAT_STR])


def _uwind_dataarray(
    values: list[float] | np.ndarray,
    lats: list[float] | np.ndarray,
) -> xr.DataArray:
    """Create a zonal wind DataArray with latitude coordinate."""
    return xr.DataArray(
        np.array(values, dtype=float),
        coords={LAT_STR: np.array(lats, dtype=float)},
        dims=[LAT_STR],
    )


# ---------------------------------------------------------------------------
# TestCoriolisParam
# ---------------------------------------------------------------------------


class TestCoriolisParam:
    """Tests for coriolis_param."""

    def test_equator_is_zero(self) -> None:
        """Coriolis parameter is zero at the equator."""
        result = coriolis_param(0.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_north_pole(self) -> None:
        """Coriolis parameter at the North Pole equals 2*Omega."""
        result = coriolis_param(90.0)
        np.testing.assert_allclose(result, 2.0 * ROT_RATE_EARTH)

    def test_south_pole(self) -> None:
        """Coriolis parameter at the South Pole equals -2*Omega."""
        result = coriolis_param(-90.0)
        np.testing.assert_allclose(result, -2.0 * ROT_RATE_EARTH)

    def test_antisymmetric(self) -> None:
        """Coriolis parameter is antisymmetric about the equator."""
        f_30n = coriolis_param(30.0)
        f_30s = coriolis_param(-30.0)
        np.testing.assert_allclose(f_30n, -f_30s)

    def test_30n(self) -> None:
        """Coriolis parameter at 30N is Omega (sin(30)=0.5)."""
        result = coriolis_param(30.0)
        np.testing.assert_allclose(result, ROT_RATE_EARTH)

    def test_custom_rot_rate(self) -> None:
        """Works with a custom rotation rate."""
        result = coriolis_param(90.0, rot_rate=1.0)
        np.testing.assert_allclose(result, 2.0)

    def test_array_input(self) -> None:
        """Works with numpy array input."""
        lats = np.array([-90, -30, 0, 30, 90], dtype=float)
        result = coriolis_param(lats)
        assert result.shape == (5,)
        np.testing.assert_allclose(result[2], 0.0, atol=1e-15)

    def test_dataarray_input(self) -> None:
        """Works with xarray DataArray input."""
        lats = _lat_dataarray([0.0, 30.0, 60.0, 90.0])
        result = coriolis_param(lats)
        assert isinstance(result, xr.DataArray)
        np.testing.assert_allclose(result.sel(lat=0.0).item(), 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# TestPlanBurgNum
# ---------------------------------------------------------------------------


class TestPlanBurgNum:
    """Tests for plan_burg_num."""

    def test_default_earth(self) -> None:
        """Planetary Burger number with default Earth values."""
        result = plan_burg_num()
        expected = HEIGHT_TROPO * GRAV_EARTH / (ROT_RATE_EARTH * RAD_EARTH) ** 2
        np.testing.assert_allclose(result, expected)

    def test_positive(self) -> None:
        """Planetary Burger number is positive."""
        assert plan_burg_num() > 0

    def test_custom_params(self) -> None:
        """Works with custom parameters."""
        result = plan_burg_num(height=1.0, grav=1.0, rot_rate=1.0, radius=1.0)
        np.testing.assert_allclose(result, 1.0)

    def test_scales_with_height(self) -> None:
        """Burger number scales linearly with height."""
        bu1 = plan_burg_num(height=1e4)
        bu2 = plan_burg_num(height=2e4)
        np.testing.assert_allclose(bu2 / bu1, 2.0)


# ---------------------------------------------------------------------------
# TestThermRossNum
# ---------------------------------------------------------------------------


class TestThermRossNum:
    """Tests for therm_ross_num."""

    def test_zero_delta_h(self) -> None:
        """Thermal Rossby number is zero when delta_h is zero."""
        result = therm_ross_num(0.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_positive_for_positive_delta_h(self) -> None:
        """Thermal Rossby number is positive for positive delta_h."""
        result = therm_ross_num(1.0 / 6.0)
        assert result > 0

    def test_default_lat_max_90(self) -> None:
        """Default lat_max of 90 gives sin(90)=1, so result equals delta_h * Bu."""
        delta_h = 0.2
        result = therm_ross_num(delta_h)
        expected = delta_h * plan_burg_num()
        np.testing.assert_allclose(result, expected)

    def test_scales_with_delta_h(self) -> None:
        """Thermal Rossby number scales linearly with delta_h."""
        r1 = therm_ross_num(0.1)
        r2 = therm_ross_num(0.2)
        np.testing.assert_allclose(r2 / r1, 2.0)

    def test_array_delta_h(self) -> None:
        """Works with array-valued delta_h."""
        delta_h = np.array([0.1, 0.2, 0.3])
        result = therm_ross_num(delta_h)
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# TestAbsAngMom
# ---------------------------------------------------------------------------


class TestAbsAngMom:
    """Tests for abs_ang_mom."""

    def test_zero_wind_equator(self) -> None:
        """At the equator with u=0, angular momentum is Omega*a^2."""
        u = _uwind_dataarray([0.0], [0.0])
        result = abs_ang_mom(u)
        expected = ROT_RATE_EARTH * RAD_EARTH**2
        np.testing.assert_allclose(result.item(), expected)

    def test_zero_wind_pole(self) -> None:
        """At the pole with u=0, angular momentum is zero (cos(90)=0)."""
        u = _uwind_dataarray([0.0], [90.0])
        result = abs_ang_mom(u)
        np.testing.assert_allclose(result.item(), 0.0, atol=1e-5)

    def test_nonzero_wind(self) -> None:
        """Angular momentum with nonzero wind at the equator."""
        u_val = 10.0
        u = _uwind_dataarray([u_val], [0.0])
        result = abs_ang_mom(u)
        expected = RAD_EARTH * (ROT_RATE_EARTH * RAD_EARTH + u_val)
        np.testing.assert_allclose(result.item(), expected)

    def test_explicit_lat(self) -> None:
        """Works when lat is explicitly provided."""
        u = _uwind_dataarray([0.0], [45.0])
        result = abs_ang_mom(u, lat=45.0)
        assert isinstance(result, xr.DataArray)

    def test_multiple_latitudes(self) -> None:
        """Works with multiple latitudes."""
        lats = [0.0, 30.0, 60.0, 90.0]
        u = _uwind_dataarray([0.0, 0.0, 0.0, 0.0], lats)
        result = abs_ang_mom(u)
        assert result.shape == (4,)
        # Monotonically decreasing from equator to pole for u=0
        assert result.values[0] > result.values[-1]


# ---------------------------------------------------------------------------
# TestAbsVortVertComp
# ---------------------------------------------------------------------------


class TestAbsVortVertComp:
    """Tests for abs_vort_vert_comp."""

    def test_returns_dataarray(self) -> None:
        """Returns an xarray DataArray."""
        lats = np.linspace(-80, 80, 41)
        u = _uwind_dataarray(np.zeros_like(lats), lats)
        m = abs_ang_mom(u)
        result = abs_vort_vert_comp(m)
        assert isinstance(result, xr.DataArray)

    def test_shape_preserved(self) -> None:
        """Output shape matches input shape."""
        lats = np.linspace(-80, 80, 41)
        u = _uwind_dataarray(np.zeros_like(lats), lats)
        m = abs_ang_mom(u)
        result = abs_vort_vert_comp(m)
        assert result.shape == m.shape

    def test_solid_body_matches_coriolis(self) -> None:
        """For u=0 (solid-body rotation), abs vorticity equals Coriolis parameter."""
        lats = np.linspace(-80, 80, 41)
        u = _uwind_dataarray(np.zeros_like(lats), lats)
        m = abs_ang_mom(u)
        result = abs_vort_vert_comp(m)
        expected = coriolis_param(_lat_dataarray(lats))
        # Finite-difference derivative introduces error at endpoints; compare interior
        np.testing.assert_allclose(
            result.values[2:-2], expected.values[2:-2], rtol=1e-2
        )


# ---------------------------------------------------------------------------
# TestAbsVortFromU
# ---------------------------------------------------------------------------


class TestAbsVortFromU:
    """Tests for abs_vort_from_u."""

    def test_zero_wind(self) -> None:
        """With u=0, absolute vorticity equals planetary vorticity (2*Omega*sin(lat))."""
        lats = np.linspace(-80, 80, 41)
        u = _uwind_dataarray(np.zeros_like(lats), lats)
        result = abs_vort_from_u(u)
        expected = coriolis_param(_lat_dataarray(lats))
        np.testing.assert_allclose(result.values, expected.values, atol=1e-12)

    def test_returns_dataarray(self) -> None:
        """Returns an xarray DataArray."""
        lats = np.linspace(-60, 60, 31)
        u = _uwind_dataarray(np.ones(31) * 5.0, lats)
        result = abs_vort_from_u(u)
        assert isinstance(result, xr.DataArray)

    def test_positive_nh(self) -> None:
        """Absolute vorticity is positive in NH for zero wind."""
        lats = np.linspace(10, 80, 15)
        u = _uwind_dataarray(np.zeros(15), lats)
        result = abs_vort_from_u(u)
        assert (result.values > 0).all()


# ---------------------------------------------------------------------------
# TestRelVortFromU
# ---------------------------------------------------------------------------


class TestRelVortFromU:
    """Tests for rel_vort_from_u."""

    def test_zero_wind(self) -> None:
        """Relative vorticity is zero for zero wind."""
        lats = np.linspace(-80, 80, 41)
        u = _uwind_dataarray(np.zeros(41), lats)
        result = rel_vort_from_u(u)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-15)

    def test_returns_dataarray(self) -> None:
        """Returns an xarray DataArray."""
        lats = np.linspace(-60, 60, 31)
        u = _uwind_dataarray(np.ones(31) * 10.0, lats)
        result = rel_vort_from_u(u)
        assert isinstance(result, xr.DataArray)

    def test_shape_preserved(self) -> None:
        """Output shape matches input shape."""
        lats = np.linspace(-80, 80, 41)
        u = _uwind_dataarray(np.ones(41) * 5.0, lats)
        result = rel_vort_from_u(u)
        assert result.shape == u.shape


# ---------------------------------------------------------------------------
# TestRossNumFromUwind
# ---------------------------------------------------------------------------


class TestRossNumFromUwind:
    """Tests for ross_num_from_uwind."""

    def test_zero_wind(self) -> None:
        """Rossby number is zero for zero wind (away from equator)."""
        lats = np.linspace(10, 80, 15)
        u = _uwind_dataarray(np.zeros(15), lats)
        result = ross_num_from_uwind(u)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-10)

    def test_returns_dataarray(self) -> None:
        """Returns an xarray DataArray."""
        lats = np.linspace(10, 80, 15)
        u = _uwind_dataarray(np.ones(15) * 20.0, lats)
        result = ross_num_from_uwind(u)
        assert isinstance(result, xr.DataArray)

    def test_explicit_lat(self) -> None:
        """Works when lat is explicitly provided."""
        lats = np.linspace(10, 80, 15)
        u = _uwind_dataarray(np.zeros(15), lats)
        result = ross_num_from_uwind(u, lat=lats)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# TestRossNumGen
# ---------------------------------------------------------------------------


class TestRossNumGen:
    """Tests for ross_num_gen."""

    def test_returns_dataarray(self) -> None:
        """Returns an xarray DataArray."""
        lats = np.linspace(10, 80, 15)
        levs = np.array([500.0, 700.0, 850.0])
        uwind = xr.DataArray(
            np.ones((15, 3)),
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        vwind = xr.DataArray(
            np.ones((15, 3)) * 0.5,
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        omega = xr.DataArray(
            np.ones((15, 3)) * 0.01,
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        result = ross_num_gen(uwind, vwind, omega)
        assert isinstance(result, xr.DataArray)

    def test_hpa_to_pa_flag(self) -> None:
        """hpa_to_pa flag scales du_dp by 1e-2."""
        lats = np.linspace(10, 80, 15)
        levs = np.array([500.0, 700.0, 850.0])
        # Use non-uniform uwind so du_dp is nonzero.
        u_vals = np.outer(np.ones(15), np.array([5.0, 10.0, 15.0]))
        uwind = xr.DataArray(
            u_vals,
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        vwind = xr.DataArray(
            np.ones((15, 3)) * 0.5,
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        omega = xr.DataArray(
            np.ones((15, 3)) * 0.01,
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        r_pa = ross_num_gen(uwind, vwind, omega, hpa_to_pa=False)
        r_hpa = ross_num_gen(uwind, vwind, omega, hpa_to_pa=True)
        # They should differ because hpa_to_pa scales the du_dp term
        assert not np.allclose(r_pa.values, r_hpa.values)


# ---------------------------------------------------------------------------
# TestBruntVaisalaFreq
# ---------------------------------------------------------------------------


class TestBruntVaisalaFreq:
    """Tests for brunt_vaisala_freq."""

    def test_positive_for_stable(self) -> None:
        """Brunt-Vaisala frequency is real and positive for stable stratification."""
        result = brunt_vaisala_freq(3e-3)
        assert result > 0

    def test_zero_for_neutral(self) -> None:
        """Brunt-Vaisala frequency is zero for neutral stratification."""
        result = brunt_vaisala_freq(0.0)
        np.testing.assert_allclose(result, 0.0)

    def test_custom_params(self) -> None:
        """Works with custom theta_ref and grav."""
        result = brunt_vaisala_freq(1.0, theta_ref=1.0, grav=1.0)
        np.testing.assert_allclose(result, 1.0)

    def test_array_input(self) -> None:
        """Works with array inputs."""
        dtheta_dz = np.array([3e-3, 4e-3, 5e-3])
        result = brunt_vaisala_freq(dtheta_dz)
        assert result.shape == (3,)

    def test_scales_with_sqrt(self) -> None:
        """Frequency scales with sqrt of dtheta_dz."""
        n1 = brunt_vaisala_freq(1e-3)
        n4 = brunt_vaisala_freq(4e-3)
        np.testing.assert_allclose(n4 / n1, 2.0)


# ---------------------------------------------------------------------------
# TestRossbyRadius
# ---------------------------------------------------------------------------


class TestRossbyRadius:
    """Tests for rossby_radius."""

    def test_positive(self) -> None:
        """Rossby radius is positive in the NH."""
        result = rossby_radius(45.0, dtheta_dz=3e-3)
        assert result > 0

    def test_larger_at_lower_lat(self) -> None:
        """Rossby radius is larger at lower latitudes (weaker f)."""
        r_low = rossby_radius(20.0, dtheta_dz=3e-3)
        r_high = rossby_radius(60.0, dtheta_dz=3e-3)
        assert r_low > r_high

    def test_larger_with_stronger_stratification(self) -> None:
        """Rossby radius increases with stronger stratification."""
        r_weak = rossby_radius(45.0, dtheta_dz=1e-3)
        r_strong = rossby_radius(45.0, dtheta_dz=4e-3)
        assert r_strong > r_weak


# ---------------------------------------------------------------------------
# TestZonalFricInferredSteady
# ---------------------------------------------------------------------------


class TestZonalFricInferredSteady:
    """Tests for zonal_fric_inferred_steady."""

    def test_zero_inputs(self) -> None:
        """Zero fluxes and wind give zero friction."""
        lats = np.linspace(-80, 80, 41)
        levs = np.array([500.0, 700.0, 850.0])
        zeros = xr.DataArray(
            np.zeros((41, 3)),
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        result = zonal_fric_inferred_steady(zeros, zeros, zeros)
        assert isinstance(result, xr.DataArray)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# TestZFromHypso
# ---------------------------------------------------------------------------


class TestZFromHypso:
    """Tests for z_from_hypso."""

    def test_returns_dataarray(self) -> None:
        """Returns an xarray DataArray."""
        lats = np.array([0.0, 30.0, 60.0])
        levs = np.array([100000.0, 85000.0, 50000.0, 20000.0])
        temp = xr.DataArray(
            np.full((3, 4), 250.0),
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        result = z_from_hypso(temp)
        assert isinstance(result, xr.DataArray)

    def test_height_increases_with_lower_pressure(self) -> None:
        """Height increases as pressure decreases (levs ordered high-to-low)."""
        lats = np.array([0.0])
        levs = np.array([100000.0, 85000.0, 50000.0, 20000.0])
        temp = xr.DataArray(
            np.full((1, 4), 250.0),
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        result = z_from_hypso(temp)
        heights = result.isel({LAT_STR: 0}).values
        valid = heights[~np.isnan(heights)]
        # Heights should monotonically increase toward lower pressure (index 0→3)
        assert valid.size > 1
        assert np.all(np.diff(valid) >= 0)

    def test_warmer_atmosphere_gives_larger_heights(self) -> None:
        """A warmer atmosphere yields larger heights at the same pressure."""
        lats = np.array([0.0])
        levs = np.array([100000.0, 50000.0, 20000.0])
        temp_cold = xr.DataArray(
            np.full((1, 3), 200.0),
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        temp_warm = xr.DataArray(
            np.full((1, 3), 300.0),
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        z_cold = z_from_hypso(temp_cold)
        z_warm = z_from_hypso(temp_warm)
        # Compare at lowest-pressure level (highest altitude)
        idx = {LAT_STR: 0, LEV_STR: -1}
        assert z_warm.isel(idx).item() > z_cold.isel(idx).item()


# ---------------------------------------------------------------------------
# TestUBci2layerQg
# ---------------------------------------------------------------------------


class TestUBci2layerQg:
    """Tests for u_bci_2layer_qg."""

    def test_positive_nh(self) -> None:
        """Critical shear is positive in the Northern Hemisphere."""
        result = u_bci_2layer_qg(45.0)
        assert result > 0

    def test_larger_near_equator(self) -> None:
        """Critical shear is larger near the equator (weaker f)."""
        r_low = u_bci_2layer_qg(10.0)
        r_high = u_bci_2layer_qg(60.0)
        assert r_low > r_high

    def test_symmetric(self) -> None:
        """Critical shear magnitude is symmetric about the equator."""
        # cos(lat)/sin^2(lat) is positive for lat > 0, negative for lat < 0
        # but the function uses cos/sin^2, so the sign depends on cos(lat)
        r_n = u_bci_2layer_qg(45.0)
        r_s = u_bci_2layer_qg(-45.0)
        # cos(-45) = cos(45), sin(-45)^2 = sin(45)^2, so should be equal
        np.testing.assert_allclose(r_n, r_s)

    def test_array_input(self) -> None:
        """Works with array input."""
        lats = np.array([10.0, 30.0, 45.0, 60.0])
        result = u_bci_2layer_qg(lats)
        assert result.shape == (4,)

    def test_dataarray_input(self) -> None:
        """Works with DataArray input."""
        lats = _lat_dataarray([10.0, 30.0, 45.0, 60.0])
        result = u_bci_2layer_qg(lats)
        assert isinstance(result, xr.DataArray)


# ---------------------------------------------------------------------------
# TestBulkStatStab
# ---------------------------------------------------------------------------


class TestBulkStatStab:
    """Tests for bulk_stat_stab."""

    def test_zero_for_uniform(self) -> None:
        """Bulk static stability is zero for uniform potential temperature."""
        levs = np.array([200.0, 500.0, 700.0, 850.0, 1000.0])
        pot_temp = xr.DataArray(
            np.full(5, 300.0), coords={LEV_STR: levs}, dims=[LEV_STR]
        )
        result = bulk_stat_stab(pot_temp)
        np.testing.assert_allclose(result.item(), 0.0)

    def test_positive_for_stable(self) -> None:
        """Positive static stability when upper level is warmer in theta."""
        levs = np.array([200.0, 500.0, 700.0, 850.0, 1000.0])
        # Potential temperature increasing with height (decreasing pressure)
        theta_vals = np.array([350.0, 320.0, 305.0, 295.0, 290.0])
        pot_temp = xr.DataArray(theta_vals, coords={LEV_STR: levs}, dims=[LEV_STR])
        result = bulk_stat_stab(pot_temp)
        assert result.item() > 0

    def test_custom_levels(self) -> None:
        """Works with custom upper/lower levels."""
        levs = np.array([200.0, 500.0, 700.0, 850.0, 1000.0])
        theta_vals = np.array([350.0, 320.0, 305.0, 295.0, 290.0])
        pot_temp = xr.DataArray(theta_vals, coords={LEV_STR: levs}, dims=[LEV_STR])
        result = bulk_stat_stab(pot_temp, lev_upper=200, lev_lower=1000)
        expected = (350.0 - 290.0) / 300.0
        np.testing.assert_allclose(result.item(), expected)

    def test_custom_ref(self) -> None:
        """Works with custom reference potential temperature."""
        levs = np.array([500.0, 850.0])
        theta_vals = np.array([320.0, 295.0])
        pot_temp = xr.DataArray(theta_vals, coords={LEV_STR: levs}, dims=[LEV_STR])
        result = bulk_stat_stab(pot_temp, pot_temp_ref=250.0)
        expected = (320.0 - 295.0) / 250.0
        np.testing.assert_allclose(result.item(), expected)

    def test_2d_input(self) -> None:
        """Works with 2D (lat x lev) input."""
        lats = np.array([0.0, 30.0, 60.0])
        levs = np.array([200.0, 500.0, 700.0, 850.0, 1000.0])
        theta = xr.DataArray(
            np.tile([350.0, 320.0, 305.0, 295.0, 290.0], (3, 1)),
            coords={LAT_STR: lats, LEV_STR: levs},
            dims=[LAT_STR, LEV_STR],
        )
        result = bulk_stat_stab(theta)
        assert result.shape == (3,)
