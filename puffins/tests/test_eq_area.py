"""Tests for eq_area module.

Known-value tests reconstruct expected outputs from raw numpy per the repo
test convention.  The equal-area residual tests are the strongest checks:
they verify that each analytical solution (cell edge, equatorial potential
temperature, meridional profile) satisfies the two defining equal-area
constraints (temperature continuity at the cell edge, zero net energy
input over the cell) rather than re-deriving any closed form.

Numerical-solver anchor values (cell edges, inner-edge temperatures) were
pinned from converged solves with residuals below 1e-8, verified 2026-07-15.

Every parameter is passed explicitly: eq_area functions default to
different tau/height/theta_ref values than sibling modules, so relying on
defaults invites silent inconsistencies.
"""

import types
from collections.abc import Callable

import numpy as np
import pytest
import scipy.optimize

from puffins.eq_area import (
    _eq_area_cqe_fixed_tt,
    _equal_area_model_bouss,
    _equal_area_model_cqe,
    _lh88_model,
    _lh88_model_fixed_tt,
    cell_edge_lin_ro_lata0_approx,
    cell_edge_lin_ro_lata0_full,
    cell_edge_mean_ro,
    eq_pot_temp_lin_ro_lata0_small_ang,
    eq_pot_temp_mean_ro,
    equal_area_bouss,
    equal_area_cqe,
    equal_area_cqe_fixed_tt,
    equal_area_lh88,
    equal_area_lh88_fixed_temp_tropo,
    heat_flux_mean_ro,
    mom_flux_mean_ro,
    pot_temp_lin_ro_eq_area,
    pot_temp_mean_ro,
    u_sfc_mean_ro,
)

# ---------------------------------------------------------------------------
# Explicit test parameters (never rely on module defaults).
# ---------------------------------------------------------------------------

THETA_REF = 330.0
DELTA_H = 1.0 / 3.0
DELTA_V = 1.0 / 8.0
RADIUS = 6.371e6
ROT_RATE = 7.292e-5
HEIGHT = 12.0e3
TAU = 30.0 * 86400.0
DRAG_COEFF = 0.008
C_P = 1003.5

_IGNORE_QUAD_WARNINGS = pytest.mark.filterwarnings(
    "ignore::scipy.integrate.IntegrationWarning"
)
# The root solver's trial iterates can transiently leave |sin(lat)| <= 1
# (or drive a sqrt argument negative in the fixed-tt temperature field),
# emitting "invalid value encountered" RuntimeWarnings that CI promotes to
# errors.  The residual assertions on the final solution are what guarantee
# correctness, so these excursions are expected root-finding noise.
_IGNORE_SOLVER_WANDER = pytest.mark.filterwarnings(
    "ignore:invalid value encountered:RuntimeWarning"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rce_small_angle(
    lat_rad: np.ndarray, delta_h: float, theta_ref: float
) -> np.ndarray:
    """RCE profile, quadratic in latitude (radians): ms/HH80 forcing."""
    return theta_ref * (1 + delta_h / 3 - delta_h * lat_rad**2)


def _rce_lh88_func(
    sinlat_0: float, theta_ref: float, delta_h: float
) -> Callable[[float], float]:
    """RCE profile as a function of sin(lat), LH88 Eq. (1b) form."""

    def _rce(sinlat: float) -> float:
        return theta_ref * (1 + delta_h / 3 - delta_h * (sinlat - sinlat_0) ** 2)

    return _rce


def _edge_lin_ro_full_raw_rad(therm: float, ro_a: float, ro_d: float) -> float:
    """Cell edge (radians) from the linear-Ro quartic, via raw numpy."""
    delro = ro_a - ro_d
    c4 = ro_a**2 / 7 - ro_a * delro / 6 + 4 * delro**2 / 81
    c2 = 2 * ro_a / 5 - 2 * delro / 9
    c0 = -2 * therm / 3
    latd_sq = (-c2 + np.sqrt(c2**2 - 4 * c4 * c0)) / (2 * c4)
    return float(np.sqrt(latd_sq))


def _equal_area_residuals(
    lat_rad: np.ndarray, theta: np.ndarray, delta_h: float, theta_ref: float
) -> tuple[float, float]:
    """Continuity residual (K) at the edge and relative energy-integral residual."""
    diff = theta - _rce_small_angle(lat_rad, delta_h, theta_ref)
    scale = np.abs(diff).max() * lat_rad[-1]
    return float(diff[-1]), float(np.trapezoid(diff, lat_rad) / scale)


def _fake_failed_root(*args: object, **kwargs: object) -> types.SimpleNamespace:
    """Stand-in for scipy.optimize.root reporting non-convergence."""
    x0 = np.asarray(args[1], dtype=float)
    return types.SimpleNamespace(success=False, x=x0, message="mocked non-convergence")


# ---------------------------------------------------------------------------
# Uniform-Ro closed forms
# ---------------------------------------------------------------------------


class TestCellEdgeMeanRo:
    """Tests for cell_edge_mean_ro."""

    def test_known_value(self) -> None:
        """Raw-numpy reconstruction of ms Eq. (13)."""
        ro, therm = 0.5, 0.1
        expected = np.rad2deg(np.sqrt(5 * therm / (3 * ro)))
        np.testing.assert_allclose(cell_edge_mean_ro(ro, therm), expected, rtol=1e-13)

    def test_reduces_to_hh80_at_unit_ro(self) -> None:
        """Ro=1 recovers the Held-Hou 1980 small-angle edge."""
        therm = 0.1
        np.testing.assert_allclose(
            cell_edge_mean_ro(1.0, therm),
            np.rad2deg(np.sqrt(5 * therm / 3)),
            rtol=1e-13,
        )

    def test_inverse_sqrt_ro_scaling(self) -> None:
        """Edge scales as Ro^(-1/2): quartering Ro doubles the edge."""
        therm = 0.1
        np.testing.assert_allclose(
            cell_edge_mean_ro(0.25, therm),
            2 * cell_edge_mean_ro(1.0, therm),
            rtol=1e-13,
        )


class TestEqPotTempMeanRo:
    """Tests for eq_pot_temp_mean_ro."""

    def test_known_value(self) -> None:
        """Raw-numpy reconstruction of ms Eq. (15)."""
        ro, therm = 0.5, 0.1
        expected = THETA_REF * (1 + DELTA_H / 3 - 5 * therm * DELTA_H / (18 * ro))
        np.testing.assert_allclose(
            eq_pot_temp_mean_ro(ro, therm, DELTA_H, theta_ref=THETA_REF),
            expected,
            rtol=1e-13,
        )

    def test_cooler_as_ro_decreases(self) -> None:
        """Equatorial temperature drops monotonically as Ro decreases."""
        therm = 0.1
        ros = np.array([1.0, 0.8, 0.5, 0.27])
        temps = np.array(
            [eq_pot_temp_mean_ro(ro, therm, DELTA_H, theta_ref=THETA_REF) for ro in ros]
        )
        assert np.all(np.diff(temps) < 0)


class TestUniformRoEqualArea:
    """The uniform-Ro solution satisfies the equal-area constraints."""

    @pytest.mark.parametrize("therm", [0.02, 0.1, 0.3])
    @pytest.mark.parametrize("ro", [1.0, 0.8, 0.5, 0.27])
    def test_continuity_and_energy_constraints(self, ro: float, therm: float) -> None:
        """Profile meets RCE at the edge; net energy over the cell is zero."""
        edge_rad = np.deg2rad(cell_edge_mean_ro(ro, therm))
        lat_rad = np.linspace(0.0, edge_rad, 4001)
        theta = pot_temp_mean_ro(
            np.rad2deg(lat_rad), ro, therm, DELTA_H, theta_ref=THETA_REF
        )
        cont, integ_rel = _equal_area_residuals(lat_rad, theta, DELTA_H, THETA_REF)
        np.testing.assert_allclose(cont, 0.0, atol=1e-9)
        assert abs(integ_rel) < 1e-6

    def test_profile_known_value(self) -> None:
        """Raw-numpy reconstruction of ms Eqs. (10) + (15)."""
        ro, therm = 0.8, 0.1
        lats = np.array([0.0, 8.0, 15.0])
        theta_eq = THETA_REF * (1 + DELTA_H / 3 - 5 * therm * DELTA_H / (18 * ro))
        expected = theta_eq - (
            THETA_REF * 0.5 * ro * DELTA_H / therm * np.deg2rad(lats) ** 4
        )
        actual = pot_temp_mean_ro(lats, ro, therm, DELTA_H, theta_ref=THETA_REF)
        np.testing.assert_allclose(actual, expected, rtol=1e-13)


# ---------------------------------------------------------------------------
# Uniform-Ro flux and surface-wind fields
# ---------------------------------------------------------------------------


class TestHeatFluxMeanRo:
    """Tests for heat_flux_mean_ro."""

    def test_known_value(self) -> None:
        """Raw-numpy reconstruction of ms Eq. (16)."""
        ro, therm = 0.5, 0.1
        lats = np.array([5.0, 12.0, 20.0])
        edge_rad = np.sqrt(5 * therm / (3 * ro))
        x = np.deg2rad(lats) / edge_rad
        prefac = (
            (5 / 18)
            * (5 / 3) ** 0.5
            * RADIUS
            * HEIGHT
            * DELTA_H
            / TAU
            * (therm / ro) ** 1.5
        )
        expected = THETA_REF * prefac * (x - 2 * x**3 + x**5)
        actual = heat_flux_mean_ro(
            lats,
            therm,
            ro,
            theta_ref=THETA_REF,
            radius=RADIUS,
            height=HEIGHT,
            delta_h=DELTA_H,
            tau=TAU,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_vanishes_at_equator_and_edge(self) -> None:
        """Heat flux is zero at the equator and at the cell edge."""
        ro, therm = 0.5, 0.1
        edge = cell_edge_mean_ro(ro, therm)
        vals = heat_flux_mean_ro(
            np.array([0.0, edge]),
            therm,
            ro,
            theta_ref=THETA_REF,
            radius=RADIUS,
            height=HEIGHT,
            delta_h=DELTA_H,
            tau=TAU,
        )
        np.testing.assert_allclose(vals, 0.0, atol=1e-8)

    def test_positive_within_cell(self) -> None:
        """Poleward (positive) heat flux everywhere inside the cell."""
        ro, therm = 0.5, 0.1
        edge = cell_edge_mean_ro(ro, therm)
        lats = np.linspace(0.02, 0.98, 40) * edge
        vals = heat_flux_mean_ro(
            lats,
            therm,
            ro,
            theta_ref=THETA_REF,
            radius=RADIUS,
            height=HEIGHT,
            delta_h=DELTA_H,
            tau=TAU,
        )
        assert np.all(vals > 0)

    def test_increases_as_ro_decreases(self) -> None:
        """At fixed latitude within the cell, smaller Ro gives more flux."""
        therm = 0.1
        ro_hi, ro_lo = 1.0, 0.5
        lats = np.linspace(0.05, 0.9, 20) * cell_edge_mean_ro(ro_hi, therm)
        kwargs = dict(
            theta_ref=THETA_REF,
            radius=RADIUS,
            height=HEIGHT,
            delta_h=DELTA_H,
            tau=TAU,
        )
        flux_hi = heat_flux_mean_ro(lats, therm, ro_hi, **kwargs)
        flux_lo = heat_flux_mean_ro(lats, therm, ro_lo, **kwargs)
        assert np.all(flux_lo > flux_hi)


class TestMomFluxMeanRo:
    """Tests for mom_flux_mean_ro."""

    def test_known_value(self) -> None:
        """Raw-numpy reconstruction of the expanded ms Eq. (17)."""
        ro, therm = 0.5, 0.1
        lats = np.array([5.0, 12.0, 20.0])
        latrad = np.deg2rad(lats)
        prefac = ROT_RATE * RADIUS**2 * HEIGHT * DELTA_H / (6 * TAU * DELTA_V)
        expected = (
            prefac
            * latrad**3
            * (5 * therm / 3 - ro * latrad**2 * (2 - 3 * ro * latrad**2 / (5 * therm)))
        )
        actual = mom_flux_mean_ro(
            lats,
            therm,
            ro,
            radius=RADIUS,
            rot_rate=ROT_RATE,
            height=HEIGHT,
            delta_h=DELTA_H,
            delta_v=DELTA_V,
            tau=TAU,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_consistent_with_heat_flux(self) -> None:
        """ms Eq. (17): mom flux = Ro*Omega*a*lat^2/Delta_v * heat flux/theta0.

        This is the self-similarity identity relating the two fluxes, so it
        cross-checks the two implementations against each other through an
        independent algebraic route.
        """
        ro, therm = 0.5, 0.1
        lats = np.linspace(0.5, 0.95, 7) * cell_edge_mean_ro(ro, therm)
        heat = heat_flux_mean_ro(
            lats,
            therm,
            ro,
            theta_ref=THETA_REF,
            radius=RADIUS,
            height=HEIGHT,
            delta_h=DELTA_H,
            tau=TAU,
        )
        mom = mom_flux_mean_ro(
            lats,
            therm,
            ro,
            radius=RADIUS,
            rot_rate=ROT_RATE,
            height=HEIGHT,
            delta_h=DELTA_H,
            delta_v=DELTA_V,
            tau=TAU,
        )
        identity = (
            ro * ROT_RATE * RADIUS * np.deg2rad(lats) ** 2 / DELTA_V * heat / THETA_REF
        )
        np.testing.assert_allclose(mom, identity, rtol=1e-12)

    def test_increases_as_ro_decreases(self) -> None:
        """At fixed latitude within the cell, smaller Ro gives more momentum flux.

        The counterintuitive prediction discussed in the fixed-Ro manuscript:
        d(flux)/d(Ro) = -2*prefac*lat^5*(1 - (lat/edge)^2) < 0 inside the cell.
        """
        therm = 0.1
        ro_hi, ro_lo = 1.0, 0.5
        lats = np.linspace(0.05, 0.9, 20) * cell_edge_mean_ro(ro_hi, therm)
        kwargs = dict(
            radius=RADIUS,
            rot_rate=ROT_RATE,
            height=HEIGHT,
            delta_h=DELTA_H,
            delta_v=DELTA_V,
            tau=TAU,
        )
        flux_hi = mom_flux_mean_ro(lats, therm, ro_hi, **kwargs)
        flux_lo = mom_flux_mean_ro(lats, therm, ro_lo, **kwargs)
        assert np.all(flux_lo > flux_hi)


class TestUSfcMeanRo:
    """Tests for u_sfc_mean_ro."""

    def test_known_value(self) -> None:
        """Raw-numpy reconstruction of ms Eq. (18)."""
        ro, therm = 0.5, 0.1
        lats = np.array([5.0, 12.0, 20.0])
        edge = np.rad2deg(np.sqrt(5 * therm / (3 * ro)))
        x = lats / edge
        prefac = (
            -25
            * ROT_RATE
            * RADIUS
            * HEIGHT
            * DELTA_H
            / (18 * DRAG_COEFF * TAU * DELTA_V)
        )
        expected = (
            prefac * (therm / ro) ** 2 * (x**2 - (10 / 3) * x**4 + (7 / 3) * x**6)
        )
        actual = u_sfc_mean_ro(
            lats,
            therm,
            ro,
            radius=RADIUS,
            rot_rate=ROT_RATE,
            height=HEIGHT,
            delta_h=DELTA_H,
            delta_v=DELTA_V,
            tau=TAU,
            drag_coeff=DRAG_COEFF,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_easterly_westerly_transition(self) -> None:
        """Easterlies for lat < sqrt(3/7)*edge, westerlies poleward to the edge.

        The bracket factors as x^2*(1 - x^2)*(1 - 7x^2/3), giving interior
        zeros at x = sqrt(3/7) and x = 1.
        """
        ro, therm = 0.5, 0.1
        edge = cell_edge_mean_ro(ro, therm)
        x_zero = np.sqrt(3 / 7)
        kwargs = dict(
            radius=RADIUS,
            rot_rate=ROT_RATE,
            height=HEIGHT,
            delta_h=DELTA_H,
            delta_v=DELTA_V,
            tau=TAU,
            drag_coeff=DRAG_COEFF,
        )
        easterly = u_sfc_mean_ro(
            np.linspace(0.05, 0.95, 20) * x_zero * edge, therm, ro, **kwargs
        )
        westerly = u_sfc_mean_ro(
            np.linspace(1.05 * x_zero, 0.98, 20) * edge, therm, ro, **kwargs
        )
        at_zero = u_sfc_mean_ro(np.array([x_zero * edge]), therm, ro, **kwargs)
        typical = np.abs(easterly).max()
        assert np.all(easterly < 0)
        assert np.all(westerly > 0)
        np.testing.assert_allclose(at_zero, 0.0, atol=1e-10 * typical)


# ---------------------------------------------------------------------------
# Linear-Ro closed forms
# ---------------------------------------------------------------------------


class TestCellEdgeLinRo:
    """Tests for cell_edge_lin_ro_lata0_full and _approx."""

    def test_full_known_value(self) -> None:
        """Raw-numpy quadratic-formula reconstruction of ms Eq. (7)."""
        therm, ro_a, ro_d = 0.1, 0.7, 0.3
        expected = np.rad2deg(_edge_lin_ro_full_raw_rad(therm, ro_a, ro_d))
        np.testing.assert_allclose(
            cell_edge_lin_ro_lata0_full(therm, ro_a, ro_d),
            expected,
            rtol=1e-12,
        )

    def test_approx_known_value(self) -> None:
        """Raw-numpy reconstruction of ms Eq. (8)."""
        therm, ro_a, ro_d = 0.1, 0.7, 0.3
        expected = np.rad2deg(np.sqrt(15 * therm / (4 * ro_a + 5 * ro_d)))
        np.testing.assert_allclose(
            cell_edge_lin_ro_lata0_approx(therm, ro_a, ro_d),
            expected,
            rtol=1e-12,
        )

    def test_approx_reduces_to_uniform_when_flat(self) -> None:
        """delta_Ro = 0 recovers the uniform-Ro edge exactly."""
        therm, ro = 0.1, 0.5
        np.testing.assert_allclose(
            cell_edge_lin_ro_lata0_approx(therm, ro, ro),
            cell_edge_mean_ro(ro, therm),
            rtol=1e-13,
        )

    def test_full_close_to_uniform_when_flat(self) -> None:
        """delta_Ro = 0 full solution is near (but poleward-truncated from)
        the uniform-Ro edge, which drops the quartic term."""
        therm, ro = 0.02, 0.5
        full = cell_edge_lin_ro_lata0_full(therm, ro, ro)
        unif = cell_edge_mean_ro(ro, therm)
        assert full < unif
        np.testing.assert_allclose(full, unif, rtol=0.02)

    def test_full_equatorward_of_approx(self) -> None:
        """Retaining the (positive-coefficient) quartic term pulls the edge
        equatorward, by under 1 degree for manuscript-typical parameters."""
        for therm, ro_a, ro_d in [(0.02, 1.0, 0.0), (0.1, 1.0, 0.0), (0.1, 0.7, 0.3)]:
            full = cell_edge_lin_ro_lata0_full(therm, ro_a, ro_d)
            approx = cell_edge_lin_ro_lata0_approx(therm, ro_a, ro_d)
            assert full < approx
            assert approx - full < 1.0

    def test_edge_moves_poleward_with_delta_ro(self) -> None:
        """At fixed Ro_a, increasing delta_Ro moves the edge poleward."""
        therm, ro_a = 0.1, 1.0
        edges_full = [
            cell_edge_lin_ro_lata0_full(therm, ro_a, ro_a - delro)
            for delro in (0.0, 0.3, 0.6, 1.0)
        ]
        edges_approx = [
            cell_edge_lin_ro_lata0_approx(therm, ro_a, ro_a - delro)
            for delro in (0.0, 0.3, 0.6, 1.0)
        ]
        assert np.all(np.diff(edges_full) > 0)
        assert np.all(np.diff(edges_approx) > 0)

    def test_full_returns_real_scalar(self) -> None:
        """The full solution is a real (not complex) scalar."""
        edge = cell_edge_lin_ro_lata0_full(0.1, 0.7, 0.3)
        assert np.isrealobj(edge)

    def test_zero_thermal_ro_gives_zero_edge(self) -> None:
        """No thermal forcing means no cell."""
        np.testing.assert_allclose(
            cell_edge_lin_ro_lata0_full(0.0, 0.7, 0.3), 0.0, atol=1e-13
        )

    def test_full_raises_when_no_root(self) -> None:
        """Ro_a = Ro_d = 0 with nonzero forcing leaves the cell-edge
        polynomial with no non-negative real root."""
        with pytest.raises(ValueError, match="non-negative real root"):
            cell_edge_lin_ro_lata0_full(0.1, 0.0, 0.0)


class TestEqPotTempLinRo:
    """Tests for eq_pot_temp_lin_ro_lata0_small_ang."""

    def test_known_value_via_edge_continuity(self) -> None:
        """Independent reconstruction: RCE at the (raw-numpy) edge plus the
        gradient-balance drop from ms Eq. (6) evaluated at the edge."""
        therm, ro_a, ro_d = 0.1, 1.0, 0.5
        delro = ro_a - ro_d
        burg = therm / DELTA_H
        latd = _edge_lin_ro_full_raw_rad(therm, ro_a, ro_d)
        latd2 = latd**2
        bracket = (
            ro_a / 2
            - 4 * delro / 15
            + ro_a**2 * latd2 / 6
            - 4 * ro_a * delro * latd2 / 21
            + delro**2 * latd2 / 18
        )
        rce_at_edge = THETA_REF * (1 + DELTA_H / 3 - DELTA_H * latd2)
        expected = rce_at_edge + THETA_REF * latd2**2 / burg * bracket
        actual = eq_pot_temp_lin_ro_lata0_small_ang(
            therm, ro_a, ro_d, DELTA_H, theta_ref=THETA_REF
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13)


class TestLinRoEqualArea:
    """The linear-Ro solution satisfies the equal-area constraints.

    This exercises the full chain: quartic cell edge (ms Eq. 7),
    edge-continuity equatorial temperature, and the Eq. (6) profile,
    all of which must use the same Burger number Ro_th/delta_h.
    """

    @pytest.mark.parametrize("therm", [0.02, 0.1])
    @pytest.mark.parametrize(
        "ro_a,ro_d", [(1.0, 0.0), (0.7, 0.3), (0.5, 0.5), (1.0, 0.5)]
    )
    def test_continuity_and_energy_constraints(
        self, ro_a: float, ro_d: float, therm: float
    ) -> None:
        """Profile meets RCE at the edge; net energy over the cell is zero."""
        edge_rad = np.deg2rad(cell_edge_lin_ro_lata0_full(therm, ro_a, ro_d))
        lat_rad = np.linspace(0.0, edge_rad, 4001)
        theta = pot_temp_lin_ro_eq_area(
            np.rad2deg(lat_rad),
            therm,
            ro_a,
            ro_d,
            delta_h=DELTA_H,
            theta_ref=THETA_REF,
        )
        cont, integ_rel = _equal_area_residuals(lat_rad, theta, DELTA_H, THETA_REF)
        np.testing.assert_allclose(cont, 0.0, atol=1e-9)
        assert abs(integ_rel) < 1e-6


# ---------------------------------------------------------------------------
# Numerical equal-area solvers
# ---------------------------------------------------------------------------


@_IGNORE_QUAD_WARNINGS
@_IGNORE_SOLVER_WANDER
class TestEqualAreaLh88:
    """Tests for equal_area_lh88."""

    def test_equinoctial_symmetric(self) -> None:
        """Equatorial forcing gives a mirror-symmetric two-cell solution
        whose edges match the HH80 small-angle prediction."""
        therm = 0.02
        guess_edge = np.sin(np.sqrt(5 * therm / 3))
        init = [-guess_edge, 0.0, guess_edge, THETA_REF * (1 + DELTA_H / 3)]
        sol = equal_area_lh88(init, 0.0, THETA_REF, DELTA_H, therm)
        resid = _lh88_model(sol, 0.0, THETA_REF, DELTA_H, therm)
        np.testing.assert_allclose(np.abs(resid).max(), 0.0, atol=1e-6)
        np.testing.assert_allclose(sol[0], -sol[2], atol=1e-9)
        np.testing.assert_allclose(sol[1], 0.0, atol=1e-6)
        edge_deg = np.rad2deg(np.arcsin(sol[2]))
        hh80 = np.rad2deg(np.sqrt(5 * therm / 3))
        np.testing.assert_allclose(edge_deg, hh80, rtol=0.02)

    def test_solstitial_asymmetric(self) -> None:
        """Off-equatorial forcing: the cells' shared edge lies well poleward
        of the forcing maximum (LH88's nonlinear amplification), and the
        winter cell is far wider than the summer cell."""
        therm = 0.1
        sinlat_0 = np.sin(np.deg2rad(6.0))
        init = [-0.5, 0.25, 0.55, THETA_REF * (1 + DELTA_H / 3)]
        sol = equal_area_lh88(init, sinlat_0, THETA_REF, DELTA_H, therm)
        resid = _lh88_model(sol, sinlat_0, THETA_REF, DELTA_H, therm)
        np.testing.assert_allclose(np.abs(resid).max(), 0.0, atol=1e-6)
        assert sol[1] > sinlat_0
        winter_width = abs(sol[0] - sol[1])
        summer_width = abs(sol[2] - sol[1])
        assert winter_width > 2 * summer_width
        # Anchor from a converged solve (residuals ~1e-9), 2026-07-15.
        np.testing.assert_allclose(np.rad2deg(np.arcsin(sol[1])), 19.63, atol=0.2)


@_IGNORE_QUAD_WARNINGS
@_IGNORE_SOLVER_WANDER
class TestEqualAreaBouss:
    """Tests for equal_area_bouss."""

    def test_matches_lh88_solver(self) -> None:
        """With the LH88 RCE profile it reproduces equal_area_lh88."""
        therm = 0.02
        sinlat_0 = 0.0
        guess_edge = np.sin(np.sqrt(5 * therm / 3))
        init = [-guess_edge, 0.0, guess_edge, THETA_REF * (1 + DELTA_H / 3)]
        sol_lh88 = equal_area_lh88(init, sinlat_0, THETA_REF, DELTA_H, therm)
        sol_bouss = equal_area_bouss(
            init,
            THETA_REF,
            DELTA_H / therm,
            _rce_lh88_func(sinlat_0, THETA_REF, DELTA_H),
        )
        np.testing.assert_allclose(sol_bouss, sol_lh88, atol=1e-7)

    @pytest.mark.parametrize("ro", [1.0, 0.5])
    def test_ro_scaled_matches_small_angle_analytics(self, ro: float) -> None:
        """Scaling del_h_over_ro by Ro solves the fixed-Ro equal-area model;
        at small Ro_th it must agree with the small-angle closed forms for
        the cell edge and equatorial temperature."""
        therm = 0.005
        edge_analytic = cell_edge_mean_ro(ro, therm)
        theta_eq_analytic = eq_pot_temp_mean_ro(ro, therm, DELTA_H, theta_ref=THETA_REF)
        guess_edge = np.sin(np.deg2rad(edge_analytic))
        init = [-guess_edge, 0.0, guess_edge, THETA_REF * (1 + DELTA_H / 3)]
        sol = equal_area_bouss(
            init,
            THETA_REF,
            ro * DELTA_H / therm,
            _rce_lh88_func(0.0, THETA_REF, DELTA_H),
        )
        resid = _equal_area_model_bouss(
            sol,
            THETA_REF,
            ro * DELTA_H / therm,
            _rce_lh88_func(0.0, THETA_REF, DELTA_H),
        )
        np.testing.assert_allclose(np.abs(resid).max(), 0.0, atol=1e-6)
        edge_num = np.rad2deg(np.arcsin(sol[2]))
        np.testing.assert_allclose(edge_num, edge_analytic, rtol=0.02)
        np.testing.assert_allclose(sol[3], theta_eq_analytic, atol=0.01)


@_IGNORE_QUAD_WARNINGS
@_IGNORE_SOLVER_WANDER
class TestEqualAreaCqe:
    """Tests for equal_area_cqe."""

    def test_equinoctial_symmetric(self) -> None:
        """Equatorial forcing gives a converged, mirror-symmetric solution."""
        theta_b_ref = 345.0
        delta_h_b = 1.0 / 6.0
        vert_diff = 100.0
        rce_b = _rce_lh88_func(0.0, theta_b_ref, delta_h_b)
        guess_edge = np.sin(np.deg2rad(20.0))
        init = [-guess_edge, 0.0, guess_edge, theta_b_ref * (1 + delta_h_b / 3)]
        sol = equal_area_cqe(init, vert_diff, C_P, RADIUS, ROT_RATE, rce_b)
        resid = _equal_area_model_cqe(sol, vert_diff, C_P, RADIUS, ROT_RATE, rce_b)
        np.testing.assert_allclose(np.abs(resid).max(), 0.0, atol=1e-6)
        np.testing.assert_allclose(sol[0], -sol[2], atol=1e-8)
        np.testing.assert_allclose(sol[1], 0.0, atol=1e-8)
        # Anchor from a converged solve (residuals ~2e-9), 2026-07-15.
        np.testing.assert_allclose(np.rad2deg(np.arcsin(sol[2])), 19.31, atol=0.1)


@_IGNORE_QUAD_WARNINGS
@_IGNORE_SOLVER_WANDER
class TestEqualAreaLh88FixedTempTropo:
    """Tests for equal_area_lh88_fixed_temp_tropo."""

    def test_equinoctial_symmetric_nontrivial(self) -> None:
        """Converges to the nontrivial symmetric solution from a reasonable
        initial guess (poor guesses can land on the degenerate zero-width
        fixed point, which also satisfies the constraints)."""
        theta_ref, delta_h = 290.0, 1.0 / 6.0
        guess_edge = np.sin(np.deg2rad(20.0))
        init = [-guess_edge, 0.0, guess_edge, 302.0]
        sol = equal_area_lh88_fixed_temp_tropo(
            init,
            0.0,
            theta_ref=theta_ref,
            delta_h=delta_h,
            gamma=1.0,
            dtheta_dts=1.0,
            temp_tropo=200.0,
            rot_rate=ROT_RATE,
            radius=RADIUS,
            c_p=C_P,
        )
        resid = _lh88_model_fixed_tt(
            sol,
            0.0,
            theta_ref,
            delta_h,
            1.0,
            1.0,
            200.0,
            ROT_RATE,
            RADIUS,
            C_P,
        )
        np.testing.assert_allclose(np.abs(resid).max(), 0.0, atol=1e-6)
        np.testing.assert_allclose(sol[0], -sol[2], atol=1e-8)
        edge_deg = np.rad2deg(np.arcsin(sol[2]))
        assert edge_deg > 5.0
        # Anchors from a converged solve (residuals ~1e-9), 2026-07-15.
        np.testing.assert_allclose(edge_deg, 19.92, atol=0.1)
        np.testing.assert_allclose(sol[3], 305.12, atol=0.1)


@_IGNORE_QUAD_WARNINGS
@_IGNORE_SOLVER_WANDER
class TestEqualAreaCqeFixedTt:
    """Tests for equal_area_cqe_fixed_tt."""

    def test_equinoctial_converges_symmetric(self) -> None:
        """Converges to a symmetric, non-degenerate solution with small
        residuals (regression test for the numpy>=2.0 float() failure on
        1-element DataArrays inside the energy integrand)."""
        theta_b_ref = 345.0
        delta_h_b = 1.0 / 6.0
        rce_b = _rce_lh88_func(0.0, theta_b_ref, delta_h_b)
        theta_guesses = np.linspace(260.0, 420.0, 33)
        # A theta_1 guess at the RCE equatorial value lands on the
        # degenerate zero-width fixed point; start slightly below it.
        guess_edge = np.sin(np.deg2rad(25.0))
        init = [-guess_edge, 0.0, guess_edge, 362.0]
        sol = equal_area_cqe_fixed_tt(
            init,
            rce_b,
            theta_guesses,
            temp_tropo=200.0,
            rel_hum=0.7,
            pressure=1.0e5,
            c_p=C_P,
            radius=RADIUS,
            rot_rate=ROT_RATE,
        )
        resid = _eq_area_cqe_fixed_tt(
            sol,
            rce_b,
            theta_guesses,
            200.0,
            0.7,
            1.0e5,
            C_P,
            RADIUS,
            ROT_RATE,
        )
        np.testing.assert_allclose(
            np.abs(np.asarray(resid, dtype=float)).max(), 0.0, atol=1e-6
        )
        np.testing.assert_allclose(sol[0], -sol[2], atol=1e-6)
        edge_deg = np.rad2deg(np.arcsin(sol[2]))
        assert edge_deg > 5.0
        # Anchors from a converged solve (residuals ~6e-10), 2026-07-15.
        np.testing.assert_allclose(edge_deg, 23.79, atol=0.1)
        np.testing.assert_allclose(sol[3], 362.50, atol=0.1)


class TestSolverConvergenceChecks:
    """All equal-area solvers raise on scipy non-convergence."""

    @pytest.mark.parametrize(
        "caller",
        [
            pytest.param(
                lambda: equal_area_lh88(
                    [-0.3, 0.0, 0.3, 360.0], 0.0, THETA_REF, DELTA_H, 0.02
                ),
                id="lh88",
            ),
            pytest.param(
                lambda: equal_area_bouss(
                    [-0.3, 0.0, 0.3, 360.0],
                    THETA_REF,
                    DELTA_H / 0.02,
                    _rce_lh88_func(0.0, THETA_REF, DELTA_H),
                ),
                id="bouss",
            ),
            pytest.param(
                lambda: equal_area_cqe(
                    [-0.3, 0.0, 0.3, 360.0],
                    100.0,
                    C_P,
                    RADIUS,
                    ROT_RATE,
                    _rce_lh88_func(0.0, 345.0, 1.0 / 6.0),
                ),
                id="cqe",
            ),
            pytest.param(
                lambda: equal_area_lh88_fixed_temp_tropo(
                    [-0.3, 0.0, 0.3, 302.0],
                    0.0,
                    theta_ref=290.0,
                    delta_h=1.0 / 6.0,
                    gamma=1.0,
                    dtheta_dts=1.0,
                    temp_tropo=200.0,
                    rot_rate=ROT_RATE,
                    radius=RADIUS,
                    c_p=C_P,
                ),
                id="lh88-fixed-tt",
            ),
            pytest.param(
                lambda: equal_area_cqe_fixed_tt(
                    [-0.3, 0.0, 0.3, 364.0],
                    _rce_lh88_func(0.0, 345.0, 1.0 / 6.0),
                    np.linspace(250.0, 420.0, 10),
                    temp_tropo=200.0,
                    rel_hum=0.7,
                    pressure=1.0e5,
                    c_p=C_P,
                    radius=RADIUS,
                    rot_rate=ROT_RATE,
                ),
                id="cqe-fixed-tt",
            ),
        ],
    )
    def test_raises_runtime_error(
        self,
        caller: Callable[[], np.ndarray],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed scipy root solve raises instead of returning garbage."""
        monkeypatch.setattr(scipy.optimize, "root", _fake_failed_root)
        with pytest.raises(RuntimeError, match="mocked non-convergence"):
            caller()

    @_IGNORE_QUAD_WARNINGS
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_fixed_tt_bad_guess_raises(self) -> None:
        """A guess that drives the AMC temperature field complex (NaN
        residuals) must raise rather than silently return the guess."""
        init = [-np.sin(np.deg2rad(40.0)), 0.0, np.sin(np.deg2rad(40.0)), 295.0]
        with pytest.raises(RuntimeError):
            equal_area_lh88_fixed_temp_tropo(
                init,
                0.0,
                theta_ref=290.0,
                delta_h=1.0 / 6.0,
                gamma=1.0,
                dtheta_dts=1.0,
                temp_tropo=200.0,
                rot_rate=ROT_RATE,
                radius=RADIUS,
                c_p=C_P,
            )
