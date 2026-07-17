"""Tests for the lcl (lifting condensation level) module.

Reference values are reconstructed independently from raw numpy/scipy using the
closed-form expressions of Romps 2017, not from the module's own helpers, so the
tests pin the full coefficient chain (see CLAUDE.md test-teeth convention).

Romps, David M. 2017. "Exact Expression for the Lifting Condensation Level."
Journal of the Atmospheric Sciences 74 (12): 3891-3900.
https://doi.org/10.1175/JAS-D-17-0102.1.

"""

import numpy as np
import pytest
import scipy.special
import xarray as xr

from puffins._typing import ArrayLike
from puffins.constants import (
    C_PD,
    C_PV,
    C_VL,
    C_VV,
    E_0V,
    GRAV_EARTH,
    P_TRIP,
    R_D,
    R_V,
    T_TRIP,
)
from puffins.lcl import (
    gas_const_moist_air,
    lift_cond_level,
    pres_lift_cond_level,
    sat_vap_press_liq_wat,
    spec_heat_const_press_moist_air,
    temp_lift_cond_level,
)

# A representative unsaturated surface parcel.
PRESS = 1.0e5
TEMP = 300.0
REL_HUM = 0.7


# --- Independent reference implementation (raw numpy/scipy) ---------------


def _ref_sat_vap_press(
    temp: float, p_trip: float = P_TRIP, e_0v: float = E_0V
) -> float:
    """Saturation vapor pressure over liquid water, Romps 2017 Eq. (4).

    `p_trip` and `e_0v` are overridable so the forwarding tests can probe that
    non-default constants actually reach this expression.
    """
    return (
        p_trip
        * (temp / T_TRIP) ** ((C_PV - C_VL) / R_V)
        * np.exp((e_0v - (C_VV - C_VL) * T_TRIP) / R_V * (1.0 / T_TRIP - 1.0 / temp))
    )


def _ref_q_v(press: float, vap_press: float) -> float:
    return R_D * vap_press / (R_V * press + (R_D - R_V) * vap_press)


def _ref_r_m(press: float, vap_press: float) -> float:
    q_v = _ref_q_v(press, vap_press)
    return (1.0 - q_v) * R_D + q_v * R_V


def _ref_c_pm(press: float, vap_press: float) -> float:
    q_v = _ref_q_v(press, vap_press)
    return (1.0 - q_v) * C_PD + q_v * C_PV


def _ref_temp_lcl(
    press: float,
    temp: float,
    rel_hum: float,
    p_trip: float = P_TRIP,
    e_0v: float = E_0V,
) -> float:
    vap_press = rel_hum * _ref_sat_vap_press(temp, p_trip=p_trip, e_0v=e_0v)
    r_m = _ref_r_m(press, vap_press)
    c_pm = _ref_c_pm(press, vap_press)
    a = (C_VL - C_PV) / R_V + c_pm / r_m
    b = -(e_0v - (C_VV - C_VL) * T_TRIP) / (R_V * temp)
    c = b / a
    return float(
        c * temp / scipy.special.lambertw(rel_hum ** (1.0 / a) * c * np.exp(c), -1).real
    )


def _ref_lcl_height(
    press: float,
    temp: float,
    rel_hum: float,
    z_0: float = 0.0,
    p_trip: float = P_TRIP,
    e_0v: float = E_0V,
) -> float:
    vap_press = rel_hum * _ref_sat_vap_press(temp, p_trip=p_trip, e_0v=e_0v)
    c_pm = _ref_c_pm(press, vap_press)
    temp_lcl = _ref_temp_lcl(press, temp, rel_hum, p_trip=p_trip, e_0v=e_0v)
    return z_0 + c_pm / GRAV_EARTH * (temp - temp_lcl)


def _ref_pres_lcl(
    press: float,
    temp: float,
    rel_hum: float,
    p_trip: float = P_TRIP,
    e_0v: float = E_0V,
) -> float:
    """Romps 2017 Eq. (22b): p_lcl = press * (T_lcl / T) ** (c_pm / R_m)."""
    vap_press = rel_hum * _ref_sat_vap_press(temp, p_trip=p_trip, e_0v=e_0v)
    r_m = _ref_r_m(press, vap_press)
    c_pm = _ref_c_pm(press, vap_press)
    temp_lcl = _ref_temp_lcl(press, temp, rel_hum, p_trip=p_trip, e_0v=e_0v)
    return press * (temp_lcl / temp) ** (c_pm / r_m)


# Representative (press, temp) points spanning the lower troposphere, used to
# broaden the known-value checks beyond a single parcel.
KNOWN_POINTS = [
    (1.0e5, 300.0),
    (0.9e5, 290.0),
    (0.7e5, 270.0),
]


# --- sat_vap_press_liq_wat -------------------------------------------------


class TestSatVapPressLiqWat:
    def test_known_value(self) -> None:
        """Full closed-form reconstruction from raw numpy."""
        result = sat_vap_press_liq_wat(TEMP)
        np.testing.assert_allclose(result, _ref_sat_vap_press(TEMP), rtol=1e-12)

    def test_equals_p_trip_at_triple_point(self) -> None:
        """At the triple-point temperature the expression collapses to P_TRIP."""
        np.testing.assert_allclose(sat_vap_press_liq_wat(T_TRIP), P_TRIP, rtol=1e-12)

    def test_monotonic_increasing(self) -> None:
        """Saturation vapor pressure rises with temperature."""
        temps = np.array([260.0, 280.0, 300.0, 320.0])
        svp = sat_vap_press_liq_wat(temps)
        assert np.all(np.diff(svp) > 0)


# --- gas_const_moist_air ---------------------------------------------------


class TestGasConstMoistAir:
    def test_known_value(self) -> None:
        vap_press = REL_HUM * _ref_sat_vap_press(TEMP)
        result = gas_const_moist_air(PRESS, vap_press)
        np.testing.assert_allclose(result, _ref_r_m(PRESS, vap_press), rtol=1e-12)

    def test_dry_limit_is_r_d(self) -> None:
        """With no water vapor the moist gas constant reduces to R_D."""
        np.testing.assert_allclose(gas_const_moist_air(PRESS, 0.0), R_D, rtol=1e-12)

    def test_between_r_d_and_r_v(self) -> None:
        """A moist parcel lies strictly between the dry and vapor limits."""
        vap_press = REL_HUM * _ref_sat_vap_press(TEMP)
        r_m = gas_const_moist_air(PRESS, vap_press)
        assert R_D < r_m < R_V


# --- spec_heat_const_press_moist_air ---------------------------------------


class TestSpecHeatConstPressMoistAir:
    def test_known_value(self) -> None:
        vap_press = REL_HUM * _ref_sat_vap_press(TEMP)
        result = spec_heat_const_press_moist_air(PRESS, vap_press)
        np.testing.assert_allclose(result, _ref_c_pm(PRESS, vap_press), rtol=1e-12)

    def test_dry_limit_is_c_pd(self) -> None:
        """With no water vapor the moist specific heat reduces to C_PD."""
        np.testing.assert_allclose(
            spec_heat_const_press_moist_air(PRESS, 0.0), C_PD, rtol=1e-12
        )


# --- temp_lift_cond_level --------------------------------------------------


class TestTempLiftCondLevel:
    @pytest.mark.parametrize("press,temp", KNOWN_POINTS)
    def test_known_value(self, press: float, temp: float) -> None:
        """Full coefficient chain reconstructed from raw numpy/scipy."""
        result = temp_lift_cond_level(press, temp, REL_HUM)
        np.testing.assert_allclose(
            result, _ref_temp_lcl(press, temp, REL_HUM), rtol=1e-12
        )

    def test_cooler_than_parcel(self) -> None:
        """Unsaturated ascent to the LCL cools the parcel."""
        assert temp_lift_cond_level(PRESS, TEMP, REL_HUM) < TEMP

    def test_saturated_parcel_no_cooling(self) -> None:
        """A saturated parcel is already at its LCL, so T_lcl == T."""
        np.testing.assert_allclose(
            temp_lift_cond_level(PRESS, TEMP, 1.0), TEMP, rtol=1e-10
        )

    def test_drier_is_cooler(self) -> None:
        """Lower relative humidity means a colder (higher) LCL."""
        assert temp_lift_cond_level(PRESS, TEMP, 0.3) < temp_lift_cond_level(
            PRESS, TEMP, 0.9
        )


# --- lift_cond_level -------------------------------------------------------


class TestLiftCondLevel:
    @pytest.mark.parametrize("press,temp", KNOWN_POINTS)
    def test_known_value(self, press: float, temp: float) -> None:
        result = lift_cond_level(press, temp, REL_HUM)
        np.testing.assert_allclose(
            result, _ref_lcl_height(press, temp, REL_HUM), rtol=1e-12
        )

    def test_saturated_parcel_at_surface(self) -> None:
        """A saturated parcel has its LCL at the launch height z_0."""
        np.testing.assert_allclose(lift_cond_level(PRESS, TEMP, 1.0), 0.0, atol=1e-6)

    def test_z0_offset(self) -> None:
        """z_0 shifts the LCL height by a constant offset."""
        z_0 = 500.0
        base = lift_cond_level(PRESS, TEMP, REL_HUM)
        shifted = lift_cond_level(PRESS, TEMP, REL_HUM, z_0=z_0)
        np.testing.assert_allclose(shifted - base, z_0, rtol=1e-10)

    def test_drier_is_higher(self) -> None:
        """A drier parcel must rise farther to reach saturation."""
        assert lift_cond_level(PRESS, TEMP, 0.3) > lift_cond_level(PRESS, TEMP, 0.9)


# --- pres_lift_cond_level --------------------------------------------------


class TestPresLiftCondLevel:
    @pytest.mark.parametrize("press,temp", KNOWN_POINTS)
    def test_known_value(self, press: float, temp: float) -> None:
        """Full reconstruction of Romps 2017 Eq. (22b) from raw numpy/scipy."""
        result = pres_lift_cond_level(press, temp, REL_HUM)
        np.testing.assert_allclose(
            result, _ref_pres_lcl(press, temp, REL_HUM), rtol=1e-12
        )

    def test_saturated_parcel_at_launch_pressure(self) -> None:
        """A saturated parcel is already at its LCL, so p_lcl == press."""
        np.testing.assert_allclose(
            pres_lift_cond_level(PRESS, TEMP, 1.0), PRESS, rtol=1e-10
        )

    def test_below_launch_pressure(self) -> None:
        """An unsaturated parcel's LCL sits above launch, at lower pressure."""
        assert pres_lift_cond_level(PRESS, TEMP, REL_HUM) < PRESS

    def test_drier_is_lower_pressure(self) -> None:
        """A drier parcel reaches its LCL higher up, at lower pressure."""
        assert pres_lift_cond_level(PRESS, TEMP, 0.3) < pres_lift_cond_level(
            PRESS, TEMP, 0.9
        )


# --- constant forwarding into the SVP call ---------------------------------


class TestConstantForwarding:
    """Thermodynamic constants must reach the internal saturation-vapor-pressure
    call, not only the Lambert-W coefficients (PR #52 review).

    `p_trip` enters the LCL expressions *only* through `sat_vap_press_liq_wat`,
    so it is a clean probe: were it dropped on the way to the SVP call (the
    original bug), a non-default `p_trip` would leave the result unchanged.
    """

    P_TRIP_ALT = P_TRIP * 1.1
    E_0V_ALT = E_0V * 1.02

    def test_temp_lift_cond_level_forwards_p_trip(self) -> None:
        alt = temp_lift_cond_level(PRESS, TEMP, REL_HUM, p_trip=self.P_TRIP_ALT)
        # A dropped-forwarding bug returns a byte-identical default, so exact
        # inequality is the precise discriminator (the effect is small).
        assert alt != temp_lift_cond_level(PRESS, TEMP, REL_HUM)
        np.testing.assert_allclose(
            alt, _ref_temp_lcl(PRESS, TEMP, REL_HUM, p_trip=self.P_TRIP_ALT), rtol=1e-12
        )

    def test_lift_cond_level_forwards_p_trip(self) -> None:
        alt = lift_cond_level(PRESS, TEMP, REL_HUM, p_trip=self.P_TRIP_ALT)
        assert alt != lift_cond_level(PRESS, TEMP, REL_HUM)
        np.testing.assert_allclose(
            alt,
            _ref_lcl_height(PRESS, TEMP, REL_HUM, p_trip=self.P_TRIP_ALT),
            rtol=1e-12,
        )

    def test_pres_lift_cond_level_forwards_p_trip(self) -> None:
        alt = pres_lift_cond_level(PRESS, TEMP, REL_HUM, p_trip=self.P_TRIP_ALT)
        assert alt != pres_lift_cond_level(PRESS, TEMP, REL_HUM)
        np.testing.assert_allclose(
            alt,
            _ref_pres_lcl(PRESS, TEMP, REL_HUM, p_trip=self.P_TRIP_ALT),
            rtol=1e-12,
        )

    def test_e_0v_forwarded_into_svp(self) -> None:
        """`e_0v` appears in both the SVP and the Lambert-W coefficients; a full
        reconstruction stays consistent only if it reaches the SVP call too."""
        alt = temp_lift_cond_level(PRESS, TEMP, REL_HUM, e_0v=self.E_0V_ALT)
        np.testing.assert_allclose(
            alt, _ref_temp_lcl(PRESS, TEMP, REL_HUM, e_0v=self.E_0V_ALT), rtol=1e-12
        )


# --- ArrayLike input support ----------------------------------------------


@pytest.mark.parametrize(
    "func",
    [temp_lift_cond_level, pres_lift_cond_level, lift_cond_level],
    ids=["temp_lift_cond_level", "pres_lift_cond_level", "lift_cond_level"],
)
@pytest.mark.parametrize(
    "press,temp,rel_hum",
    [
        (PRESS, TEMP, REL_HUM),
        (
            np.array([PRESS, 0.9e5]),
            np.array([TEMP, 290.0]),
            np.array([REL_HUM, 0.5]),
        ),
        (
            xr.DataArray([PRESS, 0.9e5], dims=["x"]),
            xr.DataArray([TEMP, 290.0], dims=["x"]),
            xr.DataArray([REL_HUM, 0.5], dims=["x"]),
        ),
    ],
    ids=["scalar", "ndarray", "DataArray"],
)
def test_array_like_inputs(
    func: object, press: ArrayLike, temp: ArrayLike, rel_hum: ArrayLike
) -> None:
    """Entry points accept float, np.ndarray, and xr.DataArray inputs."""
    result = func(press, temp, rel_hum)  # type: ignore[operator]
    assert np.all(np.isfinite(np.asarray(result)))
