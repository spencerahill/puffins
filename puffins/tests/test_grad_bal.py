"""Tests for grad_bal module (linear-Ro profile Burger-number handling)."""

import numpy as np

from puffins.grad_bal import pot_temp_lin_ro_lata0_small_ang

THETA_REF = 330.0
GRAV = 9.81
RADIUS = 6.371e6
ROT_RATE = 7.292e-5
HEIGHT = 12.0e3


class TestPotTempLinRoLata0SmallAng:
    """Tests for pot_temp_lin_ro_lata0_small_ang."""

    @staticmethod
    def _raw_profile(
        lat_deg: np.ndarray,
        lat_descent_deg: float,
        ro_a: float,
        ro_d: float,
        pot_temp_lat0: float,
        burg_num: float,
    ) -> np.ndarray:
        """ms Eq. (6) profile reconstructed from raw numpy."""
        delro = ro_a - ro_d
        lat = np.deg2rad(lat_deg)
        latd = np.deg2rad(lat_descent_deg)
        bracket = (
            ro_a / 2
            - 4 * delro * lat / (15 * latd)
            + ro_a**2 * lat**2 / 6
            - 4 * delro * ro_a * lat**3 / (21 * latd)
            + delro**2 * lat**4 / (18 * latd**2)
        )
        return pot_temp_lat0 - THETA_REF * lat**4 / burg_num * bracket

    def test_explicit_burg_num(self) -> None:
        """An explicit burg_num overrides the planetary default, so the
        profile can be made consistent with a thermal-Rossby-number
        parameterized equal-area solution (Bu = Ro_th / delta_h)."""
        burg = 0.06
        lats = np.array([0.0, 10.0, 20.0, 30.0])
        expected = self._raw_profile(lats, 33.0, 1.0, 0.5, 365.0, burg)
        actual = pot_temp_lin_ro_lata0_small_ang(
            lats,
            33.0,
            1.0,
            0.5,
            365.0,
            theta_ref=THETA_REF,
            burg_num=burg,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13)

    def test_default_burg_num_is_planetary(self) -> None:
        """Without burg_num, the Burger number comes from the planetary
        parameters, gH/(Omega*a)^2, preserving the original behavior."""
        burg_planet = GRAV * HEIGHT / (ROT_RATE * RADIUS) ** 2
        lats = np.array([0.0, 10.0, 20.0, 30.0])
        expected = self._raw_profile(lats, 33.0, 1.0, 0.5, 365.0, burg_planet)
        actual = pot_temp_lin_ro_lata0_small_ang(
            lats,
            33.0,
            1.0,
            0.5,
            365.0,
            rot_rate=ROT_RATE,
            radius=RADIUS,
            theta_ref=THETA_REF,
            grav=GRAV,
            height=HEIGHT,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13)
