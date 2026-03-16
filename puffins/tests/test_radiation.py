"""Tests for radiation module."""

import numpy as np

from puffins.constants import SPEED_OF_LIGHT, STEF_BOLTZ_CONST
from puffins.radiation import (
    planck_frequency,
    planck_wavelength,
    wien_peak_frequency,
    wien_peak_wavelength,
)


class TestPlanckWavelength:
    """Tests for planck_wavelength."""

    def test_peak_at_wien(self) -> None:
        """Peak of B_lambda should occur near Wien's peak wavelength."""
        temp = 5800.0
        lam_peak = wien_peak_wavelength(temp)
        lam = np.linspace(0.1e-6, 20e-6, 10000)
        b_lam = planck_wavelength(lam, temp)
        lam_at_max = lam[np.argmax(b_lam)]
        np.testing.assert_allclose(lam_at_max, lam_peak, rtol=0.01)

    def test_positive_values(self) -> None:
        """Planck function should be positive for positive wavelength and T."""
        b = planck_wavelength(1e-6, 300.0)
        assert b > 0

    def test_scalar_input(self) -> None:
        """Should accept scalar inputs and return a scalar."""
        b = planck_wavelength(0.5e-6, 5800.0)
        assert np.isscalar(b) or b.ndim == 0


class TestPlanckFrequency:
    """Tests for planck_frequency."""

    def test_peak_at_wien(self) -> None:
        """Peak of B_nu should occur near Wien's peak frequency."""
        temp = 5800.0
        nu_peak = wien_peak_frequency(temp)
        nu = np.linspace(1e12, 3e15, 10000)
        b_nu = planck_frequency(nu, temp)
        nu_at_max = nu[np.argmax(b_nu)]
        np.testing.assert_allclose(nu_at_max, nu_peak, rtol=0.01)

    def test_positive_values(self) -> None:
        """Planck function should be positive for positive frequency and T."""
        b = planck_frequency(1e14, 300.0)
        assert b > 0


class TestStefanBoltzmannIntegral:
    """Verify that integrating pi * B_lambda recovers sigma * T^4."""

    def test_integral_at_300k(self) -> None:
        temp = 300.0
        lam = np.linspace(1e-7, 1e-3, 100000)
        b_lam = planck_wavelength(lam, temp)
        flux = np.pi * np.trapezoid(b_lam, lam)
        expected = STEF_BOLTZ_CONST * temp**4
        np.testing.assert_allclose(flux, expected, rtol=2e-3)

    def test_integral_at_5800k(self) -> None:
        temp = 5800.0
        lam = np.linspace(1e-8, 1e-3, 100000)
        b_lam = planck_wavelength(lam, temp)
        flux = np.pi * np.trapezoid(b_lam, lam)
        expected = STEF_BOLTZ_CONST * temp**4
        np.testing.assert_allclose(flux, expected, rtol=2e-3)


class TestSpectralConsistency:
    """B_lambda d_lambda = B_nu d_nu at corresponding wavelength/frequency."""

    def test_consistency(self) -> None:
        temp = 5800.0
        lam = 0.5e-6  # meters
        nu = SPEED_OF_LIGHT / lam
        b_lam = planck_wavelength(lam, temp)
        b_nu = planck_frequency(nu, temp)
        # B_lambda = B_nu * |d_nu/d_lambda| = B_nu * c / lambda^2
        b_lam_from_nu = b_nu * SPEED_OF_LIGHT / lam**2
        np.testing.assert_allclose(b_lam, b_lam_from_nu, rtol=1e-10)
