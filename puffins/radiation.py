#! /usr/bin/env python
"""Blackbody radiation: Planck function and Wien's displacement law."""

import numpy as np

from .constants import (
    BOLTZ_CONST,
    PLANCK_CONST,
    SPEED_OF_LIGHT,
    WIEN_DISP_CONST,
)

# Wien displacement constant for the frequency form: nu_peak = WIEN_FREQ * T.
# Derived from x * exp(x) / (exp(x) - 1) = 3, giving x ≈ 2.821.
WIEN_FREQ_FACTOR = 2.821 * BOLTZ_CONST / PLANCK_CONST


def planck_wavelength(
    wavelength: np.ndarray | float,
    temp: np.ndarray | float,
) -> np.ndarray | float:
    """Planck spectral radiance B_lambda (W m-2 sr-1 m-1).

    Parameters
    ----------
    wavelength : Wavelength in meters.
    temp : Temperature in Kelvin.

    """
    lam = np.asarray(wavelength, dtype=float)
    return (2 * PLANCK_CONST * SPEED_OF_LIGHT**2 / lam**5) / (
        np.exp(PLANCK_CONST * SPEED_OF_LIGHT / (lam * BOLTZ_CONST * temp)) - 1
    )


def planck_frequency(
    freq: np.ndarray | float,
    temp: np.ndarray | float,
) -> np.ndarray | float:
    """Planck spectral radiance B_nu (W m-2 sr-1 Hz-1).

    Parameters
    ----------
    freq : Frequency in Hz.
    temp : Temperature in Kelvin.

    """
    nu = np.asarray(freq, dtype=float)
    return (2 * PLANCK_CONST * nu**3 / SPEED_OF_LIGHT**2) / (
        np.exp(PLANCK_CONST * nu / (BOLTZ_CONST * temp)) - 1
    )


def wien_peak_wavelength(temp: np.ndarray | float) -> np.ndarray | float:
    """Peak wavelength from Wien's displacement law (m).

    Parameters
    ----------
    temp : Temperature in Kelvin.

    """
    return WIEN_DISP_CONST / temp


def wien_peak_frequency(temp: np.ndarray | float) -> np.ndarray | float:
    """Peak frequency from Wien's displacement law (Hz).

    Parameters
    ----------
    temp : Temperature in Kelvin.

    """
    return WIEN_FREQ_FACTOR * temp
