#! /usr/bin/env python
"""Blackbody radiation: Planck function and Wien's displacement law."""

import numpy as np
import xarray as xr

from .constants import (
    BOLTZ_CONST,
    PLANCK_CONST,
    SPEED_OF_LIGHT,
)

ArrayLike = xr.DataArray | np.ndarray | float


def planck_wavelength(
    wavelength: ArrayLike,
    temp: ArrayLike,
) -> ArrayLike:
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
    freq: ArrayLike,
    temp: ArrayLike,
) -> ArrayLike:
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


def wien_peak_wavelength(temp: ArrayLike) -> ArrayLike:
    """Peak wavelength from Wien's displacement law (m).

    Parameters
    ----------
    temp : Temperature in Kelvin.

    """
    wien_disp_const = 2.898e-3  # Wien displacement constant (m K).
    return wien_disp_const / temp


def wien_peak_frequency(temp: ArrayLike) -> ArrayLike:
    """Peak frequency from Wien's displacement law (Hz).

    Parameters
    ----------
    temp : Temperature in Kelvin.

    """
    # Frequency-form Wien factor: x * exp(x) / (exp(x) - 1) = 3, giving x ≈ 2.821.
    wien_freq_factor = 2.821 * BOLTZ_CONST / PLANCK_CONST
    return wien_freq_factor * temp
