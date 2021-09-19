#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""General functions for working with photometry.

This module contains functions for converting magnitudes to fluxes, and
the reverse and a function for de-reddening spectra. Also contained are
the zero-points for common filters.
"""

from enum import Enum

import astropy.units as u

# Zero-point enumerator --------------------------------------------------------


class ZeroPointUnits(Enum):
    freq = "Freq."
    wavelength = "Lambda"


# Photometry -------------------------------------------------------------------

FILTERS = {
    # Swift UVOT: Vega
    "V": {
        "Freq.": 3636.18,
        "Lambda": 3.72e-9,
        "EffLambda": 5410,
        "system": "Vega",
        "instrument": "Swift UVOT"
    },
    "B": {
        "Freq.": 4059.85,
        "Lambda": 6.44e-9,
        "EffLambda": 4321,
        "system": "Vega",
        "instrument": "Swift UVOT"
    },
    "U": {
        "Freq.": 1480.55,
        "Lambda": 3.58e-9,
        "EffLambda": 3442,
        "system": "Vega",
        "instrument": "Swift UVOT"
    },
    "W1": {
        "Freq.": 981.360,
        "Lambda": 4.08e-9,
        "EffLambda": 2486,
        "system": "Vega",
        "instrument": "Swift UVOT"
    },
    "M2": {
        "Freq.": 770.290,
        "Lambda": 4.58e-9,
        "EffLambda": 2221,
        "system": "Vega",
        "instrument": "Swift UVOT"
    },
    "W2": {
        "Freq.": 759.990,
        "Lambda": 5.24e-9,
        "EffLambda": 1991,
        "system": "Vega",
        "instrument": "Swift UVOT"
    },
    # Liverpool Telescope: AB
    "z": {
        "Freq.": None,
        "Lambda": 1.35201e-9,
        "EffLambda": 8972.92,
        "system": "AB",
        "instrument": "Liverpool Telescope"
    },
    "i": {
        "Freq.": None,
        "Lambda": 1.88946e-9,
        "EffLambda": 7590.22,
        "system": "AB",
        "instrument": "Liverpool Telescope"
    },
    "r": {
        "Freq.": None,
        "Lambda": 2.90493e-9,
        "EffLambda": 6121.47,
        "system": "AB",
        "instrument": "Liverpool Telescope"
    },
    "g": {
        "Freq.": None,
        "Lambda": 4.73581e-9,
        "EffLambda": 4794.31,
        "system": "AB",
        "instrument": "Liverpool Telescope"
    },
    "u": {
        "Freq.": None,
        "Lambda": 8.81680e-9,
        "EffLambda": 3513.73,
        "system": "AB",
        "instrument": "Liverpool Telescope"
    }
}

filters = FILTERS


def error_in_flux(e_magnitude, flux):
    """Calculate the error from flux from the error in magnitude.

    Parameters
    ----------
    e_magnitude: float
        The error in the magnitude calculation.
    flux: float
        The value of the flux.

    Returns
    -------
    error: float
        The error in the flux conversion.
    """
    error_upper = flux * (10**(0.4 * e_magnitude) - 1.0)
    error_lower = flux * (1 - 10**(-0.4 * e_magnitude))
    error = 0.5 * (error_lower + error_upper)

    return error


def magnitude_to_flux(magnitude, filter, error=None, host_magnitude=None, flux_type="Lambda"):
    """Convert a magnitude value to a flux.

    If m - m0 = -2.5 * log10(F / F0), then F = F_0 * 10^(-m / 2.5) where
    F_0 is flux for the zero point magnitude of the instrument/filter.

    Parameters
    ----------
    magnitude: float
        The magnitude of the point.
    filter: str
        The filter the point was taken in.
    error: float
        The error on the measurement.
    host_magnitude: float
        The magnitude of the host galaxy.
    flux_type: str
        Whether the flux is per unit wavelength or frequency.

    Returns
    -------
    flux: float
        The magnitude converted into a flux.
    """
    if filter not in FILTERS.keys():
        raise ValueError(f"unknown {filter}: known are {FILTERS.keys()}")
    if flux_type not in ["Freq.", "Lambda"]:
        raise ValueError(f"unknown flux units {flux_type}: know are Freq. or Lambda")
    if FILTERS[filter][flux_type] is None:
        raise NotImplementedError("This filter does not have a flux zero point in Freq. units")

    flux = FILTERS[filter][flux_type] * 10**(-magnitude / 2.5)
    if host_magnitude:
        host_flux = FILTERS[filter][flux_type] * 10**(-host_magnitude / 2.5)
        flux -= host_flux

    if error:
        error = error_in_flux(error, flux)
        return flux, error
    else:
        return flux


# Observed spectra


def deredden(wavelength, flux, r_v, e_bv, curve="CCM89"):
    """Deredden a spectrum.

    Remove the interstellar/dust reddening from a spectrum, given the colour
    excess (E(B-V)) and selective extinction (Rv).

    This function assumes that the flux is in units of erg s^-1 cm^-2 AA^-1

    Parameters
    ----------
    wavelength: np.ndarray
        The wavelength bins of the spectrum.
    flux: np.ndarray
        The flux bins of the spectrum.
    r_v: float
        The selective extinction coefficient.
    e_bv: float
        The color excess.
    curve: str
        The name of the extinction curve to use, either CCM89 or F99.

    Returns
    -------
    flux: np.ndarray
        The corrected flux.
    """

    from dust_extinction.parameter_averages import CCM89, F99

    wavelength *= u.angstrom
    flux *= u.erg / u.s / u.cm / u.cm / u.AA

    if curve == "CCM89":
        curve = CCM89(Rv=r_v)
    elif curve == "F99":
        curve = F99(Rv=r_v)
    else:
        raise ValueError("Unknown extinction curve {curve}: CCM89 and F99 are available.")

    flux /= curve.extinguish(wavelength, Ebv=e_bv)

    return flux
