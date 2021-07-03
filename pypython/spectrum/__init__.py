#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions for creating and analyzing spectra.

Python has the ability to create a "delay_dump" file, which is a dump of
all extracted photons. Using this file, a spectrum can be created with
various physical processes remove, i.e. resonance scattering. This
module also includes functions for measuring line width and equivalent
widths.
"""
import astropy.units as u
from dust_extinction.parameter_averages import CCM89, F99

from pypython.spectrum import create, lines, photometry


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
