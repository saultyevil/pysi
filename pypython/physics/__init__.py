#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate basic quantities.

This sub-module is intended to house various functions for calculating
quantities governed my various laws and equations, i.e. gravitational
radius. It also includes functions for converting quantities into other
quantities.
"""

from pypython.constants import ANGSTROM, C
from pypython.physics import accretion, blackbody, blackhole, models


def angstrom_to_hz(wavelength):
    """Convert a wavelength from Angstroms into a frequency.

    Parameters
    ----------
    wavelength: float
        The wavelength in Angstroms.

    Returns
    -------
    The frequency in Hertz.
    """

    return C / (wavelength * ANGSTROM)


def hz_to_angstrom(frequency):
    """Convert a frequency in Hz to a wavelength in Angstroms.

    Parameters
    ----------
    frequency: float
        The frequency in Hz.

    Returns
    -------
    The wavelength in Angstroms.
    """

    return C / frequency / ANGSTROM
