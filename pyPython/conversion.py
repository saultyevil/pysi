#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The functions in this file are concerned with converting quantities from one
unit to another.
"""


from .constants import ANGSTROM, C


def angstrom_to_hz(wl: float):
    """
    Convert a wavelength from Angstroms into a frequency.

    Parameters
    ----------
    wl: float
        The wavelength in Angstroms.

    Returns
    -------
    The frequency in Hertz.
    """

    return C / (wl * ANGSTROM)


def hz_to_angstrom(freq: float):
    """
    Convert a frequency in Hz to a wavelength in Angstroms.

    Parameters
    ----------
    freq: float
        The frequency in Hz.

    Returns
    -------
    The wavelength in Angstroms.

    """

    return C / freq / ANGSTROM
