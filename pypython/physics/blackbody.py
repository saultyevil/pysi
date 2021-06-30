#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Blackbody functions for wavelength and frequency.

In wavelength space, the blackbody function assumes that the wavelength
is given in units of Angstroms.
"""

import numpy as np

from pypython.constants import (ANGSTROM, BOLTZMANN, VLIGHT, WIEN_FREQUENCY, WIEN_WAVELENGTH, H)


def planck_lambda(temperature, lamda):
    """Calculate the monochromatic intensity for a black body given a
    temperature and frequency of interest.

    Parameters
    ----------
    temperature: float
        The temperature of the blackbody.
    lamda: np.ndarray or float
        The wavelength points to calculate the value at, in Angstroms.

    Returns
    -------
    b_lamda: float
        The value of the Planck function with the provided temperature and
        wavelength. Has units ergs s^-1 cm^-2 A^-1.
    """

    with np.errstate(all="ignore"):
        lcm = lamda * ANGSTROM
        x = H * VLIGHT / lcm / BOLTZMANN / temperature
        y = 2 * H * VLIGHT**2 / lcm**5
        b_lamda = y / (np.exp(x) - 1)

    return b_lamda


def planck_nu(temperature, frequency):
    """Calculate the monochromatic intensity for a black body given a
    temperature and frequency of interest.

    Parameters
    ----------
    temperature: float
        The temperature of the blackbody.
    frequency: np.ndarray or float
        The frequency points to calculate the vale at, in units of Hz.

    Returns
    -------
    b_nu: float
        The value of the Planck function with the provided temperature and
        frequency. Has units ergs s^-1 cm^-2 Hz^-1.
    """

    with np.errstate(all="ignore"):
        x = H * frequency / (BOLTZMANN * temperature)
        b_nu = (2 * H * frequency**3) / (VLIGHT**2 * (np.exp(x) - 1))

    return b_nu


def wien_law(temperature, freq_space=False):
    """Calculate the peak wavelength of a blackbody curve.

    Parameters
    ----------
    temperature: float
        The temperature of the blackbody.
    freq_space: bool [optional]
        Return the peak in frequency space.

    Returns
    -------
    The wavelength (in Angstrom) or frequency where the blackbody curve is
    at maximum.
    """

    if freq_space:
        return WIEN_FREQUENCY * temperature
    else:
        return WIEN_WAVELENGTH / temperature / 1e-10
