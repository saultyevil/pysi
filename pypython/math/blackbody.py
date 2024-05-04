#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Blackbody functions for wavelength and frequency.

In wavelength space, the blackbody function assumes that the wavelength
is given in units of Angstroms.
"""
from math import pi

import numpy as np
from astropy import units
from astropy.constants import c, h, k_B, sigma_sb  # pylint: disable=no-name-in-module


from pypython.math.constants import ANGSTROM, WIEN_FREQUENCY, WIEN_WAVELENGTH


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

    temperature *= units.K
    lcm = lamda * ANGSTROM * units.cm
    x = h.cgs * c.cgs / lcm / k_B.cgs / temperature
    y = 2 * h.cgs * c.cgs**2 / lcm**5
    b_lamda = y / (np.exp(x) - 1)

    return b_lamda


def planck_nu(temperature, frequency, factor=1):
    """Calculate the monochromatic intensity for a black body given a
    temperature and frequency of interest.

    Parameters
    ----------
    temperature: float
        The temperature of the blackbody.
    frequency: np.ndarray or float
        The frequency points to calculate the vale at, in units of Hz.
    factor: float
        The colour correction factor.

    Returns
    -------
    b_nu: float
        The value of the Planck function with the provided temperature and
        frequency. Has units ergs s^-1 cm^-2 Hz^-1.
    """

    x = h.cgs * frequency / (factor * k_B.cgs * temperature)
    b_nu = (2 * h.cgs * frequency**3) / (factor**4 * c.cgs**2 * (np.exp(x) - 1))

    return b_nu


def stefan_boltzmann(radius, temperature):
    """Calculate the luminosity for a spherical blackbody following from
    Stefan-Boltzmann.

    Parameters
    ----------
    radius: float
        The radius of the sphere.
    temperature: float
        The temperature of the blackbody.

    Returns
    -------
    lum: float
        The luminosity of the sphere.
    """
    return 4 * pi * radius**2 * sigma_sb.cgs * temperature**4


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
