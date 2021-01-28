#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Contains functions for calculating the properties of a blackbody.
"""

import numpy as np
from typing import Union
from .constants import BOLTZMANN, H, VLIGHT, ANGSTROM


def planck_nu(
    temperature: float, frequency: Union[np.ndarray, float]
) -> Union[np.ndarray, float]:
    """Calculate the monochromatic intensity for a black body given a temperature
    and frequency of interest.

    Parameters
    ----------
    temperature: float
        The temperature to calculate the function at.
    frequency: float
        The frequency to calculate the function at.

    Returns
    -------
    b_nu: float
        The value of the Planck function with the provided temperature and
        frequency. Has units ergs s^-1 cm^-2 Hz^-1."""

    with np.errstate(all="ignore"):
        x = H * frequency / (BOLTZMANN * temperature)
        b_nu = (2 * H * frequency ** 3) / (VLIGHT ** 2 * (np.exp(x) - 1))

    return b_nu


def planck_lambda(
    temperature: float, lamda: Union[np.ndarray, float]
) -> Union[np.ndarray, float]:
    """Calculate the monochromatic intensity for a black body given a temperature
    and frequency of interest.

    Parameters
    ----------
    temperature: float
        The temperature to calculate the function at.
    lamda: float
        The frequency to calculate the function at.

    Returns
    -------
    b_lamda: float
        The value of the Planck function with the provided temperature and
        wavelength. Has units ergs s^-1 cm^-2 A^-1."""

    with np.errstate(all="ignore"):
        lcm = lamda * ANGSTROM
        x = H * VLIGHT / lcm / BOLTZMANN / temperature
        y = 2 * H * VLIGHT ** 2 / lcm ** 5
        b_lamda = y / (np.exp(x) - 1)

    return b_lamda
