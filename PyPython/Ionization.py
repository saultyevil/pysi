#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions to calculate the ionization and level populations.
"""


import numpy as np

from .Constants import PI, MELEC, BOLTZMANN, H


def saha_equation_population_ratio(
    ne: float, g_upper: float, g_lower: float, energy_upper: float, energy_lower: float, temperature: float
):
    """
    Calculate the ratio of n_i+1 / n_i, using the Saha-Boltzman equation.

    Parameters
    ----------
    ne: float
        The electron density of the plasma, in cm^-2.
    g_upper: float
        The statistical weight of the upper level.
    g_lower: float
        The statistical weight of the lower level.
    energy_upper: float
        The ionisation potential of the upper level, in ergs.
    energy_lower: upper
        The ionisation potential of the lower level, in ergs.
    temperature: float
        The temperature of the plasma in K.

    Returns
    -------
    N_i+1 / N_i: float
        The ratio of the population of the upper ionisation and ground state
        of the atom.
    """

    gratio = 2 * g_upper / g_lower
    saha = ((2 * PI * MELEC * BOLTZMANN * temperature) / H ** 2) ** (3 / 2)

    return saha * gratio * np.exp(-(energy_upper - energy_lower) / (BOLTZMANN * temperature)) / ne
