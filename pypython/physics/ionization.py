#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate various atomic quantities, such as the ratio of the level population
for two atomic levels.
"""

import numpy as np
from constants import PI, MELEC, BOLTZMANN, H


def saha_population_ratio(
    electron_density: float, g_upper: float, g_lower: float, energy_upper: float, energy_lower: float,
    temperature: float
) -> float:
    """Calculate the ratio of two level populations n_i+1 / n_i, using the
    Saha-Boltzmann equation.

    Parameters
    ----------
    electron_density: float
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
    n_i+1 / n_i: float
        The ratio of the population of the upper ionisation and ground state
        of the atom."""

    gratio = 2 * g_upper / g_lower
    saha = ((2 * PI * MELEC * BOLTZMANN * temperature) / H ** 2) ** (3 / 2)
    saha *= gratio * np.exp(-(energy_upper - energy_lower) / (BOLTZMANN * temperature)) / electron_density

    return saha
