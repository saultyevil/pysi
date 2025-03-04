"""Blackbody functions for wavelength and frequency.

In wavelength space, the blackbody function assumes that the wavelength
is given in units of Angstroms.
"""

import numpy as np
from astropy import units
from astropy.constants import c, h, k_B

from pysi.math.constants import ANGSTROM


def planck_lambda(temperature: float, lamda: float | np.ndarray) -> float | np.ndarray:
    """Compute the Planck function for a blackbody, B_lambda.

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

    return y / (np.exp(x) - 1)


def planck_nu(temperature: float, frequency: float | np.ndarray, colour_factor: float = 1) -> float | np.ndarray:
    """Compute the Placnk function for a blockbody, B_nu.

    Parameters
    ----------
    temperature: float
        The temperature of the blackbody.
    frequency: np.ndarray or float
        The frequency points to calculate the vale at, in units of Hz.
    colour_factor: float
        The colour correction factor.

    Returns
    -------
    b_nu: float
        The value of the Planck function with the provided temperature and
        frequency. Has units ergs s^-1 cm^-2 Hz^-1.

    """
    x = h.cgs * frequency / (colour_factor * k_B.cgs * temperature)

    return (2 * h.cgs * frequency**3) / (colour_factor**4 * c.cgs**2 * (np.exp(x) - 1))
