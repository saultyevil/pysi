#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Functions pertaining to accretion discs and accretion in general live in here.
For example, there are functions for the temperature profile for an alpha-disc,
as well as functions to calculate the Eddington luminosity or accretion rate.
"""

from typing import Union
import numpy as np
import pandas as pd

from .Blackbody import planck_lambda, planck_nu
from .Constants import STEFAN_BOLTZMANN, C, MPROT, THOMPSON, G, PI, MSOL, MSOL_PER_YEAR


def alpha_disc_effective_temperature(
    ri: Union[np.ndarray, float], rstar: float, mbh: float, mdot: float
) -> np.ndarray:
    """Standard alpha-disc effective temperature profile."""

    mbh *= MSOL
    mdot *= MSOL_PER_YEAR

    teff4 = (3 * G * mbh * mdot) / (8 * np.pi * ri ** 3 * STEFAN_BOLTZMANN)
    teff4 *= 1 - (rstar / ri) ** 0.5

    return teff4 ** 0.25


def eddington_critical_disc_effective_temperature(
    ri: Union[np.ndarray, float], mbh: float, mdot: float, ledd: float, rg: float, risco: float
):
    """The effective temperature profile from Strubbe and Quataert 2009."""

    mbh *= MSOL
    mdot *= MSOL_PER_YEAR

    fnt = 1 - np.sqrt(risco / ri)
    teff4 = (3 * G * mbh * mdot * fnt) / (8 * PI * ri ** 3 * STEFAN_BOLTZMANN)
    teff4 *= (0.5 + (0.25 + 6 * fnt * (mdot * C ** 2 / ledd) ** 2 * (ri / rg) ** -2) ** 0.5) ** -1

    return teff4 ** 0.25


def eddington_accretion_limit(
    mbh: float, efficiency: float
) -> float:
    """
    Calculate the Eddington accretion limit for a black hole. Note that the
    accretion rate can be larger than the Eddington accretion rate. See, for
    example, Foundations of High-Energy Astrophysics by Mario Vietri.

    Parameters
    ----------
    mbh: float
        The mass of the black hole in units of msol.
    efficiency: float
        The efficiency of the accretion process. Less than 1.

    Returns
    -------
    The Eddington accretion rate in units of grams / second.
    """

    mbh *= MSOL

    return (4 * PI * G * mbh * MPROT) / (efficiency * C * THOMPSON)


def eddington_luminosity_limit(
        mbh: float
) -> float:
    """
    Calculate the Eddington luminosity for accretion onto a black hole.

    Parameters
    ----------
    mbh: float
        The mass of the black hole in units of msol.

    Returns
    -------
    The Eddington luminosity for the black hole in units of ergs / second.
    """

    mbh *= MSOL

    return (4 * PI * G * mbh * C * MPROT) / THOMPSON


def generate_simple_disc_spectrum(
    mbh: float, mdot: float, xmin: float, xmax: float, rinner: float, router: float, freq_units: bool = False,
    npoints: int = 500
) -> np.array:
    """
    Generate a crude accretion disc spectrum given the mass of the black hole
    and its accretion rate.
    """

    mbh *= MSOL
    mdot *= MSOL_PER_YEAR

    if freq_units:
        xlabel = "Freq."
    else:
        xlabel = "Lambda"

    radii = np.logspace(np.log10(rinner), np.log10(router), npoints)
    xrange = np.logspace(np.log10(xmin), np.log10(xmax), npoints)
    s = pd.DataFrame(columns=[xlabel, "Flux"])

    # Initialise the data frame
    s[xlabel] = xrange
    s["Flux"] = np.zeros(npoints)

    # TODO: this can probably be vectorised to be faster
    for i in range(npoints - 1):
        # Use midpoint of annulus as point on r grid
        r = (radii[i + 1] + radii[i]) * 0.5
        area = PI * (radii[i + 1] ** 2 - radii[i] ** 2)
        teff = float(alpha_disc_effective_temperature(r, rinner, mbh, mdot))

        if freq_units:
            f = planck_nu(teff, xrange)
        else:
            f = planck_lambda(teff, xrange)

        s["Flux"] += f * area * PI

    return s
