#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Equations for accretion discs.

Includes temperature profiles, calculations for the Eddington luminosity
and accretion rate and also a function to create a crude accretion disc
spectrum.
"""

import numpy as np
import pandas as pd
from numba import jit, njit

from pypython.constants import (MPROT, MSOL, MSOL_PER_YEAR, PI, STEFAN_BOLTZMANN, THOMPSON, C, G)
from pypython.physics.blackbody import planck_lambda, planck_nu
from pypython.physics.blackhole import (gravitational_radius, innermost_stable_circular_orbit)


@njit
def _calculate_disc_spectrum(m_co, mdot, radius, frequency_bins, modified_teff, freq_units):
    """Calculate the accretion disc spectrum using jit.
    
    Parameters
    ----------
    m_co: float
        The mass of the central object, in Msol.
    mdot: float
        The accretion rate onto the central object, in Msol/yr.
    radius: np.ndarray
        The radial points of the annuli of the disc.
    frequency_bins: np.ndarray
        The frequency bins to calculate the blackbody intensity for.
    modified_teff: bool
        Whether to use a modified Eddington T_eff profile or not.
    freq_units: bool
        Whether to return in frequency or wavelength units.

    Returns
    -------
    lum: np.ndarray
        The luminosity at the frequency bins.
    """
    
    n_rings = len(radius)  
    lum = np.zeros_like(frequency_bins)

    for i in range(n_rings - 1):
        # Use midpoint of annulus as point on r grid
        r = (radius[i + 1] + radius[i]) * 0.5
        area_annulus = PI * (radius[i + 1]**2 - radius[i]**2)

        if modified_teff:
            t_eff = modified_eddington_alpha_disc_effective_temperature(r, m_co, mdot)
        else:
        
            t_eff = alpha_disc_effective_temperature(r, radius[0], m_co, mdot)

        if freq_units:
            f = planck_nu(t_eff, frequency_bins)
        else:
            f = planck_lambda(t_eff, frequency_bins)

        lum += f * area_annulus * PI

    return lum


@jit
def alpha_disc_effective_temperature(r, r_co, m_co, mdot):
    """Standard alpha-disc effective temperature profile.

    Parameters
    ----------
    r: np.ndarray or float
        An array of radii or a single radius to calculate the temperature at.
    r_co: float
        The radius of the central object.
    m_co: float
        The mass of the central object in units of solar masses.
    mdot: float
        The accretion rate onto the central object in units of solar masses per
        year.

    Returns
    -------
    teff: np.ndarray or float
        The effective temperature at the provided radius or radii.
    """

    m_co *= MSOL
    mdot *= MSOL_PER_YEAR

    teff4 = (3 * G * m_co * mdot) / (8 * np.pi * r**3 * STEFAN_BOLTZMANN)
    teff4 *= 1 - (r_co / r)**0.5

    return teff4**0.25


def create_disc_spectrum(m_co, mdot, r_in, r_out, freq_min, freq_max, n_freq=5000, n_rings=5000, modified_teff=False,
                         freq_units=True):
    """Create a crude accretion disc spectrum. This works by approximating an
    accretion disc as being a collection of annuli radiating at different
    temperatures and treats them as a blackbody. The emerging spectrum is then
    an ensemble of these blackbodies.

    Parameters
    ----------
    m_co: float
        The mass of the central object in solar masses.
    mdot: float
        The accretion rate onto the central object in solar masses per year.
    r_in: float
        The inner radius of the accretion disc in cm.
    r_out: float
        The outer radius of the accretion disc in cm.
    freq_min: float
        The low frequency boundary of the spectrum to create.
    freq_max: float
        The high frequency boundary of the spectrum to create.
    freq_units: float
        Calculate the spectrum in frequency units, or wavelength units if False.
    n_freq: int
        The number of frequency bins in the spectrum.
    n_rings: int
        The number of rings used to model the accretion disc.

    Returns
    -------
    s: pd.DataFrame
        The accretion disc spectrum. If in frequency units, the columns are
        "Freq." (Hz) and "Lum" (ergs/s/cm^2/Hz). If in wavelength units, the columns are
        "Lambda" (A) and "Lum" (ergs/s/cm^2/A).
    """

    if freq_units:
        xlabel = "Freq."
    else:
        xlabel = "Lambda"

    radius = np.logspace(np.log10(r_in), np.log10(r_out), n_rings)
    frequency_bins = np.linspace(freq_min, freq_max, n_freq)
    s = pd.DataFrame(columns=[xlabel, "Lum."])

    # Initialise the data frame
    s[xlabel] = frequency_bins
    s["Lum."] = _calculate_disc_spectrum(m_co, float(mdot), radius, frequency_bins, modified_teff, freq_units)

    return s


def eddington_accretion_limit(mbh, efficiency):
    """Calculate the Eddington accretion limit for a black hole. Note that the
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


@jit
def eddington_luminosity_limit(mbh):
    """Calculate the Eddington luminosity for accretion onto a black hole.

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


@jit
def modified_eddington_alpha_disc_effective_temperature(r, m_co, mdot):
    """The effective temperature profile from Strubbe and Quataert 2009.

    Parameters
    ----------
    r: np.ndarray or float
        An array of radii or a single radius to calculate the temperature at.
    m_co: float
        The mass of the central object in units of solar masses.
    mdot: float
        The accretion rate onto the central object in units of solar masses per
        year.

    Returns
    -------
    teff: np.ndarray or float
        The effective temperature at the provided radius or radii.
    """

    r_isco = innermost_stable_circular_orbit(m_co)
    rg = gravitational_radius(m_co)
    l_edd = eddington_luminosity_limit(m_co)

    m_co *= MSOL
    mdot *= MSOL_PER_YEAR

    fnt = 1 - np.sqrt(r_isco / r)
    teff4 = (3 * G * m_co * mdot * fnt) / (8 * PI * r**3 * STEFAN_BOLTZMANN)
    teff4 *= (0.5 + (0.25 + 6 * fnt * (mdot * C**2 / l_edd)**2 * (r / rg)**-2)**0.5)**-1

    return teff4**0.25
