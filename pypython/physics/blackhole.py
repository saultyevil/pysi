#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate parameters relating to Schwarzchild black holes.

All functions require the mass to be in units of solar masses and
accretion rates in solar masses per year.
"""

from pypython.constants import GRAV, MSOL, RSOL, VLIGHT


def innermost_stable_circular_orbit(m_bh):
    """Calculate the radius of the innermost stable circular orbit of a black
    hole of given mass Mbh.

    Parameters
    ----------
    m_bh: float
        The mass of the black hole in solar masses.

    Returns
    -------
    The radius of the innermost stable circular orbit in cm.
    """
    return 3 * schwarzschild_radius(m_bh)


def gravitational_radius(m_bh):
    """Calculate the gravitational radius of a black hole of given mass Mbh.

    Parameters
    ----------
    m_bh: float
        The mass of the black hole in solar masses.

    Returns
    -------
    The gravitational radius in cm.
    """
    return GRAV * m_bh * MSOL / VLIGHT**2


def schwarzschild_radius(m_bh):
    """Calculate the Schwarzschild radius of a black hole of given mass Mbh.

    Parameters
    ----------
    m_bh: float
        The mass of the black hole in solar masses.

    Returns
    -------
    The Schwarzschild radius in cm.
    """
    return 2 * gravitational_radius(m_bh)


def tidal_disruption_radius(m_bh, m_star, r_star):
    """Calculate the disruption radius of a black hole for a given solar mass.

    Parameters
    ----------
    m_bh: float
        The mass of the black hole in solar masses.
    m_star: float
        The mass of the disrupted star in solar masses.
    r_star: float
        The radius of the disrupted star in solar radii.
    """
    return r_star * (m_bh / m_star)**(1 / 3) * RSOL
