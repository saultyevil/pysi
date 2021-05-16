#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate parameters relating to Schwarzchild black holes.

All functions require the mass to be in units of solar masses and
accretion rates in solar masses per year.
"""

from pypython.constants import GRAV, MSOL, VLIGHT


def gravitational_radius(m_bh):
    """Calculate the gravitational radius of a black hole of given mass Mbh.

    Parameters
    ----------
    m_bh: float
        The mass of the black hole.

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
        The mass of the black hole.

    Returns
    -------
    The Schwarzschild radius in cm.
    """

    return 2 * gravitational_radius(m_bh)


def innermost_stable_circular_orbit(m_bh):
    """Calculate the radius of the innermost stable circular orbit of a black
    hole of given mass Mbh.

    Parameters
    ----------
    m_bh: float
        The mass of the black hole.

    Returns
    -------
    The radius of the innermost stable circular orbit in cm.
    """

    return 3 * schwarzschild_radius(m_bh)
