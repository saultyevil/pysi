#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate various parameters relating to black holes.
"""

from .constants import MSOL, GRAV, VLIGHT


def schwarzchild_radius(
    m_bh: float
) -> float:
    """Calculate the Schwarzschild radius of a black hole of given mass Mbh.

    Parameters
    ----------
    m_bh: float
        The mass of the black hole."""

    m_bh *= MSOL
    r_s = 2 * GRAV * m_bh / VLIGHT ** 2

    return r_s


def gravitational_radius(
    m_bh: float
) -> float:
    """Calculate the gravitational radius of a black hole of given mass Mbh.

    Parameters
    ----------
    m_bh: float
        The mass of the black hole."""

    m_bh *= MSOL
    r_g = GRAV * m_bh / VLIGHT ** 2

    return r_g


def innermost_stable_orbit(
    m_bh: float
) -> float:
    """Calculate the radius of the innermost stable circular orbit of a
    black hole of given mass Mbh.


    Parameters
    ----------
    m_bh: float
        The mass of the black hole."""

    r_s = schwarzchild_radius(m_bh)
    r_isco = 3 * r_s

    return r_isco
