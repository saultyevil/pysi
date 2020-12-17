#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate various parameters relating to black holes.
"""

from .constants import MSOL, GRAV, VLIGHT


def schwarzchild_radius(Mbh: float):
    """
    Calculated the Schwarzschild radius of a black hole of given mass Mbh.

    Parameters
    ----------
    Mbh: float
        The mass of the black hole.
    """

    Mbh *= MSOL
    r_s = 2 * GRAV * Mbh / VLIGHT ** 2

    return r_s


def gravitational_radius(Mbh: float):
    """
    Calculate the gravitational radius of a black hole of given mass Mbh.

    Parameters
    ----------
    Mbh: float
        The mass of the black hole.
    """

    Mbh *= MSOL
    r_g = GRAV * Mbh / VLIGHT ** 2

    return r_g


def innermost_stable_orbit(Mbh: float):
    """
    Calculate the radius of the innermost stable circular orbit of a
    black hole of given mass Mbh.


    Parameters
    ----------
    Mbh: float
        The mass of the black hole.
    """

    r_s = schwarzchild_radius(Mbh)
    r_isco = 3 * r_s

    return r_isco
