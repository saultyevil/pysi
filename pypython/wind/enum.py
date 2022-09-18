#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Enumerators for the wind classes."""


from enum import Enum


class Units(Enum):
    """The possible unit types for spatial coordinates and velocity."""

    CENTIMETRES = "cm"
    METRES = "m"
    GRAVITATIONAL = "rg"
    CENTIMETRES_PER_SECOND = "cms"


class WindCoordSystem(Enum):
    """Possible grid coordinate systems in Python."""

    cylindrical = "rectilinear"
    polar = "polar"
    spherical = "spherical"
    unknown = "unknown"


class WindVelocityUnits(Enum):
    """Possible velocity conversions for Wind objects.

    Default is cms.
    """

    kms = "kms"
    cms = "cms"
    light = "c"


class WindDistanceUnits(Enum):
    """Possible distance conversions for Wind objects.

    Default is cm.
    """

    cm = "cm"
    m = "m"
    km = "km"
    rg = "rg"


class CellModelType(Enum):
    """Possible model types for cell SED."""

    powerlaw = 1
    exponential = 2
