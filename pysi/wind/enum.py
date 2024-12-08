"""Enumerators for the wind classes."""

from enum import Enum, auto


class CoordSystem(Enum):
    """Possible grid coordinate systems in Python."""

    CYLINDRICAL = auto()
    POLAR = auto()
    SPHERICAL = auto()
    UNKNOWN = auto()


class VelocityUnits(Enum):
    """Possible velocity conversions for Wind objects."""

    CENTIMETRES_PER_SECOND = auto()
    METRES_PER_SECOND = auto()
    KILOMETRES_PER_SECOND = auto()
    SPEED_OF_LIGHT = auto()


class DistanceUnits(Enum):
    """Possible distance conversions for Wind objects."""

    CENTIMETRES = auto()
    METRES = auto()
    KILOMETRES = auto()
    GRAVITATIONAL_RADIUS = auto()


class WindCellPosition(Enum):
    """Wind cells can either be fully "inwind" or partially inwind."""

    INWIND = 0
    PARTIALLY_INWIND = 1
