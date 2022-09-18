#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contains the user facing Wind class."""

from enum import Enum
from typing import Union

import pypython.wind_class.plot


class Units(Enum):
    """The possible unit types for spatial coordinates and velocity."""

    CENTIMETRES = "cm"
    METRES = "m"
    GRAVITATIONAL = "rg"
    CENTIMETRES_PER_SECOND = "cms"


class Wind(pypython.wind_class.plot.WindPlot):
    """Main wind class for pypython.

    This class includes...
    """
    def __init__(self, root: str, directory: str, **kwargs) -> None:
        """Initialize the class."""

        super().__init__(root, directory, **kwargs)
        self.spatial_units = Units.CENTIMETRES
        self.velocity_units = Units.CENTIMETRES_PER_SECOND

    def change_units(self, new_units: Union[str, Units]) -> None:
        """Change the spatial or velocity units."""
        pass
