"""Contains the user facing Wind class."""

from .enum import CoordSystem, DistanceUnits, VelocityUnits, WindCellPosition
from .model.plot import WindPlot


class Wind(WindPlot):
    """Main wind class for PyPython."""
