"""pysi - making using Python a wee bit easier.

pysi is a companion python package to handle and analyse the data which
comes out of a Python simulation.
"""

__version__ = "4.1.2"

from . import math, sim, spec, util, wind
from .error import (
    CoordError,
    DimensionError,
    InvalidFileContentsError,
    InvalidParameterError,
    ShellRunError,
    SIROCCOError,
)
from .spec import Spectrum
from .wind import Wind
