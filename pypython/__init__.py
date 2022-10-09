#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pypython - making using Python a wee bit easier.

pypython is a companion python package to handle and analyse the data which
comes out of a Python simulation.
"""


import numpy as np

import pypython.error as err
import pypython.math
import pypython.observations
import pypython.physics
import pypython.plot
import pypython.simulation
import pypython.spectrum
import pypython.util
import pypython.wind

# Import all the things which will be able to be seen


# Functions --------------------------------------------------------------------


# Load in all the submodules ---------------------------------------------------

Spectrum = pypython.spectrum.Spectrum
Wind = pypython.wind.Wind

__all__ = [
    # sub-modules
    "dump",
    "math",
    "observations",
    "physics",
    "plot",
    "simulation",
    "spectrum",
    "util",
    "wind",
    # classes
    "Wind",
    "Spectrum",
    # other things
]
