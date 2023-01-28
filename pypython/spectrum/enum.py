#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Enumerators for spectra."""

from enum import Enum, auto


class SpectrumSpectralAxis(Enum):
    """The possible spatial units for a spectrum.

    Either wavelength or frequency, and these WILL ALWAYS be in units of
    Angstroms and Hz, respectively.
    """

    FREQUENCY = auto()
    WAVELENGTH = auto()
    NONE = auto()


class SpectrumUnits(Enum):
    """Possible units for the spectra created in Python.

    Note the typo in the per wavelength units. This is due to a typo in
    Python.
    """

    L_NU = auto()
    L_LAM = auto()
    F_NU = auto()
    F_LAM = auto()
    F_LAM_LEGACY = auto()
    NONE = auto()


class SpectrumType(Enum):
    """The possible types of spectra which can be read in.

    This should cover all the types, and should be interchangable
    between the linear and logarithmic versions of the spectra.
    """

    SPEC = auto()
    SPEC_TOT = auto()
    SPEC_WIND = auto()
    SPEC_TOT_WIND = auto()
    SPEC_TAU = auto()
