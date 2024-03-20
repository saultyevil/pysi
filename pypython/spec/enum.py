#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Enumerators for spectra."""

from enum import auto
from aenum import MultiValueEnum


# pylint: disable=too-few-public-methods
class SpectrumSpectralAxis(MultiValueEnum):
    """The possible spatial units for a spectrum.

    Either wavelength or frequency, and these WILL ALWAYS be in units of
    Angstroms and Hz, respectively.
    """

    FREQUENCY = auto(), "Hz"
    WAVELENGTH = auto(), "Angstrom"
    NONE = auto()


class SpectrumUnits(MultiValueEnum):
    """Possible units for the spectra created in Python.

    Note the typo in the per wavelength units. This is due to a typo in
    Python.
    """

    TAU_NU = auto(), "\tau_{\nu}"
    L_NU = auto(), "erg/s/Hz"
    L_LAM = auto(), "erg/s/A"
    F_NU = auto(), "erg/s/cm^-2/Hz"
    F_LAM = auto(), "erg/s/cm^-2/A"
    F_LAM_LEGACY = auto(), "erg/s/cm^2/A"  # TODO, need to check what this is
    NONE = auto()


class SpectrumType(MultiValueEnum):
    """The possible types of spectra which can be read in.

    This should cover all the types, and should be interchangable
    between the linear and logarithmic versions of the spectra.
    """

    SPEC = auto(), "spec"
    SPEC_TOT = auto(), "spec_tot"
    SPEC_WIND = auto(), "spec_wind"
    SPEC_TOT_WIND = auto(), "spec_tot_wind"
    SPEC_TAU = auto(), "spec_tau"
