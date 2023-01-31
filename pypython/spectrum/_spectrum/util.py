#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utility fuctionality for the spectrum class."""

from scipy.integrate import simpson

from pypython import physics
from pypython import utilities
from pypython.spectrum import enum
from pypython.spectrum._spectrum import base


class SpectrumUtil(base.SpectrumBase):
    """Utility functions."""

    def __init__(self, root, directory, **kwargs):
        """Initialize the class."""
        super().__init__(root, directory, **kwargs)

    def integrate(self, name, spec_type=None):
        """Integrate the entire spectrum.

        Parameters
        ----------
        name: str
            The name of the spectrum to integrate, i.e. "60", "Emitted".
        spec_type: str [optional]
            The spectrum type to use. If this is None, then spectrum.current is
            used

        Returns
        -------
        The integral of the entire spectrum
        """
        x_min = None
        x_max = None

        return self.integrate_between_limits(name, x_min, x_max, spec_type)

    def integrate_between_limits(self, name, xmin, xmax, spec_type=None) -> float:
        """Integrate a sub-range of a spectrum.

        By integrating a spectrum in luminosity units between [xmin, xmax], it
        is possible to calculate the total luminosity of a given wavelength band.
        For example, by using xmin, xmax = 3000, 8000 Angstroms, the total optical
        luminosity can be estimated.

        This function uses Simpson's rule to approximate the integral given the
        wavelength/frequency bins (used as the sample points) and the luminosity
        bins.

        Parameters
        ----------
        name: str
            The name of the spectrum to integrate, i.e. "60", "Emitted".
        xmin: float
            The lower integration bound, in Angstroms.
        xmax: float
            The upper integration bound, in Angstroms.
        spec_type: str [optional]
            The spectrum type to use. If this is None, then spectrum.current is
            used

        Returns
        -------
        The integral between of the spectrum between xmin and xmax.
        """
        spec_key = spec_type if spec_type else self.current

        if self.spectra[self.scaling][spec_key].units in [enum.SpectrumUnits.L_LAM, enum.SpectrumUnits.F_LAM]:
            x_points = self.spectra[self.scaling][spec_key]["Lambda"]
        else:
            x_points = self.spectra[self.scaling][spec_key]["Freq."]
            tmp = xmin  # temp variable whilst we convert to frequency space, since the order swaps
            xmin = physics.angstrom_to_hz(xmax)
            xmax = physics.angstrom_to_hz(tmp)

        x_points, y_points = utilities.xy_subset(x_points, self.spectra[self.scaling][spec_key][name], xmin, xmax)

        return simpson(y_points, x_points)
