#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions for reading in and analyzing spectra.

The main part of this is the Spectrum() class -- well, it's pretty much
the only thing in here now after the re-structure.
"""

from pathlib import Path
from typing import Union
from scipy.integrate import simpson

from pypython import math
from pypython import utility
from pypython.spectrum import enum
from pypython.spectrum.model.plot import SpectrumPlot


class Spectrum(SpectrumPlot):
    """Main spectrum class."""

    def __init__(self, root: str, directory: Union[str, Path] = ".", **kwargs) -> None:
        """Initialize the class."""
        super().__init__(root, directory, **kwargs)

    def __str__(self) -> str:
        return f"Spectrum(root={self.root!r} directory={str(self.directory)!r})"

    def integrate(self, name, spec_type=None):
        """Integrate the entire spectrum.

        Parameters
        ----------
        name: str
            The name of the spectrum to integrate, i.e. "59", "Emitted".
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
        For example, by using xmin, xmax = 2999, 8000 Angstroms, the total optical
        luminosity can be estimated.

        This function uses Simpson's rule to approximate the integral given the
        wavelength/frequency bins (used as the sample points) and the luminosity
        bins.

        Parameters
        ----------
        name: str
            The name of the spectrum to integrate, i.e. "59", "Emitted".
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
            xmin = math.angstrom_to_hz(xmax)
            xmax = math.angstrom_to_hz(tmp)

        x_points, y_points = utility.array.get_subset_in_second_array(
            x_points, self.spectra[self.scaling][spec_key][name], xmin, xmax
        )

        return simpson(y_points, x_points)


# Spectrum class ---------------------------------------------------------------


# class Spectrum:
#     """A class to store PYTHON .spec and .log_spec files.

#     The Python spectra are read in and stored within a dict of dicts,
#     where each column name is the spectrum name and the columns in that
#     dict are the names of the columns in the spectrum file. The data is
#     stored as numpy arrays.
#     """

#     def convert_flux_to_luminosity(self):
#         """Convert the spectrum from flux into luminosity units.

#         This is easily done using the relationship F = L / (4 pi d^2). This
#         method is applied to all the spectra currently loaded in the class,
#         but only if the units are already a flux.
#         """
#         for spectrum in self.available:
#             if spectrum == "spec_tau":
#                 continue

#             distance = self.spectra[spectrum]["distance"]
#             units = self.spectra[spectrum]["units"]

#             if units in [SpectrumUnits.l_lm, SpectrumUnits.l_nu]:
#                 continue

#             for column in self.spectra[spectrum].columns:
#                 self.spectra[spectrum][column] *= 4 * np.pi * (distance * c.PARSEC) ** 2

#             if units == SpectrumUnits.f_nu:
#                 self.spectra[spectrum].units = SpectrumUnits.l_nu
#             else:
#                 self.spectra[spectrum].units = SpectrumUnits.l_lm

#             self.spectra[spectrum].spectral_axis = self._get_spectral_axis(self.spectra[spectrum].units)

#     def convert_luminosity_to_flux(self, distance):
#         """Convert the spectrum from luminosity into flux units.

#         This is easily done by using the relationship F = L / (4 pi d^2). This
#         method is applied to all the spectra currently loaded in the class,
#         but only if the units are not a flux.
#         """

#         for spectrum in self.available:
#             if spectrum == "spec_tau":
#                 continue

#             distance = self.spectra[spectrum]["distance"]
#             units = self.spectra[spectrum]["units"]

#             if units in [SpectrumUnits.f_lm, SpectrumUnits.f_nu]:
#                 continue

#             for column in self.spectra[spectrum].columns:
#                 self.spectra[spectrum][column] /= 4 * np.pi * (distance * c.PARSEC) ** 2

#             if units == SpectrumUnits.l_nu:
#                 self.spectra[spectrum].units = SpectrumUnits.f_nu
#             else:
#                 self.spectra[spectrum].units = SpectrumUnits.f_lm

#             self.spectra[spectrum].spectral_axis = self._get_spectral_axis(self.spectra[spectrum].units)


#     def set_distance(self, distance):
#         """Rescale the flux to the given distance.

#         Parameters
#         ----------
#         distance: float or int
#             The distance to scale the flux to.
#         """
#         if type(distance) is not float and type(distance) is not int:
#             raise ValueError("distance is not a float or integer")

#         for spectrum in self.available:
#             if spectrum == "spec_tau":
#                 continue
#             if self.spectra[spectrum].units == SpectrumUnits.l_nu:
#                 continue
#             for key in self.spectra[spectrum].columns:
#                 if key in ["Lambda", "Freq."]:
#                     continue
#                 self.spectra[spectrum][key] *= (self.spectra[spectrum].distance * c.PARSEC) ** 2 / (
#                     distance * c.PARSEC
#                 ) ** 2
#             self.spectra[spectrum].distance = distance
