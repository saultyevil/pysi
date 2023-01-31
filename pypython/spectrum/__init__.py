#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions for reading in and analyzing spectra.

The main part of this is the Spectrum() class -- well, it's pretty much
the only thing in here now after the re-structure.
"""

from pathlib import Path
from typing import Union

from pypython.spectrum._spectrum import plot


class Spectrum(plot.SpectrumPlot):
    """Main spectrum class."""

    def __init__(self, root: str, directory: Union[str, Path] = ".", **kwargs) -> None:
        """Initialize the class."""
        super().__init__(root, directory, **kwargs)

    def __str__(self) -> str:
        return f"Spectrum(root={self.root!r} directory={str(self.directory)!r})"


# Spectrum class ---------------------------------------------------------------


# class Spectrum:
#     """A class to store PYTHON .spec and .log_spec files.

#     The Python spectra are read in and stored within a dict of dicts,
#     where each column name is the spectrum name and the columns in that
#     dict are the names of the columns in the spectrum file. The data is
#     stored as numpy arrays.

#     TODO: Read in linear and logarithmic version of the spectra
#     TODO: Create Enum for spectrum types
#     TODO: Create Enum for "spatial type", i.e. for frequency or lambda monochromatic things
#     """

#     def __init__(self, root, fp=".", log=True, smooth=None, distance=None, default=None, delim=None):
#         """Create the Spectrum object.

#         Construct the file path of the spectrum files given the
#         root, directory and whether the logarithmic spectrum is used or not.
#         The different spectra are then read in, with either the .spec or the
#         first spectrum file read in being the default index choice.

#         Parameters
#         ----------
#         root: str
#             The root name of the model.
#         fp: str [optional]
#             The directory containing the model.
#         default: str [optional]
#             The default spectrum to make the available spectrum for indexing.
#         log: bool [optional]
#             Read in the logarithmic version of the spectra.
#         smooth: int [optional]
#             The amount of smoothing to use.
#         distance: float [optional]
#             The distance of the spectrum flux, in units of parsec.
#         delim: str [optional]
#             The deliminator in the spectrum file.
#         """
#         root, fp = pypython._cleanup_root(root, fp)

#         self.root = root
#         self.fp = path.expanduser(fp)
#         if self.fp[-1] != "/":
#             self.fp += "/"
#         self.pf = self.fp + self.root + ".pf"

#         self.log_spec = log
#         if default and self.log_spec:
#             if not default.startswith("log_") and default != "spec_tau":
#                 default = "log_" + default

#         # Initialize the important members
#         # Each spectrum is stored as a key in a dictionary. Each dictionary
#         # contains keys for each column in the spectrum file, as well as other
#         # meta info such as the distance of the spectrum

#         self.spectra = {}
#         self._original_spectra = None
#         self.available = []

#         # Now we can read in the spectra and set the default/target spectrum
#         # for the object. We can also re-scale to a different distance.

#         self.read_in_spectra(delim)

#         if default:
#             self.current = self._get_spec_key(default)
#         else:
#             self.current = self.available[0]

#         self.set(self.current)

#         if distance:
#             self.set_distance(distance)

#         # Smooth all the spectra. A copy of the unsmoothed spectra is kept
#         # in the member self.original.

#         if smooth:
#             self.smooth(smooth)

#     # Private methods ----------------------------------------------------------

#     @staticmethod
#     def _get_spectral_axis(units):
#         """Get the spectral axis units of a spectrum.

#         Determines the spectral axis, given the units of the spectrum.

#         Parameters
#         ----------
#         units: SpectrumUnits
#             The units of the spectrum.

#         Returns
#         -------
#         spectral_axis: SpectrumSpectralAxis
#             The spectral axis units of the spectrum
#         """
#         if units in [SpectrumUnits.f_lm, SpectrumUnits.l_lm]:
#             spectral_axis = SpectrumSpectralAxis.wavelength
#         elif units in [SpectrumUnits.f_nu, SpectrumUnits.l_nu]:
#             spectral_axis = SpectrumSpectralAxis.frequency
#         else:
#             spectral_axis = SpectrumSpectralAxis.none

#         return spectral_axis

#     def _plot_observer_spectrum(self, label_lines=False):
#         """Plot the spectrum components and observer spectra on a 1x2 panel
#         plot. The left panel has the components, whilst the right panel has the
#         observer spectrum.

#         Parameters
#         ----------
#         label_lines: bool
#             Plot line IDs.
#         """
#         name = self._get_spec_key("spec")
#         if name not in self.available:
#             raise IOError("A .spec/.log_spec file was not read in, cannot use this function")

#         fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey="row")

#         # Plot the components of the observer spectrum, i.e Emitted, Created,
#         # Disc, etc.

#         for component in self.columns[: -self.n_inclinations]:
#             if component in ["Lambda", "Freq."]:
#                 continue
#             ax[0] = self._plot_thing(component, name, label_lines=label_lines, ax_update=ax[0])

#         for line in ax[0].get_lines():  # Set the different spectra to have a transparency
#             line.set_alpha(0.7)
#         ax[0].legend(ncol=2, loc="upper right").set_zorder(0)

#         # Now plot the observer spectra

#         for inclination in self.inclinations:
#             ax[1] = self._plot_thing(inclination, name, label_lines=label_lines, ax_update=ax[1])

#         for label, line in zip(self.inclinations, ax[1].get_lines()):
#             line.set_alpha(0.7)
#             line.set_label(str(label) + r"$^{\circ}$")
#         ax[1].set_ylabel("")
#         ax[1].legend(ncol=2, loc="upper right").set_zorder(0)

#         # Final clean up to make a nice spectrum

#         ax[0].set_title("Components")
#         ax[1].set_title("Observer spectra")
#         fig = pyplt.finish_figure(fig, wspace=0)

#         return fig, ax

#     def _plot_thing(self, thing, spec_type, scale="loglog", xmin=None, xmax=None, label_lines=False, ax_update=None):
#         """Plot a specific column in a spectrum file.

#         Parameters
#         ----------
#         thing: str
#             The name of the thing to be plotted.
#         scale: str
#             The scale of the axes, i.e. loglog, logx, logy, linlin.
#         xmin: float
#             The lower boundary for the x axis.
#         xmax: float
#             The upper boundary for the x axis.
#         label_lines: bool
#             Plot line IDs.
#         ax_update: plt.Axes
#             An plt.Axes object to update, i.e. to plot on.
#         """
#         if ax_update:
#             ax = ax_update
#         else:
#             fig, ax = plt.subplots(figsize=(9, 5))

#         if spec_type:
#             key = spec_type
#         else:
#             key = self.current

#         spec_axis = self.spectra[key].spectral_axis
#         distance = self.spectra[key].distance
#         ax = pyplt.set_axes_scales(ax, scale)
#         ax = splt.set_axes_labels(ax, units=spec_axis, distance=distance)

#         # How things are plotted depends on the units of the spectrum

#         if spec_axis == SpectrumSpectralAxis.wavelength:
#             x_thing = "Lambda"
#         else:
#             x_thing = "Freq."

#         label = thing
#         if thing.isdigit():
#             label += r"$^{\circ}$"

#         x, y = pypython.get_xy_subset(self.spectra[key][x_thing], self.spectra[key][thing], xmin, xmax)
#         ax.plot(x, y, label=label, zorder=0)

#         if label_lines:
#             ax = plot.add_line_ids(ax, plot.common_lines(spectral_axis=spec_axis), linestyle="none", fontsize=10)

#         if ax_update:
#             return ax
#         else:
#             fig = pyplt.finish_figure(fig)
#             return fig, ax

#     # Methods ------------------------------------------------------------------

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

#     def plot(self, names=None, spec_type=None, scale="loglog", xmin=None, xmax=None, label_lines=False):
#         """Plot the spectra or a single component in a single figure. By
#         default this creates a 1 x 2 of the components on the left and the
#         observer spectra on the right. Useful for when in an interactive
#         session.

#         Parameters
#         ----------
#         names: str
#             The name of the thing to plot.
#         spec_type: str
#             The spectrum the thing to plot belongs in.
#         scale: str
#             The scale of the axes, i.e. loglog, logx, logy or linlin.
#         xmin: float
#             The lower boundary for the x axis.
#         xmax: float
#             The upper boundary for the x axis.
#         label_lines: bool
#             Plot line IDs.
#         """
#         # If name is given, then plot that column of the spectrum. Otherwise
#         # assume we just want to plot all columns in the spec file

#         if names:
#             if type(names) is not list:
#                 names = [names]
#             fig, ax = self._plot_thing(str(names[0]), spec_type, scale, xmin, xmax, label_lines)
#             if len(names) > 1:
#                 for name in names[1:]:
#                     ax = self._plot_thing(str(name), spec_type, scale, xmin, xmax, label_lines, ax_update=ax)
#             ax.legend()
#         else:
#             fig, ax = self._plot_observer_spectrum(label_lines)

#         return fig, ax

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

#     @staticmethod
#     def show(block=True):
#         """Show a plot which has been created.

#         Wrapper around pyplot.show().

#         Parameters
#         ----------
#         block: bool
#             Use blocking or non-blocking figure display.
#         """
#         plt.show(block=block)

#     def smooth(self, width=5):
#         """Smooth the spectrum flux/luminosity bins.

#         If this is used after the spectrum has already been smoothed, then the
#         "original" is copied back into the spectrum before smoothing again. This
#         way the function does not smooth an already smoothed spectrum.

#         Parameters
#         ----------
#         width: int [optional]
#             The width of the boxcar filter (in bins).
#         """
#         if self._original_spectra is None:
#             self._original_spectra = copy.deepcopy(self.spectra)
#         else:
#             self.spectra = copy.deepcopy(self._original_spectra)

#         # Loop over each available spectrum and smooth it

#         for spectrum in self.available:
#             if spectrum == "spec_tau":  # todo: cleaner way to skip spec_tau
#                 continue

#             for thing_to_smooth in self.spectra[spectrum].columns:
#                 try:
#                     self.spectra[spectrum][thing_to_smooth] = pypython.smooth_array(
#                         self.spectra[spectrum][thing_to_smooth], width
#                     )
#                 except KeyError:
#                     pass  # some spectra do not have the inclination angles...

#     def restore_original_spectra(self):
#         """Restore the spectrum to its original unsmoothed form."""
#         self.spectra = copy.deepcopy(self._original_spectra)
