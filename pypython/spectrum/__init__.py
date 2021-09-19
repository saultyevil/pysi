#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions for creating and analyzing spectra."""
import copy
import textwrap
from enum import Enum
from os import path

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson

import pypython
import pypython.constants as c
import pypython.physics
import pypython.plot as pyplt
import pypython.spectrum.plot as splt

# Units enumerators ------------------------------------------------------------


class SpectrumThing(Enum):

    frequency = "Hz"
    wavelength = "Angstrom"


class SpectrumUnits(Enum):
    """Possible units for the spectra created in Python.

    Note the typo in the per wavelength units. This is due to a typo in
    Python.
    """
    l_nu = "erg/s/Hz"
    l_lm = "erg/s/A"
    f_nu = "erg/s/cm^-2/Hz"
    f_lm = "erg/s/cm^-2/A"
    unknown = "unknown"


# Spectrum class ---------------------------------------------------------------


class Spectrum:
    """A class to store PYTHON .spec and .log_spec files.

    The Python spectra are read in and stored within a dict of dicts,
    where each column name is the spectrum name and the columns in that
    dict are the names of the columns in the spectrum file. The data is
    stored as numpy arrays.
    """
    def __init__(self, root, fp=".", log_spec=True, smooth=None, distance=None, default=None, delim=None):
        """Create the Spectrum object.

        Construct the file path of the spectrum files given the
        root, directory and whether the logarithmic spectrum is used or not.
        The different spectra are then read in, with either the .spec or the
        first spectrum file read in being the default index choice.

        Parameters
        ----------
        root: str
            The root name of the model.
        fp: str [optional]
            The directory containing the model.
        default: str [optional]
            The default spectrum to make the available spectrum for indexing.
        log_spec: bool [optional]
            Read in the logarithmic version of the spectra.
        smooth: int [optional]
            The amount of smoothing to use.
        distance: float [optional]
            The distance of the spectrum flux, in units of parsec.
        delim: str [optional]
            The deliminator in the spectrum file.
        """
        root, fp = pypython._cleanup_root(root, fp)

        self.root = root
        self.fp = path.expanduser(fp)
        if self.fp[-1] != "/":
            self.fp += "/"
        self.pf = self.fp + self.root + ".pf"

        self.log_spec = log_spec
        if default and self.log_spec:
            if not default.startswith("log_") and default != "spec_tau":
                default = "log_" + default

        # Initialize the important members
        # Each spectrum is stored as a key in a dictionary. Each dictionary
        # contains keys for each column in the spectrum file, as well as other
        # meta info such as the distance of the spectrum

        self.spectra = pypython._AttributeDict({})
        self._original_spectra = None
        self.available = []

        # Now we can read in the spectra and set the default/target spectrum
        # for the object. We can also re-scale to a different distance.

        self.get_spectra(delim)

        if default:
            self.current = self._get_spec_key(default)
        else:
            self.current = self.available[0]

        self.set(self.current)

        if distance:
            self.set_distance(distance)

        # Smooth all the spectra. A copy of the unsmoothed spectra is kept
        # in the member self.original.

        if smooth:
            self.smooth(smooth)

    # Private methods ----------------------------------------------------------

    def _get_spec_key(self, name):
        """Get the key depending on if the log or linear version of a spectrum
        was read in.

        Parameters
        ----------
        name: str
            The name of the spectrum to get the key for.

        Returns
        -------
        name: str
            The key for the spectrum requested.
        """
        if self.log_spec and not name.startswith("log_") and name != "spec_tau":
            name = "log_" + name

        return name

    def _plot_observer_spectrum(self, label_lines=False):
        """Plot the spectrum components and observer spectra on a 1x2 panel
        plot. The left panel has the components, whilst the right panel has the
        observer spectrum.

        Parameters
        ----------
        label_lines: bool
            Plot line IDs.
        """
        name = self._get_spec_key("spec")
        if name not in self.available:
            raise IOError("A .spec/.log_spec file was not read in, cannot use this function")

        fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey="row")

        # Plot the components of the observer spectrum, i.e Emitted, Created,
        # Disc, etc.

        for component in self.columns[:-self.n_inclinations]:
            if component in ["Lambda", "Freq."]:
                continue
            ax[0] = self._plot_thing(component, name, label_lines=label_lines, ax_update=ax[0])

        for line in ax[0].get_lines():  # Set the different spectra to have a transparency
            line.set_alpha(0.7)
        ax[0].legend(ncol=2, loc="upper right").set_zorder(0)

        # Now plot the observer spectra

        for inclination in self.inclinations:
            ax[1] = self._plot_thing(inclination, name, label_lines=label_lines, ax_update=ax[1])

        for label, line in zip(self.inclinations, ax[1].get_lines()):
            line.set_alpha(0.7)
            line.set_label(str(label) + r"$^{\circ}$")
        ax[1].set_ylabel("")
        ax[1].legend(ncol=2, loc="upper right").set_zorder(0)

        # Final clean up to make a nice spectrum

        ax[0].set_title("Components")
        ax[1].set_title("Observer spectra")
        fig = pyplt.finish_figure(fig, wspace=0)

        return fig, ax

    def _plot_thing(self, thing, spec_type, scale="loglog", label_lines=False, ax_update=None):
        """Plot a specific column in a spectrum file.

        Parameters
        ----------
        thing: str
            The name of the thing to be plotted.
        scale: str
            The scale of the axes, i.e. loglog, logx, logy, linlin.
        label_lines: bool
            Plot line IDs.
        ax_update: plt.Axes
            An plt.Axes object to update, i.e. to plot on.
        """
        if ax_update:
            ax = ax_update
        else:
            fig, ax = plt.subplots(figsize=(9, 5))

        if spec_type:
            key = spec_type
        else:
            key = self.current

        units = self.spectra[key].units
        distance = self.spectra[key].distance
        ax = pyplt.set_axes_scales(ax, scale)
        ax = splt.set_axes_labels(ax, units=units, distance=distance)

        # How things are plotted depends on the units of the spectrum

        if units == SpectrumUnits.f_lm or units == SpectrumUnits.l_lm:
            x_thing = "Lambda"
        else:
            x_thing = "Freq."

        label = thing
        if thing.isdigit():
            label += r"$^{\circ}$"

        ax.plot(self.spectra[key][x_thing], self.spectra[key][thing], label=label, zorder=0)

        if label_lines:
            ax = plot.add_line_ids(ax, plot.common_lines(units), linestyle="none", fontsize=10)

        if ax_update:
            return ax
        else:
            fig = pyplt.finish_figure(fig)
            return fig, ax

    # Methods ------------------------------------------------------------------

    def convert_flux_to_luminosity(self):
        """Convert the spectrum from flux into luminosity units.

        This is easily done using the relationship F = L / (4 pi d^2). This
        method is applied to all the spectra currently loaded in the class,
        but only if the units are already a flux.
        """
        for spectrum in self.available:
            if spectrum == "spec_tau":
                continue

            distance = self.spectra[spectrum]["distance"]
            units = self.spectra[spectrum]["units"]

            if units in [SpectrumUnits.l_lm, SpectrumUnits.l_nu]:
                continue

            for column in self.spectra[spectrum].columns:
                self.spectra[spectrum][column] *= 4 * np.pi * (distance * c.PARSEC)**2

            if units == SpectrumUnits.f_nu:
                self.spectra[spectrum].units = SpectrumUnits.l_nu
            else:
                self.spectra[spectrum].units = SpectrumUnits.l_lm

    def convert_luminosity_to_flux(self, distance):
        """Convert the spectrum from luminosity into flux units.

        This is easily done by using the relationship F = L / (4 pi d^2). This
        method is applied to all the spectra currently loaded in the class,
        but only if the units are not a flux.
        """

        for spectrum in self.available:
            if spectrum == "spec_tau":
                continue

            distance = self.spectra[spectrum]["distance"]
            units = self.spectra[spectrum]["units"]

            if units in [SpectrumUnits.f_lm, SpectrumUnits.f_nu]:
                continue

            for column in self.spectra[spectrum].columns:
                self.spectra[spectrum][column] /= 4 * np.pi * (distance * c.PARSEC)**2

            if units == SpectrumUnits.l_nu:
                self.spectra[spectrum].units = SpectrumUnits.f_nu
            else:
                self.spectra[spectrum].units = SpectrumUnits.f_lm

    def get_spectra(self, delim=None):
        """Read in a spectrum file given in self.filepath. The spectrum is
        stored as a dictionary in self.spectra where each key is the name of
        the columns.

        Parameters
        ----------
        delim: str [optional]
            A custom delimiter, useful for reading in files which have sometimes
            between delimited with commas instead of spaces.
        """
        n_read = 0
        files_to_read = ["spec", "spec_tot", "spec_tot_wind", "spec_wind", "spec_tau"]

        # Read in each spec file type, and store each spectrum as a key in
        # self.avail_spec, etc.

        for spec_type in files_to_read:
            fp = self.fp + self.root + "."
            if self.log_spec and spec_type != "spec_tau":
                spec_type = "log_" + spec_type
            fp += spec_type
            if not path.exists(fp):
                continue

            n_read += 1

            self.spectra[spec_type] = pypython._AttributeDict({
                "units": SpectrumUnits.unknown,
            })

            with open(fp, "r") as f:
                spectrum_file = f.readlines()

            # Read in the spectrum file. Ignore empty lines and lines which have
            # been commented out by #

            spectrum = []

            for line in spectrum_file:
                line = line.strip()
                if delim:
                    line = line.split(delim)
                else:
                    line = line.split()
                if "Units:" in line:
                    self.spectra[spec_type]["units"] = SpectrumUnits(line[4][1:-1])
                    if self.spectra[spec_type]["units"] in [SpectrumUnits.f_lm, SpectrumUnits.f_nu]:
                        self.spectra[spec_type]["distance"] = float(line[6])
                    else:
                        self.spectra[spec_type]["distance"] = 0

                if len(line) == 0 or line[0] == "#":
                    continue

                spectrum.append(line)

            # Extract the header columns of the spectrum. This assumes the first
            # read line in the spectrum is the header.

            header = []  # wish this was short enough to do in a list comprehension

            for i, column_name in enumerate(spectrum[0]):
                if column_name[0] == "A":
                    j = column_name.find("P")
                    column_name = column_name[1:j].lstrip("0")  # remove leading 0's for, i.e., 01 degrees
                header.append(column_name)

            columns = [column for column in header if column not in ["Freq.", "Lambda"]]
            spectrum = np.array(spectrum[1:], dtype=np.float64)

            # Add the spectrum to self.avail_spectrum[spec_type]. The keys of
            # the dictionary are the column names in the spectrum, i.e. what
            # is in the header

            for i, column_name in enumerate(header):
                self.spectra[spec_type][column_name] = spectrum[:, i]

            inclinations = []  # this could almost be a list comprehension...

            for col in header:
                if col.isdigit() and col not in inclinations:
                    inclinations.append(col)

            self.spectra[spec_type]["columns"] = tuple(columns)
            self.spectra[spec_type]["inclinations"] = tuple(inclinations)
            self.spectra[spec_type]["n_inclinations"] = len(inclinations)

        if n_read == 0:
            raise IOError(f"Unable to open any spectrum files for {self.root} in {self.fp}")

        self.available = tuple(self.spectra.keys())

    def plot(self, names=None, spec_type=None, scale="loglog", label_lines=False):
        """Plot the spectra or a single component in a single figure. By
        default this creates a 1 x 2 of the components on the left and the
        observer spectra on the right. Useful for when in an interactive
        session.

        Parameters
        ----------
        names: str
            The name of the thing to plot.
        spec_type: str
            The spectrum the thing to plot belongs in.
        scale: str
            The scale of the axes, i.e. loglog, logx, logy or linlin.
        label_lines: bool
            Plot line IDs.
        """
        # If name is given, then plot that column of the spectrum. Otherwise
        # assume we just want to plot all columns in the spec file

        if names:
            if type(names) is not list:
                names = [names]
            fig, ax = self._plot_thing(str(names[0]), spec_type, scale, label_lines)
            if len(names) > 1:
                for name in names[1:]:
                    ax = self._plot_thing(str(name), spec_type, scale, label_lines, ax_update=ax)
            ax.legend()
        else:
            fig, ax = self._plot_observer_spectrum(label_lines)

        return fig, ax

    def set(self, name):
        """Set a spectrum as the default.

        Sets a different spectrum to be the currently available spectrum for
        indexing.

        Parameters
        ----------
        name: str
            The name of the spectrum, i.e. log_spec or spec_tot, etc. The
            available spectrum types are stored in self.available.
        """
        name = self._get_spec_key(name)
        if name not in self.available:
            raise IndexError(f"spectrum {name} is not available: available are {self.available}")

        self.current = name

    def set_distance(self, distance):
        """Rescale the flux to the given distance.

        Parameters
        ----------
        distance: float or int
            The distance to scale the flux to.
        """
        if type(distance) is not float and type(distance) is not int:
            raise ValueError("distance is not a float or integer")

        for spectrum in self.available:
            if spectrum == "spec_tau":
                continue
            if self.spectra[spectrum].units == SpectrumUnits.l_nu:
                continue
            for key in self.spectra[spectrum].columns:
                if key in ["Lambda", "Freq."]:
                    continue
                self.spectra[spectrum][key] *= \
                    (self.spectra[spectrum].distance * c.PARSEC) ** 2 / (distance * c.PARSEC) ** 2
            self.spectra[spectrum].distance = distance

    @staticmethod
    def show(block=True):
        """Show a plot which has been created.

        Wrapper around pyplot.show().

        Parameters
        ----------
        block: bool
            Use blocking or non-blocking figure display.
        """
        plt.show(block=block)

    def smooth(self, width=5):
        """Smooth the spectrum flux/luminosity bins.

        If this is used after the spectrum has already been smoothed, then the
        "original" is copied back into the spectrum before smoothing again. This
        way the function does not smooth an already smoothed spectrum.

        Parameters
        ----------
        width: int [optional]
            The width of the boxcar filter (in bins).
        """
        if self._original_spectra is None:
            self._original_spectra = copy.deepcopy(self.spectra)
        else:
            self.spectra = copy.deepcopy(self._original_spectra)

        # Loop over each available spectrum and smooth it

        for spectrum in self.available:
            if spectrum == "spec_tau":  # todo: cleaner way to skip spec_tau
                continue

            for thing_to_smooth in self.spectra[spectrum].columns:
                try:
                    self.spectra[spectrum][thing_to_smooth] = pypython.smooth_array(self.spectra[spectrum][thing_to_smooth],
                                                                                    width)
                except KeyError:
                    pass  # some spectra do not have the inclination angles...

    def restore_original_spectra(self):
        """Restore the spectrum to its original unsmoothed form."""
        self.spectra = copy.deepcopy(self._original_spectra)

    # Built in stuff -----------------------------------------------------------

    def __getattr__(self, key):
        return self.spectra[self.current][key]

    def __getitem__(self, key):
        if self._get_spec_key(key) in self.available:
            return self.spectra[self._get_spec_key(key)]
        else:
            return self.spectra[self.current][key]

    # def __setattr__(self, name, value):
    #     self.spectra[self.current][name] = value

    def __setitem__(self, key, value):
        if self._get_spec_key(key) in self.available:
            self.spectra[self._get_spec_key(key)] = value
        else:
            self.spectra[self.current][key] = value

    def __str__(self):
        msg = f"Spectrum for the model {self.root}\n"
        msg += f"\nDirectory {self.fp}"
        msg += f"\nAvailable spectra {self.available}"
        msg += f"\nCurrent spectrum {self.current}"
        if "spec" in self.available or "log_spec" in self.available:
            if self.log_spec:
                key = "log_spec"
            else:
                key = "spec"
            msg += f"\nSpectrum inclinations: {self.spectra[key].inclinations}"
        if "tau_spec" in self.available:
            msg += f"\nOptical depth inclinations {self.spectra.tau_spec.inclinations}"

        return textwrap.dedent(msg)


# Functions --------------------------------------------------------------------


def integrate(spectrum, name, xmin, xmax, spec_type=None):
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
    spectrum: pypython.Spectrum
        The spectrum class containing the spectrum to integrate.
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
    if spec_type:
        key = spec_type
    else:
        key = spectrum.current

    if spectrum[key].units == SpectrumUnits.l_lm or spectrum[key].units == SpectrumUnits.f_lm:
        sample_points = spectrum[key]["Lambda"]
    else:
        sample_points = spectrum[key]["Freq."]
        tmp = xmin
        xmin = pypython.physics.angstrom_to_hz(xmax)
        xmax = pypython.physics.angstrom_to_hz(tmp)

    sample_points, y = pypython.get_xy_subset(sample_points, spectrum[key][name], xmin, xmax)

    return simpson(y, sample_points)

