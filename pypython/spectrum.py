#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the spectrum object, as well as utility and plotting functions for
spectra.
"""

import os
import copy
import textwrap
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import boxcar, convolve

from .extrautil.error import InvalidParameter
from .physics.constants import PARSEC
from .plotutil import (ax_add_line_ids, common_lines, get_y_lims_for_x_lims,
                       normalize_figure_style, photoionization_edges,
                       remove_extra_axes, subplot_dims)
from .util import get_root_from_filepath, smooth_array

UNITS_LNU = "erg/s/Hz"
UNITS_FNU = "erg/s/cm^-2/Hz"
UNITS_FLAMBDA = "erg/s/cm^-2/A"


class Spectrum:
    """A class to store PYTHON .spec and .log_spec files.
    The PYTHON spectrum is read in and stored within a dict, where each column
    name is a key and the data is stored as a numpy array."""
    def __init__(self,
                 root: str,
                 cd: str = ".",
                 default: str = None,
                 log: bool = False,
                 smooth: int = None,
                 delim: str = None):
        """Initialise a Spectrum object. This method will construct the file path
        of the spectrum file given the root, containing directory and whether
        the logarithmic spectrum is used or not. The spectrum is then read in.

        Parameters
        ----------
        root: str
            The root name of the model.
        cd: str [optional]
            The directory containing the model.
        default: str [optional]
            The default spectrum to make the available spectrum for indexing.
        log: bool [optional]
            Read in the logarithmic spectrum.
        smooth: int [optional]
            The amount of smoothing to use.
        delim: str [optional]
            The deliminator in the spectrum file."""

        self.root = root
        self.cd = cd
        self.logspec = log

        if log and not default.startswith("log_"):
            default = "log_" + default

        if self.cd[-1] != "/":
            self.cd += "/"

        self.all_spectrum = {}
        self.all_columns = {}
        self.all_inclinations = {}
        self.all_n_inclinations = {}
        self.all_units = {}

        # self.unsmoothed is a variable which keeps a copy of the spectrum for
        # safe keeping if it is smoothed

        self.unsmoothed = None

        # The next method call reads in the spectrum and initializes the above
        # member variables. We also keep track of what spectra have been loaded
        # in and set the "target" spectrum for indexing

        self.read_in_spectra(delim)
        self.available = tuple(self.all_spectrum.keys())

        # Now set the units, etc., to the target spectrum. If default is
        # provided, then this is used as the default spectrum.

        if default:
            if default in self.available:
                self.current = default
            else:
                raise ValueError(
                    f"{self.root}.{default} is not available as it has not been read in"
                )
        else:
            self.current = self.available[0]

        self.spectrum = self.all_spectrum[self.current]
        self.columns = self.all_columns[self.current]
        self.inclinations = self.all_inclinations[self.current]
        self.n_inclinations = self.all_n_inclinations[self.current]
        self.units = self.all_units[self.current]

        if smooth:
            self.smooth(smooth)

    def read_in_spectra(self, delim: str = None):
        """Read in a spectrum file given in self.filepath. The spectrum is stored
        as a dictionary in self.spectrum where each key is the name of the
        columns.

        Parameters
        ----------
        delim: str [optional]
            A custom delimiter, useful for reading in files which have sometimes
            between delimited with commas instead of spaces."""

        n_read = 0
        files_to_read = [
            "spec", "spec_tot", "spec_tot_wind", "spec_wind", "spec_tau"
        ]

        for spec_type in files_to_read:
            fpath = self.cd + self.root + "."
            if self.logspec and spec_type != "spec_tau":
                fpath += "log_"
            fpath += spec_type

            if not os.path.exists(fpath):
                continue

            n_read += 1
            self.all_spectrum[spec_type] = {}
            self.all_units[spec_type] = "unknown"

            with open(fpath, "r") as f:
                spectrum_file = f.readlines()

            # Read in the spectrum file, ignoring empty lines and lines which have
            # been commented out by # at the beginning
            # todo: need some method to detect incorrect syntax

            spectrum = []

            for line in spectrum_file:
                line = line.strip()
                if delim:
                    line = line.split(delim)
                else:
                    line = line.split()
                if "Units:" in line:
                    self.all_units[spec_type] = line[4][1:-1]
                if len(line) == 0 or line[0] == "#":
                    continue
                spectrum.append(line)

            # Extract the header columns of the spectrum. This assumes the first
            # read line in the spectrum is the header. If no header is found, then
            # the columns are numbered instead

            header = []

            if spectrum[0][0] == "Freq." or spectrum[0][0] == "Lambda":
                for i, column_name in enumerate(spectrum[0]):
                    if column_name[0] == "A":
                        j = column_name.find("P")
                        column_name = column_name[1:j]
                    header.append(column_name)
                spectrum = np.array(spectrum[1:], dtype=np.float)
            else:
                header = np.arange(len(spectrum[0]))

            # Add the actual spectrum to the spectrum dictionary, the keys of the
            # dictionary are the column names as given above. Set the header and
            # also the inclination angles here as well

            for i, column_name in enumerate(header):
                self.all_spectrum[spec_type][column_name] = spectrum[:, i]

            inclinations = []

            for col in header:
                if col.isdigit() and col not in inclinations:
                    inclinations.append(col)

            self.all_columns[spec_type] = tuple(header)
            self.all_inclinations[spec_type] = tuple(inclinations)
            self.all_n_inclinations[spec_type] = len(inclinations)

        if n_read == 0:
            raise IOError(f"Unable to open any spectrum files in {self.cd}")

    def smooth(self,
               width: int = 5,
               to_smooth: Union[List[str], Tuple[str], str] = None):
        """Smooth the spectrum flux/luminosity bins.

        Parameters
        ----------
        width: int [optional]
            The width of the boxcar filter (in bins).
        to_smooth: list or tuple of strings [optional]
            A list or tuple"""

        # Create a backup of the unsmoothed array before it is smoothed it

        if self.unsmoothed is None:
            self.unsmoothed = copy.deepcopy(self.spectrum)

        # Get the input parameters for smoothing and make sure it's good input

        if type(width) is not int:
            try:
                width = int(width)
            except ValueError:
                print(f"Unable to cast {width} into an int")
                return

        if to_smooth is None:
            to_smooth = ("Created", "WCreated", "Emitted", "CenSrc", "Disk",
                         "Wind", "HitSurf", "Scattered") + tuple(
                             self.inclinations)
        elif type(to_smooth) is str:
            to_smooth = to_smooth,
        else:
            raise ValueError(
                f"unknown format for to_smooth, must be a tuple of strings or string"
            )

        # Loop over each available spectrum and smooth it

        for key in self.available:
            for thing_to_smooth in to_smooth:
                try:
                    self.spectrum[key][thing_to_smooth] = convolve(
                        self.spectrum[key][thing_to_smooth],
                        boxcar(width) / float(width),
                        mode="same")
                except KeyError:
                    continue

    def unsmooth(self):
        """Restore the spectrum to its unsmoothed form."""

        self.spectrum = copy.deepcopy(self.unsmoothed)

    def _plot_specific(self,
                       name: str,
                       label_lines: bool = False,
                       ax_update: plt.Axes = None):
        """Plot a specific column in a spectrum file.

        Parameters
        ----------
        label_lines: bool
            Plot line IDs.
        ax_update: plt.Axes
            An plt.Axes object to update, i.e. to plot on."""

        normalize_figure_style()

        if not ax_update:
            fig, ax = plt.subplots(figsize=(9, 5))
        else:
            ax = ax_update

        ax.set_yscale("log")
        ax.set_xscale("log")

        if self.units == UNITS_FLAMBDA:
            ax.plot(self.spectrum["Lambda"],
                    self.spectrum[name],
                    label=name)
            ax.set_xlabel(r"Wavelength [\AA]")
            ax.set_ylabel(
                r"Flux Density 100 pc [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]")
            if label_lines:
                ax = ax_add_line_ids(ax, common_lines(False), logx=True)
        else:
            ax.plot(self.spectrum["Freq."],
                    self.spectrum[name],
                    label=name)
            ax.set_xlabel("Frequency [Hz]")
            if self.units == UNITS_LNU:
                ax.set_ylabel(r"Luminosity 100 pc [erg s$^{-1}$ Hz$^{-1}$]")
            else:
                ax.set_ylabel(
                    r"Flux Density 100 pc [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
            if label_lines:
                ax = ax_add_line_ids(ax, common_lines(True), logx=True)

        if not ax_update:
            fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
            return fig, ax
        else:
            return ax

    def _spec_plot_all(self, label_lines: bool = False):
        """Plot the spectrum components and observer spectra on a 1x2 panel
        plot. The left panel has the components, whilst the right panel has
        the observer spectrum.

        Parameters
        ----------
        label_lines: bool
            Plot line IDs."""

        normalize_figure_style()

        fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey="row")

        for component in self.columns[:-self.n_inclinations]:
            if component in ["Lambda", "Freq."]:
                continue
            ax[0] = self._plot_specific(component, label_lines, ax[0])

        for line in ax[0].get_lines():
            line.set_alpha(0.7)
        ax[0].legend(ncol=2, loc="upper right").set_zorder(0)

        for inclination in self.inclinations:
            ax[1] = self._plot_specific(inclination, label_lines, ax[1])

        for label, line in zip(self.inclinations,
                               ax[1].get_lines()):
            line.set_alpha(0.7)
            line.set_label(str(label) + r"$^{\circ}$")
        ax[1].set_ylabel("")
        ax[1].legend(ncol=2, loc="upper right").set_zorder(0)

        ax[0].set_title("Components")
        ax[1].set_title("Observer spectra")

        fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
        fig.subplots_adjust(wspace=0)

        return fig, ax

    def plot(self, name: str = None, label_lines: bool = False):
        """Plot the spectra or a single component in a single figure. By default
        this creates a 1 x 2 of the components on the left and the observer
        spectra on the right. Useful for when in an interactive session.

        Parameters
        ----------
        name: str
            The name of the thing to plot.
        label_lines: bool
            Plot line IDs."""

        # todo:
        # This is some badness inspired by Python. This is done, for now, as
        # I haven't implemented a way to plot other spectra quickly this way

        ot = self.current
        self.current = "spec"

        if name:
            if name not in self.columns:
                print(f"{name} is not in the spectrum columns")
                return
            fig, ax = self._plot_specific(name, label_lines)
            if name.isdigit():
                name += r"$^{\circ}$"
            ax.set_title(name.replace("_", r"\_"))
        else:
            # todo: update with more functions to plot spec_tot w/o name etc
            if "spec" not in self.available and "log_spec" not in self.available:
                raise IOError(
                    f"Unable to plot without parameter 'name' as there is no {self.root}.spec file")
            fig, ax = self._spec_plot_all(label_lines)

        self.current = ot

        return fig, ax

    def show(self, block=True):
        """Show any plots which have been generated."""

        plt.show(block=block)

    def set(self, name):
        """Set a different spectrum to be the target."""

        if self.logspec and not name.startswith("log_"):
            name = "log_" + name

        if name not in self.available:
            raise ValueError(
                f"Spectrum {name} is not available: available {self.available}"
            )

        self.current = name
        self.spectrum = self.all_spectrum[self.current]
        self.columns = self.all_columns[self.current]
        self.inclinations = self.all_inclinations[self.current]
        self.n_inclinations = self.all_n_inclinations[self.current]
        self.units = self.all_units[self.current]

    def __getitem__(self, key):
        """Return an array in the spectrum dictionary when indexing."""

        if key not in self.available:
            return self.spectrum[self.current][key]
        else:
            return self.spectrum[key]

    def __setitem__(self, key, value):
        """Allows to modify the arrays in the spectrum dictionary."""

        if key not in self.available:
            self.spectrum[self.current][key] = value
        else:
            self.spectrum[key] = value

    def __str__(self):
        """Print the basic details about the spectrum."""

        msg = f"Spectrum for the model {self.root} in {self.cd}\n"
        msg += f"Available spectra: {self.available}\n"
        msg += f"Current spectrum {self.current}\n"
        if "spec" in self.available or "log_spec" in self.available:
            msg += f"Spectrum inclinations: {self.inclinations['spec']}\n"
        if "tau_spec" in self.available:
            msg += f"Optical depth inclinations {self.inclinations['tau_spec']}\n"

        return textwrap.dedent(msg)


# Utility functions ------------------------------------------------------------


def get_spectrum_files(this_root: str = None,
                       cd: str = ".",
                       ignore_delay_dump_spec: bool = True) -> List[str]:
    """Find root.spec files recursively in the provided directory.

    Parameters
    ----------
    this_root: str [optional]
        If root is set, then only .spec files with this root name will be
        returned
    cd: str [optional]
        The path to recursively search from
    ignore_delay_dump_spec: [optional] bool
        When True, root.delay_dump.spec files will be ignored

    Returns
    -------
    spec_files: List[str]
        The file paths of various .spec files"""

    spec_files = []

    for filepath in Path(cd).glob("**/*.spec"):
        str_filepath = str(filepath)
        if ignore_delay_dump_spec and str_filepath.find(
                ".delay_dump.spec") != -1:
            continue
        if this_root:
            root, cd = get_root_from_filepath(str_filepath)
            if root != this_root:
                continue
        spec_files.append(str_filepath)

    return spec_files


# Plotting functions -----------------------------------------------------------

MIN_SPEC_COMP_FLUX = 1e-15
DEFAULT_PYTHON_DISTANCE = 100 * PARSEC


def _plot_panel_subplot(ax: plt.Axes, x_values: np.ndarray, spectrum: Spectrum,
                        units: str, things_to_plot: Union[List[str], str],
                        xlims: Tuple[Union[float, int, None],
                                     Union[float, int, None]], sm: int,
                        alpha: float, scale: str, frequency_space: bool,
                        skip_sparse: bool) -> plt.Axes:
    """Create a subplot panel for a figure given the spectrum components names
    in the list dname.

    Parameters
    ----------
    ax: plt.Axes
        The plt.Axes object for the subplot
    x_values: np.array[float]
        The x-axis data, i.e. wavelength or frequency
    spectrum: pd.DataFrame
        The spectrum data file as a pandas DataFrame
    units: str
        The units of the spectrum
    things_to_plot: list[str]
        The name of the spectrum components to add to the subplot panel
    xlims: Tuple[float, float]
        The lower and upper x-axis boundaries (xlower, xupper)
    sm: int
        The size of the boxcar filter to smooth the spectrum components
    alpha: float
        The alpha value of the spectrum to be plotted.
    scale: bool
        Set the scale for the plot axes
    frequency_space: bool
        Create the figure in frequency space instead of wavelength space
    skip_sparse: bool
        If True, then sparse spectra will not be plotted
    n: str
        The name of the calling function

    Returns
    -------
    ax: pyplot.Axes
        The pyplot.Axes object for the subplot"""

    if type(things_to_plot) == str:
        things_to_plot = [things_to_plot]

    n_skip = 0

    for thing in things_to_plot:
        try:
            fl = smooth_array(spectrum[thing], sm)
        except KeyError:
            print("unable to find data column with label {}".format(thing))
            continue

        # Skip sparse spec components to make prettier plot

        if skip_sparse and len(fl[fl < MIN_SPEC_COMP_FLUX]) > 0.7 * len(fl):
            n_skip += 1
            continue

        # If plotting in frequency space, of if the units then the flux needs
        # to be converted in nu F nu

        if frequency_space and units == UNITS_FLAMBDA:
            fl *= spectrum["Lambda"]
        elif frequency_space and units == UNITS_FNU:
            fl *= spectrum["Freq."]

        # If the spectrum units are Lnu then plot nu Lnu

        if units == UNITS_LNU:
            fl *= spectrum["Freq."]

        ax.plot(x_values, fl, label=thing, alpha=alpha)

        if scale == "logx" or scale == "loglog":
            ax.set_xscale("log")
        if scale == "logy" or scale == "loglog":
            ax.set_yscale("log")

    if n_skip == len(things_to_plot):
        return ax

    ax.set_xlim(xlims[0], xlims[1])
    if frequency_space:
        ax.set_xlabel(r"Frequency (Hz)")
        if units == UNITS_LNU:
            ax.set_ylabel(r"$\nu L_{\nu}$ (erg s$^{-1}$ Hz$^{-1})$")
        else:
            ax.set_ylabel(r"$\nu F_{\nu}$ (erg s$^{-1}$ cm$^{-2})$")
    else:
        ax.set_xlabel(r"Wavelength ($\AA$)")
        ax.set_ylabel(r"$F_{\lambda}$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)")

    ax.legend(loc="lower left")

    return ax


def plot(x: np.ndarray,
         y: np.ndarray,
         xmin: float = None,
         xmax: float = None,
         xlabel: str = None,
         ylabel: str = None,
         scale: str = "logy",
         fig: plt.Figure = None,
         ax: plt.Axes = None,
         label: str = None,
         alpha: float = 1.0,
         display: bool = False) -> Tuple[plt.Figure, plt.Axes]:
    """This is a simple plotting function designed to give you the bare minimum.
    It will create a figure and axes object for a single panel and that is
    it. It is mostly designed for quick plotting of models and real data.

    Parameters
    ----------
    x: np.ndarray
        The wavelength or x-axis data to plot.
    y: np.ndarray
        The flux or y-axis data to plot.
    xmin: float [optional]
        The smallest number to display on the x-axis
    xmax: float [optional]
        The largest number to display on the x-axis
    xlabel: str [optional]
        The data label for the x-axis.
    ylabel: str [optional]
        The data label for the y-axis.
    scale: str [optional]
        The scale of the axes for the plot.
    fig: plt.Figure [optional]
        A matplotlib Figure object of which to use to create the plot.
    ax: plt.Axes [optional]
        A matplotlib Axes object of which to use to create the plot.
    label: str [optional]
        A label for the data being plotted.
    alpha: float [optional]
        The alpha value for the data to be plotted.
    display: bool [optional]
        If set to True, then the plot will be displayed.

    Returns
    -------
    fig: plt.Figure
        The figure object for the plot.
    ax: plt.Axes
        The axes object containing the plot."""

    # It doesn't make sense to provide only fig and not ax, or ax and not fig
    # so at this point we will throw an error message and return

    normalize_figure_style()

    if fig and not ax:
        raise InvalidParameter(
            "fig has been provided, but ax has not. Both are required.")
    if not fig and ax:
        raise InvalidParameter(
            "fig has not been provided, but ax has. Both are required.")
    if not fig and not ax:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    if label is None:
        label = ""

    ax.plot(x, y, label=label, alpha=alpha)

    # Set the scales of the aes

    if scale == "loglog" or scale == "logx":
        ax.set_xscale("log")
    if scale == "loglog" or scale == "logy":
        ax.set_yscale("log")

    # If axis labels are provided, then set them

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Set the x and y axis limits. For the y axis, we use a function to try and
    # figure out appropriate values for the axis limits to display the data
    # sensibly

    xlims = [x.min(), x.max()]
    if not xmin:
        xmin = xlims[0]
    if not xmax:
        xmax = xlims[1]
    xlims = (xmin, xmax)
    ax.set_xlim(xlims[0], xlims[1])

    ymin, ymax = get_y_lims_for_x_lims(x, y, xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_optical_depth(root: str,
                       wd: str,
                       inclinations: List[str] = "all",
                       xmin: float = None,
                       xmax: float = None,
                       scale: str = "loglog",
                       show_absorption_edge_labels: bool = True,
                       frequency_space: bool = True,
                       display: bool = False) -> Tuple[plt.Figure, plt.Axes]:
    """Create an optical depth spectrum for a given Python simulation. This figure
    can be created in both wavelength or frequency space and with various
    choices of axes scaling.

    This function will return the Figure and Axes object used to create the
    plot.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    wd: str
        The absolute or relative path containing the Python simulation
    inclinations: List[str] [optional]
        A list of inclination angles to plot
    xmin: float [optional]
        The lower x boundary for the figure
    xmax: float [optional]
        The upper x boundary for the figure
    scale: str [optional]
        The scale of the axes for the plot.
    show_absorption_edge_labels: bool [optional]
        Label common absorption edges of interest onto the figure
    frequency_space: bool [optional]
        Create the figure in frequency space instead of wavelength space
    display: bool [optional]
        Display the final plot if True.

    Returns
    -------
    fig: pyplot.Figure
        The pyplot.Figure object for the created figure
    ax: pyplot.Axes
        The pyplot.Axes object for the created figure"""

    normalize_figure_style()

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    if type(inclinations) == str:
        inclinations = [inclinations]
    spectrum = Spectrum(root, wd, "spec_tau")
    spec_angles = spectrum.inclinations
    n_angles = len(spec_angles)
    n_plots = len(
        inclinations)  # Really have no clue what this does in hindsight...

    # Ignore all if other inclinations are passed - assume it was a mistake to pass all

    if inclinations[0] == "all" and len(inclinations) > 1:
        inclinations = inclinations[1:]
        n_plots = len(inclinations)
    if inclinations[0] == "all":
        inclinations = spec_angles
        n_plots = n_angles

    # Set wavelength or frequency boundaries

    xlabel = "Lambda"
    if frequency_space:
        xlabel = "Freq."

    x = spectrum[xlabel]
    if not xmin:
        xmin = np.min(spectrum[xlabel])
    if not xmax:
        xmax = np.max(spectrum[xlabel])

    # This loop will plot the inclinations provided by the user

    for i in range(n_plots):
        if inclinations[0] != "all" and inclinations[
                i] not in spec_angles:  # Skip inclinations which don't exist
            continue
        ii = str(inclinations[i])
        label = ii + r"$^{\circ}$"
        n_non_zero = np.count_nonzero(spectrum[ii])

        # Skip inclinations which look through vacuum

        if n_non_zero == 0:
            continue

        ax.plot(x, spectrum[ii], linewidth=2, label=label)
        if scale == "logx" or scale == "loglog":
            ax.set_xscale("log")
        if scale == "logy" or scale == "loglog":
            ax.set_yscale("log")

    ax.set_ylabel(r"Optical Depth, $\tau$")
    if frequency_space:
        ax.set_xlabel(r"Frequency, [Hz]")
    else:
        ax.set_xlabel(r"Wavelength, [$\AA$]")
    ax.set_xlim(xmin, xmax)
    ax.legend(loc="lower left")

    if show_absorption_edge_labels:
        if scale == "loglog" or scale == "logx":
            logx = True
        else:
            logx = False
        ax_add_line_ids(ax, photoionization_edges(frequency_space), logx=logx)

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_spectrum_physics_process_contributions(
        contribution_spectra: dict,
        inclination: str,
        root: str,
        wd: str = ".",
        xmin: float = None,
        xmax: float = None,
        ymin: float = None,
        ymax: float = None,
        scale: str = "logy",
        line_labels: bool = True,
        sm: int = 5,
        lw: int = 2,
        alpha: float = 0.75,
        file_ext: str = "png",
        display: bool = False) -> Tuple[plt.Figure, plt.Axes]:
    """Description of the function.
    todo: some of these things really need re-naming..... it seems very confusing

    Parameters
    ----------

    Returns
    -------
    fig: plt.Figure
        The plt.Figure object for the created figure
    ax: plt.Axes
        The plt.Axes object for the created figure"""

    normalize_figure_style()

    fig, ax = plt.subplots(figsize=(12, 8))

    for name, spectrum in contribution_spectra.items():
        ax.plot(spectrum["Lambda"],
                smooth_array(spectrum[inclination], sm),
                label=name,
                linewidth=lw,
                alpha=alpha)

    if scale == "logx" or scale == "loglog":
        ax.set_xscale("log")
    if scale == "logy" or scale == "loglog":
        ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(loc="upper center", ncol=len(contribution_spectra))
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel(r"Flux F$_{\lambda}$ [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")

    if line_labels:
        if scale == "logx" or scale == "loglog":
            logx = True
        else:
            logx = False
        ax = ax_add_line_ids(ax, common_lines(), logx=logx)

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig("{}/{}_spec_processes.{}".format(wd, root, file_ext), dpi=300)
    if file_ext != "png":
        fig.savefig("{}/{}_spec_processes.png".format(wd, root), dpi=300)

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_spectrum_components(
        root: str,
        wd: str,
        spec_tot: bool = False,
        wind_tot: bool = False,
        xmin: float = None,
        xmax: float = None,
        smooth_amount: int = 5,
        scale: str = "loglog",
        alpha: float = 0.6,
        frequency_space: bool = False,
        display: bool = False) -> Tuple[plt.Figure, plt.Axes]:
    """Create a figure of the different spectrum components of a Python spectrum
    file. Note that all of the spectrum components added together DO NOT have
    to equal the output spectrum or the emitted spectrum (don't ask).

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    wd: str
        The absolute or relative path containing the Python simulation
    spec_tot: bool [optional]
        If True, the root.log_spec_tot file will be plotted
    wind_tot: bool [optional]
        If True, the root.log_wind_tot file will be plotted
    xmin: float [optional]
        The lower x boundary for the figure
    xmax: float [optional]
        The upper x boundary for the figure
    smooth_amount: int [optional]
        The size of the boxcar filter to smooth the spectrum components
    scale: bool [optional]
        The scale to use for the axes. Allowed values are linlin, logx, logy and
        loglog.
    alpha: float [optional]
        The alpha value used for plotting the spectra.
    frequency_space: bool [optional]
        Create the figure in frequency space instead of wavelength space
    display: bool [optional]
        Display the final plot if True.

    Returns
    -------
    fig: pyplot.Figure
        The pyplot.Figure object for the created figure
    ax: pyplot.Axes
        The pyplot.Axes object for the created figure"""

    normalize_figure_style()

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    # Determine the type of spectrum to read in

    if spec_tot:
        scale = "loglog"
        frequency_space = True
        logspec = True
        spectype = "spec_tot"
    elif wind_tot:
        scale = "loglog"
        frequency_space = True
        logspec = True
        spectype = "spec_tot_wind"
    else:
        spectype = None
        logspec = False

    spectrum = Spectrum(root, wd, spectype, logspec)
    if frequency_space:
        x = spectrum["Freq."]
    else:
        x = spectrum["Lambda"]
    xlims = (None, None)

    ax[0] = _plot_panel_subplot(ax[0], x, spectrum, spectrum.units,
                                ["Created", "WCreated", "Emitted"], xlims,
                                smooth_amount, alpha, scale, frequency_space,
                                True)
    ax[1] = _plot_panel_subplot(ax[1], x, spectrum, spectrum.units,
                                ["CenSrc", "Disk", "Wind", "HitSurf"], xlims,
                                smooth_amount, alpha, scale, frequency_space,
                                True)

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_spectrum_inclinations_in_subpanels(
        root: str,
        wd: str,
        xmin: float = None,
        xmax: float = None,
        smooth_amount: int = 5,
        add_line_ids: bool = True,
        frequency_space: bool = False,
        scale: str = "logy",
        figsize: Tuple[float, float] = None,
        display: bool = False) -> Tuple[plt.Figure, plt.Axes]:
    """Creates a figure which plots all of the different inclination angles in
    different panels.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    wd: str
        The absolute or relative path containing the Python simulation
    xmin: float [optional]
        The lower x boundary for the figure
    xmax: float [optional]
        The upper x boundary for the figure
    smooth_amount: int [optional]
        The size of the boxcar filter to smooth the spectrum components.
    add_line_ids: bool [optional]
        Plot labels for common line transitions.
    frequency_space: bool [optional]
        Create the figure in frequency space instead of wavelength space
    scale: bool [optional]
        Set the scales for the axes in the plot
    figsize: Tuple[float, float] [optional]
        The size of the Figure in matplotlib units (inches?)
    display: bool [optional]
        Display the final plot if True.

    Returns
    -------
    fig: pyplot.Figure
        The pyplot.Figure object for the created figure
    ax: pyplot.Axes
        The pyplot.Axes object for the created figure"""

    spectrum = Spectrum(root, wd)
    spectrum_units = spectrum.units
    spectrum_inclinations = spectrum.inclinations
    n_inclinations = spectrum.n_inclinations
    plot_dimensions = subplot_dims(n_inclinations)

    if figsize:
        size = figsize
    else:
        size = (12, 10)

    normalize_figure_style()

    fig, ax = plt.subplots(plot_dimensions[0],
                           plot_dimensions[1],
                           figsize=size,
                           squeeze=False)
    fig, ax = remove_extra_axes(fig, ax, n_inclinations,
                                plot_dimensions[0] * plot_dimensions[1])

    # Use either frequency or wavelength and set the plot limits respectively

    if frequency_space:
        x = spectrum["Freq."]
    else:
        x = spectrum["Lambda"]
    xlims = [x.min(), x.max()]
    if not xmin:
        xmin = xlims[0]
    if not xmax:
        xmax = xlims[1]
    xlims = (xmin, xmax)

    inclination_index = 0
    for i in range(plot_dimensions[0]):
        for j in range(plot_dimensions[1]):
            if inclination_index > n_inclinations - 1:
                break
            name = str(spectrum_inclinations[inclination_index])
            ax[i,
               j] = _plot_panel_subplot(ax[i, j], x, spectrum, spectrum_units,
                                        name, xlims, smooth_amount, 1, scale,
                                        frequency_space, False)
            ymin, ymax = get_y_lims_for_x_lims(x, spectrum[name], xmin, xmax)
            ax[i, j].set_ylim(ymin, ymax)

            if add_line_ids:
                if scale == "loglog" or scale == "logx":
                    logx = True
                else:
                    logx = False
                ax[i, j] = ax_add_line_ids(ax[i, j],
                                           common_lines(frequency_space),
                                           logx=logx)
            inclination_index += 1

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_single_spectrum_inclination(
        root: str,
        wd: str,
        inclination: Union[str, float, int],
        xmin: float = None,
        xmax: float = None,
        smooth_amount: int = 5,
        scale: str = "logy",
        frequency_space: bool = False,
        display: bool = False) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
    """Create a plot of an individual spectrum for the provided inclination
    angle.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    wd: str
        The absolute or relative path containing the Python simulation
    inclination: str, float, int
        The specific inclination angle to plot for
    xmin: float [optional]
        The lower x boundary for the figure
    xmax: float [optional]
        The upper x boundary for the figure
    smooth_amount: int [optional]
        The size of the boxcar filter to smooth the spectrum components
    scale: str [optional]
        The scale of the axes for the plot.
    frequency_space: bool [optional]
        Create the figure in frequency space instead of wavelength space
    display: bool [optional]
        Display the final plot if True.

    Returns
    -------
    fig: pyplot.Figure
        The pyplot.Figure object for the created figure
    ax: pyplot.Axes
        The pyplot.Axes object for the created figure"""

    normalize_figure_style()

    s = Spectrum(root, wd, smooth=smooth_amount)

    if frequency_space:
        x = s["Freq."]
    else:
        x = s["Lambda"]
    xlims = [x.min(), x.max()]
    if not xmin:
        xmin = xlims[0]
    if not xmax:
        xmax = xlims[1]
    xlims = (xmin, xmax)

    if type(inclination) != str:
        try:
            inclination = str(inclination)
        except ValueError:
            print("unable to convert into string")
            return
    y = s[inclination]
    if frequency_space:
        xax = r"Frequency [Hz]"
        yax = r"$\nu F_{\nu}$ (erg s$^{-1}$ cm$^{-2}$)"
        y *= s["Lambda"]
    else:
        xax = r"Wavelength [$\AA$]"
        yax = r"$F_{\lambda}$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)"

    fig, ax = plot(x, y, xlims[0], xlims[1], xax, yax, scale)

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_multiple_model_spectra(
        output_name: str,
        spectra_filepaths: list,
        inclination_angle: str,
        wd: str = ".",
        x_min: float = None,
        x_max: float = None,
        frequency_space: bool = False,
        axes_scales: str = "logy",
        smooth_amount: int = 5,
        plot_common_lines: bool = False,
        file_ext: str = "png",
        display: bool = False) -> Tuple[plt.Figure, plt.Axes]:
    """Plot multiple spectra, from multiple models, given in the list of spectra
    provided.
    todo: when using "all", create separate plot for each inclination

    Parameters
    ----------
    output_name: str
        The name to use for the created plot.
    spectra_filepaths: List[str]
        A list of spectrum file paths.
    inclination_angle: str
        The inclination angle(s) to plot
    wd: [optional] str
        The working directory containing the Python simulation
    x_min: [optional] float
        The smallest value on the x axis.
    x_max: [optional] float
        The largest value on the x axis.
    frequency_space: [optional] bool
        Create the plot in frequency space and use nu F_nu instead.
    axes_scales: [optional] str
        The scaling of the x and y axis. Allowed logx, logy, linlin, loglog
    smooth_amount: [optional] int
        The amount of smoothing to use.
    plot_common_lines: [optional] bool
        Add line labels to the figure.
    file_ext: [optional] str
        The file extension of the output plot.
    display: [optional] bool
        Show the plot when finished

    Returns
    -------
    fig: plt.Figure
        Figure object.
    ax: plt.Axes
        Axes object."""

    normalize_figure_style()

    spectrum_objects = []
    for spectrum in spectra_filepaths:
        root, cd = get_root_from_filepath(spectrum)
        spectrum_objects.append(Spectrum(root, cd, smooth=smooth_amount))

    if inclination_angle == "all":
        inclinations = []
        for spectrum in spectrum_objects:
            inclinations += spectrum.inclinations
        inclinations = sorted(list(dict.fromkeys(inclinations)))
        figsize = (12, 12)
    else:
        inclinations = [inclination_angle]
        figsize = (12, 5)

    n_inclinations = len(inclinations)
    n_rows, n_cols = subplot_dims(n_inclinations)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig, ax = remove_extra_axes(fig, ax, n_inclinations, n_rows * n_cols)
    ax = ax.flatten()  # for safety...

    y_min = +1e99
    y_max = -1e99

    for i, inclination in enumerate(inclinations):

        for spectrum in spectrum_objects:

            # Ignore spectra which are from continuum only models...

            if spectrum.filepath.find("continuum") != -1:
                continue

            if frequency_space:
                x = spectrum["Freq."]
            else:
                x = spectrum["Lambda"]
            try:
                if frequency_space:
                    y = spectrum["Lambda"] * spectrum[inclination]
                else:
                    y = spectrum[inclination]
            except KeyError:
                continue

            ax[i].plot(x,
                       y,
                       label=spectrum.filepath.replace("_", r"\_"),
                       alpha=0.75)

            # Calculate the y-axis limits to keep all spectra within the
            # plot area

            if not x_min:
                x_min = x.min()
            if not x_max:
                x_max = x.max()
            this_y_min, this_y_max = get_y_lims_for_x_lims(x, y, x_min, x_max)
            if this_y_min < y_min:
                y_min = this_y_min
            if this_y_max > y_max:
                y_max = this_y_max

        if y_min == +1e99:
            y_min = None
        if y_max == -1e99:
            y_max = None

        ax[i].set_title(f"{inclinations[i]}" + r"$^{\circ}$")

        x_lims = list(ax[i].get_xlim())
        if not x_min:
            x_min = x_lims[0]
        if not x_max:
            x_max = x_lims[1]
        ax[i].set_xlim(x_min, x_max)
        ax[i].set_ylim(y_min, y_max)

        if axes_scales == "loglog" or axes_scales == "logx":
            ax[i].set_xscale("log")
        if axes_scales == "loglog" or axes_scales == "logy":
            ax[i].set_yscale("log")

        if frequency_space:
            ax[i].set_xlabel(r"Frequency [Hz]")
            ax[i].set_ylabel(r"$\nu F_{\nu}$ (erg s$^{-1}$ cm$^{-2}$")
        else:
            ax[i].set_xlabel(r"Wavelength [$\AA$]")
            ax[i].set_ylabel(
                r"$F_{\lambda}$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)")

        if plot_common_lines:
            if axes_scales == "logx" or axes_scales == "loglog":
                logx = True
            else:
                logx = False
            ax[i] = ax_add_line_ids(ax[i], common_lines(), logx=logx)

    ax[0].legend(loc="upper left")
    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    if inclination_angle != "all":
        name = "{}/{}_i{}".format(wd, output_name, inclination_angle)
    else:
        name = "{}/{}".format(wd, output_name)

    fig.savefig("{}.{}".format(name, file_ext))
    if file_ext == "pdf" or file_ext == "eps":
        fig.savefig("{}.png".format(name))

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax
