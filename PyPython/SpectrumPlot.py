#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains functions which can be used to plot the various different
output files from Python. The general design of the functions is to return
figure and axes objects, just in case anything else wants to be changed before
being saved to disk or displayed.
"""

from .Constants import PARSEC
from .SpectrumUtils import photo_edges_list, common_lines_list, plot_line_ids, smooth
from .SpectrumUtils import read_spec_file, get_spec_inclinations, get_ylims, get_spec_units
from .SpectrumUtils import UNITS_FLAMBDA, UNITS_FNU, UNITS_LNU
from .PythonUtils import subplot_dims
from .Error import InvalidParameter

import pandas as pd
import numpy as np
from typing import List, Tuple, Union
from matplotlib import pyplot as plt


plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.labelsize'] = 15


MIN_SPEC_COMP_FLUX = 1e-15
DEFAULT_PYTHON_DISTANCE = 100 * PARSEC


def __panel_subplot(
    ax: plt.Axes, x: np.ndarray, spec: pd.DataFrame, units: str, dname: Union[List[str], str], xlims: Tuple[float, float],
    smooth_amount: int, scale: str, frequency_space: bool, skip_sparse: bool, n: str
) -> plt.Axes:
    """
    Create a subplot panel for a figure given the spectrum components names
    in the list dname.

    Parameters
    ----------
    ax: pyplot.Axes
        The pyplot.Axes object for the subplot
    x: np.array[float]
        The x-axis data, i.e. wavelength or frequency
    spec: pd.DataFrame
        The spectrum data file as a pandas DataFrame
    units: str
        The units of the spectrum
    dname: list[str]
        The name of the spectrum components to add to the subplot panel
    xlims: Tuple[float, float]
        The lower and upper x-axis boundaries (xlower, xupper)
    smooth: int
        The size of the boxcar filter to smooth the spectrum components
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
        The pyplot.Axes object for the subplot
    """

    if type(dname) == str:
        dname = [dname]

    for i in range(len(dname)):

        try:
            fl = smooth(spec[dname[i]].values, smooth_amount)
        except KeyError:
            print("{}: unable to find data column with label {}".format(n, dname[i]))
            continue

        # Skip sparse spec components to make prettier plot
        if skip_sparse and len(fl[fl < MIN_SPEC_COMP_FLUX]) > 0.7 * len(fl):
            continue

        # If plotting in frequency space, of if the units then the flux needs
        # to be converted in nu F nu
        if frequency_space and units == UNITS_FLAMBDA:
            fl *= spec["Lambda"].values
        elif frequency_space and units == UNITS_FNU:
            fl *= spec["Freq."].values

        # If the spectrum units are Lnu then plot nu Lnu
        if units == UNITS_LNU:
            fl *= spec["Freq."].values

        ax.plot(x, fl, label=dname[i])

        if scale == "logx" or scale == "loglog":
            ax.set_xscale("log")
        if scale == "logy" or scale == "loglog":
            ax.set_yscale("log")

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

    ax.legend()

    return ax


def plot_simple(
    x: np.ndarray, y: np.ndarray, xmin: float = None, xmax: float = None, xlabel: str = None, ylabel: str = None,
    scale: str = "logy", fig: plt.Figure = None, ax: plt.Axes = None, display: bool = False, label: str = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    This is a simple plotting function designed to give you the bare minimum.
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
    display: bool [optional]
        If set to True, then the plot will be displayed.

    Returns
    -------
    fig: plt.Figure
        The figure object for the plot.
    ax: plt.Axes
        The axes object containing the plot.
    """

    n = plot_simple.__name__

    nrows = ncols = 1

    # It doesn't make sense to provide only fig and not ax, or ax and not fig
    # so at this point we will throw an error message and return

    if fig and not ax:
        print("{}: fig has been provided, but ax has not. Both are required.".format(n))
        raise InvalidParameter()

    if not fig and ax:
        print("{}: fig has not been provided, but ax has. Both are required.".format(n))
        raise InvalidParameter()

    if not fig and not ax:
        fig, ax = plt.subplots(nrows, ncols, figsize=(12, 5))

    if label is None:
        label = ""

    ax.plot(x, y, label=label)

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

    ymin, ymax = get_ylims(x, y, xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_optical_depth(
    root: str, wd: str, inclinations: List[str] = "all", xmin: float = None, xmax: float = None, scale: str = "loglog",
    show_absorption_edge_labels: bool = True, frequency_space: bool = True, axes_label_fontsize: float = 15,
    display: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create an optical depth spectrum for a given Python simulation. This figure
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
    axes_label_fontsize: float [optional]
        The fontsize for labels on the plot

    Returns
    -------
    fig: pyplot.Figure
        The pyplot.Figure object for the created figure
    ax: pyplot.Axes
        The pyplot.Axes object for the created figure
    """

    n = plot_optical_depth.__name__
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    fname = "{}/diag_{}/{}.tau_spec.diag".format(wd, root, root)

    if type(inclinations) == str:
        inclinations = [inclinations]

    try:
        s = read_spec_file(fname)
    except IOError:
        print("{}: unable to find the optical depth spectrum {}".format(n, fname))
        return fig, ax

    xlabel = "Lambda"
    if frequency_space:
        xlabel = "Freq."

    # Set wavelength or frequency boundaries
    x = s[xlabel].values
    if not xmin:
        xmin = np.min(s[xlabel])
    if not xmax:
        xmax = np.max(s[xlabel])

    spec_angles = get_spec_inclinations(s)
    nangles = len(spec_angles)

    # Determine the number of inclinations requested in a convoluted way :^)
    nplots = len(inclinations)

    # Ignore all if other inclinations are passed - assume it was a mistake to pass all
    if inclinations[0] == "all" and len(inclinations) > 1:
        inclinations = inclinations[1:]
        nplots = len(inclinations)
    if inclinations[0] == "all":
        inclinations = spec_angles
        nplots = nangles

    # This loop will plot the inclinations provided by the user
    for i in range(nplots):
        if inclinations[0] != "all" and inclinations[i] not in spec_angles:  # Skip inclinations which don't exist
            continue
        ii = str(inclinations[i])

        label = r"$i$ = " + ii + r"$^{\circ}$"
        n_non_zero = np.count_nonzero(s[ii])
        # Skip inclinations which look through vacuum
        if n_non_zero == 0:
            continue

        ax.plot(x, s[ii], linewidth=2, label=label)

        if scale == "logx" or scale == "loglog":
            ax.set_xscale("log")
        if scale == "logy" or scale == "loglog":
            ax.set_yscale("log")

    ax.set_ylabel(r"Optical Depth, $\tau$", fontsize=axes_label_fontsize)
    if frequency_space:
        ax.set_xlabel(r"Frequency, [Hz]", fontsize=axes_label_fontsize)
    else:
        ax.set_xlabel(r"Wavelength, [$\AA$]", fontsize=axes_label_fontsize)

    ax.set_xlim(xmin, xmax)
    ax.legend()

    if show_absorption_edge_labels:
        if scale == "loglog" or scale == "logx":
            logx = True
        else:
            logx = False
        plot_line_ids(ax, photo_edges_list(frequency_space), logx, fontsize=15)

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_spectrum_components(
    root: str, wd: str, spec_tot: bool = False, wind_tot: bool = False, xmin: float = None, xmax: float = None,
    smooth_amount: int = 5, scale: str = "loglog", frequency_space: bool = False, display: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a figure of the different spectrum components of a Python spectrum
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
    logy: bool [optional]
        Use a log scale for the y axis of the figure
    frequency_space: bool [optional]
        Create the figure in frequency space instead of wavelength space

    Returns
    -------
    fig: pyplot.Figure
        The pyplot.Figure object for the created figure
    ax: pyplot.Axes
        The pyplot.Axes object for the created figure
    """

    n = plot_spectrum_components.__name__

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    if spec_tot:
        scale = "loglog"
        frequency_space = True
        extension = "log_spec_tot"
    elif wind_tot:
        scale = "loglog"
        frequency_space = True
        extension = "log_spec_tot_wind"
    else:
        extension = "spec"

    fname = "{}/{}.{}".format(wd, root, extension)

    try:
        s = read_spec_file(fname)
    except IOError:
        print("{}: unable to open .spec file with name {}".format(n, fname))
        return fig, ax

    if frequency_space:
        x = s["Freq."].values
    else:
        x = s["Lambda"].values

    xlims = [x.min(), x.max()]
    if not xmin:
        xmin = xlims[0]
    if not xmax:
        xmax = xlims[1]
    xlims = (xmin, xmax)

    ax[0] = __panel_subplot(
        ax[0], x, s, get_spec_units(fname), ["Created", "Emitted"], xlims, smooth_amount, scale, frequency_space, True,
        n
    )
    ax[1] = __panel_subplot(
        ax[1], x, s, get_spec_units(fname), ["CenSrc", "Disk", "Wind", "HitSurf", "Scattered"], xlims, smooth_amount,
        scale, frequency_space, True, n
    )

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_all_spectra(
    root: str, wd: str, xmin: float = None, xmax: float = None, smooth_amount: int = 5, add_line_ids: bool = True,
    frequency_space: bool = False, scale: str = "logy", figsize: Tuple[float, float] = None, display: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Creates a figure which plots all of the different inclination angles in
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
    axes_label_fontsize: float [optional]
        The fontsize for labels on the plot
    figsize: Tuple[float, float] [optional]
        The size of the Figure in matplotlib units (inches?)

    Returns
    -------
    fig: pyplot.Figure
        The pyplot.Figure object for the created figure
    ax: pyplot.Axes
        The pyplot.Axes object for the created figure
        :param add_line_ids:
    """

    n = plot_all_spectra.__name__

    fname = "{}/{}.spec".format(wd, root)
    try:
        s = read_spec_file(fname)
    except IOError:
        print("{}: unable to open .spec file with name {}".format(n, fname))
        return

    units = get_spec_units(fname)
    inclinations = get_spec_inclinations(s)
    panel_dims = subplot_dims(len(inclinations))
    size = (12, 10)
    if figsize:
        size = figsize
    fig, ax = plt.subplots(panel_dims[0], panel_dims[1], figsize=size, squeeze=False)
    # ax = ax.flatten()

    # Use either frequency or wavelength and set the plot limits respectively
    if frequency_space:
        x = s["Freq."].values
    else:
        x = s["Lambda"].values

    xlims = [x.min(), x.max()]
    if not xmin:
        xmin = xlims[0]
    if not xmax:
        xmax = xlims[1]
    xlims = (xmin, xmax)

    ii = 0
    for i in range(panel_dims[0]):
        for j in range(panel_dims[1]):
            if ii > len(inclinations) - 1:
                break
            name = str(inclinations[ii])
            ax[i, j] = __panel_subplot(
                ax[i, j], x, s, units, name, xlims, smooth_amount, scale, frequency_space, False, n)
            ymin, ymax = get_ylims(x, s[name].values, xmin, xmax)
            ax[i, j].set_ylim(ymin, ymax)

            if add_line_ids:
                if scale == "loglog" or scale == "logx":
                    logx = True
                else:
                    logx = False
                ax[i, j] = plot_line_ids(ax[i, j], common_lines_list(frequency_space), logx)
            ii += 1

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_spectrum(
    root: str, wd: str, inclination: Union[str, float, int], xmin: float = None, xmax: float = None,
    smooth_amount: int = 5, scale: str = "logy", frequency_space: bool = False, display: bool = False
) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
    """
    Create a plot of an individual spectrum for the provided inclination angle.

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

    Returns
    -------
    fig: pyplot.Figure
        The pyplot.Figure object for the created figure
    ax: pyplot.Axes
        The pyplot.Axes object for the created figure
    """

    n = plot_spectrum.__name__

    fname = "{}/{}.spec".format(wd, root)
    try:
        s = read_spec_file(fname)
    except IOError:
        print("{}: unable to open .spec file with name {}".format(n, fname))
        return

    if frequency_space:
        x = s["Freq."].values
    else:
        x = s["Lambda"].values

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
            print("{}: unable to convert into string".format(n, inclination))
            return

    y = s[inclination].values

    if frequency_space:
        xax = r"Frequency [Hz]"
        yax = r"$\nu F_{\nu}$ (erg s$^{-1}$ cm$^{-2}$)"
        # Convert into lambda F_lambda which is (I hope) the same as nu F_nu
        y *= s["Lambda"].values
    else:
        xax = r"Wavelength [$\AA$]"
        yax = r"$F_{\lambda}$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)"

    fig, ax = plot_simple(x, smooth(y, smooth_amount), xlims[0], xlims[1], xax, yax, scale)

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax
