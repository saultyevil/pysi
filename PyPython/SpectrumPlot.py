#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

from .Error import InvalidFileContents
from .Constants import C, ANGSTROM, PARSEC
from .SpectrumUtils import absorption_edges, plot_line_ids, smooth_spectrum, read_spec

import pandas as pd
import numpy as np
from typing import List, Tuple
from matplotlib import pyplot as plt


MIN_SPEC_COMP_FLUX = 1e-17
DEFAULT_PYTHON_DISTANCE = 100 * PARSEC


def spec_plot_tau_spec(root: str, wd: str, inclination: List[str] = "all", wmin: float = None, wmax: float = None,
                       logy: bool = True, loglog: bool = False, show_absorption_edge_labels: bool = True,
                       frequency_space: bool = False, axes_label_fontsize: float = 15) -> Tuple[plt.Figure, plt.Axes]:
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
    inclination: List[str] [optional]
        A list of inclination angles to plot
    wmin: float [optional]
        The lower wavelength boundary for the figure
    wmax: float [optional]
        The upper wavelength boundary for the figure
    logy: bool [optional]
        Use a log scale for the y axis of the figure
    loglog: bool [optional]
        Use a log scale for both the x and y axes of the figure
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

    n = spec_plot_tau_spec.__name__
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fname = "{}/diag_{}/{}.tau_spec.diag".format(wd, root, root)

    if type(inclination) == str:
        inclination = [inclination]

    try:
        tspec = np.loadtxt(fname, dtype=float)
    except IOError:
        print("{}: unable to find the optical depth spectrum {}".format(n, fname))
        return fig, ax

    # Set wavelength or frequency boundaries
    if not wmin:
        wmin = np.min(tspec[:, 0])
    if not wmax:
        wmax = np.max(tspec[:, 0])
    
    if frequency_space:
        tspec[:, 0] = C / (tspec[:, 0] * ANGSTROM)
        wl_wmax = wmax
        wl_wmin = wmin
        wmin = C / (wl_wmax * ANGSTROM)
        wmax = C / (wl_wmin * ANGSTROM)

    # Determine the inclinations which are available
    with open(fname, "r") as f:
        angles = f.readline().split()
    if angles[0] == "#":
        angles = angles[3:]
    else:
        raise InvalidFileContents("{}: provided file is possibly not an optical depth spectrum".format(n))
    nangles = len(angles)
    for i in range(nangles):
        angles[i] = angles[i][1:3]

    # Determine the number of inclinations requested in a convoluted way :^)
    ninclinations = len(inclination)
    if inclination[0] == "all" and len(inclination) > 1:  # Ignore all if other inclinations are passed
        inclination = inclination[1:]
        ninclinations = len(inclination)
    if inclination[0] == "all":
        ninclinations = nangles

    # This loop will plot the inclinations provided by the user
    for i in range(ninclinations):
        if inclination[0] != "all" and inclination[i] not in angles:  # Skip inclinations which don't exist
            continue
        if inclination[0] == "all":
            ii = i + 2
        else:
            ii = angles.index(inclination[i]) + 2
        lab = r"$i$ = " + angles[ii - 2] + r"$^{\circ}$"
        n_non_zero = np.count_nonzero(tspec[:, ii])
        if n_non_zero == 0:  # Skip inclinations which look through empty space hence no optical depth
            continue
        if loglog:
            ax.loglog(tspec[:, 0], tspec[:, ii], label=lab)
        elif logy:
            ax.semilogy(tspec[:, 0], smooth_spectrum(tspec[:, ii], 1), label=lab)
        else:
            ax.plot(tspec[:, 0], tspec[:, ii], label=lab)

    if frequency_space:
        if loglog:
            ax.set_xlabel(r"Log[Frequency], [Hz]", fontsize=axes_label_fontsize)
        else:
            ax.set_xlabel(r"Frequency, [Hz]", fontsize=axes_label_fontsize)
    else:
        ax.set_xlabel(r"Wavelength, [$\AA$]", fontsize=axes_label_fontsize)
    ax.set_ylabel(r"Optical Depth, $\tau$", fontsize=axes_label_fontsize)
    ax.set_xlim(wmin, wmax)
    ax.legend()

    if show_absorption_edge_labels:
        plot_line_ids(ax, absorption_edges(freq=frequency_space, log=loglog))

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    return fig, ax


def __spec_plot_components_sub(ax: plt.Axes, x: np.ndarray, spec: pd.DataFrame, dname: List[str],
                               xlims: Tuple[float, float], smooth: int, logy: bool, frequency_space: bool,
                               axes_label_fontsize: float, verbose: bool, n: str):
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
    dname: list[str]
        The name of the spectrum components to add to the subplot panel
    xlims: Tuple[float, float]
        The lower and upper x-axis boundaries (xlower, xupper)
    smooth: int [optional]
        The size of the boxcar filter to smooth the spectrum components
    logy: bool [optional]
        Use a log scale for the y axis of the figure
    frequency_space: bool [optional]
        Create the figure in frequency space instead of wavelength space
    axes_label_fontsize: float [optional]
        The fontsize for labels on the plot
    verbose: bool [optional]
        Enable verbose output to screen

    Returns
    -------
    ax: pyplot.Axes
        The pyplot.Axes object for the subplot
    """

    for i in range(len(dname)):
        if verbose:
            print("{}: plotting {}".format(n, dname[i]))
        try:
            fl = smooth_spectrum(spec[dname[i]].values.astype(float), smooth)
        except KeyError:
            print("{}: unable to find spectrum component {}".format(n, dname[i]))
            continue
        if len(fl[fl < MIN_SPEC_COMP_FLUX]) > 0.7 * len(fl):  # Skip sparse spec components to make prettier plot
            if verbose:
                print("{}: most of {} less than MIN_SPEC_COM_FLUX ({}) hence skipping"
                      .format(n, dname[i], MIN_SPEC_COMP_FLUX))
            continue
        if frequency_space:
            ax.loglog(x, fl, label=dname[i])
        elif logy:
            ax.semilogy(x, fl, label=dname[i])
        else:
            ax.plot(x, fl, label=dname[i])
    ax.set_xlim(xlims[0], xlims[1])
    if frequency_space:
        ax.set_xlabel(r"Frequency ([Hz])", fontsize=axes_label_fontsize)
    else:
        ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=axes_label_fontsize)
    ax.set_ylabel(r"$F_{\lambda}$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)", fontsize=axes_label_fontsize)
    ax.legend()

    return ax


def spec_plot_components(root: str, wd: str, wmin: float = None, wmax: float = None, smooth: int = 5,
                         logy: bool = True, frequency_space: bool = False, axes_label_fontsize: float = 15,
                         verbose: bool = False) -> Tuple[plt.Figure, plt.Axes]:
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
    wmin: float [optional]
        The lower wavelength boundary for the figure
    wmax: float [optional]
        The upper wavelength boundary for the figure
    smooth: int [optional]
        The size of the boxcar filter to smooth the spectrum components
    logy: bool [optional]
        Use a log scale for the y axis of the figure
    frequency_space: bool [optional]
        Create the figure in frequency space instead of wavelength space
    axes_label_fontsize: float [optional]
        The fontsize for labels on the plot
    verbose: bool [optional]
        Enable verbose output to screen

    Returns
    -------
    fig: pyplot.Figure
        The pyplot.Figure object for the created figure
    ax: pyplot.Axes
        The pyplot.Axes object for the created figure
    """

    n = spec_plot_components.__name__

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    fname = "{}/{}.spec".format(wd, root)
    try:
        spec = read_spec(fname)
    except IOError:
        print("{}: unable to open .spec file with name {}".format(n, fname))
        return fig, ax

    # Use either frequency or wavelength and set the plot limits respectively
    if frequency_space:
        x = spec["Freq."].values.astype(float)
    else:
        x = spec["Lambda"].values.astype(float)
    xlims = [x.min(), x.max()]
    if wmin:
        if frequency_space:
            xlims[0] = C / (wmin * ANGSTROM)
        else:
            xlims[0] = wmin
    if wmax:
        if frequency_space:
            xlims[1] = C / (wmax * ANGSTROM)
        else:
            xlims[1] = wmax
    xlims = (xlims[0], xlims[1])

    ax[0] = __spec_plot_components_sub(ax[0], x, spec, ["Created", "Emitted"], xlims, smooth, logy, frequency_space,
                                       axes_label_fontsize, verbose, n)
    ax[1] = __spec_plot_components_sub(ax[1], x, spec, ["CenSrc", "Disk", "Wind", "HitSurf", "Scattered"], xlims,
                                       smooth, logy, frequency_space, axes_label_fontsize, verbose, n)

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    return fig, ax


def spec_plot_single(wl: np.ndarray, fl: np.ndarray, xlabel: str = None, ylabel: str = None, scale: str = "logy",
                     display: bool = False, **kwargs):
    """
    This is a simple plotting function designed to give you the bare minimum.
    It will create a figure and axes object for a single panel and that is
    it. It is mostly designed for quick plotting of models and real data.

    Parameters
    ----------
    wl: np.ndarray
        The wavelength or x-axis data to plot.
    fl: np.ndarray
        The flux or y-axis data to plot.
    xlabel: str [optional]
        The data label for the x-axis.
    ylabel: str [optional]
        The data label for the y-axis.
    scale: str [optional]
        The scale of the axes for the plot.
    display: bool [optional]
        If set to True, then the plot will be displayed.

    Returns
    -------
    fig: plt.Figure
        The figure object for the plot.
    ax: plt.Axes
        The axes object containing the plot.
    """

    figsize = (12, 5)
    if "figsize" in kwargs:
        figsize = kwargs["figsize"]

    nrows = ncols = 1
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    ax.plot(wl, fl, **kwargs)
    if scale == "loglog" or scale == "logx":
        ax.set_xscale("log")
    if scale == "loglog" or scale == "logy":
        ax.set_yscale("log")

    if xlabel:
        ax.set_xlabel(xlabel, **kwargs)
    if ylabel:
        ax.set_ylabel(ylabel, **kwargs)

    if display:
        plt.show()

    return fig, ax
