#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various functions used to create visualisations of the
synthetic spectra which are output from Python. This includes utility functions
from finding these spectrum files, as well as functions to create plots of the
spectra.
"""

from .Error import DimensionError, InvalidFileContents
from .Constants import C, ANGSTROM
from .Util import split_root_directory

from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Union, Tuple
from scipy.signal import convolve, boxcar
from matplotlib import pyplot as plt


MIN_SPEC_COMP_FLUX = 1e-17


def find_specs(root: str = None, path: str = "./") -> List[str]:
    """
    Find root.spec files recursively in provided directory path.

    Parameters
    ----------
    root: str [optional]
        If root is set, then only .spec files with this root name will be
        returned
    path: str [optional]
        The path to recursively search from

    Returns
    -------
    spec_files: List[str]
        The file paths of various .spec files
    """

    spec_files = []
    for filename in Path(path).glob("**/*.spec"):
        fname = str(filename)
        if root:
            froot, wd = split_root_directory(fname)
            if froot == root:
                spec_files.append(fname)
            continue
        spec_files.append(fname)

    return spec_files


def read_spec(file_name: str, delim: str = None, numpy: bool = False) -> Union[np.ndarray, pd.DataFrame]:
    """
    Read in data from an external file, line by line whilst ignoring comments.
        - Comments begin with #
        - The default delimiter is assumed to be a space

    Parameters
    ----------
    file_name: str
        The directory path to the spec file to be read in
    delim: str [optional]
        The delimiter between values in the file, by default a space is assumed
    numpy:bool [optional]
        If True, a Numpy array of strings will be used instead :-(

    Returns
    -------
    lines: np.ndarray or pd.DataFrame
        The .spec file as a Numpy array or a Pandas DataFrame
    """

    n = read_spec.__name__

    try:
        with open(file_name, "r") as f:
            flines = f.readlines()
    except IOError:
        raise Exception("{}: cannot open spec file {}".format(n, file_name))

    lines = []
    for i in range(len(flines)):
        line = flines[i].strip()
        if delim:
            line = line.split(delim)
        else:
            line = line.split()
        if len(line) > 0:
            if line[0] == "#":
                continue
            if line[0] == "Freq.":
                for j in range(len(line)):
                    if line[j][0] == "A":
                        index = line[j].find("P")
                        line[j] = line[j][1:index]
            lines.append(line)

    if numpy:
        return np.array(lines)

    return pd.DataFrame(lines[1:], columns=lines[0])


def spec_inclinations(spec_paths: List[str]) -> np.array:
    """
    Find the unique inclination angles for a set of Python .spec files given
    the path for multiple .spec files.

    Parameters
    ----------
    spec_paths: List[str]
        The directory path to Python .spec files

    Returns
    -------
    inclinations: List[int]
        All of the unique inclination angles found in the Python .spec files
    """

    n = spec_inclinations.__name__
    inclinations = []

    # Find the viewing angles in each .spec file
    for i in range(len(spec_paths)):
        spec = read_spec(spec_paths[i])

        if type(spec) == pd.core.frame.DataFrame:
            col_names = spec.columns.values
        elif type(spec) == np.ndarray:
            col_names = spec[0, :]
        else:
            raise TypeError("{}: bad data type {}: require pd.DataFrame or np.array".format(n, type(spec)))

        # Go over the columns and look for viewing angles
        for j in range(len(col_names)):
            if col_names[j].isdigit() is True:
                angle = int(col_names[j])
                duplicate_flag = False
                for va in inclinations:  # Check for duplicate angle
                    if angle == va:
                        duplicate_flag = True
                if duplicate_flag is False:
                    inclinations.append(angle)

    return inclinations


def check_inclination(inclination: str, spec: Union[pd.DataFrame, np.ndarray]) -> bool:
    """
    Check that an inclination angle is in a spectrum.

    Parameters
    ----------
    inclination: str
        The inclination angle to check
    spec: np.ndarray
        The spectrum array to read -- assume that it is a np.array of strings
        Note that tde_spec_plot has a similar routine for pd.DataFrame's, whoops!

    Returns
    -------
    allowed: bool
        If True, angle is a legal angle, otherwise false
    """

    n = check_inclination.__name__
    allowed = False

    if type(spec) == pd.core.frame.DataFrame:
        headers = spec.columns.values
    elif type(spec) == np.ndarray:
        headers = spec[0, :]
    else:
        raise TypeError("{}: unknown data type {} for function".format(n, type(spec)))

    if type(inclination) != str:
        try:
            inclination = str(inclination)
        except ValueError:
            raise TypeError("{}: could not convert {} into string".format(n, inclination))

    if inclination in headers:
        allowed = True

    return allowed


def smooth_spectrum(flux: np.ndarray, smooth: Union[int, float]) -> np.ndarray:
    """
    Smooth a 1D array of data using a boxcar filter of width smooth pixels.

    Parameters
    ----------
    flux: np.array[float]
        The data to smooth using the boxcar filter
    smooth: int
        The size of the window for the boxcar filter

    Returns
    -------
    smoothed: np.ndarray
        The smoothed data
    """

    n = smooth_spectrum.__name__

    if type(flux) != list and type(flux) != np.ndarray:
        raise TypeError("{}: expecting list or np.ndarray".format(n))

    if type(flux) == list:
        flux = np.array(flux, dtype=float)

    if len(flux.shape) > 2:
        raise DimensionError("{}: data is not 1 dimensional but has shape {}".format(n, flux.shape))

    if type(smooth) != int:
        try:
            smooth = int(smooth)
        except ValueError as e:
            print(e)
            print("{}: could not convert smooth = {} into an integer. Returning original array.".format(n, smooth))
            return flux

    flux = np.reshape(flux, (len(flux),))
    smoothed = convolve(flux, boxcar(smooth) / float(smooth), mode="same")

    return smoothed


def ylims(wavelength: np.array, flux: np.array, wmin: float, wmax: float, scale: float = 10,) \
        -> Union[Tuple[float, float], Tuple[bool, bool]]:
    """
    Determine the lower and upper limit for the flux given a restricted
    wavelength range (wmin, wmax).

    Parameters
    ----------
    wavelength: np.array[float]
        An array containing all wavelengths in a spectrum
    flux: np.array[float]
        An array containing the flux at each wavelength point
    wmin: float
        The shortest wavelength which is being plotted
    wmax: float
        The longest wavelength which is being plotted
    scale: float [optional]
        The scaling factor for white space around the data

    Returns
    -------
    yupper: float
        The upper y limit to use with the wavelength range
    ylower: float
        The lower y limit to use with the wavelength range
    """

    n = ylims.__name__

    if wavelength.shape[0] != flux.shape[0]:
        raise DimensionError("{}: wavelength and flux are of different dimensions wavelength {} flux {}"
                                   .format(n, wavelength.shape, flux.shape))

    if type(wavelength) != np.ndarray or type(flux) != np.ndarray:
        raise TypeError("{}: wavelength or flux array not a numpy arrays wavelength {} flux {}"
                        .format(n, type(wavelength), type(flux)))

    yupper = ylower = None

    if wmin and wmax:
        idx_wmin = wavelength < wmin
        idx_wmax = wavelength > wmax
        flux_lim_wav = np.where(idx_wmin == idx_wmax)[0]
        yupper = np.max(flux[flux_lim_wav]) * scale
        ylower = np.min(flux[flux_lim_wav]) / scale

    return yupper, ylower


def common_lines(freq: bool = False, log: bool = False) -> list:
    """
    Return a list containing the names of line transitions and the
    wavelength of the transition in Angstroms. Instead of returning the
    wavelength, the frequency can be returned instead. It is also possible to
    return in log space.

    Parameters
    ----------
    freq: bool [optional]
        Label the transitions in frequency space
    log: bool [optional]
        Label the transitions in log space

    Returns
    -------
    line: List[List[str, float]]
        A list of lists where each element of the list is the name of the
        transition/edge and the rest wavelength of that transition in Angstroms.
    """

    lines = [
        ["HeII Edge", 229],
        ["Lyman Edge", 912],
        ["P V", 1118],
        [r"Ly$\alpha$/N V", 1216],
        ["", 1240],
        ["O V/Si IV", 1371],
        ["", 1400],
        ["N IV", 1489],
        ["C IV", 1549],
        ["He II", 1640],
        ["N III]", 1750],
        ["Al III", 1854],
        ["C III]", 1908],
        ["Mg II", 2798],
        ["Balmer Edge", 3646],
        ["He II", 4686],
        [r"H$_{\beta}$", 4861],
        [r"H$_{\alpha}$", 6564],
        ["Paschen Edge", 8204]
    ]

    if freq:
        for i in range(len(lines)):
            lines[i][1] = C / (lines[i][1] * ANGSTROM)

    if log:
        for i in range(len(lines)):
            lines[i][1] = np.log10(lines[i][1])

    return lines


def absorption_edges(freq: bool = False, log: bool = False) -> list:
    """
    Return a list containing the names of line transitions and the
    wavelength of the transition in Angstroms. Instead of returning the
    wavelength, the frequency can be returned instead. It is also possible to
    return in log space.

    Parameters
    ----------
    freq: bool [optional]
        Label the transitions in frequency space
    log: bool [optional]
        Label the transitions in log space

    Returns
    -------
    edges: List[List[str, float]]
        A list of lists where each element of the list is the name of the
        transition/edge and the rest wavelength of that transition in Angstroms.
    """

    edges = [
        ["HeII Edge", 229],
        ["Lyman Edge", 912],
        ["Balmer Edge", 3646],
        ["Paschen Edge", 8204],
    ]

    if freq:
        for i in range(len(edges)):
            edges[i][1] = C / (edges[i][1] * ANGSTROM)

    if log:
        for i in range(len(edges)):
            edges[i][1] = np.log10(edges[i][1])

    return edges


def plot_line_ids(ax: plt.Axes, lines: list, rotation: str = "vertical", fontsize: int = 10) -> plt.Axes:
    """
    Plot line IDs onto a figure. This should probably be used after the x-limits
    have been set on the figure which these labels are being plotted onto.

    Parameters
    ----------
    ax: plt.Axes
        The plot object to add line IDs to
    lines: list
        A list containing the line name and wavelength in Angstroms
        (ordered by wavelength)
    rotation: str [optional]
        Vertical or horizontal rotation for text ids
    fontsize: int [optional]
        The fontsize of the labels

    Returns
    -------
    ax: plt.Axes
        The plot object now with lines IDs :-)
    """

    nlines = len(lines)
    xlims = ax.get_xlim()

    for i in range(nlines):
        x = lines[i][1]
        if x < xlims[0]:
            continue
        if x > xlims[1]:
            continue
        label = lines[i][0]
        ax.axvline(x, linestyle="--", linewidth=0.5, color="k", zorder=1)
        x = x - 50
        xnorm = (x - xlims[0]) / (xlims[1] - xlims[0])
        ax.text(xnorm, 0.90, label, ha="center", va="center", rotation=rotation, fontsize=fontsize,
                transform=ax.transAxes)

    return ax


def plot_tau_spectrum(root: str, wd: str, inclination: List[str] = ["all"], wmin: float = None, wmax: float = None,
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

    n = plot_tau_spectrum.__name__
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fname = "{}/diag_{}/{}.tau_spec.diag".format(wd, root, root)

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
        if wmin:
            wmin = C / (wmin * ANGSTROM)
        if wmax:
            wmax = C / (wmax * ANGSTROM)

    # Determine the inclinations which are available
    with open(fname, "r") as f:
        angles = f.readline().split()
    if angles[0] == "#":
        angles = angles[2:]
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
            ii = i + 1
        else:
            ii = angles.index(inclination[i]) + 1
        l = r"$i$ = " + angles[ii - 1] + r"$^{\circ}$"
        n_non_zero = np.count_nonzero(tspec[:, ii])
        if n_non_zero == 0:  # Skip inclinations which look through empty space hence no optical depth
            continue
        if loglog:
            ax.loglog(np.log10(tspec[:, 0]), tspec[:, ii], label=l)
        elif logy:
            ax.semilogy(tspec[:, 0], tspec[:, ii], label=l)
        else:
            ax.plot(tspec[:, 0], tspec[:, ii], label=l)

    if frequency_space:
        if loglog:
            ax.set_xlabel(r"Log[Frequency], Hz", fontsize=axes_label_fontsize)
        else:
            ax.set_xlabel(r"Frequency, Hz", fontsize=axes_label_fontsize)
    else:
        ax.set_xlabel(r"Wavelength, $\AA$", fontsize=axes_label_fontsize)
    ax.set_ylabel(r"Optical Depth, $\tau$", fontsize=axes_label_fontsize)
    ax.set_xlim(wmin, wmax)
    ax.legend()

    if show_absorption_edge_labels:
        plot_line_ids(ax, absorption_edges(freq=frequency_space, log=loglog))

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    return fig, ax


def plot_spectrum_components(root: str, wd: str, wmin: float = None, wmax: float = None, smooth: int = 5,
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

    def plot_data(ax: plt.Axes, x: np.ndarray, spec: pd.DataFrame, dname: List[str], xlims: Tuple[float, float]):
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

        Returns
        -------
        ax: pyplot.Axes
            The pyplot.Axes object for the subplot
        """

        for i in range(len(dname)):
            if verbose:
                print("{}: plotting {}".format(n, dname[i]))
            fl = smooth_spectrum(spec[dname[i]].values.astype(float), smooth)
            if len(fl[fl < MIN_SPEC_COMP_FLUX]) > 0.7 * len(fl):  # Skip sparse spec components to make prettier plot
                if verbose:
                    print("{}: most of {} less than MIN_SPEC_COM_FLUX hence skipping".format(n, dname[i]))
                continue
            if frequency_space:
                ax.loglog(x, fl, label=dname[i])
            elif logy:
                ax.semilogy(x, fl, label=dname[i])
            else:
                ax.plot(x, fl, label=dname[i])
        ax.set_xlim(xlims[0], xlims[1])
        if frequency_space:
            ax.set_xlabel(r"Frequency (Hz)", fontsize=axes_label_fontsize)
        else:
            ax.set_xlabel(r"Wavelength ($\AA$)", fontsize=axes_label_fontsize)
        ax.set_ylabel(r"$F_{\lambda}$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)", fontsize=axes_label_fontsize)
        ax.legend()

        return ax

    n = plot_spectrum_components.__name__
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

    # Now plot the data
    ax[0] = plot_data(ax[0], x, spec, ["Created", "Emitted"], xlims)
    ax[1] = plot_data(ax[1], x, spec, ["CenSrc", "Disk", "Wind", "HitSurf", "Scattered"], xlims)

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    return fig, ax


def plot_spectrum():
    pass
