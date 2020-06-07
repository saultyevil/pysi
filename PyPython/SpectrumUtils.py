#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various functions used to create visualisations of the
synthetic spectra which are output from Python. This includes utility functions
from finding these spectrum files.
"""


from .Error import DimensionError
from .Constants import C, ANGSTROM, PARSEC
from .PythonUtils import split_root_directory

from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Union, Tuple
from scipy.signal import convolve, boxcar
from matplotlib import pyplot as plt


def find_specs(root: str = None, path: str = ".") -> List[str]:
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
        # todo this is bad, need cleaner way to omit these files
        if fname.find(".delay_dump.spec") != -1:
            continue
        if root:
            froot, wd = split_root_directory(fname)
            if froot == root:
                spec_files.append(fname)
            continue
        spec_files.append(fname)

    return spec_files


def read_spec(fname: str, delim: str = None, numpy: bool = False) -> Union[int, np.ndarray, pd.DataFrame]:
    """
    Read in data from an external file, line by line whilst ignoring comments.
        - Comments begin with #
        - The default delimiter is assumed to be a space

    Parameters
    ----------
    fname: str
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

    with open(fname, "r") as f:
        flines = f.readlines()

    lines = []
    for i in range(len(flines)):
        line = flines[i].strip()
        if delim:
            line = line.split(delim)
        else:
            line = line.split()
        if len(line) == 0:
            continue
        if line[0] == "#":
            continue
        if line[0] == "Freq." or line[0] == "Lambda":
            for j in range(len(line)):
                if line[j][0] == "A":
                    index = line[j].find("P")
                    line[j] = line[j][1:index]
        lines.append(line)

    if numpy:
        try:
            spec = np.array(lines, dtype=float)
            return spec
        except ValueError:
            return np.array(lines)

    return pd.DataFrame(lines[1:], columns=lines[0]).astype(float)


def spec_inclinations(spec: Union[pd.DataFrame, np.ndarray, List[str], str]) -> list:
    """
    Find the unique inclination angles for a set of Python .spec files given
    the path for multiple .spec files.

    Parameters
    ----------
    spec: List[str]
        A spectrum in the form of pd.DataFrame/np.ndarray or a list of
        directories to Python .spec files

    Returns
    -------
    inclinations: List[int]
        All of the unique inclination angles found in the Python .spec files
    """

    n = spec_inclinations.__name__
    inclinations = []

    nspec = 1
    readin = False

    if type(spec) == list:
        nspec = len(spec)
        readin = True
    elif type(spec) == str:
        spec = [spec]
        readin = True
    elif type(spec) != pd.core.frame.DataFrame and type(spec) != np.ndarray:
        raise TypeError("{}: spec passed is of unknown type {}".format(n, type(spec)))

    # Find the viewing angles in each .spec file
    for i in range(nspec):
        if readin:
            spec = read_spec(spec[i])

        if type(spec) == pd.core.frame.DataFrame:
            col_names = spec.columns.values
        elif type(spec) == np.ndarray:
            col_names = spec[0, :]
        else:
            raise TypeError("{}: bad data type {}: require pd.DataFrame or np.array".format(n, type(spec)))

        for j in range(len(col_names)):
            if col_names[j].isdigit() is True and col_names[j] not in inclinations:
                inclinations.append(col_names[j])

    inclinations = sorted(inclinations)

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
            print("{}: could not convert {} into string".format(n, inclination))
            return allowed

    if inclination in headers:
        allowed = True

    return allowed


def smooth(input: Union[np.ndarray, List[Union[float, int]]], smooth_amount: Union[int, float]) -> np.ndarray:
    """
    Smooth a 1D array of data using a boxcar filter of width smooth pixels.

    Parameters
    ----------
    input: np.array[float]
        The data to smooth using the boxcar filter
    smooth_amount: int
        The size of the window for the boxcar filter

    Returns
    -------
    smoothed: np.ndarray
        The smoothed data
    """

    n = smooth.__name__

    input = np.array(input)

    if type(input) != list and type(input) != np.ndarray:
        raise TypeError("{}: expecting list or np.ndarray".format(n))

    if type(input) == list:
        input = np.array(input, dtype=float)

    if len(input.shape) > 2:
        raise DimensionError("{}: data is not 1 dimensional but has shape {}".format(n, input.shape))

    if type(smooth_amount) != int:
        try:
            smooth_amount = int(smooth_amount)
        except ValueError:
            print("{}: could not convert smooth = {} into an integer. Returning original array.".format(n, smooth_amount))
            return input

    input = np.reshape(input, (len(input),))
    smoothed = convolve(input, boxcar(smooth_amount) / float(smooth_amount), mode="same")

    return smoothed


def ylims(wavelength: np.array, flux: np.array, wmin: Union[float, None], wmax: Union[float, None],
          scale: float = 10,) -> Union[Tuple[float, float], Tuple[bool, bool]]:
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
    ylower: float
        The lower y limit to use with the wavelength range
    yupper: float
        The upper y limit to use with the wavelength range
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

    return ylower, yupper


def common_lines(freq: bool = False) -> list:
    """
    Return a list containing the names of line transitions and the
    wavelength of the transition in Angstroms. Instead of returning the
    wavelength, the frequency can be returned instead. It is also possible to
    return in log space.

    Parameters
    ----------
    freq: bool [optional]
        Label the transitions in frequency space

    Returns
    -------
    line: List[List[str, float]]
        A list of lists where each element of the list is the name of the
        transition/edge and the rest wavelength of that transition in Angstroms.
    """

    lines = [
        ["He II Edge", 229],
        ["N III/O III", 305],
        ["He I Edge", 504],
        ["Lyman Edge", 912],
        ["P V", 1118],
        [r"Ly$\alpha$/N V", 1216],
        ["", 1242],
        ["O V/Si IV", 1371],
        ["", 1400],
        ["N IV", 1489],
        ["C IV", 1548],
        ["", 1550],
        ["He II", 1640],
        ["N III]", 1750],
        ["Al III", 1854],
        ["C III]", 1908],
        ["Mg II", 2798],
        ["Balmer Edge", 3646],
        ["Ca II", 3934],
        ["", 3969],
        [r"H$_{\delta}$", 4101],
        [r"H$_{\gamma}$", 4340],
        ["He II", 4686],
        [r"H$_{\beta}$", 4861],
        ["Na I", 5891],
        ["", 5897],
        [r"H$_{\alpha}$", 6564],
        ["Paschen Edge", 8204]
    ]

    if freq:
        for i in range(len(lines)):
            lines[i][1] = C / (lines[i][1] * ANGSTROM)

    return lines


def absorption_edges(freq: bool = False) -> list:
    """
    Return a list containing the names of line transitions and the
    wavelength of the transition in Angstroms. Instead of returning the
    wavelength, the frequency can be returned instead. It is also possible to
    return in log space.

    Parameters
    ----------
    freq: bool [optional]
        Label the transitions in frequency space

    Returns
    -------
    edges: List[List[str, float]]
        A list of lists where each element of the list is the name of the
        transition/edge and the rest wavelength of that transition in Angstroms.
    """

    edges = [
        ["He II Edge", 229],
        ["He I Edge", 504],
        ["Lyman Edge", 912],
        ["Balmer Edge", 3646],
        ["Paschen Edge", 8204],
    ]

    if freq:
        for i in range(len(edges)):
            edges[i][1] = C / (edges[i][1] * ANGSTROM)

    return edges


def plot_line_ids(ax: plt.Axes, lines: list, logx: bool = False, offset: float = 25, rotation: str = "vertical",
                  fontsize: int = 10) \
        -> plt.Axes:
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
    logx: bool [optional]
        If the x-axis is logarithmic, then we need to calculate the xnorm
        value slightly differently
    offset: float [optional]
        The amount to offset line labels along the x-axis
    rotation: str [optional]
        Vertical or horizontal rotation for text ids
    fontsize: int [optional]
        The fontsize of the labels

    Returns
    -------
    ax: plt.Axes
        The plot object now with lines IDs :-)
        :param logx:
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
        x = x - offset

        # Calculate the x location of the label in axes coordinates

        if logx:
            xnorm = (np.log10(x) - np.log10(xlims[0])) / (np.log10(xlims[1]) - np.log10(xlims[0]))
        else:
            xnorm = (x - xlims[0]) / (xlims[1] - xlims[0])

        ax.text(xnorm, 0.90, label, ha="center", va="center", rotation=rotation, fontsize=fontsize,
                transform=ax.transAxes)

    return ax


def get_wavelength_index(wl: float, target_wl: float) -> int:
    """
    Return the index for an array, generally this is used to find the index
    for a certain wavelength value.

    Parameters
    ----------
    wl: float
        The array containing the wavelength values.
    target_wl: float
        The target wavelength or value.

    Returns
    -------
    index: int
        The index for where the target is in the provided array.
    """

    index = np.abs(wl - target_wl).argmin()

    return index
