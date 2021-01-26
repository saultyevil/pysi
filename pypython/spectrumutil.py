#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various functions used to create visualisations of the
synthetic spectra which are output from Python. This includes utility functions
from finding these spectrum files.
"""


from .error import DimensionError, EXIT_FAIL
from .constants import C, ANGSTROM
from .util import get_root

from sys import exit
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Union, Tuple
from scipy.signal import convolve, boxcar
from matplotlib import pyplot as plt


UNITS_FLAMBDA = "erg/s/cm^-2/A"
UNITS_FNU = "erg/s/cm^-2/Hz"
UNITS_LNU = "erg/s/Hz"


def read_spectrum(
    fname: str, delim: str = None, numpy: bool = False
) -> Union[int, np.ndarray, pd.DataFrame]:
    """
    Read in a spectrum .spec file. The spectrum is read in line by line and
    ignores comments which being with #.

    TODO update to be consistent with the root wd approach

    Parameters
    ----------
    fname: str
        The path to the .spec file
    delim: str [optional]
        The value delimiter.
    numpy:bool [optional]
        Return a Numpy array of strings instead. This is the less ideal option,
        but is retained for legacy reasons.

    Returns
    -------
    spectrum: np.ndarray or pd.DataFrame
        The read in spectrum either as a numpy array or pandas DataFrame.
    """

    n = read_spectrum.__name__

    try:
        with open(fname, "r") as f:
            lines = f.readlines()
    except IOError:
        print("{}: unable to open file {}".format(n, fname))
        exit(EXIT_FAIL)

    spectrum = []

    for i in range(len(lines)):
        line = lines[i].strip()

        if delim:
            line = line.split(delim)
        else:
            line = line.split()

        # Ignore empty and comment lines

        if len(line) == 0:
            continue
        elif line[0] == "#":
            continue

        # Extract the header line

        if line[0] == "Freq." or line[0] == "Lambda":
            for j in range(len(line)):
                # Remove the phase from headers, I don't care about them
                if line[j][0] == "A":
                    index = line[j].find("P")
                    line[j] = line[j][1:index]

        spectrum.append(line)

    if numpy:
        spectrum = np.array(spectrum, dtype=float)
        return spectrum
    else:
        return pd.DataFrame(spectrum[1:], columns=spectrum[0]).astype(float)


def find_spec_files(
    root: str = None, wd: str = ".", ignore_delay_dump_spec: bool = True
) -> List[str]:
    """
    Find root.spec files recursively in the provided directory.

    Parameters
    ----------
    root: str [optional]
        If root is set, then only .spec files with this root name will be
        returned
    wd: str [optional]
        The path to recursively search from
    ignore_delay_dump_spec: [optional] bool
        When True, root.delay_dump.spec files will be ignored

    Returns
    -------
    spec_files: List[str]
        The file paths of various .spec files
    """

    spec_files = []

    for filename in Path(wd).glob("**/*.spec"):

        fname = str(filename)

        if ignore_delay_dump_spec and fname.find(".delay_dump.spec") != -1:
            continue

        if root:
            t_root, wd = get_root(fname)
            if t_root == root:
                spec_files.append(fname)
            else:
                continue
        spec_files.append(fname)

    return spec_files


def get_spectrum_units(
    fname: str
):
    """
    Get the units of a Python spectrum. This information is contained in the
    top matter of the spectrum file.

    Parameters
    ----------
    fname: str
        The path to the spectrum file.

    Returns
    -------
    units: str
        The units of the spectrum.
    """

    with open(fname, "r") as f:
        lines = f.readlines()

    units = "unknown"

    for i in range(len(lines)):
        line = lines[i]
        if line.find("# Units:") != -1:
            units = line.split()[4][1:-1]
            break

    return units


def get_spectrum_inclinations(
    spectra: Union[pd.DataFrame, np.ndarray, List[str], str]
) -> list:
    """
    Return the inclination angles for a single or multiple spectrum files. A
    single spectrum or multiple spectra can be provided, either as a numpy array
    of strings, a list of file names or as pandas DataFrames.

    Parameters
    ----------
    spectra: List[str, np.array[str], pd.DataFrame]
        A spectrum in the form of pd.DataFrame/np.ndarray or a list of
        directories to Python .spec files

    Returns
    -------
    inclinations: List[int]
        All of the unique inclination angles found in the Python .spec files
    """

    n = get_spectrum_inclinations.__name__

    n_spec = 1
    b_read_in_spec = False
    inclinations = []

    if type(spectra) == list:
        b_read_in_spec = True
        n_spec = len(spectra)
    elif type(spectra) == str:
        b_read_in_spec = True
        spectra = [spectra]
    elif type(spectra) != pd.DataFrame and type(spectra) != np.ndarray:
        raise TypeError("{}: spec passed is of unknown type {}".format(n, type(spectra)))

    # Find the viewing angles in each .spec file
    for i in range(n_spec):
        if b_read_in_spec:
            spectra = read_spectrum(spectra[i])

        # I only know what to do when I expect the spectrum to be a pd.DataFrame
        # or a np.array

        if type(spectra) == pd.DataFrame:
            col_names = spectra.columns.values
        elif type(spectra) == np.ndarray:
            col_names = spectra[0, :]
        else:
            raise TypeError("{}: bad data type {}: require pd.DataFrame or np.array".format(n, type(spectra)))

        for j in range(len(col_names)):
            if col_names[j].isdigit() is True and col_names[j] not in inclinations:
                inclinations.append(col_names[j])

    inclinations = sorted(inclinations)

    return inclinations


def check_inclination_valid(
    inclination: str, spectrum: Union[pd.DataFrame, np.ndarray, str]
) -> bool:
    """
    Check that an inclination angle is one which exists in the spectrum.

    Parameters
    ----------
    inclination: str
        The inclination angle to check
    spectrum: str, pd.DataFrame, np.ndarray
        The spectrum to check against. Can either be a pandas DataFrame, numpy
        array of strings or the file name of the spectrum.

    Returns
    -------
    allowed: bool
        If True, angle is a legal angle, otherwise false
    """

    n = check_inclination_valid.__name__

    is_allowed = False

    if type(spectrum) == pd.DataFrame:
        headers = spectrum.columns.values
    elif type(spectrum) == np.ndarray:
        headers = spectrum[0, :]
    elif type(spectrum) == str:
        headers = read_spectrum(spectrum).columns.values
    else:
        raise TypeError("{}: unknown data type {} for function".format(n, type(spectrum)))

    if type(inclination) != str:
        inclination = str(inclination)

    if inclination in headers:
        is_allowed = True

    return is_allowed


def smooth(
    array: Union[np.ndarray, List[Union[float, int]]], smooth_amount: Union[int, float]
) -> np.ndarray:
    """
    Smooth a 1D array of data using a boxcar filter.

    Parameters
    ----------
    array: np.array[float]
        The array to be smoothed.
    smooth_amount: int
        The size of the boxcar filter.

    Returns
    -------
    smoothed: np.ndarray
        The smoothed array
    """

    n = smooth.__name__

    # If smooth_amount is None, then the user has indicated they didn't want
    # to use any smoothing so return the original array. Though, I think using
    # a smoothing window of 1 has the same effect............... dunno

    if smooth_amount is None:
        return array
    elif int(smooth_amount) == 1:
        return array
    elif type(smooth_amount) != int:
        try:
            smooth_amount = int(smooth_amount)
        except ValueError:
            print("{}: could not convert smooth {} into an integer. Returning original array.".format(n, smooth_amount))
            return array

    # Now we need to make sure the array is actually an array and not a list and
    # convert it. We also check the dimensions of the array.

    if type(array) != np.ndarray:
        array = np.array(array)

    if len(array.shape) > 2:
        raise DimensionError("{}: data is not 1 dimensional but has shape {}".format(n, array.shape))

    array = np.reshape(array, (len(array),))  # because fuck me, why does it have to be this form?
    smoothed = convolve(array, boxcar(smooth_amount) / float(smooth_amount), mode="same")

    return smoothed


def calculate_axis_y_limits(
    x: np.array, y: np.array, xmin: float, xmax: float, scale: float = 10
) -> Union[Tuple[float, float], Tuple[None, None]]:
    """
    Determine the lower and upper y limits for a matplotlib plot given a
    restricted x range, since matplotlib doesn't do this automatically.


    Parameters
    ----------
    x: np.array[float]
        The array of x axis points.
    y: np.array[float]
        The array of y axis points.
    xmin: float
        The lowest x value.
    xmax: float
        The largest x value.
    scale: float [optional]
        The scaling factor for white space around the data

    Returns
    -------
    ymin: float
        The lowest y value.
    ymax: float
        The highest y value.
    """

    n = calculate_axis_y_limits.__name__

    if x.shape[0] != y.shape[0]:
        raise DimensionError(
            "{}: wavelength and flux are of different dimensions wavelength {} flux {}".format(n, x.shape, y.shape)
        )

    if type(x) == pd.Series or type(y) == pd.Series:
        try:
            x = np.array(x)  # x = x.values
            y = np.array(y)  # y = y.values
        except ValueError:
            raise TypeError("{}: x or y not a numpy array or pandas series x {} y {}" .format(n, type(x), type(y)))
    elif type(x) != np.ndarray or type(y) != np.ndarray:
        raise TypeError("{}: x or y not a numpy array or pandas series x {} y {}" .format(n, type(x), type(y)))

    if not xmin or not xmax:
        return None, None

    # Determine indices which are within the wavelength range

    id_xmin = x < xmin
    id_xmax = x > xmax

    # Extract flux which is in the wavelength range, remove 0 values and then
    # find min and max value and scale

    y_lim_x = np.where(id_xmin == id_xmax)[0]

    y = y[y_lim_x]
    y = y[y > 0]

    ymin = np.min(y) / scale
    ymax = np.max(y) * scale

    return ymin, ymax


def ax_add_line_id(
    ax: plt.Axes, lines: list, linestyle: str = "dashed", ynorm: float = 0.90, logx: bool = False, offset: float = 25,
    rotation: str = "vertical", fontsize: int = 10
) -> plt.Axes:
    """
    Add labels for line transitions or other regions of interest onto a
    matplotlib figure. Labels are placed at the top of the panel and dashed
    lines, with zorder = 0, are drawn from top to bottom.

    Parameters
    ----------
    ax: plt.Axes
        The axes (plt.Axes) to add line labels too
    lines: list
        A list containing the line name and wavelength in Angstroms
        (ordered by wavelength)
    linestyle: str [optional]
        The type of line to draw to show where the transitions are. Allowed
        values [none, dashed, top]
    ynorm: float [optional]
        The normalized y coordinate to place the label.
    logx: bool [optional]
        Use when the x-axis is logarithmic
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
        if linestyle == "dashed":
            ax.axvline(x, linestyle="--", linewidth=0.5, color="k", zorder=1)
        if linestyle == "thick":
            ax.axvline(x, linestyle="-", linewidth=2, color="k", zorder=1)
        elif linestyle == "top":
            pass  # TODO: to implement
        x = x - offset

        # Calculate the x location of the label in axes coordinates

        if logx:
            xnorm = (np.log10(x) - np.log10(xlims[0])) / (np.log10(xlims[1]) - np.log10(xlims[0]))
        else:
            xnorm = (x - xlims[0]) / (xlims[1] - xlims[0])

        ax.text(
            xnorm, ynorm, label, ha="center", va="center", rotation=rotation, fontsize=fontsize, transform=ax.transAxes
        )

    return ax


def common_lines_list(
    freq: bool = False
) -> list:
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
        ["N III/O III", 305],
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
        ["Ca II", 3934],
        ["", 3969],
        [r"H$_{\delta}$", 4101],
        [r"H$_{\gamma}$", 4340],
        ["He II", 4389],
        ["He II", 4686],
        [r"H$_{\beta}$", 4861],
        ["Na I", 5891],
        ["", 5897],
        [r"H$_{\alpha}$", 6564],
    ]

    if freq:
        for i in range(len(lines)):
            lines[i][1] = C / (lines[i][1] * ANGSTROM)

    return lines


def photo_edges_list(
    freq: bool = False
) -> list:
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
        ["He II", 229],
        ["He I", 504],
        ["Lyman", 912],
        # ["Ca I", 2028],
        # ["Al I", 2071],
        ["Balmer", 3646],
        ["Paschen", 8204],
    ]

    if freq:
        for i in range(len(edges)):
            edges[i][1] = C / (edges[i][1] * ANGSTROM)

    return edges
