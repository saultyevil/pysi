#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt

from pypython.error import DimensionError

from ..physics.constants import ANGSTROM, C


def normalize_figure_style():
    """Set default pypython matplotlib parameters."""

    parameters = {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.serif": "cm",
        "font.size": 18,
        "legend.fontsize": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "axes.linewidth": 2,
        "lines.linewidth": 2.2,
        "xtick.bottom": True,
        "xtick.minor.visible": True,
        "xtick.direction": "out",
        "xtick.major.width": 1.5,
        "xtick.minor.width": 1,
        "xtick.major.size": 4,
        "xtick.minor.size": 3,
        "ytick.left": True,
        "ytick.minor.visible": True,
        "ytick.direction": "out",
        "ytick.major.width": 1.5,
        "ytick.minor.width": 1,
        "ytick.major.size": 4,
        "ytick.minor.size": 3,
        "savefig.dpi": 300,
        "pcolor.shading": "auto"
    }

    plt.rcParams.update(parameters)

    return parameters


def subplot_dims(n_plots):
    """Return the dimensions for a plot with multiple subplot panels.
    todo: include more plot subdivisions

    Parameters
    ----------
    n_plots: int
        The number of subplots which will be plotted

    Returns
    -------
    dims: Tuple[int, int]
        The dimensions of the subplots returned as (nrows, ncols)"""

    n_rows = n_cols = 1

    if n_plots < 1:
        return n_rows, n_cols

    if n_plots > 2:
        n_cols = 2
        n_rows = (1 + n_plots) // n_cols
    elif n_plots > 9:
        n_cols = 3
        n_rows = (1 + n_plots) // n_cols
    else:
        n_cols = 1
        n_rows = n_plots

    return n_rows, n_cols


def remove_extra_axes(fig, ax, n_wanted, n_panel):
    """Remove additional axes which are included in a plot. This can be used if you
    have 4 x 2 = 8 panels but only want to use 7 of tha panels. The 8th panel
    will be removed.

    Parameters
    ----------
    fig: plt.Figure
        The Figure object to modify.
    ax: plt.Axes
        The Axes objects to modify.
    n_wanted: int
        The actual number of plots/panels which are wanted.
    n_panel: int
        The number of panels which are currently in the Figure and Axes objects.

    Returns
    -------
    fig: plt.Figure
        The modified Figure.
    ax: plt.Axes
        The modified Axes."""

    if type(ax) != np.ndarray:
        return fig, ax
    elif len(ax) == 1:
        return fig, ax

    # Flatten the axes array to make life easier with indexing

    shape = ax.shape
    ax = ax.flatten()

    if n_panel > n_wanted:
        for i in range(n_wanted, n_panel):
            fig.delaxes(ax[i])

    # Return ax to the shape it was passed as

    ax = np.reshape(ax, (shape[0], shape[1]))

    return fig, ax


def get_y_lims_for_x_lims(x, y, xmin, xmax, scale=10):
    """Determine the lower and upper y limits for a matplotlib plot given a
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

    n = get_y_lims_for_x_lims.__name__

    if x.shape[0] != y.shape[0]:
        raise DimensionError("{}: x and y are of different dimensions x {} y {}".format(n, x.shape, y.shape))

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


def common_lines(freq=False):
    """Return a list containing the names of line transitions and the
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
        transition/edge and the rest wavelength of that transition in
        Angstroms.
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


def photoionization_edges(freq=False):
    """Return a list containing the names of line transitions and the
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
        transition/edge and the rest wavelength of that transition in
        Angstroms.
    """

    edges = [
        ["He II", 229],
        ["He I", 504],
        ["Lyman", 912],
        ["Balmer", 3646],
        ["Paschen", 8204],
    ]

    if freq:
        for i in range(len(edges)):
            edges[i][1] = C / (edges[i][1] * ANGSTROM)

    return edges


def ax_add_line_ids(ax, lines, linestyle="dashed", ynorm=0.90, logx=False, offset=25, rotation="vertical", fontsize=10):
    """Add labels for line transitions or other regions of interest onto a
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
        The plot object now with lines IDs :-)"""

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
            pass  # todo: implement
        x = x - offset

        # Calculate the x location of the label in axes coordinates

        if logx:
            xnorm = (np.log10(x) - np.log10(xlims[0])) / (np.log10(xlims[1]) - np.log10(xlims[0]))
        else:
            xnorm = (x - xlims[0]) / (xlims[1] - xlims[0])

        ax.text(xnorm,
                ynorm,
                label,
                ha="center",
                va="center",
                rotation=rotation,
                fontsize=fontsize,
                transform=ax.transAxes)

    return ax
