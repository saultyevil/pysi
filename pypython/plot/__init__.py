#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The basic/universal plotting functions of pypython.

The module includes functions for normalizing the style, as well as ways
to finish the plot. Included in pypython are functions to plot the
spectrum files and the wind save tables.
"""

import numpy as np
from matplotlib import pyplot as plt

from pypython import SPECTRUM_UNITS_FNU, SPECTRUM_UNITS_LNU, get_array_index
from pypython.constants import ANGSTROM, C
from pypython.error import DimensionError, InvalidParameter

# Generic plotting function ----------------------------------------------------


def plot(x,
         y,
         xmin=None,
         xmax=None,
         xlabel=None,
         ylabel=None,
         scale="logy",
         fig=None,
         ax=None,
         label=None,
         alpha=1.0,
         display=False):
    """Wrapper function around plotting a simple line graph.

    Parameters
    ----------

    Returns
    -------
    """

    # It doesn't make sense to provide only fig and not ax, or ax and not fig
    # so at this point we will throw an error message and return

    normalize_figure_style()

    if fig and not ax:
        raise InvalidParameter("fig has been provided, but ax has not. Both are required.")
    elif not fig and ax:
        raise InvalidParameter("fig has not been provided, but ax has. Both are required.")
    elif not fig and not ax:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.plot(x, y, label=label, alpha=alpha)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.set_xlim(xmin, xmax, auto=True)
    ax.set_ylim(auto=True)
    ax = set_axes_scales(ax, scale)

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


# Helper functions -------------------------------------------------------------


def _check_axes_scale_string(scale):
    """Check that the axes scales passed are recognised.

    Parameters
    ----------
    scale: str
        The scaling of the axes to check.
    """

    if scale not in ["logx", "logy", "linlin", "loglog"]:
        raise ValueError(f"{scale} is an unknown axes scale choice, allowed: logx, logy, linlin, loglog")


def set_axes_scales(ax, scale):
    """Set the scale for axes.

    Parameters
    ----------
    ax: plt.Axes
        The matplotlib Axes to update.
    scale: str
        The axes scaling to use.
    """
    _check_axes_scale_string(scale)

    if scale == "logx" or scale == "loglog":
        ax.set_xscale("log")
    if scale == "logy" or scale == "loglog":
        ax.set_yscale("log")

    return ax


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


set_style = set_figure_style = normalize_figure_style


def finish_figure(fig, title=None, hspace=None, wspace=None):
    """Add finishing touches to a figure.

    This function can be used to add a title or adjust the spacing between
    subplot panels. The subplots will also be given a tight layout.

    Parameters
    ----------
    fig:
        The figure object to update.
    title: str
        The title of the figure. Underscores are automatically modified so
        LaTeX doesn't complain.
    hspace: float
        The amount of vertical space between subplots.
    wspace: float
        The amount of horizontal space between subplots.
    """

    if title:
        fig.suptitle(title.replace("_", r"\_"))

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    if hspace:
        fig.subplots_adjust(hspace=hspace)
    if wspace:
        fig.subplots_adjust(wspace=wspace)

    return fig


def subplot_dims(n_plots):
    """Get the number of rows and columns for the give number of plots.

    Returns how many rows and columns should be used to have the correct
    number of figures available. This doesn't return anything larger than
    3 columns, but the number of rows can be large.

    Parameters
    ----------
    n_plots: int
        The number of subplots which will be plotted

    Returns
    -------
    dims: Tuple[int, int]
        The dimensions of the subplots returned as (nrows, ncols)
    """
    if n_plots > 9:
        n_cols = 3
        n_rows = (1 + n_plots) // n_cols
    elif n_plots < 2:
        n_rows = n_cols = 1
    else:
        n_cols = 2
        n_rows = (1 + n_plots) // n_cols

    return n_rows, n_cols


def remove_extra_axes(fig, ax, n_wanted, n_panel):
    """Remove additional axes which are included in a plot.

    This should be used if you have 4 x 2 = 8 panels but only want to use 7 of
    the panels, in this case the 8th panel will be removed.

    Parameters
    ----------
    fig: plt.Figure
        The Figure object to modify.
    ax: np.ndarray of plt.Axes
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
        The modified Axes.
    """

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


def _check_sorted(x):
    """Check if an array is sorted in ascending order.

    Parameters
    ----------
    x: np.ndarray, list
        The array to check.
    """
    return np.all(np.diff(x) >= 0)


def get_x_subset(x, y, xmin, xmax):
    """Get a subset of values from two array given xmin and xmax.

    Parameters
    ----------
    x: np.ndarray
        The first array to get the subset from, set by xmin and xmax.
    y: np.ndarray
        The second array to get the subset from.
    xmin: float
        The minimum x value
    xmax: float
        The maximum x value

    Returns
    -------
    x, y: np.ndarray
        The subset arrays.
    """

    ascending = True

    if not _check_sorted(x):
        if _check_sorted(x.copy()[::-1]):
            ascending = False
        else:
            raise ValueError("cannot use get_x_subset on an unsorted array")

    if ascending:
        if xmin:
            idx = get_array_index(x, xmin)
            x = x[idx:]
            y = y[idx:]
        if xmax:
            idx = get_array_index(x, xmax)
            x = x[:idx]
            y = y[:idx]
    else:
        if xmin:
            idx = get_array_index(x, xmin)
            x = x[:idx]
            y = y[:idx]
        if xmax:
            idx = get_array_index(x, xmax)
            x = x[idx:]
            y = y[idx:]

    return x, y


def get_y_lims_for_x_lims(x, y, xmin, xmax, scale=10):
    """Determine the lower and upper y for the given x range.

    Useful as matplotlib does not rescale the y limits when the x range is
    restricted.

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

    if xmin is None or xmax is None:
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


def _convert_labels_to_frequency_space(lines, freq=False, spectrum=None):
    """Convert the given list of lines/edges from Angstrom to Hz.

    Parameters
    ----------
    lines: List[str, float]
        The list of labels to convert from wavelength to frequency space.
    freq: bool
        The flag to indicate to convert to frequency space
    spectrum: pypython.Spectrum
        A spectrum object, used to find the units of the spectrum.
    """

    if spectrum:
        if spectrum.units in [SPECTRUM_UNITS_FNU, SPECTRUM_UNITS_LNU]:
            for i in range(len(lines)):
                lines[i][1] = C / (lines[i][1] * ANGSTROM)
    elif freq:
        for i in range(len(lines)):
            lines[i][1] = C / (lines[i][1] * ANGSTROM)
    # else:
    #     raise ValueError("_convert_labels_to_frequency_space: Unable to convert to label locations to frequency space "
    #                      "when arguments freq or spectrum aren't provided")

    return lines


def common_lines(freq=False, spectrum=None):
    """Return a list containing the names of line transitions and the
    wavelength of the transition in Angstroms. Instead of returning the
    wavelength, the frequency can be returned instead. It is also possible to
    return in log space.

    Parameters
    ----------
    freq: bool [optional]
        Label the transitions in frequency space
    spectrum: pypython.Spectrum
        The spectrum object. Used to get the units.

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

    lines = _convert_labels_to_frequency_space(lines, freq, spectrum)

    return lines


def photoionization_edges(freq=False, spectrum=False):
    """Return a list containing the names of line transitions and the
    wavelength of the transition in Angstroms. Instead of returning the
    wavelength, the frequency can be returned instead. It is also possible to
    return in log space.

    Parameters
    ----------
    freq: bool [optional]
        Label the transitions in frequency space
    spectrum: pypython.Spectrum
        The spectrum object. Used to get the units.

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

    edges = _convert_labels_to_frequency_space(edges, freq, spectrum)

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


# This is placed here due to a circular dependency -----------------------------

from pypython.plot import misc, spectrum, wind
