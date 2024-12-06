"""The basic/universal plotting functions of pysi.

The module includes functions for normalizing the style, as well as ways
to finish the plot. Included in pysi are functions to plot the
spectrum files and the wind save tables.
"""

import shutil

import numpy as np
from matplotlib import pyplot as plt

import pysi.error as err
from pysi.util import array

LARGE_NUM_PLOTS = 9
SMALL_NUM_PLOTS = 2


def _check_axes_scale_string(scale: str) -> None:
    """Check that the axes scales passed are recognised.

    Parameters
    ----------
    scale: str
        The scaling of the axes to check.

    Raises
    ------
    ValueError
        If the scale is not recognised.

    """
    if scale not in ["logx", "logy", "linlin", "loglog"]:
        raise ValueError(f"{scale} is an unknown axes scale choice, allowed: logx, logy, linlin, loglog")


def plot(  # noqa: PLR0913
    x: np.ndarray,
    y: np.ndarray,
    *,
    xmin: float | None = None,
    xmax: float | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    scale: str = "logy",
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    label: str | None = None,
    alpha: float = 1.0,
    display: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a set of x and y data.

    This function acts as a big wrapper around matplotlib, to plot and create
    a nice set of figures. By providing a Figure and Axes object, one can
    add additional data to an already existing plot.

    Parameters
    ----------
    x: array-like
        The x data to plot.
    y: array-like
        The y data to ploy.
    xmin: float
        The lower boundary of the x axis.
    xmax: flaot
        The upper boundary of the x axis.
    xlabel: str
        The label for the x axis.
    ylabel: str
        The label for the y axis.
    scale: str
        The scalings for the axes, i.e. logx, loglog, linlin.
    fig: plt.Figure
        A figure object to update.
    ax: plt.Axes
        An axes object to update.
    label: str
        The label to give the data being plotted.
    alpha: float
        The transparency of the line for the data being plotted.
    display: bool
        If True, the figure will be displayed.

    Returns
    -------
    fig: plt.Figure
        The figure object.
    ax: plt.Axes
        The axes object.

    """
    # It doesn't make sense to provide only fig and not ax, or ax and not fig
    # so at this point we will throw an error message and return
    if fig and not ax:
        msg = "fig has been provided, but ax has not. Both are required."
        raise err.InvalidParameter(msg)
    if not fig and ax:
        msg = "fig has not been provided, but ax has. Both are required."
        raise err.InvalidParameter(msg)
    if not fig and not ax:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    x, y = array.get_subset_in_second_array(x, y, xmin, xmax)
    ax.plot(x, y, label=label, alpha=alpha)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax = set_axes_scales(ax, scale)

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def finish_figure(
    fig: plt.Figure, *, title: str | None = None, hspace: float | None = None, wspace: float | None = None
) -> plt.Figure:
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

    Returns
    -------
    plt.Figure
        The updated Figure object.

    """
    if title:
        fig.suptitle(title.replace("_", r"\_"))
    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    if hspace is not None:
        fig.subplots_adjust(hspace=hspace)
    if wspace is not None:
        fig.subplots_adjust(wspace=wspace)

    return fig


def set_figure_style() -> dict:
    """Set default pysi matplotlib parameters.

    Returns
    -------
    dict
        The parameters dictionary.

    """
    parameters = {
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
        "pcolor.shading": "auto",
    }

    if shutil.which("pdflatex"):
        parameters["text.usetex"] = True
        parameters["text.latex.preamble"] = r"\usepackage{amsmath}"
    plt.rcParams.update(parameters)

    return parameters


def remove_extra_axee(fig: plt.Figure, ax: plt.Axes, n_wanted: int, n_panel: int) -> tuple[plt.Figure, plt.Axes]:
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
    if type(ax) is np.ndarray or len(ax) == 1:
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


def set_axes_scales(ax: plt.Axes, scale: str) -> plt.Axes:
    """Set the axes scaling for an Axes object.

    Parameters
    ----------
    ax: plt.Axes
        The matplotlib Axes to update.
    scale: str
        The axes scaling to use.

    Returns
    -------
    plt.Axes
        The updated matplotlib Axes.

    """
    _check_axes_scale_string(scale)
    if scale in ("logx", "loglog"):
        ax.set_xscale("log")
    if scale in ("logy", "loglog"):
        ax.set_yscale("log")

    return ax


def subplot_dims(n_plots: int) -> tuple[int, int]:
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
    dims: tuple[int, int]
        The dimensions of the subplots returned as (nrows, ncols)

    """
    if n_plots > LARGE_NUM_PLOTS:
        n_cols = 3
        n_rows = (1 + n_plots) // n_cols
    elif n_plots < SMALL_NUM_PLOTS:
        n_rows = n_cols = 1
    else:
        n_cols = 2
        n_rows = (1 + n_plots) // n_cols

    return n_rows, n_cols
