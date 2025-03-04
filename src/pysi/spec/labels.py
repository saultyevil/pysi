"""Decorate a figure with IDs for common atomic transitions."""

import numpy
from astropy.constants import c
from matplotlib import pyplot as plt

from pysi.math.constants import ANGSTROM


def _convert_to_frequency_space(lines: dict | list, convert: bool) -> dict | list:  # noqa: FBT001
    """Convert a dictionary of lines from wavelength to frequency space.

    Parameters
    ----------
    lines : dict | list
        The dict containing the lines and their wavelength.
    convert : bool
        Convert to frequency space if True, otherwise return the original.

    Returns
    -------
    dict | list
        The updated dict.

    """
    if not convert:
        return lines

    if isinstance(lines, dict):
        for k, v in lines.items():
            lines[k] = c.cgs / (v * ANGSTROM)
    elif isinstance(lines, list):
        lines = [[v[0], c.cgs / (v[1] * ANGSTROM)] for v in lines]
    else:
        msg = "lines must be a dict or list"
        raise TypeError(msg)

    return lines


def get_common_transition_lines(*, frequency_space: bool = False) -> list:
    """Get a list of common transition lines.

    Parameters
    ----------
    frequency_space : bool, optional
        Return the absorption edges in frequency space, by default False

    Returns
    -------
    list
        The absorption edges

    """
    lines = [
        [r"N \textsc{iii} / O \textsc{iii}", 305],
        [r"P \textsc{v}", 1118],
        [r"Ly$\alpha$ / N \textsc{v}", 1216],
        ["", 1242],
        [r"O \textsc{v} / Si \textsc{iv}", 1371],
        ["", 1400],
        [r"N \textsc{iv}", 1489],
        [r"C \textsc{iv}", 1548],
        ["", 1550],
        [r"He \textsc{ii}", 1640],
        [r"N \textsc{iii]}", 1750],
        [r"Al \textsc{iii}", 1854],
        [r"C \textsc{iii]}", 1908],
        [r"Mg \textsc{ii}", 2798],
        [r"Ca \textsc{ii}", 3934],
        ["", 3969],
        [r"H$_{\delta}$", 4101],
        [r"H$_{\gamma}$", 4340],
        [r"He \textsc{ii}", 4389],
        [r"He \textsc{ii}", 4686],
        [r"H$_{\beta}$", 4861],
        [r"He \textsc{i}", 5877],
        ["", 5897],
        [r"H$_{\alpha}$", 6564],
        [r"He \textsc{i}", 7067],
    ]

    return _convert_to_frequency_space(lines, frequency_space)


def get_common_absorption_edges(*, frequency_space: bool = False) -> list:
    """Get a list of common absorption edges.

    Parameters
    ----------
    frequency_space : bool, optional
        Return the absorption edges in frequency space, by default False

    Returns
    -------
    list
        The absorption edges

    """
    absorption_edges = [
        [r"O \textsc{viii}", 14],
        [r"O \textsc{vii}", 16],
        [r"O \textsc{vi} / O \textsc{v}", 98],
        ["", 105],
        [r"O \textsc{iv}", 160],
        [r"He \textsc{ii}", 227],
        [r"He \textsc{i}", 504],
        ["Lyman", 912],
        ["Balmer", 3646],
        ["Paschen", 8204],
    ]

    return _convert_to_frequency_space(absorption_edges, frequency_space)


def add_transition_labels_to_ax(  # noqa: PLR0913
    ax: plt.Axes,
    transitions_to_add: list,
    *,
    label_linestyle: str = "none",
    label_loc: float = 0.90,
    label_offset: float = 0,
    label_rotation: str = "vertical",
    label_fontsize: float = 15,
    whitespace_scale: float = 2,
) -> plt.Axes:
    """Add transition lines and labels to a plot.

    This function adds vertical lines and corresponding labels to a
    matplotlib Axes object, indicating specific transition points.

    Parameters
    ----------
    ax : plt.Axes
        The Axes object to update with transition lines and labels.
    transitions_to_add : list
        A list of transitions, where each transition is a list containing
        a label (str) and a position (float).
    label_linestyle : str, optional
        The style of the line to use for labels. Options are "none",
        "dashed", "thick", or "top". Default is "none".
    label_loc : float, optional
        The vertical location for the label text in Axes coordinates.
        Default is 0.90.
    label_offset : float, optional
        The offset to apply to the label position on the x-axis. Default is 0.
    label_rotation : str, optional
        The rotation angle for the label text. Default is "vertical".
    label_fontsize : float, optional
        The font size for the label text. Default is 15.
    whitespace_scale : float, optional
        The scale factor for the y-axis limits to add whitespace. Default is 2.

    Returns
    -------
    plt.Axes
        The updated Axes object with transition lines and labels.

    """
    num_lines = len(transitions_to_add)
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()

    for i in range(num_lines):
        label = transitions_to_add[i][0]
        x = transitions_to_add[i][1]

        if x < x_lims[0]:
            continue
        if x > x_lims[1]:
            continue

        if label_linestyle == "dashed":
            ax.axvline(x, linestyle="--", linewidth=0.5, color="k", zorder=1)
        elif label_linestyle == "thick":
            ax.axvline(x, linestyle="-", linewidth=2, color="k", zorder=1)
        elif label_linestyle == "top":
            raise NotImplementedError

        x = x - label_offset
        if ax.get_xscale() == "log":
            xnorm = (numpy.log10(x) - numpy.log10(x_lims[0])) / (numpy.log10(x_lims[1]) - numpy.log10(x_lims[0]))
        else:
            xnorm = (x - x_lims[0]) / (x_lims[1] - x_lims[0])

        ax.text(
            xnorm,
            label_loc,
            label,
            ha="center",
            va="center",
            rotation=label_rotation,
            fontsize=label_fontsize,
            transform=ax.transAxes,
        )

    ax.set_ylim(y_lims[0], y_lims[1] * whitespace_scale)

    return ax
