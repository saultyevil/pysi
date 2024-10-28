#!/usr/bin/env python

"""Module containing functions to add labels for common atomic transitions.
TODO: add type hinting to function signatures
"""

import numpy
from astropy.constants import c

from pysi.math.constants import ANGSTROM
from pysi.spec.enum import SpectrumSpectralAxis


def _convert_labels_to_frequency_space(lines, spectral_axis=None, spectrum=None):
    """Convert the given list of lines/edges from Angstrom to Hz.

    Parameters
    ----------
    lines: List[str, float]
        The list of labels to convert from wavelength to frequency space.
    spectrum: pysi.Spectrum
        A spectrum object, used to find the units of the spectrum.

    """
    if spectrum is None and spectral_axis is None:
        return lines

    if spectrum:
        spectral_axis = spectrum["spectral_axis"]

    if spectral_axis == SpectrumSpectralAxis.FREQUENCY:
        for i in range(len(lines)):
            lines[i][1] = c.cgs / (lines[i][1] * ANGSTROM)

    return lines


def add_line_ids(
    ax, lines, linestyle="none", ynorm=0.90, offset=0, rotation="vertical", fontsize=15, whitespace_scale=2
):
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
    offset: float or int [optional]
        The amount to offset line labels along the x-axis
    rotation: str [optional]
        Vertical or horizontal rotation for text ids
    fontsize: int [optional]
        The fontsize of the labels
    whitespace_scale: float [optional]
        The amount to scale the upper y limit of the plot by, to add whitespace
        for the line labels.

    Returns
    -------
    ax: plt.Axes
        The updated axes object.

    """
    nlines = len(lines)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    for i in range(nlines):
        label = lines[i][0]
        x = lines[i][1]

        if x < xlims[0]:
            continue
        if x > xlims[1]:
            continue

        if linestyle == "dashed":
            ax.axvline(x, linestyle="--", linewidth=0.5, color="k", zorder=1)
        elif linestyle == "thick":
            ax.axvline(x, linestyle="-", linewidth=2, color="k", zorder=1)
        elif linestyle == "top":
            raise NotImplementedError

        x = x - offset

        # Calculate the x location of the label in axes coordinates

        if ax.get_xscale() == "log":
            xnorm = (numpy.log10(x) - numpy.log10(xlims[0])) / (numpy.log10(xlims[1]) - numpy.log10(xlims[0]))
        else:
            xnorm = (x - xlims[0]) / (xlims[1] - xlims[0])

        ax.text(
            xnorm, ynorm, label, ha="center", va="center", rotation=rotation, fontsize=fontsize, transform=ax.transAxes
        )

    ax.set_ylim(ylims[0], ylims[1] * whitespace_scale)

    return ax


def common_lines(spectrum=None, spectral_axis=None):
    """Return a list containing the names of line transitions and the
    wavelength of the transition in Angstroms. Instead of returning the
    wavelength, the frequency can be returned instead. It is also possible to
    return in log space.

    Parameters
    ----------
    spectral_axis: pysi.Spectrum.SpectrumSpectralAxis
        The units of the spectral axis
    spectrum: pysi.Spectrum
        The spectrum object. Used to get the spectral axis units.

    Returns
    -------
    line: List[List[str, float]]
        A list of lists where each element of the list is the name of the
        transition/edge and the rest wavelength of that transition in
        Angstroms.

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

    return _convert_labels_to_frequency_space(lines, spectral_axis, spectrum)


def photoionization_edges(spectrum=None, spectral_axis=None):
    """Return a list containing the names of line transitions and the
    wavelength of the transition in Angstroms. Instead of returning the
    wavelength, the frequency can be returned instead. It is also possible to
    return in log space.

    Parameters
    ----------
    spectral_axis: pysi.Spectrum.SpectrumSpectralAxis
        The units of the spectral axis
    spectrum: Spectrum [optional]
        The spectrum object. Used to get the spectral axis units.

    Returns
    -------
    edges: List[List[str, float]]
        A list of lists where each element of the list is the name of the
        transition/edge and the rest wavelength of that transition in
        Angstroms.

    """
    edges = [
        [r"O \textsc{viii}", 14],
        [r"O \textsc{vii}", 16],
        [r"O \textsc{vi} / O \textsc{v}", 98],
        [r"", 105],
        [r"O \textsc{iv}", 160],
        [r"He \textsc{ii}", 227],
        [r"He \textsc{i}", 504],
        ["Lyman", 912],
        ["Balmer", 3646],
        ["Paschen", 8204],
    ]

    return _convert_labels_to_frequency_space(edges, spectral_axis, spectrum)
