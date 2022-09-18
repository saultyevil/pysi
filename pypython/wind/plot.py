#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sub-class containing plotting functions.
"""

from typing import Tuple

import matplotlib.pyplot as plt

from pypython import plot
from pypython.wind import properties


class WindPlot(properties.WindProperties):
    """An extension to the WindGrid base class which adds various plotting
    functionality.
    """

    def __init__(self, root: str, directory: str, **kwargs):
        super().__init__(root, directory, **kwargs)

    def plot_parameter(self, thing: str) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a wind parameter.

        Parmeters
        ---------
        thing: str
            The name of the parameter to plot.

        Returns
        -------
        fig: plt.Figure
            The create Figure object, containing the axes.
        ax: plt.Axes
            The axes object for the plot.
        """
        fig, ax = plt.subplots(1, 1)

        return fig, ax

    def plot_cell_spectrum(self, i: int, j: int = 0, axes_scales: str = "loglog") -> Tuple[plt.Figure, plt.Axes]:
        """Plot a spectrum for a wind cell.

        Creates (and returns) a figure

        Parameters
        ----------
        i: int
            The i-th cell index
        j: int [optional]
            The j-th cell index
        axes_scales: str
            The scale types for each axis.

        Returns
        -------
        fig: plt.Figure
            The create Figure object, containing the axes.
        ax: plt.Axes
            The axes object for the plot.
        """
        fig, ax = plt.subplots(1, 1)

        if self.coord_type == "polar":
            spectrum = self.parameters["spec_flux"][i]
            freq = self.parameters["spec_freq"][i]
        else:
            spectrum = self.parameters["spec_flux"][i, j]
            freq = self.parameters["spec_freq"][i, j]

        ax.plot(freq, freq * spectrum, label="Spectrum")
        ax.set_xlabel(r"Rest-frame Frequency [$\nu$]")
        ax.set_ylabel(r"$\nu ~ J_{\nu}$ [ergs s$^{-1}$ cm$^{-2}$]")

        ax = plot.set_axes_scales(ax, axes_scales)
        fig = plot.finish_figure(fig, f"Cell ({i}, {j}) spectrum")

        return fig, ax

    def plot_cell_model(self, i: int, j: int = 0, axes_scales: str = "loglog") -> Tuple[plt.Figure, plt.Axes]:
        """Plot a spectrum for a wind cell.

        Creates (and returns) a figure

        Parameters
        ----------
        i: int
            The i-th cell index
        j: int [optional]
            The j-th cell index
        axes_scales: str
            The scale types for each axis.

        Returns
        -------
        fig: plt.Figure
            The create Figure object, containing the axes.
        ax: plt.Axes
            The axes object for the plot.
        """
        fig, ax = plt.subplots(1, 1)

        return fig, ax

    def show_figures(self) -> None:
        """Show any plot windows."""
        plt.show()
