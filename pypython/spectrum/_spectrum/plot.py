#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Class extension for plotting spectra."""

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt

from pypython.utilities import plot
from pypython.spectrum._spectrum import util


class SpectrumPlot(util.SpectrumUtil):
    """Class for plotting spectra."""

    def __init__(self, root: str, directory: Union[str, Path], **kwargs):
        """Initialize the class."""
        super().__init__(root, directory, **kwargs)

    def plot_extracted_spectrum(self, thing, fig=None, ax=None, figsize=(10, 6), axes_scales="loglog"):
        """Plot a spectrum, from the spectral cycles."""

        if not fig and not ax:
            fig, ax = plt.subplots(figsize=figsize)
        elif not fig and ax or fig and not ax:
            raise ValueError("fig and ax need to be supplied together")

        spectral_axis = self["spec"]["Lambda"]
        flux = self["spec"][thing]

        ax.plot(spectral_axis, flux, label=f"{thing}" + r"$^{\circ}$")
        ax.set_xlabel("Spectral axis")
        ax.set_ylabel("Flux")
        ax.legend()

        ax = plot.set_axes_scales(ax, axes_scales)
        fig = plot.finish_figure(fig, "spectrum")

        return fig, ax

    def plot_diagnostic_spectrum(self, thing):
        """Plot a diagnostic spectrum, from the ionization cycle."""

    @staticmethod
    def show_figures() -> None:
        """Show plotted figures."""
        plt.show()
