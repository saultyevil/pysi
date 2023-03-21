#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Class extension for plotting spectra."""

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt

from pypython.utilities import plot
from pypython.spectrum._spectrum import util
from pypython.spectrum._spectrum import plot_util


class SpectrumPlot(util.SpectrumUtil):
    """Class for plotting spectra."""

    def __init__(self, root: str, directory: Union[str, Path], **kwargs):
        """Initialize the class."""
        super().__init__(root, directory, **kwargs)

    def plot_extracted_spectrum(self, thing, fig=None, ax=None, axes_scales="loglog"):
        """Plot a spectrum, from the spectral cycles."""

        if not fig and not ax:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        elif not fig and ax or fig and not ax:
            raise ValueError("fig and ax need to be supplied together")

        ax = plot_util.plot_spectrum(ax, self, thing, None, None, 1.0, "loglog", False)

        # if label_lines:
        #     ax = add_line_ids(ax, common_lines(spectrum=spectrum), "none")

        fig = plot.finish_figure(fig, "spectrum")

        return fig, ax

    def plot_diagnostic_spectrum(self, thing):
        """Plot a diagnostic spectrum, from the ionization cycle."""

    @staticmethod
    def show_figures() -> None:
        """Show plotted figures."""
        plt.show()
