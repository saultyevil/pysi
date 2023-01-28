#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Class extension for plotting spectra."""

from pathlib import Path
from typing import Union

from pypython.spectrum import util


class SpectrumPlot(util.SpectrumUtil):
    """Class for plotting spectra."""

    def __init__(self, root: str, directory: Union[str, Path], **kwargs):
        """Initialize the class."""
        super().__init__(root, directory, **kwargs)

    def plot_extracted_spectrum(self, thing):
        """Plot a spectrum, from the spectral cycles."""

    def plot_diagnostic_spectrum(self, thing):
        """Plot a diagnostic spectrum, from the ionization cycle."""
