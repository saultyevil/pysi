"""Contains the user facing Spectrum class."""

from . import labels
from .enum import SpectrumSpectralAxis, SpectrumType, SpectrumUnits
from .model.plot import SpectrumPlot


class Spectrum(SpectrumPlot):
    """Main spectrum class."""
