#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Base class for reading in a spectrum."""

from pathlib import Path
from typing import Union


class SpectrumBase:
    """The base class."""

    def __init__(
        self, root: str, directory: Union[str, Path] = Path("."), default_spectrum: str = None, boxcar_width: int = 5
    ) -> None:
        """Initialize the class.

        Parameters
        ----------
        root: str
            The root name of the model.
        directory: Union[str, patlib.Path]
            The directory contaning the model.
        default_spectrum: str
            The spectrum to use as the current. Used as the default choice when
            indexing the class.
        boxcar_width: int
            The boxcar filter width to smooth spectra.
        """
        self.root = root
        self.directory = Path(directory)

        self.spectrum = {}

        if default_spectrum:
            self.current = default_spectrum
        else:
            self.current = None

        self.set_spectrum(self.current)

        if boxcar_width:
            self.smooth_all_spectra(boxcar_width)

    def set_spectrum(self, spec_key: str) -> None:
        """Set the default spectrum index.

        Parameters
        ----------
        spec_key: str
            The name of the spectrum to set default.
        """
        if spec_key not in self.spectrum.keys():
            raise KeyError(f"{spec_key} not available in spectrum. Allowed: {','.join(list(self.spectrum.keys()))}")

        self.current = spec_key

    def smooth_all_spectra(self, boxcar_width: int) -> None:
        """Smooth all the spectra, using a boxcar filter.

        Parameters
        ----------
        boxcar_width: int
            The width of the boxcar filter.
        """

    def smooth_spectrum(self, spec_key: str, boxcar_width: int) -> None:
        """Smooth a spectrum, using a boxcar filter.

        Parameters
        ----------
        spec_key: str
            The name of the spectrum to smooth.
        boxcar_width: int
            The width of the boxcar filter.
        """

    def read_in_spectra(self):
        """Read in the spectra."""
