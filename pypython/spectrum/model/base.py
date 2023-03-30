#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Base class for reading in a spectrum."""

import copy
from pathlib import Path
from typing import Union

import numpy

from pypython.utility import array
from pypython.spectrum import enum


class SpectrumBase:
    """The base class."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        root: str,
        directory: Union[str, Path] = Path("."),
        default_scaling: str = "log",
        default_spectrum: str = None,
        smooth_width: int = 0,
    ) -> None:
        """Initialize the class.

        Parameters
        ----------
        root: str
            The root name of the model.
        directory: Union[str, pathlib.Path]
            The directory containing the model.
        default_scaling: str
            The default bin scaling to use, either "log" or "lin".
        default_spectrum: str
            The spectrum to use as the current. Used as the default choice when
            indexing the class.
        boxcar_width: int
            The boxcar filter width to smooth spectra.
        """
        self.root = root
        self.directory = Path(directory)

        self.spectra = {"log": {}, "lin": {}}
        self.read_in_spectra()

        if default_scaling:
            self.scaling = default_scaling
        else:
            self.scaling = "log"

        if default_spectrum:
            self.current = default_spectrum
        else:
            self.current = list(self.spectra[self.scaling].keys())[0]

        self.set_scaling(self.scaling)
        self.set_spectrum(self.current)

        self.__original_spectra = None

        if smooth_width:
            self.smooth_all_spectra(smooth_width)

    def __getitem__(self, key):
        # this should catch "log" or "lin", so then you can do something like
        # Spectrum["log"]["spec"]
        if key in self.spectra:
            return self.spectra[key]
        # this should catch keys like "spec" or "spec_tot"
        if key in self.spectra[self.scaling]:
            return self.spectra[self.scaling][key]
        # if you just ask for "Freq." or "WCreated", etc, then this should catch
        # that
        return self.spectra[self.scaling][self.current][key]

    # Private methods ----------------------------------------------------------

    def __check_line_for_units(self, line: str, spec_type: str, extension: str) -> None:
        """_summary_

        Parameters
        ----------
        line : str
            _description_
        spec_type : str
            _description_
        extension : str
            _description_

        Raises
        ------
        KeyError
            _description_
        IOError
            _description_
        """
        if "Units:" in line:
            self.spectra[extension][spec_type]["units"] = enum.SpectrumUnits(line[4][1:-1])

            # convert old F_lm typo to new units
            if self.spectra[extension][spec_type]["units"] == enum.SpectrumUnits.F_LAM_LEGACY:
                self.spectra[extension][spec_type]["units"] = enum.SpectrumUnits.F_LAM

            # If a flux, get the distance from the same line
            if self.spectra[extension][spec_type]["units"] in [enum.SpectrumUnits.F_LAM, enum.SpectrumUnits.F_NU]:
                self.spectra[extension][spec_type]["distance"] = float(line[6])
            else:
                self.spectra[extension][spec_type]["distance"] = 0

    @staticmethod
    def _get_spectral_axis(units: enum.SpectrumUnits):
        """Get the spectral axis units of a spectrum.

        Determines the spectral axis, given the units of the spectrum.

        Parameters
        ----------
        units: SpectrumUnits
            The units of the spectrum.

        Returns
        -------
        spectral_axis: SpectrumSpectralAxis
            The spectral axis units of the spectrum
        """
        if units in [enum.SpectrumUnits.F_LAM, enum.SpectrumUnits.F_LAM_LEGACY, enum.SpectrumUnits.L_LAM]:
            return enum.SpectrumSpectralAxis.WAVELENGTH

        if units in [enum.SpectrumUnits.F_NU, enum.SpectrumUnits.L_NU]:
            return enum.SpectrumSpectralAxis.FREQUENCY

        return enum.SpectrumSpectralAxis.NONE

    # Public methods -----------------------------------------------------------

    def set_scaling(self, scaling: str) -> None:
        """Set the scaling of the spectrum.

        Parameters
        ----------
        scaling : str
            The scaling, either log or linear.
        """
        scaling_lower = scaling.lower()
        if scaling_lower not in ["log", "linear"]:
            raise ValueError(f"{scaling} is not a valid scaling. Choose either 'log' or 'linear'.")
        self.scaling = scaling_lower

    def set_spectrum(self, spec_key: str) -> None:
        """Set the default spectrum index.

        Parameters
        ----------
        spec_key: str
            The name of the spectrum to set default.
        """
        if spec_key not in self.spectra[self.scaling]:
            raise KeyError(f"{spec_key} not available in spectrum. Choose: {','.join(list(self.spectra.keys()))}")
        self.current = spec_key

    def smooth_all_spectra(self, boxcar_width: int) -> None:
        """Smooth all the spectra, using a boxcar filter.

        The loop over all of the spectra is quite complex... for example if you
        want the log spectrum for an observer at 10 degrees, you need to index
        in the following way: self.spectra["log"]["spec"]["10"]. This
        unforunately means we need a triple nested loop.

        Parameters
        ----------
        boxcar_width: int
            The width of the boxcar filter.
        """
        if not self.__original_spectra:
            self.__original_spectra = copy.deepcopy(self.spectra)

        for scaling, spec_types in self.spectra.items():
            for spec_type, spec in spec_types.items():
                for thing, values in spec.items():
                    if isinstance(values, numpy.ndarray) and thing not in ["Freq.", "Lambda"]:
                        # pylint: disable=unnecessary-dict-index-lookup
                        self.spectra[scaling][spec_type][thing] = array.smooth_array(values, boxcar_width)

    def unsmooth_all_spectra(self) -> None:
        """Reset the spectra to the original unsmoothed versions."""
        if not self.__original_spectra:
            return

        self.spectra = copy.deepcopy(self.__original_spectra)

    def read_in_spectra(self, files_to_read=("spec", "spec_tot", "spec_tot_wind", "spec_wind", "spec_tau")) -> None:
        """Read in all of the spectrum from the simulation.

        This function (by default) will read in both the linear and logarithmic
        spectra for the following types of spectra,
            1. spec;
            2. spec_tot;
            3. spec_tot_wind;
            4. spec_wind, and;
            5. spec_tau.
        The files which are read in is controlled by the "files_to_read"
        argument. If a file cannot be read in, then this is skipped without
        raising an exception.

        TODO, refactor to lower the number of branches

        Parameters
        ----------
        files_to_read: iterable
            An iterable containing the file extensions of the spectrum files
            to be read in. By default this includes spec, spec_tot,
            spec_tot_wind, spec_wind and spec_tau.

        Raises
        ------
        IOError
            Raised when no spectrum files could be found/read in.
        """
        n_read = 0

        for scaling in ["log", "lin"]:
            # try to read in each type of spectrum for log and lin scaling
            for spec_type in files_to_read:
                # TODO, need a better implementation to get log/linear
                if scaling == "log":
                    extension = "log_"
                else:
                    extension = ""

                path = Path(f"{self.directory}/{self.root}.{extension}{spec_type}")
                if not path.exists():
                    continue

                n_read += 1

                self.spectra[scaling][spec_type] = {
                    "units": enum.SpectrumUnits.NONE,
                    "spectral_axis": enum.SpectrumSpectralAxis.NONE,
                }

                with open(path, "r", encoding="utf-8") as file_in:
                    spectrum_file = file_in.readlines()

                spectrum_lines = []
                for line in spectrum_file:
                    line = line.strip().split()
                    # have to check before a comment, because the units line starts
                    # with a comment character
                    self.__check_line_for_units(line, spec_type, scaling)
                    if len(line) == 0 or line[0] == "#":
                        continue
                    spectrum_lines.append(line)

                if spec_type == "spec_tau":  # this is only created with frequency as the x axis
                    self.spectra[scaling][spec_type]["spectral_axis"] = enum.SpectrumSpectralAxis.FREQUENCY
                else:
                    self.spectra[scaling][spec_type]["spectral_axis"] = self._get_spectral_axis(
                        self.spectra[scaling][spec_type]["units"]
                    )

                # Extract the header columns of the spectrum. This assumes the first
                # read line in the spectrum is the header.
                spectrum_header = []
                for i, column_name in enumerate(spectrum_lines[0]):
                    if column_name[0] == "A":
                        j = column_name.find("P")
                        column_name = column_name[1:j].lstrip("0")  # remove leading 0's for, i.e., 01 degrees
                    spectrum_header.append(column_name)
                column_names = [column for column in spectrum_header if column not in ["Freq.", "Lambda"]]

                # spectrum[1:] is to cut out the header, which does not have a
                # comment character at the start so gets read in
                spectrum_columns = numpy.array(spectrum_lines[1:], dtype=numpy.float64)
                for i, column_name in enumerate(spectrum_header):
                    self.spectra[scaling][spec_type][column_name] = spectrum_columns[:, i]

                inclinations_in_spectrum = []
                for name in column_names:
                    if name.isdigit() and name not in inclinations_in_spectrum:
                        inclinations_in_spectrum.append(name)
                self.spectra[scaling][spec_type]["columns"] = tuple(column_names)
                self.spectra[scaling][spec_type]["inclinations"] = tuple(inclinations_in_spectrum)
                self.spectra[scaling][spec_type]["num_inclinations"] = len(inclinations_in_spectrum)

        if n_read == 0:
            raise IOError(f"Unable to open any spectrum files for {self.root} in {self.directory}")
