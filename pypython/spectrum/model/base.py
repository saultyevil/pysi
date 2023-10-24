#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Base class for reading in a spectrum."""

import copy
from pathlib import Path
from typing import Union, List

import numpy
from astropy import constants

from pypython.utility import array
from pypython.spectrum import enum


class SpectrumBase:
    """The base class."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        root: str,
        directory: Union[str, Path] = Path("."),
        default_scale: str = "log",
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
        default_scale: str
            The default bin scaling to use, either "log" or "lin".
        default_spectrum: str
            The spectrum to use as the current. Used as the default choice when
            indexing the class.
        smooth_width: int
            The boxcar filter width to smooth spectra.
        """
        self.root = root
        self.directory = Path(directory)

        self.spectra = {"log": {}, "lin": {}}
        self.load_spectra()

        if default_scale:
            self.scaling = default_scale
        else:
            self.scaling = "log"

        if default_spectrum:
            self.current = default_spectrum
        else:
            self.current = list(self.spectra[self.scaling].keys())[0]

        self.set_scale(self.scaling)
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

    def __check_line_for_units(self, line: List[str], spec_type: str, extension: str) -> None:
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
                try:
                    self.spectra[extension][spec_type]["distance"] = float(line[6])  # parsecs
                except ValueError:
                    self.spectra[extension][spec_type]["distance"] = 0
            else:
                self.spectra[extension][spec_type]["distance"] = 0

    def __get_file_contents(self, file: Path, spec_type: str, scale: str) -> List[List[str]]:
        """Returns the contents of a file as a list of words.

        In addition to reading in the contents of the file, this method will
        also set the units of the spectrum.

        Parameters
        ----------
        file: Path
            A pathlib.Path object to the file to read.
        spec_type: str
            The extension of the spectrum file.
        scale: str
            The scale of the spectrum.

        Returns
        -------
        List[List[str]]
            The contents of the file, as a list of the sentences split.
        """
        with open(file, "r", encoding="utf-8") as file_in:
            spectrum_file = file_in.readlines()

        contents = []
        for line in spectrum_file:
            line = line.strip().split()
            # have to check before a comment, because the units line starts
            # with a comment character
            self.__check_line_for_units(line, spec_type, scale)
            if len(line) == 0 or line[0] == "#":
                continue
            contents.append(line)

        return contents

    def __get_spectrum(self, spec_type: str, scale: str) -> None:
        """Read in a spectrum of `scale_spec_type`.

        Parameters
        ----------
        spec_type: str
            The extension of the spectrum to read.
        scale: str
            The scale of the spectrum, either log or linear
        """
        if scale == "log":
            extension = "log_"
        else:
            extension = ""
        extension += spec_type

        file = Path(f"{self.directory}/{self.root}.{extension}")
        if not file.exists():
            return

        self.spectra[scale][spec_type] = {
            "units": enum.SpectrumUnits.NONE,
            "spectral_axis": enum.SpectrumSpectralAxis.NONE,
        }

        spectrum_lines = self.__get_file_contents(file, spec_type, scale)

        if spec_type == "spec_tau":  # this is only created with frequency as the x-axis
            self.spectra[scale][spec_type]["spectral_axis"] = enum.SpectrumSpectralAxis.FREQUENCY
        else:
            self.spectra[scale][spec_type]["spectral_axis"] = self.__get_spectral_axis(
                self.spectra[scale][spec_type]["units"]
            )

        self.__populate_columns(spec_type, scale, spectrum_lines)

    def __populate_columns(self, spec_type: str, scale: str, spectrum_lines: List[List[str]]):
        """Get the headers for a spectrum.

        This function will retrieve the column headers in the spectrum file,
        such as the different "Spectrum Components" and the inclination angles
        for spectra generated by extract.

        Parameters
        ----------
        spec_type: str
            The file extension of the spectrum.
        scale: str
            The scale of the spectrum, either log or linear.
        spectrum_lines: List[List[str]
            A list of the contents of the spectrum file, where the line has been
            split into a list of words.
        """
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
            self.spectra[scale][spec_type][column_name] = spectrum_columns[:, i]

        inclinations_in_spectrum = []
        for name in column_names:
            if name.isdigit() and name not in inclinations_in_spectrum:
                inclinations_in_spectrum.append(name)

        self.spectra[scale][spec_type]["columns"] = tuple(column_names)
        self.spectra[scale][spec_type]["inclinations"] = tuple(inclinations_in_spectrum)
        self.spectra[scale][spec_type]["num_inclinations"] = len(inclinations_in_spectrum)

    @staticmethod
    def __get_spectral_axis(units: enum.SpectrumUnits):
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

    def apply_to_spectra(self, fn: callable) -> None:
        for scale in self.spectra:
            for spec_type, spectrum in self.spectra[scale].items():
                fn(spec_type, spectrum)

    def convert_flux_to_luminosity(self) -> None:
        """Switch to luminosity units instead of flux.

        When called, this method will iterate over all spectra in the class. The
        distance of the original spectra will not be modified.
        """

        def convert(spec_type, spectrum):
            if spec_type == "spec_tau":
                return
            if spectrum["units"] not in [
                enum.SpectrumUnits.F_LAM,
                enum.SpectrumUnits.F_LAM_LEGACY,
                enum.SpectrumUnits.F_NU,
            ]:
                return
            for column in spectrum["columns"]:
                spectrum[column] *= 4 * numpy.pi * (spectrum["distance"] * constants.pc.cgs.value) ** 2
                if spectrum["units"] in [enum.SpectrumUnits.F_LAM, enum.SpectrumUnits.F_LAM_LEGACY]:
                    spectrum["units"] = enum.SpectrumUnits.L_LAM
                else:
                    spectrum["units"] = enum.SpectrumUnits.L_NU

        self.apply_to_spectra(convert)

    def convert_luminosity_to_flux(self, distance: float | int = None) -> None:
        """Switch to flux units instead of luminosity.

        When called, this method will iterate over all spectra in the class. The
        distance of the spectra is set either through a variable or the method
        will attempt to use a value set previously.
        """

        def convert(spec_type, spectrum):
            if spec_type == "spec_tau":
                return
            if spectrum["units"] not in [
                enum.SpectrumUnits.L_LAM,
                enum.SpectrumUnits.L_NU,
            ]:
                return
            for column in spectrum["columns"]:
                if distance:
                    my_distance = distance
                    spectrum["distance"] = distance
                else:
                    my_distance = spectrum["distance"]
                spectrum[column] /= 4 * numpy.pi * (my_distance * constants.pc.cgs.value) ** 2
                if spectrum["units"] in [enum.SpectrumUnits.L_LAM, enum.SpectrumUnits.L_LAM]:
                    spectrum["units"] = enum.SpectrumUnits.F_LAM
                else:
                    spectrum["units"] = enum.SpectrumUnits.F_NU

        self.apply_to_spectra(convert)

    def set_distance(self, distance: float | int) -> None:
        """Set the distance of the spectrum.

        The distance will be scaled in parsecs from Earth. This function will
        not change the distance for any spectra which are in luminosity units.

        Parameters
        ----------
        distance : float
            The distance in parsecs.
        """

        if not isinstance(distance, (float, int)):
            raise ValueError(f"{distance} is not a valid type for distance")

        def convert(spec_type, spectrum):
            if spec_type in ["spec_tau"]:
                return
            if spectrum["units"] in [
                enum.SpectrumUnits.L_NU,
                enum.SpectrumUnits.L_LAM,
            ]:
                return
            for key in spectrum["columns"]:
                spectrum[key] *= spectrum["distance"] ** 2 / distance**2
            spectrum["distance"] = distance

        self.apply_to_spectra(convert)

    def set_scale(self, scaling: str) -> None:
        """Set the scaling of the spectrum.

        Parameters
        ----------
        scaling : str
            The scaling, either log or linear.
        """
        scaling_lower = scaling.lower()
        if scaling_lower not in ["log", "linear"]:
            raise ValueError(f"{scaling} is not a valid scale. Choose either 'log' or 'linear'.")
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

    def show_available(self) -> List[str]:
        """Prints and return a list of the available spectra.

        Returns
        -------
        List[str]
            A list containing the names of the spectra which have been
            read in.
        """
        add_scale = lambda name, scale: f"{scale}_{name}"
        available_list = tuple(
            [add_scale(key) for key in self.spectra["log"].keys()]
            + [add_scale(key) for key in self.spectra["lin"].keys()]
        )

        print(
            "The following spectra are available:\n",
            ", ".join(available_list),
        )

        return available_list

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

    def load_spectra(self, files_to_read=("spec", "spec_tot", "spec_tot_wind", "spec_wind", "spec_tau")) -> None:
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

        for scale in ["log", "lin"]:
            # try to read in each type of spectrum for log and lin scaling
            for spec_type in files_to_read:
                try:
                    self.__get_spectrum(spec_type, scale)
                    n_read += 1
                except IOError:
                    continue

        if n_read == 0:
            raise IOError(f"Unable to open any spectrum files for {self.root} in {self.directory}")
