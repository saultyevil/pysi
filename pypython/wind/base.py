#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The base class which contains variables containing the parameters of the wind,
as well as the most basic variables which describe the wind.
"""

import copy
import pathlib
import re
import textwrap
from typing import Callable, List, Tuple, Union

import numpy
from astropy import constants as const

import pypython
from pypython.wind import elements, enum

PARTIALLY_INWIND = int(1)
INWIND = int(0)
WIND_CELL_TYPES = [INWIND, PARTIALLY_INWIND]


class WindBase:
    """Base wind class for describing a wind object."""

    # Special methods ----------------------------------------------------------

    def __init__(self, root: str, directory: str, mask_value: Union[int, Callable[[int], bool]] = INWIND) -> None:
        """Initialize the class.

        Parameters
        ----------
        root: str
            The root name of the simulation.
        directory: str
            The directory file path containing the simulation.
        mask_value: int, Callable[int, int]
            The value of inwind to create a masked array with.
        """
        self.root = str(root)
        self.directory = pathlib.Path(directory)

        self.n_x = int(0)
        self.n_z = int(0)
        self.n_cells = int(0)
        self.coord_type = enum.CoordSystem.UNKNOWN
        self.n_model_freq_bands = int(0)

        self.parameters = {}
        self.__original_parameters = None
        self.mask_value = mask_value

        # These units are the default in python. In a higher level class, you
        # should be able to modify the units

        self.spatial_units = enum.DistanceUnits.CENTIMETRES
        self.velocity_units = enum.VelocityUnits.CENTIMETRES_PER_SECOND

        # Read in all the variables, spectra, etc.

        self.read_in_wind_variables()
        self.read_in_wind_ions()

        # TODO: these crash if the file is missing, or not formatted correctly
        self.read_in_cell_spectra()
        self.read_in_cell_models()

        # Create masked arrays

        if mask_value or mask_value in WIND_CELL_TYPES:
            self.mask_arrays(mask_value)

    def __getitem__(self, key: str) -> numpy.ndarray:
        # if no frac or den is no specified for an ion, default to fractional
        # populations
        if re.match("[A-Z]_i[0-9]+", key):  # matches ion specification, e.g. C_i04
            if re.match("[A-Z]_i[0-9]+$", key):  # but no type specification at the end, e.g. C_i04_frac
                key += "_frac"

        return self.parameters.get(key)

    def __str__(self) -> str:
        return textwrap.dedent(
            f"""
            This is a Wind object, containing data for:
            Root:       {self.root}
            Directory:  {self.directory}
            Mask value: {self.mask_value}
            Coord type: {self.coord_type}
            """
        )

    # Public methods -----------------------------------------------------------

    def get_elem_number_from_ij(self, i: int, j: int) -> int:
        """Get the wind element number for a given i and j index.

        Used when indexing into a 1D array, such as in Python itself.

        Parameters
        ----------
        i: int
            The i-th index of the cell.
        j: int
            The j-th index of the cell.
        """
        return int(self.n_z * i + j)

    def get_ij_from_elem_number(self, elem: int) -> Tuple[int, int]:
        """Get the i and j index for a given wind element number.

        Used when converting a wind element number into two indices for use
        in this package.

        Parameters
        ----------
        elem: int
            The element number.
        """
        i = int(elem / self.n_z)
        j = int(elem - i * self.n_z)

        return i, j

    def get_variables_for_table(self, table: str) -> Tuple[List[str], numpy.ndarray]:
        """Get variables for a specific table type.

        Parameters
        ----------
        table: str
            The type of table to read in, e.g. master, heat, etc.

        Returns
        -------
        table_header: List[str]
            The table headers for each column.
        table_parameters: numpy.ndarray
            An array of the numerical values of the table.
        """

        file_path = pathlib.Path(f"{self.directory}/{self.root}.{table}.txt")

        if file_path.is_file() is False:
            file_path = pathlib.Path(f"{str(file_path.parent)}/tables/{file_path.stem}.txt")
            if file_path.is_file() is False:
                return [], {}

        with open(file_path, "r", encoding="utf-8") as buffer:
            file_lines = [line.strip().split() for line in buffer.readlines() if not line.startswith("#")]

        if file_lines[0][0].isdigit() is True:
            raise Exception("File is formatted incorrectly and missing header")

        table_header = file_lines[0]
        table_parameters = numpy.array(file_lines[1:], dtype=numpy.float64)

        return table_header, table_parameters

    def mask_arrays(self, mask_value: Union[int, Callable[[int, int], bool]]) -> None:
        """Create masked parameter arrays.

        It is possible to remask the parameter arrays by calling this function
        again. You do not need to unmask the arrays first, this function will
        do it.

        Parameters
        ----------
        mask_value: int, Callable[int, int]
            The value of inwind to create a masked array with.
        """

        self.mask_value = mask_value

        # Create the expression to mask with, this is either a callable, such
        # as a lambda function, or an int corresponding to what we want to keep

        if callable(mask_value):
            mask_expression = mask_value(self.parameters["inwind"])
        else:
            if not isinstance(mask_value, int):
                raise ValueError("The mask_value parameter should be a callable, or an int.")
            mask_expression = self.parameters["inwind"] != mask_value

        # Create a backup of the unmasked array, so we can restore if we want
        # to remask, or unmask

        if self.__original_parameters:
            self.parameters = copy.deepcopy(self.__original_parameters)
        else:
            self.__original_parameters = copy.deepcopy(self.parameters)

        # Create a list of items we DO NOT want to mask, otherwise plots will
        # look very strange

        items_to_not_mask = [
            "x",
            "z",
            "r",
            "theta",
            "xcen",
            "zcen",
            "rcen",
            "theta_cen",
            "i",
            "j",
            "inwind",
            "spec_freq",
            "spec_flux",
            "model_freq",
            "model_flux",
        ]

        to_mask = [item for item in self.parameter_keys if item not in items_to_not_mask]

        # Finally go through each parameter in to_mask to create a masked array
        # for each item

        for item in to_mask:
            self.parameters[item] = numpy.ma.masked_where(mask_expression, self.parameters[item])

    def read_in_cell_models(self, n_freq_bins_per_band: int = 250) -> None:
        """Read in the J_nu models for each cell.

        Parameters
        ----------
        n_freq_bins: int
            The number of frequency bins to use for the model.
        """

        table_header, models = self.get_variables_for_table("spec")
        model_array = numpy.array(models, dtype=numpy.float64)
        self.n_model_freq_bands = n_bands = int(numpy.max(model_array[:, table_header.index("nband")])) + 1

        if "model_freq" not in self.parameters:
            if self.n_z > 0:
                self.parameters["model_freq"] = numpy.zeros((self.n_x, self.n_z), dtype=list)
            else:
                self.parameters["model_freq"] = numpy.zeros((self.n_x, n_bands), dtype=list)

        if "model_flux" not in self.parameters:
            if self.n_z > 0:
                self.parameters["model_flux"] = numpy.zeros((self.n_x, self.n_z), dtype=list)
            else:
                self.parameters["model_flux"] = numpy.zeros(self.n_x, dtype=list)

        # The next block will loop over each cell and constuct a model for each
        # frequency band, and put that (and the frequency bins) into an array
        # for each cell.

        for i in range(self.n_cells):
            cell_frequency = []
            cell_flux = []

            for j in range(n_bands):

                # create a dict of the parameters for band j, the table is a
                # flat list of the parameters for cell 1, 2, 3, ... for BAND 0,
                # and then the next section is the parameters for cell 1, 2,
                # 3... for BAND 1. So we have to do some funky indexing to get
                # to the correct row element in model_array

                parameters_for_band_j = {
                    col: model_array[i + j * self.n_cells, k] for k, col in enumerate(table_header)
                }

                band_freq_min = parameters_for_band_j["fmin"]
                band_freq_max = parameters_for_band_j["fmax"]

                # check first that the band hasn't broken in python or if the
                # band is empty as when empty fmin == fmax == 0

                if band_freq_max > band_freq_min:
                    band_frequency_bins = numpy.logspace(
                        numpy.log10(band_freq_min), numpy.log10(band_freq_max), n_freq_bins_per_band
                    )

                    # model_type 1 == powerlaw model, otherwise 2 == exponential
                    # this is the noclumentaure used in python :-)

                    model_type = parameters_for_band_j["spec_mod_type"]

                    if model_type == 1:
                        band_flux = 10 ** (
                            parameters_for_band_j["pl_log_w"]
                            + numpy.log10(band_frequency_bins) * parameters_for_band_j["pl_alpha"]
                        )
                    else:
                        band_flux = parameters_for_band_j["exp_w"] * numpy.exp(
                            (-1 * const.h.cgs.value * band_frequency_bins)
                            / (parameters_for_band_j["exp_temp"] * const.k_B.cgs.value)
                        )

                    cell_frequency.append(band_frequency_bins)
                    cell_flux.append(band_flux)

            if len(cell_flux) != 0:
                i_cell, j_cell = self.get_ij_from_elem_number(i)
                self.parameters["model_freq"][i_cell, j_cell] = numpy.hstack(cell_frequency)
                self.parameters["model_flux"][i_cell, j_cell] = numpy.hstack(cell_flux)

    def read_in_cell_spectra(self) -> None:
        """Read in the cell spectra"""

        spec_table_files = pypython.find("*xspec.*.txt", self.directory)
        if len(spec_table_files) == 0:
            return

        for file in spec_table_files:
            with open(file, "r", encoding="utf-8") as buffer:
                file_lines = [line.strip().split() for line in buffer.readlines() if not line.startswith("#")]

            if file_lines[0][0].isdigit() is True:
                raise Exception("File is formatted incorrectly and missing header")

            # Get the header and turn the rest of the file into an array, as
            # this makes indexing what we want a lot easier

            file_header = file_lines[0][1:]  # 1: skips the Freq.
            file_array = numpy.array(file_lines[1:], dtype=numpy.float64)

            # Populate the parameters dict

            if "spec_freq" not in self.parameters:
                if self.n_z > 0:
                    self.parameters["spec_freq"] = numpy.zeros((self.n_x, self.n_z, len(file_array[:, 0])))
                else:
                    self.parameters["spec_freq"] = numpy.zeros((self.n_x, len(file_array[:, 0])))

            if "spec_flux" not in self.parameters:
                if self.n_z > 0:
                    self.parameters["spec_flux"] = numpy.zeros((self.n_x, self.n_z, len(file_array[:, 0])))
                else:
                    self.parameters["spec_flux"] = numpy.zeros((self.n_x, len(file_array[:, 0])))

            # Go through each coord string and figure out the coords, and place
            # the spectrum into 1d/2d array

            for i, coord_string in enumerate(file_header):
                coords = numpy.array(coord_string[1:].split("_"), dtype=numpy.int32)
                if self.n_z > 0:
                    self.parameters["spec_flux"][coords[0], coords[1], :] = file_array[:, i + 1]
                    self.parameters["spec_freq"][coords[0], coords[1], :] = file_array[:, 0]
                else:
                    self.parameters["spec_flux"][coords[0], :] = file_array[:, i + 1]
                    self.parameters["spec_freq"][coords[0], :] = file_array[:, 0]

    def read_in_wind_ions(self, elements_to_read: List[str] = elements.ELEMENTS) -> None:
        """Read in the different ions in the wind.

        Parameters
        ----------
        elements_to_read: List[str], optional
            A list of atomic element names, e.g. H, He, whose ions in the wind
            will attempted to be read in. The default value is to try to read in
            all elements up to Cobalt.
        """

        n_read = 0

        # We need to loop over "frac" and "den" because ions are printed in
        # fractional populations or absolute density. The second loop is over
        # the elements passed to the function

        for ion_type in ["frac", "den"]:
            for element in elements_to_read:
                table_header, table_parameters = self.get_variables_for_table(f"{element}.{ion_type}")

                if not table_header:
                    continue

                for i, column in enumerate(table_header):
                    # the re.match here is to ignore any spatial parameters,
                    # e.g. x, z or i and j
                    if re.match("i[0-9]+", column) and column not in self.parameters:
                        self.parameters[f"{element}_{column}_{ion_type}"] = table_parameters[:, i].reshape(
                            self.n_x, self.n_z
                        )

                n_read += 1

        self.parameter_keys = self.parameters.keys()

        if n_read == 0:
            raise IOError("Have been unable to read in any wind ion tables")

    def read_in_wind_variables(self) -> None:
        """Read in the different parameters which describe state of the wind."""

        n_read = 0

        for table in ["master", "heat", "gradient", "converge"]:
            table_header, table_parameters = self.get_variables_for_table(table)

            if not table_header:
                continue

            for i, column in enumerate(table_header):
                if column not in self.parameters:
                    self.parameters[column] = table_parameters[:, i]

            n_read += 1

        if n_read == 0:
            raise IOError("Have been unable to read in any wind parameter tables")

        self.parameter_keys = self.parameters.keys()

        # Determine the number of cells in the x and z direction, and the
        # coordinate type of the grid

        self.n_x = int(numpy.max(self.parameters["i"]) + 1)
        if "z" in self.parameter_keys or "theta" in self.parameter_keys:
            self.n_z = int(numpy.max(self.parameters["j"]) + 1)
        self.n_cells = int(self.n_x * self.n_z)

        if "r" in self.parameters and "theta" in self.parameters:
            self.coord_type = enum.CoordSystem.POLAR
        elif "r" in self.parameters:
            self.coord_type = enum.CoordSystem.SPHERICAL
        else:
            self.coord_type = enum.CoordSystem.CYLINDRICAL

        # Reshape the parameters into (nx, nz) which are currently just flat
        # arrays

        self.parameters = {col: val.reshape(self.n_x, self.n_z) for col, val in self.parameters.items()}

    def unmask_arrays(self) -> None:
        """Unmask the arrays.

        Uses a copy of the original table variables to revert the masking.
        """
        self.parameters = copy.deepcopy(self.__original_parameters)
