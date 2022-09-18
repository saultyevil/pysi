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

import pypython
import pypython.wind.elements

PARTIALLY_INWIND = int(1)
INWIND = int(0)
WIND_CELL_TYPES = [INWIND, PARTIALLY_INWIND]


class WindProperties:
    """Base wind class for describing a wind object."""

    def __init__(self, root: str, directory: str = "", mask_value: Union[int, Callable[[int], bool]] = INWIND) -> None:
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

        self.nx = int(0)
        self.nz = int(0)
        self.parameters = pypython.AttributeDict()
        self.__original_parameters = None
        self.mask_value = mask_value

        self.read_in_wind_variables()
        self.read_in_wind_ions()
        self.read_in_cell_spectra()

        if mask_value or mask_value in WIND_CELL_TYPES:
            self.mask_arrays(mask_value)

    def get_variables_for_table(self, table: str) -> Tuple[list, dict]:
        """Get variables for a specific table type."""

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

        self.nx = int(numpy.max(self.parameters["i"]) + 1)
        if "z" in self.parameter_keys or "theta" in self.parameter_keys:
            self.nz = int(numpy.max(self.parameters["j"]) + 1)

    def read_in_wind_ions(self, elements: List[str] = pypython.wind.elements.ELEMENTS) -> None:
        """Read in the different ions in the wind.

        Parameters
        ----------
        elements: List[str], optional
            A list of atomic element names, e.g. H, He, whose ions in the wind
            will attempted to be read in. The default value is to try to read in
            all elements up to Cobalt.
        """

        n_read = 0

        # We need to loop over "frac" and "den" because ions are printed in
        # fractional populations or absolute density. The second loop is over
        # the elements passed to the function

        for ion_type in ["frac", "den"]:
            for element in elements:
                table_header, table_parameters = self.get_variables_for_table(f"{element}.{ion_type}")

                if not table_header:
                    continue

                for i, column in enumerate(table_header):
                    # the re.match here is to ignore any spatial parameters,
                    # e.g. x, z or i and j
                    if re.match("i[0-9]+", column) and column not in self.parameters:
                        self.parameters[f"{element}_{column}_{ion_type}"] = table_parameters[:, i]

                n_read += 1

        self.parameter_keys = self.parameters.keys()

        if n_read == 0:
            raise IOError("Have been unable to read in any wind ion tables")

    def read_in_cell_spectra(self):
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

            if "freq" not in self.parameters:
                self.parameters["freq"] = file_array[:, 0]
            if "spec" not in self.parameters:
                if self.nz > 0:
                    self.parameters["spec"] = numpy.zeros((self.nx, self.nz, len(file_array[:, 0])))
                else:
                    self.parameters["spec"] = numpy.zeros((self.nx, len(file_array[:, 0])))

            # Go through each coord string and figure out the coords, and place
            # the spectrum into 1d/2d array

            for i, coord_string in enumerate(file_header):
                coords = numpy.array(coord_string[1:].split("_"), dtype=numpy.int32)
                if self.nz > 0:
                    self.parameters["spec"][coords[0], coords[1], :] = file_array[:, i + 1]
                else:
                    self.parameters["spec"][coords[0], :] = file_array[:, i + 1]

    def mask_arrays(self, mask_value: Union[int, Callable[[int, int], bool]]):
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
            "freq",
            "spec",
        ]

        to_mask = [item for item in self.parameter_keys if item not in items_to_not_mask]

        # Finally go through each parameter in to_mask to create a masked array
        # for each item

        for item in to_mask:
            self.parameters[item] = numpy.ma.masked_where(mask_expression, self.parameters[item])

    def unmask_arrays(self):
        """Unmask the arrays.

        Uses a copy of the original table variables to revert the masking.
        """
        self.parameters = copy.deepcopy(self.__original_parameters)

    def smooth_spectra(self, amount: int):
        """Smooth the cell spectra.

        Uses a boxcar filter to smooth the cell spectra.

        Parameters
        ----------
        amount: int
            The pixel width for a boxcar filter.
        """
        for i in range(self.nx):
            for j in range(self.nz):
                self.parameters["spec"][i, j] = pypython.smooth_array(self.parameters["spec"][i, j], amount)

    def unsmooth_spectra(self):
        """Unsmooth the arrays.

        Uses a copy of the original spectra to revert the smoothing.
        """
        self.parameters["spec"] = copy.deepcopy(self.__original_parameters["spec"])

    def get_elem_number_from_ij(self, i: int, j: int):
        """Get the wind element number for a given i and j index.

        Used when indexing into a 1D array, such as in Python itself.

        Parameters
        ----------
        i: int
            The i-th index of the cell.
        j: int
            The j-th index of the cell.
        """
        return int(self.nz * i + j)

    def get_ij_from_elem_number(self, elem: int):
        """Get the i and j index for a given wind element number.

        Used when converting a wind element number into two indices for use
        in this package.

        Parameters
        ----------
        elem: int
            The element number.
        """
        i = int(elem / self.nz)
        j = int(elem - i * self.nz)

        return i, j

    # Special methods ----------------------------------------------------------

    def __getattr__(self, key: str):
        # if no frac or den is no specified for an ion, default to fractional
        # populations
        if re.match("[A-Z]_i[0-9]+", key):  # matches ion specification, e.g. C_i04
            if re.match("[A-Z]_i[0-9]+$", key):  # but no type specification at the end, e.g. C_i04_frac
                key += "_frac"

        return self.parameters.get(key)

    def __getitem__(self, key: str):
        # if no frac or den is no specified for an ion, default to fractional
        # populations
        if re.match("[A-Z]_i[0-9]+", key):  # matches ion specification, e.g. C_i04
            if re.match("[A-Z]_i[0-9]+$", key):  # but no type specification at the end, e.g. C_i04_frac
                key += "_frac"

        return self.parameters.get(key)

    def __str__(self):
        return textwrap.dedent(
            f"""
            This is a Wind object, containing data for:
            Root:       {self.root}
            Directory:  {self.directory}
            Mask value: {self.mask_value}
            """
        )
