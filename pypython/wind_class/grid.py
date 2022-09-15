#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The base class which contains variables containing the parameters of the wind,
as well as the most basic variables which describe the wind.
"""

import pathlib
import copy
import re
from typing import Tuple, Callable, Union, List
import numpy as np

import pypython
import pypython.wind.elements


PARTIALLY_INWIND = int(1)
INWIND = int(0)


class WindGrid:
    """Base wind class for describing a wind object."""

    def __init__(
        self, root: str, directory: str = "", mask_value: Union[int, Callable[[int, int], bool]] = INWIND
    ) -> None:
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
        self._original_parameters = None

        self.read_in_wind_variables()
        self.read_in_wind_ions()
        self.mask_arrays(mask_value)

    def get_variables_for_table(self, table: str) -> Tuple[list, dict]:
        """Get variables for a specific table type."""

        file_path = pathlib.Path(f"{self.directory}/{self.root}.{table}.txt")

        if file_path.is_file() is False:
            file_path = pathlib.Path(f"{str(file_path.parent)}/tables/{file_path.stem}.txt")
            if file_path.is_file() is False:
                return [], {}

        with open(file_path, "r", encoding="utf-8") as buffer:
            file_lines = [line.strip().split() for line in buffer.readlines()]

        if file_lines[0][0].isdigit() is True:
            raise Exception("File is formatted incorrectly and missing header")

        table_header = file_lines[0]
        table_parameters = np.array(file_lines[1:], dtype=np.float64)

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

        self.nx = int(np.max(self.parameters["i"]) + 1)
        if "z" in self.parameter_keys or "theta" in self.parameter_keys:
            self.nz = int(np.max(self.parameters["j"]) + 1)

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

        if self._original_parameters:
            self.parameters = copy.deepcopy(self._original_parameters)
        else:
            self._original_parameters = copy.deepcopy(self.parameters)

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
            "cell_spec",
        ]

        to_mask = [item for item in self.parameter_keys if item not in items_to_not_mask]

        # Finally go through each parameter in to_mask to create a masked array
        # for each item

        for item in to_mask:
            self.parameters[item] = np.ma.masked_where(mask_expression, self.parameters[item])

    def unmask_arrays(self):
        """Unmask the arrays.

        Uses a copy of the original table variables to revert the masking.
        """
        self.parameters = copy.deepcopy(self._original_parameters)
