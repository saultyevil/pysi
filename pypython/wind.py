#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description of file.
"""

import os
import numpy as np
from typing import List, Union, Tuple
from .util import create_wind_save_table
from .physics.constants import PI


class Wind:
    """A class to store PYTHON wind_save in memory. Contains methods to extract
    variables, as well as convert various indices into other indices."""
    def __init__(
        self, root: str, cd: str, projection: str = "rectilinear", tabletype: str = None, mask_arrays: bool = False,
        delim: str = None
    ):
        """Initialize a Wind object...

        Parameters
        ----------
        root: str
            The root name of the Python simulation.
        cd: str
            The directory containing the model."""

        self.root = root
        self.cd = cd
        self.projection = projection

        # todo: instead of .all.complete.txt, actually just read in each file on a file by file basis
        # todo: so self.filepath will be redundant, we'll just need self.cd
        if self.cd[-1] != "/":
            self.cd += "/"
        self.filepath = self.cd + self.root + "."
        if tabletype:
            allowed = ["master", "heat", "gradient", "converge", "spec"]
            if tabletype not in allowed:
                print("{} is an unknown type of table".format(tabletype))
                print("allowed: ", allowed)
                exit(1)  # todo: error code
            self.filepath += tabletype
        else:
            self.filepath += "all.complete"
        self.filepath += ".txt"

        self.nx = 0
        self.nz = 0
        self.nelem = 0
        self.x_coords = ()
        self.z_coords = ()
        self.x_cen_coords = ()
        self.z_cen_coords = ()
        self.columns = ()
        self.wind_parameters = ()
        self.wind_ions = ()
        self.variables = {}

        # The next method reads in the wind and (probably) initializes the above
        # members

        self.read_wind(delim)
        self.read_ions(delim)
        if mask_arrays:
            self.created_masked_arrays()

    def read_wind(self, delim: str = None):
        """Read in the wind parameters.
        todo: add support for polar and spherical winds"""

        if not os.path.exists(self.filepath):
            create_wind_save_table(self.root, self.cd)
        with open(self.filepath, "r") as f:
            wind_file = f.readlines()

        # Read in the wind_save table, ignoring empty lines and comments
        # todo: need some method to detect incorrect syntax

        wind = []

        for line in wind_file:
            line = line.strip()
            if delim:
                line = line.split(delim)
            else:
                line = line.split()
            if len(line) == 0 or line[0] == "#":
                continue
            wind.append(line)

        # Now construct the table, extract the column names first. If there is
        # no header for some reason, then just number the columns instead

        if wind[0][0].isdigit() is False:
            self.wind_parameters = self.columns = tuple(wind[0])
        else:
            self.wind_parameters = self.columns = tuple(np.arange(len(wind[0]), dtype=np.str))
        wind = np.array(wind[1:], dtype=np.float)
        i_col = self.columns.index("i")
        self.nx = int(np.max(wind[:, i_col]) + 1)
        if "j" in self.columns:
            j_col = self.columns.index("j")
            self.nz = int(np.max(wind[:, j_col]) + 1)
        self.nelem = int(self.nx * self.nz)  # the int() is for safety
        for index, col in enumerate(self.columns):
            self.variables[col] = wind[:, index].reshape(self.nx, self.nz)

        # Now fill in the rest of the member variables
        # todo: try to figure out how this works for a 1d system too

        self.x_coords = tuple(np.unique(self.variables["x"]))
        self.x_cen_coords = tuple(np.unique(self.variables["xcen"]))
        if "z" in self.columns:
            self.z_coords = tuple(np.unique(self.variables["z"]))
            self.z_cen_coords = tuple(np.unique(self.variables["zcen"]))

    def read_ions(self, delim: str = None, ions_to_get: Union[List[str], Tuple[str], str] = None):
        """Read in the ion parameters.
        todo: add way to load in either densities or fractions"""

        if ions_to_get is None:
            ions_to_get = ("H", "He", "C", "N", "O", "Si", "Fe")
        else:
            if type(ions_to_get) not in [str, list, tuple]:
                print("ions_to_get should be a tuple/list of strings or a string")
                exit(1)  # todo: error code

        # Read in each ion file, one by one. The ions will be stored in the
        # self.variables dict as,
        # key = ion name
        # values = dict of ion keys, i.e. i_01, i_02, etc, and the values
        # in this dict will be the values of that ion
        # todo: way to handle frac or dens

        for ion in ions_to_get:
            ion = ion.capitalize()  # for safety...
            with open(self.cd + self.root + "." + ion + ".frac" + ".txt") as f:
                ion_file = f.readlines()

            self.wind_ions += (ion,)
            self.variables[ion] = {}

            # Read in ion densities/fractions.. this can be done in a list
            # comprehension, I think, but I want to skip commented out lines
            # and I think it's better(?) to do it this way

            wind = []

            for line in ion_file:
                if delim:
                    line = line.split(delim)
                else:
                    line = line.split()
                if len(line) == 0 or line[0] == "#":
                    continue
                wind.append(line)

            # Now construct the tables, how this is done is described in some
            # of the comments above
            # todo: I should check if the header exists first

            if wind[0][0].isdigit() is False:
                columns = tuple(wind[0])
                index = columns.index("i01")
            else:
                columns = tuple(np.arrange(len(wind[0]), dtype=np.str))
                index = 0
            columns = columns[index:]
            wind = np.array(wind[1:], dtype=np.float64)
            wind = wind[:, index:]
            for index, col in enumerate(columns):
                self.variables[ion][col] = wind[:, index].reshape(self.nx, self.nz)

    def join_windsave_tables(self):
        """Join wind save tables together to create an all.complete.txt file."""
        raise NotImplementedError

    def created_masked_arrays(self, to_mask: Union[str, List[str], Tuple[str]] = None):
        """Convert each array into a masked array, where the mask is defined by
        the inwind variable."""

        if to_mask is None:
            to_mask = list(self.columns)
            for item_to_remove in ["x", "z", "xcen", "zcen", "i", "j", "inwind"]:
                try:
                    to_mask.remove(item_to_remove)
                except ValueError:
                    continue
        else:
            if type(to_mask) not in [str, list, tuple]:
                print("to_mask should be a tuple/list of strings or a string")
                exit(1)  # todo: error code
        to_mask = tuple(to_mask)
        for col in to_mask:
            self.variables[col] = np.ma.masked_where(self.variables["inwind"] < 0, self.variables[col])

    def get_variable_along_sightline(self):
        raise NotImplementedError

    def get_sightline_coordinates(self, theta: float):
        """Get the vertical z coordinates for a given set of x coordinates and
        inclination angle.

        Parameters
        ----------
        theta: float
            The angle of the sight line to extract from. Given in degrees."""
        raise self.x_coords * np.tan(PI / 2 - np.deg2rad(theta))

    def get_elem_number_from_ij(self, i: int, j: int):
        """Get the wind element number for a given i and j index."""
        raise self.nz * i + j

    def get_ij_from_elem_number(self):
        raise NotImplementedError

    def __getitem__(self, key):
        """Return an array in the variables dictionary when indexing."""
        return self.variables[key]

    def __setitem__(self, key, value):
        """Set an array in the variables dictionary."""
        self.variables[key] = value

    def __str__(self):
        """Print basic details about the wind."""
        return "NotImplementedYet:-)"
