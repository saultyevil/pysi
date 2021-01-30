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
        self, root: str, cd: str, projection: str = "rectilinear", mask_arrays: bool = True,
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
        if self.cd[-1] != "/":
            self.cd += "/"
        self.nx = 1
        self.nz = 1
        self.nelem = 1
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

        wind_all = []
        wind_columns = []

        # Read in each file, one by one, if they exist. Note that this makes
        # the assumption that all the tables are the same size.

        n_read = 0
        files_to_read = ["master", "heat", "gradient", "converge", "spec"]

        for file in files_to_read:
            file = self.cd + self.root + "." + file + ".txt"
            if not os.path.exists(file):
                # todo: throw some kinda warning, I guess?
                continue
            n_read += 1

            with open(file, "r") as f:
                wind_file = f.readlines()

            # Read in the wind_save table, ignoring empty lines and comments.
            # Each file is stored as a list of lines within a list, so a list
            # of lists.
            # todo: need some method to detect incorrect syntax

            wind_list = []

            for line in wind_file:
                line = line.strip()
                if delim:
                    line = line.split(delim)
                else:
                    line = line.split()
                if len(line) == 0 or line[0] == "#":
                    continue
                wind_list.append(line)

            # Keep track of each file header and add the wind lines for the
            # current file into wind_all, the list of lists, the master list

            if wind_list[0][0].isdigit() is False:
                wind_columns += wind_list[0]
            else:
                wind_columns += list(np.arrange(len(wind_list[0]), dtype=np.str))

            wind_all.append(np.array(wind_list[1:], dtype=np.float))

        if n_read == 0:
            print("Unable to open any wind save tables, try running windsave2table...")
            exit(1)  # todo: error code

        # Determine the number of nx and nz elements. There is a basic check to
        # only check for nz if a j column exists, i.e. if it is a 2d model.

        i_col = wind_columns.index("i")
        self.nx = int(np.max(wind_all[0][:, i_col]) + 1)
        if "j" in wind_columns:
            j_col = wind_columns.index("j")
            self.nz = int(np.max(wind_all[0][:, j_col]) + 1)
        self.nelem = int(self.nx * self.nz)  # the int() is for safety

        wind_all = np.hstack(wind_all)

        # Assign each column header to a key in the dictionary, ignoring any
        # column which is already in the dict and extract the x and z
        # coordinates

        for index, col in enumerate(wind_columns):
            if col in self.variables.keys():
                continue
            self.variables[col] = wind_all[:, index].reshape(self.nx, self.nz)
            self.columns += col,

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
