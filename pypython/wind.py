#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description of file.
"""

import os
import numpy as np
from .util import create_wind_save_table
from .physics.constants import PI


class Wind:
    """A class to store PYTHON wind_save in memory. Contains methods to extract
    variables, as well as convert various indices into other indices."""
    def __init__(self, root: str, cd: str, tabletype: str = None, delim: str = None):
        """Initialize a Wind object...

        Parameters
        ----------
        root: str
            The root name of the Python simulation.
        cd: str
            The directory containing the model."""

        self.root = root
        self.cd = cd

        if self.cd[-1] != "/":
            self.cd += "/"
        self.filepath = self.cd + self.root + "."
        if tabletype:
            allowed = ["master", "heat", "gradient", "converge", "spec"]
            if tabletype not in allowed:
                print("{} is an unknown type of table".format(tabletype))
                print("allowed: ", allowed)
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
        self.projection = ""
        self.columns = ()
        self.variables = {}

        # The next method reads in the wind and (probably) initializes the above
        # members

        self.read_wind(delim)

    def read_wind(self, delim: str = None):
        """Read in a wind_save table."""

        if os.path.exists(self.filepath):
            wind_file = open(self.filepath, "r")
        else:
            create_wind_save_table(self.root, self.cd)
            wind_file = open(self.filepath, "r")

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
            self.columns = tuple(wind[0])
        else:
            self.columns = tuple(np.arange(len(wind[0]), dtype=np.str))
        wind = np.array(wind[1:], dtype=np.float)
        for index, col in enumerate(self.columns):
            self.variables[col] = wind[:, index]

        # Now try to determine the rest of the member variables
        # todo: try to figure out how this works for a 1d system too

        self.nx = np.max(self.variables["nx"]) + 1
        try:
            self.nz = np.max(self.variables["nz"] + 1)
        except KeyError:
            self.nz = 0
        self.nelem = self.nx * self.nz

    def get_variable(self, thing: str, thing_type: str):
        raise NotImplementedError

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
        return "NotImplemented"
