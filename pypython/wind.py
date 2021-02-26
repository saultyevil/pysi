#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description of file.
"""

import os
import numpy as np
from typing import List, Union, Tuple
from .physics.constants import PI, CMS_TO_KMS, C
from .util import get_array_index
from .extrautil import vector


class Wind1D:
    """A class to store a 1D Python wind_save file. Contains methods to extract
    variables, etc."""

    def __init__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class Wind2D:
    """A class to store a 2D Python wind tables. Contains methods to extract
    variables, as well as convert various indices into other indices."""

    def __init__(
        self, root: str, cd: str = ".", coordinate_system: str = "rectilinear", velocity_units: str = "kms",
        mask_cells: bool = True, delim: str = None
    ):
        """Description of function.

        Parameters
        ----------
        root: str
            The root name of the Python simulation.
        cd: str
            The directory containing the model.
        coordinate_system: str [optional]
            The coordinate system of the wind: rectilinear or polar.
        mask_cells: bool [optional]
            Store the wind parameters as masked arrays.
        delim: str [optional]
            The delimiter used in the wind table files."""

        self.root = root
        self.cd = cd
        self.projection = coordinate_system
        if self.cd[-1] != "/":
            self.cd += "/"
        self.nx = 1
        self.nz = 1
        self.nelem = 1
        self.x_coords = ()
        self.z_coords = ()
        self.x_cen_coords = ()
        self.z_cen_coords = ()
        self.wind_parameters = ()
        self.wind_elements = ()
        self.variables = {}

        # Set up the velocity units and conversion factors

        if velocity_units not in ["cms", "kms", "c"]:
            print("unknown units: " + velocity_units)
            print("allowed: ['kms', 'cms', 'c']")
            exit(1)  # todo: error code
        self.velocity_units = velocity_units
        if velocity_units == "kms":
            self.velocity_conversion_factor = CMS_TO_KMS
        elif velocity_units == "cms":
            self.velocity_conversion_factor = 1
        else:
            self.velocity_conversion_factor = 1 / C

        # The next method reads in the wind and (probably) initializes the above
        # members

        self.read_in_wind_parameters(delim)
        self.read_in_wind_ions(delim)
        self.columns = self.wind_parameters + self.wind_elements

        # Convert velocity into desired units and also calculate the cylindrical
        # velocities

        self.project_cartesian_velocity_to_cylindrical()
        self.variables["v_x"] *= self.velocity_conversion_factor
        self.variables["v_y"] *= self.velocity_conversion_factor
        self.variables["v_z"] *= self.velocity_conversion_factor

        # Create masked cells, if that's the users deepest desire for their
        # data

        if mask_cells:
            self.mask_non_inwind_cells()

    def read_in_wind_parameters(
        self, delim: str = None
    ):
        """Read in the wind parameters.
        todo: add support for polar and spherical winds"""

        wind_all = []
        wind_columns = []

        # Read in each file, one by one, if they exist. Note that this makes
        # the assumption that all the tables are the same size.

        n_read = 0
        files_to_read = ["master", "heat", "gradient", "converge"]

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
            self.wind_parameters += col,

        self.x_coords = tuple(np.unique(self.variables["x"]))
        self.x_cen_coords = tuple(np.unique(self.variables["xcen"]))
        if "z" in self.wind_parameters:
            self.z_coords = tuple(np.unique(self.variables["z"]))
            self.z_cen_coords = tuple(np.unique(self.variables["zcen"]))

    def read_in_wind_ions(
        self, delim: str = None, elements_to_get: Union[List[str], Tuple[str], str] = None
    ):
        """Read in the ion parameters.
        todo: add way to load in either densities or fractions"""

        if elements_to_get is None:
            elements_to_get = ("H", "He", "C", "N", "O", "Si", "Fe")
        else:
            if type(elements_to_get) not in [str, list, tuple]:
                print("ions_to_get should be a tuple/list of strings or a string")
                exit(1)  # todo: error code

        # Read in each ion file, one by one. The ions will be stored in the
        # self.variables dict as,
        # key = ion name
        # values = dict of ion keys, i.e. i_01, i_02, etc, and the values
        # in this dict will be the values of that ion

        ion_types_to_get = ["frac", "den"]
        ion_types_index_names = ["fraction", "density"]

        n_elements_read = 0

        for element in elements_to_get:
            element = element.capitalize()  # for safety...
            self.wind_elements += element,

            # Each element will have a dict of two keys, either frac or den.
            # Inside each dict with be more dicts of keys where the values are
            # arrays of the

            self.variables[element] = {}

            for ion_type, ion_type_index_name in zip(ion_types_to_get, ion_types_index_names):
                file = self.cd + self.root + "." + element + "." + ion_type + ".txt"
                if not os.path.exists(file):
                    continue
                n_elements_read += 1
                with open(file, "r") as f:
                    ion_file = f.readlines()

                # Read in ion the ion file. this can be done in a list
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

                # Now construct the tables, how this is done is described in
                # some of the comments above

                if wind[0][0].isdigit() is False:
                    columns = tuple(wind[0])
                    index = columns.index("i01")
                else:
                    columns = tuple(np.arrange(len(wind[0]), dtype=np.str))
                    index = 0
                columns = columns[index:]
                wind = np.array(wind[1:], dtype=np.float64)[:, index:]

                self.variables[element][ion_type_index_name] = {}
                for index, col in enumerate(columns):
                    self.variables[element][ion_type_index_name][col] = wind[:, index].reshape(self.nx, self.nz)

        if n_elements_read == 0:
            print("Unable to open any wind save tables, try running windsave2table...")
            exit(1)

    def project_cartesian_velocity_to_cylindrical(
        self
    ):
        """Project the cartesian velocities of the wind into cylindrical coordinates."""

        v_l = np.zeros_like(self.variables["v_x"])
        v_rot = np.zeros_like(v_l)
        v_r = np.zeros_like(v_l)
        n1, n2 = v_l.shape

        for i in range(n1):
            for j in range(n2):
                cart_point = [self.variables["x"][i, j], 0, self.variables["z"][i, j]]
                # todo: don't think I need to do this check anymore
                if self.variables["inwind"][i, j] < 0:
                    v_l[i, j] = 0
                    v_rot[i, j] = 0
                    v_r[i, j] = 0
                else:
                    cart_velocity_vector = [
                        self.variables["v_x"][i, j], self.variables["v_y"][i, j], self.variables["v_z"][i, j]
                    ]
                    cyl_velocity_vector = vector.project_cartesian_to_cylindrical_coordinates(
                        cart_point, cart_velocity_vector
                    )
                    if type(cyl_velocity_vector) is int:
                        # todo: some error has happened, print a warning...
                        continue
                    v_l[i, j] = np.sqrt(cyl_velocity_vector[0] ** 2 + cyl_velocity_vector[2] ** 2)
                    v_rot[i, j] = cyl_velocity_vector[1]
                    v_r[i, j] = cyl_velocity_vector[0]

        self.variables["v_l"] = v_l * self.velocity_conversion_factor
        self.variables["v_rot"] = v_rot * self.velocity_conversion_factor
        self.variables["v_r"] = v_r * self.velocity_conversion_factor

    def mask_non_inwind_cells(
        self
    ):
        """Convert each array into a masked array, where the mask is defined by
        the inwind variable."""

        to_mask_wind = list(self.wind_parameters)

        # Remove some of the columns, as these shouldn't be masked because
        # weird things will happen when creating a plot. This doesn't need to
        # be done for the wind ions as they don't have the below items in their
        # data structures

        for item_to_remove in ["x", "z", "xcen", "zcen", "i", "j", "inwind"]:
            try:
                to_mask_wind.remove(item_to_remove)
            except ValueError:
                continue

        # First, create masked arrays for the wind parameters which is simple
        # enough.

        for col in to_mask_wind:
            self.variables[col] = np.ma.masked_where(self.variables["inwind"] < 0, self.variables[col])

        # Now, create masked arrays for the wind ions. Have to do it for each
        # element and each ion type and each ion. This is probably slow :)

        for element in self.wind_elements:
            for ion_type in self.variables[element].keys():
                for ion in self.variables[element][ion_type].keys():
                    self.variables[element][ion_type][ion] = np.ma.masked_where(
                        self.variables["inwind"] < 0, self.variables[element][ion_type][ion]
                    )

    def get_sightline_coordinates(
        self, theta: float
    ):
        """Get the vertical z coordinates for a given set of x coordinates and
        inclination angle.

        Parameters
        ----------
        theta: float
            The angle of the sight line to extract from. Given in degrees."""
        return np.array(self.x_coords, dtype=np.float) * np.tan(PI / 2 - np.deg2rad(theta))

    def get_variable_along_sightline(self, theta: float, parameter: str):
        """Extract a variable along a given sight line.
        todo: i think this only works with rectilinear grids, not polar
        todo: get this to work with ions"""

        if type(theta) is not float:
            theta = float(theta)
        z_array = np.array(self.z_coords, dtype=np.float)
        z_coords = self.get_sightline_coordinates(theta)

        values = []

        for x_index, z in enumerate(z_coords):
            z_index = get_array_index(z_array, z)
            value = self.variables[parameter][x_index, z_index]
            values.append(value)

        return np.array(self.x_coords), z_array, np.array(values, dtype=np.float)

    def get_elem_number_from_ij(
        self, i: int, j: int
    ):
        """Get the wind element number for a given i and j index."""
        raise self.nz * i + j

    def get_ij_from_elem_number(
        self, elem: int
    ):
        """Get the i and j index for a given wind element number.
        todo: check that this is row or column major in Python"""
        return np.unravel_index(elem, (self.nx, self.nz))

    def __getitem__(
        self, key
    ):
        """Return an array in the variables dictionary when indexing."""
        return self.variables[key]

    def __setitem__(
        self, key, value
    ):
        """Set an array in the variables dictionary."""
        self.variables[key] = value

    def __str__(
        self
    ):
        """Print basic details about the wind."""
        return "NotImplementedYet:-)"
