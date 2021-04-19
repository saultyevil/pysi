#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Contains the Wind object for reading in a 1D and 2D wind. Also includes plotting
functions.
"""

import os
from sys import exit
from typing import List, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt

from .extrautil import vector
from .physics.constants import CMS_TO_KMS, PI, C
from .plotutil import normalize_figure_style
from .util import get_array_index


class Wind:
    """A class to store 1D and 2D Python wind tables. Contains methods to
    extract variables, as well as convert various indices into other indices.
    todo: add dot notation for accessing dictionaries"""

    def __init__(
        self, root: str, cd: str = ".", velocity_units: str = "kms", mask_cells: bool = True, delim: str = None
    ):
        """Initialize the object.
        Parameters
        ----------
        root: str
            The root name of the Python simulation.
        cd: str
            The directory containing the model.
        mask_cells: bool [optional]
            Store the wind parameters as masked arrays.
        delim: str [optional]
            The delimiter used in the wind table files."""

        self.root = root
        self.cd = cd
        if self.cd[-1] != "/":
            self.cd += "/"
        self.nx = 1
        self.nz = 1
        self.n_elem = 1
        self.m_coords = ()
        self.n_coords = ()
        self.m_cen_coords = ()
        self.n_cem_coords = ()
        self.parameters = ()
        self.elements = ()
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
        self.columns = self.parameters + self.elements

        # Record the coordinate system and possible axes labels

        if self.nz == 1:
            self.coord_system = "spherical"
            self.axes = ["r", "r_cen"]
        elif "r" in self.parameters and "theta" in self.parameters:
            self.coord_system = "polar"
            self.axes = ["r", "theta", "r_cen", "theta_cen"]
        else:
            self.coord_system = "rectilinear"
            self.axes = ["x", "z", "x_cen", "z_cen"]

        # Convert velocity into desired units and also calculate the cylindrical
        # velocities. This doesn't work for polar or spherical coordinates as
        # they will not have these velocities

        if self.coord_system == "rectilinear":
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

        for table in files_to_read:
            fpath = self.cd + self.root + "." + table + ".txt"
            if not os.path.exists(fpath):
                fpath = self.cd + "tables/" + self.root + "." + table + ".txt"
                if not os.path.exists(fpath):
                    # todo: throw some kinda warning, I guess?
                    continue
            n_read += 1

            with open(fpath, "r") as f:
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

            wind_all.append(np.array(wind_list[1:], dtype=np.float64))

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
        self.n_elem = int(self.nx * self.nz)  # the int() is for safety

        wind_all = np.hstack(wind_all)

        # Assign each column header to a key in the dictionary, ignoring any
        # column which is already in the dict and extract the x and z
        # coordinates

        for index, col in enumerate(wind_columns):
            if col in self.variables.keys():
                continue
            self.variables[col] = wind_all[:, index].reshape(self.nx, self.nz)
            self.parameters += col,

        # Get the x/r coordinates

        if "x" in self.parameters:
            self.m_coords = tuple(np.unique(self.variables["x"]))
            self.m_cen_coords = tuple(np.unique(self.variables["xcen"]))
        else:
            self.m_coords = tuple(np.unique(self.variables["r"]))
            self.m_cen_coords = tuple(np.unique(self.variables["rcen"]))

        # Get the z coordinates

        if "z" in self.parameters:
            self.n_coords = tuple(np.unique(self.variables["z"]))
            self.n_cen_coords = tuple(np.unique(self.variables["zcen"]))

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
            self.elements += element,

            # Each element will have a dict of two keys, either frac or den.
            # Inside each dict with be more dicts of keys where the values are
            # arrays of the

            self.variables[element] = {}

            for ion_type, ion_type_index_name in zip(ion_types_to_get, ion_types_index_names):
                fpath = self.cd + self.root + "." + element + "." + ion_type + ".txt"
                if not os.path.exists(fpath):
                    fpath = self.cd + "tables/" + self.root + "." + element + "." + ion_type + ".txt"
                    if not os.path.exists(fpath):
                        continue
                n_elements_read += 1
                with open(fpath, "r") as f:
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
            print("Unable to open any ion tables, try running windsave2table...")
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

        to_mask_wind = list(self.parameters)

        # Remove some of the columns, as these shouldn't be masked because
        # weird things will happen when creating a plot. This doesn't need to
        # be done for the wind ions as they don't have the below items in their
        # data structures

        for item_to_remove in ["x", "z", "r", "theta", "xcen", "zcen", "rcen", "theta_cen", "i", "j", "inwind"]:
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

        for element in self.elements:
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

        return np.array(self.m_coords, dtype=np.float64) * np.tan(PI / 2 - np.deg2rad(theta))

    def get_variable_along_sightline(
        self, theta: float, parameter: str, fraction: bool = False
    ):
        """Extract a variable along a given sight line.
        todo: i think this only works with rectilinear grids, not polar"""
        
        if self.coord_system == "polar":
            raise NotImplementedError()
        
        if type(theta) is not float:
            theta = float(theta)
        
        z_array = np.array(self.n_coords, dtype=np.float64)
        z_coords = self.get_sightline_coordinates(theta)
        values = np.zeros_like(z_coords, dtype=np.float64)

        # Get the variable before hand, so we can deal with getting elements
        # out due to their stupid layout in memory

        element_check = parameter[:2].replace("_", "")
        ion_check = parameter[2:].replace("_", "")
        if element_check in self.elements:
            if fraction:
                variable = self.variables[element_check]["fraction"][ion_check]
            else:
                variable = self.variables[element_check]["density"][ion_check]
        else:
            variable = self.variables[parameter]

        # This is the actual work which extracts along a sight line

        for x_index, z in enumerate(z_coords):
            z_index = get_array_index(z_array, z)
            values[x_index] = variable[x_index, z_index]

        return np.array(self.m_coords), z_array, values

    def plot(
        self, variable_name: str, fraction: bool = False, log_variable: bool = True
     ):
        """Create a plot of the wind for the given variable.
        Parameters
        ----------
        variable_name: str
            The name of the variable to plot. Ions are accessed as, i.e.,
            H_i01, He_i02, etc.
        fraction: bool [optional]
            Plot ion fractions instead of density
        log_variable: bool [optional
            Plot the log10 of the variable."""

        # First, we need to get the variable. If we want to include ions, then
        # we need to do a bit of messing around

        element_check = variable_name[:2].replace("_", "")
        ion_check = variable_name[2:].replace("_", "")
        if element_check in self.elements:
            if fraction:
                variable = self.variables[element_check]["fraction"][ion_check]
            else:
                variable = self.variables[element_check]["density"][ion_check]
        else:
            variable = self.variables[variable_name]

        if log_variable:
            variable = np.log10(variable)

        # Next, we have to make sure we get the correct coordinates

        if self.coord_system == "spherical":
            m_points = self.variables["r"]
        elif self.coord_system == "rectilinear":
            m_points = self.variables["x"]
            n_points = self.variables["z"]
        else:
            n_points = np.log10(self.variables["r"])
            m_points = np.deg2rad(self.variables["theta"])

        if self.coord_system == "spherical":
            fig, ax = plot_1d_wind(m_points, variable)
        else:
            fig, ax = plot_2d_wind(m_points, n_points, variable, self.coord_system)

        # Finally, label the axes with what we actually plotted

        if len(ax) == 1:
            title = f"{variable_name}".replace("_", " ")
            if log_variable:
                title = r"$\log_{10}($" + title + r"$)$"
            if self.coord_system == "spherical":
                ax[0, 0].set_ylabel(title)
            else:
                ax[0, 0].set_title(title)

        return fig, ax

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

    def show(
        self, block=True
    ):
        """Show a plot which has been created."""

        plt.show(block=block)

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


# Plotting functions -----------------------------------------------------------


def plot_wind(
    wind: Wind, parameter: Union[str, np.ndarray], log_parameter: bool = True, ion_fraction: bool = True,
    inclinations_to_plot: List[str] = None, scale: str = "loglog", vmin: float = None, vmax: float = None,
    fig: plt.Figure = None, ax: plt.Axes = None, i: int = 0, j: int = 0
) -> Tuple[plt.Figure, Union[np.ndarray, plt.Axes]]:
    """Wrapper function for plotting a wind. Will decide if a wind is 1D or 2D
    and call the appropriate function.
    Parameters
    ----------
    wind: Wind
        The Wind object.
    parameter: np.ndarray
        The wind parameter to be plotted, in the same shape as the coordinate
        arrays. Can also be the name of the variable.
    inclinations_to_plot: List[str] [optional]
        A list of inclination angles to plot onto the ax[0, 0] sub panel. Must
        be strings and 0 < inclination < 90.
    scale: str [optional]
        The scaling of the axes: [logx, logy, loglog, linlin]
    vmin: float [optional]
        The minimum value to plot.
    vmax: float [optional]
        The maximum value to plot.
    fig: plt.Figure [optional]
        A Figure object to update, otherwise a new one will be created.
    ax: plt.Axes [optional]
        An axes array to update, otherwise a new one will be created.
    i: int [optional]
        The i index for the sub panel to plot onto.
    j: int [optional]
        The j index for the sub panel to plot onto.

    Returns
    -------
    fig: plt.Figure
        The (updated) Figure object for the plot.
    ax: plt.Axes
        The (updated) axes array for the plot.
    """

    # If the parameter is not an array, then get the variable from the wind
    # todo: put first branch into a function

    if type(parameter) is str:
        element_check = parameter[:2].replace("_", "")
        ion_check = parameter[2:].replace("_", "")
        if element_check in wind.elements:
            if ion_fraction:
                parameter_points = wind.variables[element_check]["fraction"][ion_check]
            else:
                parameter_points = wind.variables[element_check]["density"][ion_check]
        else:
            parameter_points = wind.variables[parameter]
        if log_parameter:
            parameter_points = np.log10(parameter_points)
    elif type(parameter) in [np.ndarray, np.ma.core.MaskedArray]:
        parameter_points = parameter
    else:
        print(f"Incompatible type {type(parameter)} for parameter")
        return fig, ax

    if wind.coord_system == "spherical":
        fig, ax = plot_1d_wind(
            wind["r"], parameter_points, "logx", fig, ax, i, j
        )
    else:
        fig, ax = plot_2d_wind(
            wind["x"], wind["z"], parameter_points, wind.coord_system, inclinations_to_plot, scale, vmin, vmax,
            fig, ax, i, j
        )

    return fig, ax


def plot_1d_wind(
    m_points: np.ndarray, parameter_points: np.ndarray, scale: str = "loglog", fig: plt.Figure = None,
     ax: plt.Axes = None, i: int = 0, j: int = 0
) -> Tuple[plt.Figure, Union[np.ndarray, plt.Axes]]:
    """Plot a 1D wind.

    Parameters
    ----------
    m_points: np.ndarray
        The 1st axis points, which are the r bins.
    parameter_points: np.ndarray
        The wind parameter to be plotted, in the same shape as the n_points and
        m_points arrays.
    scale: str [optional]
        The scaling of the axes: [logx, logy, loglog, linlin]
    fig: plt.Figure [optional]
        A Figure object to update, otherwise a new one will be created.
    ax: plt.Axes [optional]
        An axes array to update, otherwise a new one will be created.
    i: int [optional]
        The i index for the sub panel to plot onto.
    j: int [optional]
        The j index for the sub panel to plot onto.

    Returns
    -------
    fig: plt.Figure
        The (updated) Figure object for the plot.
    ax: plt.Axes
        The (updated) axes array for the plot."""

    normalize_figure_style()

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), squeeze=False)

    if scale not in ["logx", "logy", "loglog", "linlin"]:
        print("unknown axes scaling", scale)
        print("allwoed:", ["logx", "logy", "loglog", "linlin"])
        exit(1)  # todo: error code

    ax[i, j].plot(m_points, parameter_points)

    ax[i, j].set_xlabel(r"$r$ [cm]")
    ax[i, j].set_xlim(np.min(m_points[m_points > 0]), np.max(m_points))
    if scale == "loglog" or scale == "logx":
        ax[i, j].set_xscale("log")
    if scale == "loglog" or scale == "logy":
        ax[i, j].set_yscale("log")

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    return fig, ax


def plot_2d_wind(
    m_points: np.ndarray, n_points: np.ndarray, parameter_points: np.ndarray, coordinate_system: str = "rectilinear",
    inclinations_to_plot: List[str] = None, scale: str = "loglog", vmin: float = None, vmax: float = None,
    fig: plt.Figure = None, ax: plt.Axes = None, i: int = 0, j: int = 0
) -> Tuple[plt.Figure, Union[np.ndarray, plt.Axes]]:
    """Plot a 2D wind using a contour plot.

    Parameters
    ----------
    m_points: np.ndarray
        The 1st axis points, either x or angular (in degrees) bins.
    n_points: np.ndarray
        The 2nd axis points, either z or radial bins.
    parameter_points: np.ndarray
        The wind parameter to be plotted, in the same shape as the n_points and
        m_points arrays.
    coordinate_system: str [optional]
        The coordinate system in use, either rectilinear or polar.
    inclinations_to_plot: List[str] [optional]
        A list of inclination angles to plot onto the ax[0, 0] sub panel. Must
        be strings and 0 < inclination < 90.
    scale: str [optional]
        The scaling of the axes: [logx, logy, loglog, linlin]
    vmin: float [optional]
        The minimum value to plot.
    vmax: float [optional]
        The maximum value to plot.
    fig: plt.Figure [optional]
        A Figure object to update, otherwise a new one will be created.
    ax: plt.Axes [optional]
        An axes array to update, otherwise a new one will be created.
    i: int [optional]
        The i index for the sub panel to plot onto.
    j: int [optional]
        The j index for the sub panel to plot onto.

    Returns
    -------
    fig: plt.Figure
        The (updated) Figure object for the plot.
    ax: plt.Axes
        The (updated) axes array for the plot."""

    normalize_figure_style()

    if fig is None or ax is None:
        if coordinate_system == "rectilinear":
            fig, ax = plt.subplots(figsize=(7, 5), squeeze=False)
        elif coordinate_system == "polar":
            fig, ax = plt.subplots(figsize=(7, 5), squeeze=False, subplot_kw={"projection": "polar"})
        else:
            print("unknown wind projection", coordinate_system)
            print("allowed: ['rectilinear', 'polar']")
            exit(1)  # todo: error code

    if scale not in ["logx", "logy", "loglog", "linlin"]:
        print("unknown axes scaling", scale)
        print("allwoed:", ["logx", "logy", "loglog", "linlin"])
        exit(1)  # todo: error code

    im = ax[i, j].pcolormesh(m_points, n_points, parameter_points, shading="auto", vmin=vmin, vmax=vmax)

    if inclinations_to_plot:
        n_coords = np.unique(m_points)
        for inclination in inclinations_to_plot:
            if coordinate_system == "rectilinear":
                m_coords = n_coords * np.tan(0.5 * PI - np.deg2rad(float(inclination)))
            else:
                x_coords = np.logspace(np.log10(0), np.max(n_points))
                m_coords = x_coords * np.tan(0.5 * PI - np.deg2rad(90 - float(inclination)))
                m_coords = np.sqrt(x_coords ** 2 + m_coords ** 2)
            ax[0, 0].plot(n_coords, m_coords, label=inclination + r"$^{\circ}$")
        ax[0, 0].legend(loc="lower left")

    fig.colorbar(im, ax=ax[i, j])
    if coordinate_system == "rectilinear":
        ax[i, j].set_xlabel(r"$x$ [cm]")
        ax[i, j].set_ylabel(r"$z$ [cm]")
        ax[i, j].set_xlim(np.min(m_points[m_points > 0]), np.max(m_points))
        if scale == "loglog" or scale == "logx":
            ax[i, j].set_xscale("log")
        if scale == "loglog" or scale == "logy":
            ax[i, j].set_yscale("log")
    else:
        ax[i, j].set_theta_zero_location("N")
        ax[i, j].set_theta_direction(-1)
        ax[i, j].set_thetamin(0)
        ax[i, j].set_thetamax(90)
        ax[i, j].set_rlabel_position(90)
        ax[i, j].set_ylabel(r"$\log_{10}(R)$ [cm]")

    ax[i, j].set_ylim(np.min(n_points[n_points > 0]), np.max(n_points))
    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    return fig, ax

