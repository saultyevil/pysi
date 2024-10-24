#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Contains the user facing Wind class."""

import copy
from typing import Union

import pysi.util
from pysi.wind import enum
from pysi.wind.model import plot


def create_wind_tables(root: str, directory: str, version: str = None):
    """Force the creation of wind save tables for the model.

    Parameters
    ----------
    root : str
        The root name of the model
    directory : str
        The directory containing the model
    version : str
        The version number of Python the model was run with
    """
    pysi.util.create_wind_save_tables(root, directory, ion_density=True, version=version)
    pysi.util.create_wind_save_tables(root, directory, ion_density=False, version=version)
    pysi.util.create_wind_save_tables(root, directory, cell_spec=True, version=version)


class Wind(plot.WindPlot):
    """Main wind class for PyPython."""

    def __init__(self, root: str, directory: str = ".", **kwargs) -> None:
        """Initialize the class.

        Parameters
        ----------
        root: str
            The root name of the simulation.
        directory: str
            The directory containing the simulation. By default this is assumed
            to be the current working directory.
        """
        super().__init__(root, directory, **kwargs)
        # self.grav_radius = self.__calculate_grav_radius()

    def __str__(self) -> str:
        return f"Wind(root={self.root!r} directory={str(self.directory)!r})"

    # Public methods -----------------------------------------------------------

    def create_wind_tables(self):
        """Force the creation of wind save tables for the model.

        This is best used when a simulation has been re-run, as the
        library is unable to detect when the currently available wind
        tables do not reflect a new simulation. This function will
        create the standard wind tables, as well as the fractional and
        density ion tables and create the xspec cell spectra files.
        """
        create_wind_tables(self.root, self.directory, self.version)

    def change_units(self, new_units: Union[enum.DistanceUnits, enum.VelocityUnits]) -> None:
        """Change the spatial or velocity units.

        Parameters
        ----------
        new_units: Union[enum.DistanceUnits, enum.VelocityUnits]
            The new units to transform into.
        """
        if isinstance(new_units, enum.DistanceUnits):
            self.__change_distance_units(new_units)
        elif isinstance(new_units, enum.VelocityUnits):
            self.__change_velocity_units(new_units)
        else:
            raise ValueError(f"new_units not of type {type(enum.DistanceUnits)} or {type(enum.VelocityUnits)}")

    def smooth_cell_spectra(self, amount: int) -> None:
        """Smooth the cell spectra.

        Uses a boxcar filter to smooth the cell spectra.

        Parameters
        ----------
        amount: int
            The pixel width for a boxcar filter.
        """
        for i in range(self.n_x):
            for j in range(self.n_z):
                self.parameters["spec"][i, j] = pysi.util.array.smooth_array(self.parameters["spec"][i, j], amount)

    def unsmooth_cell_spectra(self) -> None:
        """Unsmooth the arrays.

        Uses a copy of the original spectra to revert the smoothing.
        """
        self.parameters["spec"] = copy.deepcopy(self.__original_parameters["spec"])

    # Private methods ----------------------------------------------------------

    def __calculate_grav_radius(self) -> float:
        """Calculate the gravitational radius of the model."""
        raise NotImplementedError("Method is not implemented yet.")

        # if not co_mass_in_msol:
        #     try:
        #         co_mass_in_msol = float(pysi.simulation.grid.get_parameter(self.pf, "Central_object.mass(msol)"))
        #     except Exception as ex
        #         raise ValueError("unable to find CO mass from parameter file, please supply the mass instead") from ex

        # rg = pysi.physics.blackhole.gravitational_radius(co_mass_in_msol)

        # return 0

    def __change_distance_units(self, new_units: enum.DistanceUnits):
        """Change the distance units of the wind.

        Parameters
        ----------
        new_units: enum.DistanceUnits
            The new distance units.
        """
        if self.distance_units == new_units:
            return

        distance_conv_lookup = {
            enum.DistanceUnits.CENTIMETRES: 0.01,
            enum.DistanceUnits.METRES: 1,
            enum.DistanceUnits.KILOMETRES: 1000,
            enum.DistanceUnits.GRAVITATIONAL_RADIUS: 0,
        }

        conversion_factor = distance_conv_lookup[self.distance_units] / distance_conv_lookup[new_units]

        for quant in ("x", "z", "x_cen", "z_cen", "r", "r_cen"):
            if quant not in self.parameter_keys:
                continue
            self.parameters[quant] *= conversion_factor

        # self.grav_radius *= conversion_factor
        self.distance_units = new_units

    def __change_velocity_units(self, new_units: enum.VelocityUnits) -> None:
        """Change the velocity units of the wind.

        Parameters
        ----------
        new_units: enum.VelocityUnits
            The new velocity units.
        """
        if self.velocity_units == new_units:
            return

        velocity_conv_lookup = {
            enum.VelocityUnits.CENTIMETRES_PER_SECOND: 0.01,
            enum.VelocityUnits.METRES_PER_SECOND: 1,
            enum.VelocityUnits.KILOMETRES_PER_SECOND: 1000,
            enum.VelocityUnits.SPEED_OF_LIGHT: 2.99792e8,
        }

        conversion_factor = velocity_conv_lookup[self.velocity_units] / velocity_conv_lookup[new_units]

        for quant in ("v_x", "v_y", "v_z", "v_r", "v_theta"):
            if quant not in self.parameter_keys:
                continue
            self.parameters[quant] *= conversion_factor

        self.velocity_units = new_units
