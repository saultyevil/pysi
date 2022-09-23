#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Contains the user facing Wind class."""

import copy
from typing import Union

import numpy

import pypython
from pypython.wind import enum, plot


class Wind(plot.WindPlot):
    """Main wind class for pypython.

    This class includes...
    """

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

        # self.x_coords = numpy.zeros(self.n_x)
        # self.z_coords = numpy.zeroS(self.n_z)

    def create_wind_tables(self):
        """Force the creation of wind save tables for the model.

        This is best used when a simulation has been re-run, as the
        library is unable to detect when the currently available wind
        tables do not reflect a new simulation. This function will
        create the standard wind tables, as well as the fractional and
        density ion tables and create the xspec cell spectra files.
        """

        pypython.create_wind_save_tables(self.root, self.directory, ion_density=True)
        pypython.create_wind_save_tables(self.root, self.directory, ion_density=False)
        pypython.create_wind_save_tables(
            self.root, self.directory, cell_spec=True
        )  # TODO: check if I need a seperate cell_spec call

    def change_units(self, new_units: Union[str, enum.Units]) -> None:
        """Change the spatial or velocity units."""
        pass

    # def setup_gird_properties(self) -> None:
    #     """Determine the properties of the wind grid."""

    #     # Populate the x and z coord attributes, and remove duplicate entries
    #     # so we should end up with a 1d array for both

    #     if self.coord_type in ["polar", "spherical"]:
    #         self.x_coords = self.parameters["r"]
    #     else:
    #         self.x_coords = self.parameters["x"]

    #     if self.coord_type == "polar":
    #         self.z_coords = self.parameters["theta"]
    #     elif self.coord_type == "cylindrical":
    #         self.z_coords = self.parameters["z"]
    #     else:
    #         self.z_coords = None

    #     self.x_coords = numpy.unique(self.x_coords)
    #     self.z_coords = numpy.unique(self.z_coords)

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
                self.parameters["spec"][i, j] = pypython.smooth_array(self.parameters["spec"][i, j], amount)

    def unsmooth_cell_spectra(self) -> None:
        """Unsmooth the arrays.

        Uses a copy of the original spectra to revert the smoothing.
        """
        self.parameters["spec"] = copy.deepcopy(self.__original_parameters["spec"])
