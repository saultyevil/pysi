#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility class for Wind.

Contains functions and variables which could be part of the WindBase class but
are a bit more abstracted.
"""

import copy
from typing import Callable, Union

import numpy

from pysi.wind import enum
from pysi.math import vector
from pysi.wind.model import base

WIND_CELL_TYPES = [enum.WindCellPosition.INWIND.value, enum.WindCellPosition.PARTIALLY_INWIND.value]


class WindUtil(base.WindBase):
    """Utility functions for wind stuff."""

    def __init__(
        self,
        root: str,
        directory: str,
        mask_value: Union[int, Callable[[int], bool]] = enum.WindCellPosition.INWIND.value,
        **kwargs,
    ):
        """Initialize the class."""
        super().__init__(root, directory, **kwargs)

        self.x_coords = (
            numpy.unique(self.parameters["x"]) if enum.CoordSystem.CYLINDRICAL else numpy.unique(self.parameters["r"])
        )
        if self.n_z > 1:
            self.z_coords = (
                numpy.unique(self.parameters["z"])
                if enum.CoordSystem.CYLINDRICAL
                else numpy.unique(self.parameters["theta"])
            )
        else:
            self.n_z = numpy.zeros_like(self.x_coords)

        if self.coord_type == enum.CoordSystem.CYLINDRICAL:
            self.__project_cartesian_velocity_to_cylindrical()

        # Create masked arrays

        self.__original_parameters = None

        if mask_value or mask_value in WIND_CELL_TYPES:
            self.mask_arrays(mask_value)

    # Private methods ----------------------------------------------------------

    def __get_sight_line_coordinates(self, theta: float):
        """Get the vertical z coordinates for a given set of x coordinates and
        inclination angle.

        For a wind with a polar coordinate system, this returns an array of
        the given theta the size of the x axis.

        Parameters
        ----------
        theta: float
            The angle of the sight line to extract from, in degrees.
        """
        if self.coord_type == enum.CoordSystem.POLAR:
            return numpy.ones_like(self.x_coords) * theta

        return numpy.array(self.x_coords, dtype=numpy.float64) * numpy.tan(numpy.pi * 0.5 - numpy.deg2rad(theta))

    def __project_cartesian_velocity_to_cylindrical(self):
        """Project cartesian velocities into cylindrical velocities.

        This makes the variables v_r, v_rot and v_l available in
        variables dictionary. Only works for cylindrical coordinates
        systems, which outputs the velocities in cartesian coordinates.
        """
        v_l = numpy.zeros_like(self.parameters["v_x"])
        v_rot = numpy.zeros_like(v_l)
        v_r = numpy.zeros_like(v_l)
        n_x, n_z = v_l.shape

        for i in range(n_x):
            for j in range(n_z):
                cart_point = numpy.array([self.parameters["x"][i, j], 0, self.parameters["z"][i, j]])
                if self.parameters["inwind"][i, j] < 0:
                    v_l[i, j] = 0.0
                    v_rot[i, j] = 0.0
                    v_r[i, j] = 0.0
                else:
                    cart_velocity_vector = numpy.array(
                        [self.parameters["v_x"][i, j], self.parameters["v_y"][i, j], self.parameters["v_z"][i, j]]
                    )
                    try:
                        cyl_velocity_vector = vector.project_cartesian_vec_to_cylindrical_vec(
                            cart_point, cart_velocity_vector
                        )
                    except ValueError:
                        continue
                    v_l[i, j] = numpy.sqrt(cyl_velocity_vector[0] ** 2 + cyl_velocity_vector[2] ** 2)
                    v_rot[i, j] = cyl_velocity_vector[1]
                    v_r[i, j] = cyl_velocity_vector[0]

        self.parameters["v_l"] = v_l
        self.parameters["v_rot"] = v_rot
        self.parameters["v_r"] = v_r

        # Have to do this again to include polodial velocities in the tuple
        self.parameter_keys = tuple(self.parameters.keys())

    # Public methods -----------------------------------------------------------

    def get_variable_along_sight_line(self, theta: float):
        """Get a variable along a given sightline."""
        raise NotImplementedError("Method is not implemented yet.")

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

        to_mask = [item for item in self.things_read_in if item not in items_to_not_mask]

        # Finally go through each parameter in to_mask to create a masked array
        # for each item

        for item in to_mask:
            self.parameters[item] = numpy.ma.masked_where(mask_expression, self.parameters[item])

        self.mask_value = mask_value

    def unmask_arrays(self) -> None:
        """Unmask the arrays.

        Uses a copy of the original table variables to revert the
        masking.
        """
        self.parameters = copy.deepcopy(self.__original_parameters)
