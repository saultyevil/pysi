"""Utility class for Wind.

Contains functions and variables which could be part of the WindBase class but
are a bit more abstracted.
"""

import copy
import warnings
from collections.abc import Callable

import numpy

import pysi.sim.grid
import pysi.util
from pysi.math import vector
from pysi.math.blackhole import gravitational_radius
from pysi.util.run import run_windsave2table
from pysi.wind import enum
from pysi.wind.model import base

WIND_CELL_TYPES = [enum.WindCellPosition.INWIND.value, enum.WindCellPosition.PARTIALLY_INWIND.value]

def create_wind_tables(root: str, directory: str, version: str | None = None) -> None:
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
    run_windsave2table(root, directory, ion_density=True, version=version)
    run_windsave2table(root, directory, ion_density=False, version=version)
    run_windsave2table(root, directory, cell_spec=True, version=version)


class WindUtil(base.WindBase):
    """Utility functions for wind stuff."""

    def __init__(
        self,
        root: str,
        directory: str,
        mask_value: int | Callable[[int], bool] = enum.WindCellPosition.INWIND.value,
        mass_msol: float | None = None,
        **kwargs: dict,
    ) -> None:
        """Initialize the class.

        Parameters
        ----------
        root: str
            The root name of the simulation.
        directory: str
            The directory containing the simulation.
        mask_value: int | Callable[[int], bool]
            The value to use for masking the wind cells. If this is a callable,
            it will be called with the inwind value and should return True if
            the cell should be masked.
        **kwargs
            Additional keyword arguments to pass to the WindBase class

        """
        super().__init__(root, directory, **kwargs)

        if self.coord_type == enum.CoordSystem.CYLINDRICAL:
            self._calculate_cylindrical_velocities()

        self.__original_parameters = None
        if mask_value or mask_value in WIND_CELL_TYPES:
            self.mask_arrays(mask_value)

        self.mass_msol = mass_msol
        self.grav_radius = self._calculate_grav_radius(mass_msol=self.mass_msol)

    # Private methods ----------------------------------------------------------

    def _calculate_grav_radius(self, *, mass_msol: float | None = None) -> float:
        """Calculate the gravitational radius of the model.

        Parameters
        ----------
        mass_msol: float | None
            The mass of the central object in solar masses. If None, will
            be calculated from the parameter file.

        Returns
        -------
        float
            The gravitational radius in centimetres.

        """
        if not mass_msol:
            try:
                self.mass_msol = mass_msol = float(
                    pysi.sim.grid.get_parameter_value(self.pf, "Central_object.mass(msol)")
                )
            except (OSError, IndexError, ValueError):
                warnings.warn(
                    "Unable to find central mass from parameter file, please supply the mass instead with keyword mass_msol=mass",
                    stacklevel=2,
                )
                return 0

        return gravitational_radius(mass_msol)

    def _change_distance_units(self, new_units: enum.DistanceUnits) -> None:
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
            enum.DistanceUnits.GRAVITATIONAL_RADIUS: self.grav_radius / 100,  # convert to metres
        }

        conversion_factor = distance_conv_lookup[self.distance_units] / distance_conv_lookup[new_units]

        for quant in ("x", "z", "x_cen", "z_cen", "r", "r_cen"):
            if quant not in self.parameters:
                continue
            self.parameters[quant] *= conversion_factor

        self.distance_units = new_units
        self._set_axes_coords()

    def _change_velocity_units(self, new_units: enum.VelocityUnits) -> None:
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
            if quant not in self.parameters:
                continue
            self.parameters[quant] *= conversion_factor

        self.velocity_units = new_units

    def _get_sight_line_coordinates(self, theta: float) -> numpy.ndarray:
        """Return z coordinates for a set of x coordinates and angle.

        For a wind with a polar coordinate system, this returns an array of
        the given theta the size of the x axis.

        Parameters
        ----------
        theta: float
            The angle of the sight line to extract from, in degrees.

        Returns
        -------
        numpy.ndarray
            The vertical z coordinates of the sight line.

        """
        if self.coord_type == enum.CoordSystem.POLAR:
            return numpy.ones_like(self.x_coords) * theta

        return numpy.array(self.x_coords, dtype=numpy.float64) * numpy.tan(numpy.pi * 0.5 - numpy.deg2rad(theta))

    def _calculate_cylindrical_velocities(self) -> None:
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

    def get_variable_along_sight_line(self, theta: float) -> numpy.ndarray:
        """Get a variable along a given sightline."""
        msg = "Method is not implemented yet."
        raise NotImplementedError(msg)

    def mask_arrays(self, mask_value: int | Callable[[int, int], bool]) -> None:
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
                msg = "The mask_value parameter should be a callable, or an int."
                raise TypeError(msg)
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

    def create_wind_tables(self) -> None:
        """Force the creation of wind save tables for the model.

        This is best used when a simulation has been re-run, as the
        library is unable to detect when the currently available wind
        tables do not reflect a new simulation. This function will
        create the standard wind tables, as well as the fractional and
        density ion tables and create the xspec cell spectra files.
        """
        create_wind_tables(self.root, self.directory, self.version)

    def change_units(self, new_units: enum.DistanceUnits | enum.VelocityUnits) -> None:
        """Change the spatial or velocity units.

        Parameters
        ----------
        new_units: Union[enum.DistanceUnits, enum.VelocityUnits]
            The new units to transform into.

        """
        if isinstance(new_units, enum.DistanceUnits):
            self._change_distance_units(new_units)
        elif isinstance(new_units, enum.VelocityUnits):
            self._change_velocity_units(new_units)
        else:
            raise ValueError(f"new_units not of type {type(enum.DistanceUnits)} or {type(enum.VelocityUnits)}")  # noqa: TRY004

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
