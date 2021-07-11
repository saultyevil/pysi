#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classes for analysing the wind in a Python simulation."""
import textwrap
from enum import Enum
from os import path

import numpy as np
from matplotlib import pyplot as plt

from pypython import (_AttributeDict, create_wind_save_tables, find, get_array_index)
from pypython.constants import BOLTZMANN, CMS_TO_KMS, PI, PLANCK, VLIGHT
from pypython.math import vector
from pypython.physics.blackhole import gravitational_radius
from pypython.plot import set_axes_scales
from pypython.simulation.grid import get_parameter_value


# Enumerators ------------------------------------------------------------------


class WindCoordSystem(Enum):
    """Possible grid coordinate systems in Python."""
    cylindrical = cartesian = "rectilinear"
    polar = "polar"
    spherical = "spherical"
    unknown = "unknown"


class WindVelocityUnits(Enum):
    """Possible velocity conversions for Wind objects.

    Default is cms.
    """
    kms = "kms"
    cms = "cms"
    light = "c"


class WindDistanceUnits(Enum):
    """Possible distance conversions for Wind objects.

    Default is cm.
    """
    cm = "cm"
    m = "m"
    km = "km"
    rg = "rg"


class CellModelType(Enum):
    """Possible model types for cell SED."""
    powerlaw = 1
    exponential = 2


# Constants --------------------------------------------------------------------

LEN_WHEN_1D_MODEL = 4
DEBUG_MASK_CELL_SPEC = False

# Cell spectrum ----------------------------------------------------------------


class CellSpectra:
    """A class to store the cell spectra accumulated during the ionization
    cycles.

    Cells which do not have a spectrum will return None.
    """
    def __init__(self, root, fp=".", nx=0, nz=0, force_make_spectra=False, delim=None):
        """Initialize the object.

        Reads in the cell spectra into a 1D list. This function will attempt
        to run windsave2table to create the cell spectra files if they do not
        exist.

        Parameters
        ----------
        root: str
            The root name of the Python simulation.
        fp: str [optional]
            The directory containing the model.
        nx: int [optional]
            The number of cells in the x direction.
        nz: int [optional]
            The number of cells in the z direction.
        force_make_spectra: bool [optional]
            Force windsave2table to be run to re-make the files in the
            tables directory.
        delim: str [optional]
            The delimiter used in the wind table files.
        """
        self.root = root

        self.fp = path.expanduser(fp)
        if self.fp[-1] != "/":
            self.fp += "/"
        self.pf = self.fp + self.root + ".pf"

        # Set initial conditions to create member variables

        self.nx = int(nx)
        self.nz = int(nz)
        self.header = None
        self.cells = None
        self.spectra = None

        self.original = None

        # Try to read in the spectra, if we can't then we'll try and run
        # windsave2table. It is also possible to force the re-creation of the
        # spectra files

        if force_make_spectra:
            self.create_wind_tables()

        try:
            self.get_cell_spectra(delim)
        except IOError:
            create_wind_save_tables(root, fp, cell_spec=True)
            self.get_cell_spectra(delim)

    # Methods ------------------------------------------------------------------

    def create_wind_tables(self):
        """Force the creation of wind save tables for the model.

        This is best used when a simulation has been re-run, as the
        library is unable to detect when the currently available wind
        tables do not reflect a new simulation. This function will
        create the standard wind tables, as well as the fractional and
        density ion tables and create the xspec cell spectra files.
        """

        create_wind_save_tables(self.root, self.fp, ion_density=True)
        create_wind_save_tables(self.root, self.fp, ion_density=False)
        create_wind_save_tables(self.root, self.fp, cell_spec=True)

    def get_cell_spectra(self, delim=None):
        """Read in the cell spectra.

        This function will read in spectra from across multiple files,
        if there are multiple files at least.
        """
        self.header = []
        cell_spectra = []
        frequency_bins = []

        # Loop over each file. Each time self.header is updated, but we store
        # the rest into an array which gets hstacked to make a single array

        for fp in find("*xspec.*.txt", self.fp):

            with open(fp, "r") as f:
                spectrum_file = f.readlines()

            spectra_lines = []

            for line in spectrum_file:
                if len(line) == 0 or line.startswith("#"):
                    continue
                if delim:
                    line = line.split(delim)
                else:
                    line = line.split()

                spectra_lines.append(line)

            self.header += spectra_lines[0][1:]
            array = np.array(spectra_lines[1:], dtype=np.float64)
            frequency_bins.append(array[:, 0])
            cell_spectra.append(array[:, 1:])

        cell_spectra = [np.hstack(cell_spectra)]
        frequency_bins = np.array(frequency_bins[0], dtype=np.float64)  # assuming they're all the same...
        cell_spectra = cell_spectra[0]

        # Now extract the cell indices from the header, of course done
        # differently depending on if the model is 1D or 2D.

        if len(self.header[0]) > LEN_WHEN_1D_MODEL:
            self.cells = [(int(cell[1:4]), int(cell[5:])) for cell in self.header]
        else:
            self.cells = [(int(cell[1:4]), 0) for cell in self.header]

        # If nx or nz were not given, then determine the number of coordinates
        # from the parameter files or from the cells array

        if self.nx == 0 or self.nz == 0:
            try:
                self.nx = int(get_parameter_value(self.pf, "Wind.dim.in.x_or_r.direction"))
                if len(self.header[1]) > LEN_WHEN_1D_MODEL:
                    self.nz = int(get_parameter_value(self.pf, "Wind.dim.in.z_or_theta.direction"))
            except (ValueError, IOError):
                self.nx = self.cells[-1][0] + 1
                self.nz = self.cells[-1][1] + 1

        # The final step is to create a 2D array of Nones and then populate the
        # cells which have spectra with a dict with keys Freq. and Flux

        self.spectra = np.array([None for _ in range(self.nx * self.nz)], dtype=dict).reshape(self.nx, self.nz)
        for n, (i, j) in enumerate(self.cells):
            self.spectra[i, j] = {"Freq.": np.copy(frequency_bins), "Flux": cell_spectra[:, n]}

    def get_elem_number_from_ij(self, i, j):
        """Get the wind element number for a given i and j index.

        Used when indexing into a 1D array, such as in Python itself.

        Parameters
        ----------
        i: int
            The i-th index of the cell.
        j: int
            The j-th index of the cell.
        """
        return int(self.nz * i + j)

    def get_ij_from_elem_number(self, elem):
        """Get the i and j index for a given wind element number.

        Used when converting a wind element number into two indices for use
        in this package.

        Parameters
        ----------
        elem: int
            The element number.
        """
        return np.unravel_index(elem, (self.nx, self.nz))

    def plot(self, i, j=0, energetic=True, scale="loglog", fig=None, ax=None, figsize=(12, 6)):
        """Plot the spectrum for cell (i, j).

        Simple plotting function, if you want something more advanced then
        check out pypython.plot.spectrum.

        Parameters
        ----------
        i: int
            The i-th index for the cell.
        j: int
            The j-th index for the cell.
        energetic: bool
            Plot in energetic (nu * J_nu) units.
        scale: str
            The axes scaling for the plot.
        fig: pyplot.Figure
            A matplotlib Figure object to edit. Needs ax to also be supplied.
        ax: pyplot.Axes
            A matplotlib Axes object to edit. Needs fig to also be supplied.
        figsize: tuple(int, int)
            The size of the figure (width, height) in inches (sigh).
        """
        spectrum = self.spectra[i, j]
        if spectrum is None:
            raise ValueError(f"no modelled cell spectra for cell ({i}, {j}) as cell is probably not in the wind")

        if not fig and not ax:
            fig, ax = plt.subplots(figsize=figsize)
        elif not fig and ax or fig and not ax:
            raise ValueError("fig and ax need to be supplied together")

        if energetic:
            ax.plot(spectrum["Freq."], spectrum["Freq."] * spectrum["Flux"], label="Spectrum")
            ax.set_ylabel(r"$\nu J_{\nu}$ [ergs s$^{-1}$ cm$^{-2}$]")
        else:
            ax.plot(spectrum["Freq."], spectrum["Flux"], label="Spectrum")
            ax.set_ylabel(r"$J_{\nu}$ [ergs s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")

        ax.set_xlabel("Rest-frame Frequency")
        ax = set_axes_scales(ax, scale)
        fig.suptitle(f"Spectrum in cell ({i}, {j})")

        return fig, ax

    def smooth(self):
        raise NotImplementedError()

    @staticmethod
    def show(block=True):
        """Show a plot which has been created.

        Wrapper around pyplot.show().

        Parameters
        ----------
        block: bool
            Use blocking or non-blocking figure display.
        """
        plt.show(block=block)

    # Built in stuff -----------------------------------------------------------

    def __getitem__(self, pos):
        if type(pos) is int:
            i, j = pos, 0
        else:
            i, j = pos
        return self.spectra[i, j]

    def __setitem__(self, pos, value):
        if type(pos) is int:
            i, j = pos, 0
        else:
            i, j = pos
        self.spectra[i, j] = value

    def __str__(self):
        return print(self.spectra)


# Modelled cell spectra --------------------------------------------------------


class ModelledCellSpectra:
    """A class to store the modelled cell spectra used in the ionization cycles
    to calculate various quantities.

    Cells which do not have a spectrum will return None.
    """
    def __init__(self, root, fp=".", force_make_tables=False, delim=None):
        """Initialize the object.

        Reads in the root.spec.txt file, which contains a bunch of parameters
        used to make a model spectrum.

        Parameters
        ----------
        root: str
            The root name of the Python simulation.
        fp: str [optional]
            The directory containing the model.
        force_make_tables: bool [optional]
            Force windsave2table to be run to re-make the files in the
            tables directory.
        delim: str [optional]
            The delimiter used in the wind table files.
        """
        self.root = root

        self.fp = path.expanduser(fp)
        if self.fp[-1] != "/":
            self.fp += "/"
        self.pf = self.fp + self.root + ".pf"

        # Set initial conditions to create member variables

        self.nx = 0
        self.nz = 0
        self.n_cells = 0
        self.n_bands = 0
        self.header = None
        self.cells = []
        self.model_parameters = None
        self.spectra = None

        self.n_bins_per_band = 500

        # Read in the model parameters, i.e. the root.spec.txt file, and
        # convert those into a flux. The wind tables can be forcibly created
        # if need be

        if force_make_tables:
            self.create_wind_tables()

        self.get_model_spectra(delim)
        self.construct_models()

    # Methods ------------------------------------------------------------------

    def construct_models(self):
        """Use the model parameters to create the flux models.

        This uses the model parameters, in self.model_parameters, to construct
        models of the cell spectrum. The number of frequency bins for each
        band is controlled by self.n_bins_per_band, so the number of frequency
        bins for the whole model is self.n_bins_per_band * self.nbands (which
        saw photons).
        """
        self.spectra = np.array([None for _ in range(self.n_cells)]).reshape(self.nx, self.nz)

        # We need to loop over each cell and construct the flux for each
        # band but they will be joined together at the end

        for n in range(self.n_cells):
            i, j = self.get_ij_from_elem_number(n)
            cell_model = self.model_parameters[i, j]

            cell_flux = []
            cell_frequency = []

            for band in cell_model:
                band_freq_min = band["fmin"]
                band_freq_max = band["fmax"]

                # In some cases freq_min == freq_max == 0. These are bands which
                # did not see any photons so we cannot construct a spectrum
                # from this

                if band_freq_max > band_freq_min:
                    model_type = band["spec_mod_type"]
                    frequency = np.logspace(np.log10(band_freq_min), np.log10(band_freq_max), self.n_bins_per_band)

                    # There are two model types, a power law or an exponential
                    # model

                    if model_type == CellModelType.powerlaw.value:
                        f_nu = 10**(band["pl_log_w"] + np.log10(frequency) * band["pl_alpha"])
                    elif model_type == CellModelType.exponential.value:
                        f_nu = band["exp_w"] * np.exp((-1 * PLANCK * frequency) / (band["exp_temp"] * BOLTZMANN))
                    else:
                        f_nu = np.zeros_like(frequency)

                    cell_frequency.append(frequency)
                    cell_flux.append(f_nu)

            # If we constructed a model, then set it in the array. Otherwise,
            # the cell will still contain None

            if len(cell_flux) != 0:
                cell_frequency = np.hstack(cell_frequency)
                if not np.all(np.diff(cell_frequency) >= 0):
                    raise ValueError("frequency bins are not increasing across the bands, for some reason")
                cell_flux = np.hstack(cell_flux)
                self.spectra[i, j] = {"Freq.": cell_frequency, "Flux": cell_flux}
                self.cells.append((i, j))

    def create_wind_tables(self):
        """Force the creation of wind save tables for the model.

        This is best used when a simulation has been re-run, as the
        library is unable to detect when the currently available wind
        tables do not reflect a new simulation. This function will
        create the standard wind tables, as well as the fractional and
        density ion tables and create the xspec cell spectra files.
        """

        create_wind_save_tables(self.root, self.fp, ion_density=True)
        create_wind_save_tables(self.root, self.fp, ion_density=False)
        create_wind_save_tables(self.root, self.fp, cell_spec=True)

    def get_elem_number_from_ij(self, i, j):
        """Get the wind element number for a given i and j index.

        Used when indexing into a 1D array, such as in Python itself.

        Parameters
        ----------
        i: int
            The i-th index of the cell.
        j: int
            The j-th index of the cell.
        """
        return int(self.nz * i + j)

    def get_ij_from_elem_number(self, elem):
        """Get the i and j index for a given wind element number.

        Used when converting a wind element number into two indices for use
        in this package.

        Parameters
        ----------
        elem: int
            The element number.
        """
        return np.unravel_index(elem, (self.nx, self.nz))

    def get_model_spectra(self, delim):
        """Read in the model spectrum parameters.

        The model spectrum parameters are read into self.model_parameters and
        are stored such as model_parameters[i, j][nband] where i and j are the
        cell indices and nband is a band number. Each model parameter is
        stored as a dict of the different parameters.

        Parameters
        ----------
        delim: str
            The file delimiter.
        """
        fp = self.fp + self.root + ".spec.txt"
        if not path.exists(fp):
            fp = self.fp + "tables/" + self.root + ".spec.txt"
            if not path.exists(fp):
                raise IOError(f"unable to find a {self.root}.spec file")

        # Read the file in, obviously, store all the lines into a numpy array
        # other than the header

        with open(fp, "r") as f:
            models = f.readlines()

        model_lines = []

        for line in models:
            line = line.strip()
            if delim:
                line = line.split(delim)
            else:
                line = line.split()
            if len(line) == 0 or line[0] == "#":
                continue
            model_lines.append(line)

        models = np.array(model_lines[1:], dtype=np.float64)

        # Now we can determine the parameters read in, the number of bands
        # and the number of cells in the wind

        self.header = model_lines[0]
        self.n_bands = int(np.max(models[:, self.header.index("nband")])) + 1

        self.nx = int(np.max(models[:, self.header.index("i")])) + 1
        try:
            self.nz = int(np.max(models[:, self.header.index("j")])) + 1
        except ValueError:
            self.nz = 1
        self.n_cells = self.nx * self.nz

        # Store the model parameters as model_parameters[i, j][n_band]
        # where i and j are cell indices and n_band is the band number

        self.model_parameters = np.array([None for _ in range(self.n_cells)]).reshape(self.nx, self.nz)

        for nelem in range(self.n_cells):
            bands = []
            for n in range(self.n_bands):
                band = {}
                for m, col in enumerate(self.header):
                    band[col] = models[nelem + (n * self.n_cells), m]
                bands.append(band)

            i, j = self.get_ij_from_elem_number(nelem)
            self.model_parameters[i, j] = bands

    def plot(self, i, j=0, energetic=True, scale="loglog", fig=None, ax=None, figsize=(12, 6)):
        """Plot the spectrum for cell (i, j).

        Simple plotting function, if you want something more advanced then
        check out pypython.plot.spectrum.

        Parameters
        ----------
        i: int
            The i-th index for the cell.
        j: int
            The j-th index for the cell.
        energetic: bool
            Plot in energetic (nu * J_nu) units.
        scale: str
            The axes scaling for the plot.
        fig: pyplot.Figure
            A matplotlib Figure object to edit. Needs ax to also be supplied.
        ax: pyplot.Axes
            A matplotlib Axes object to edit. Needs fig to also be supplied.
        figsize: tuple(int, int)
            The size of the figure (width, height) in inches (sigh).
        """
        model = self.spectra[i, j]
        if model is None:
            raise ValueError(f"no modelled cell spectra for cell ({i}, {j}) as cell is probably not in the wind")

        if not fig and not ax:
            fig, ax = plt.subplots(figsize=figsize)
        elif not fig and ax or fig and not ax:
            raise ValueError("fig and ax need to be supplied together")

        if energetic:
            ax.plot(model["Freq."], model["Freq."] * model["Flux"], label="Model")
            ax.set_ylabel(r"$\nu J_{\nu}$ [ergs s$^{-1}$ cm$^{-2}$]")
        else:
            ax.plot(model["Freq."], model["Flux"], label="Model")
            ax.set_ylabel(r"$J_{\nu}$ [ergs s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")

        ax.set_xlabel("Rest-frame Frequency")
        ax = set_axes_scales(ax, scale)
        fig.suptitle(f"Model in cell ({i}, {j})")

        return fig, ax

    @staticmethod
    def show(block=True):
        """Show a plot which has been created.

        Wrapper around pyplot.show().

        Parameters
        ----------
        block: bool
            Use blocking or non-blocking figure display.
        """
        plt.show(block=block)

    # Built in stuff -----------------------------------------------------------

    def __getitem__(self, pos):
        if type(pos) is int:
            i, j = pos, 0
        else:
            i, j = pos
        return self.spectra[i, j]

    def __setitem__(self, pos, value):
        if type(pos) is int:
            i, j = pos, 0
        else:
            i, j = pos
        self.spectra[i, j] = value

    def __str__(self):
        return print(self.spectra)


# Wind class -------------------------------------------------------------------


class Wind:
    """A class to store 1D and 2D Python wind tables.

    Contains methods to extract variables, as well as convert various
    indices into other indices.
    todo: I should include kwargs in a bunch of these functions
    """
    def __init__(self,
                 root,
                 fp=".",
                 distance_units="cm",
                 velocity_units="kms",
                 co_mass=None,
                 get_cell_spec=True,
                 masked=True,
                 force_make_tables=False,
                 version=None,
                 delim=None):
        """Initialize the Wind object.

        Each of the available wind save tables or ion tables are read in, and
        stored in the same dictionary. To access each paramter, using the get()
        method is perferred. However, it is also possible to index using the
        regular [ ] operator. To get an ion, the index is as follows
        w["H"]["density"]["i01"]. If using get(), it is insted as
        get("H_i01d") or get("H_i01f") for the ion fractions.

        Parameters
        ----------
        root: str
            The root name of the Python simulation.
        fp: str
            The directory containing the model.
        get_cell_spec: bool
            Load in the the cell spectra as well.
        distance_units: str
            The distance units of the wind.
        co_mass: float
            The mass of the central object, optional to use in conjunction with
            distance_units == "rg"
        velocity_units: str [optional]
            The velocity units of the wind.
        masked: bool [optional]
            Store the wind parameters as masked arrays.
        force_make_tables: bool [optional]
            Force windsave2table to be run to re-make the files in the
            tables directory.
        delim: str [optional]
            The delimiter used in the wind table files.
        """
        self.root = root

        self.fp = path.expanduser(fp)
        if self.fp[-1] != "/":
            self.fp += "/"
        self.pf = self.fp + self.root + ".pf"

        self.version = version

        # Set initial conditions for all of the important variables

        self.nx = 1
        self.nz = 1
        self.n_elem = 1
        self.x_coords = None
        self.y_coords = None
        self.x_cen_coords = None
        self.y_cen_coords = None
        self.axes = None
        self.parameters = ()
        self.elements = ()
        self.spectra = ()
        self.wind = _AttributeDict({})
        self.coord_system = WindCoordSystem.unknown
        self.gravitational_radius = -1

        # Get the conversion factors for distance and velocity units

        distance_units = WindDistanceUnits(distance_units.lower())
        self._set_velocity_conversion_factor(velocity_units)

        # Now we can read in the different elements of the wind save tables and
        # initialize most of the variables above. If no files are found, then
        # windsave2table is automatically run. If that doesn't work, then the
        # will script raise an exception. It is also possible to force the
        # re-creation of the tables.

        if force_make_tables:
            self.create_wind_tables()

        try:
            self.get_all_tables(get_cell_spec, delim)
        except IOError:
            self.create_wind_tables()
            self.get_all_tables(get_cell_spec, delim)

        self.columns = self.wind.keys()

        if masked:
            self.create_masked_arrays()

        # Now we can convert or set the units, as the wind has been read in

        self.spatial_units = "unknown"

        if distance_units == WindDistanceUnits.rg:
            self.convert_cm_to_rg(co_mass)  # self.spatial_units is set in here whenever we convert between units
        else:
            self.spatial_units = distance_units

    # Private methods ----------------------------------------------------------

    def _get_element_variable(self, element_name, ion_name):
        """Helper function to get the fraction or density of an ion depending
        on the final character of the requested variable.

        Parameters
        ----------
        element_name: str
            The element symbol, i.e. H, He.
        ion_name: str
            The name of the element, in the format i_01, i_01f, i_01d, etc.
        """

        ion_frac_or_den = ion_name[-1]
        if not ion_frac_or_den.isdigit():
            ion_name = ion_name[:-1]
            if ion_frac_or_den == "d":
                variable = self.wind[element_name]["density"][ion_name]
            elif ion_frac_or_den == "f":
                variable = self.wind[element_name]["fraction"][ion_name]
            else:
                raise ValueError(f"{ion_frac_or_den} is an unknown ion type, try f or d")
        else:
            variable = self.wind[element_name]["density"][ion_name]

        return variable

    def _get_wind_coordinates(self):
        """Get coordinates of the wind.

        This returns the 2d array of coordinate points for the grid or
        1d depending on the coordinate type. This is different from
        using self.n_coords which returns only the axes points.
        """
        if self.coord_system == WindCoordSystem.spherical:
            n_points = self.wind["r"]
            m_points = np.zeros_like(n_points)
        elif self.coord_system == WindCoordSystem.cylindrical:
            n_points = self.wind["x"]
            m_points = self.wind["z"]
        else:
            m_points = np.log10(self.wind["r"])
            n_points = np.deg2rad(self.wind["theta"])

        return n_points, m_points

    def _get_wind_indices(self):
        """Get cell indices of the grid cells of the wind.

        This returns the 2d array of grid cell indices for the grid or
        1d depending on the coordinate type.
        """
        if self.coord_system == WindCoordSystem.spherical:
            n_points = self.wind["i"]
            m_points = np.zeros_like(n_points)
        elif self.coord_system == WindCoordSystem.cylindrical:
            n_points = self.wind["i"]
            m_points = self.wind["j"]
        else:
            raise ValueError("cannot plot with the cell indices for polar winds due to how matplotlib works")

        return n_points, m_points

    def _mask_ions_for_element(self, element_to_mask):
        """Create masked arrays for a single element.

        This acts as a wrapper function to reduce the number of indented for
        loops, improving readability of created_masked_arrays().

        Parameters
        ----------
        element_to_mask: str
            The name of the element to mask.
        """
        element = self.wind[element_to_mask]

        for ion_type in element.keys():
            for ion in element[ion_type]:
                element[ion_type][ion] = np.ma.masked_where(self.wind["inwind"] < 0, element[ion_type][ion])

        self.wind[element_to_mask] = element

    @staticmethod
    def _rename_j_to_j_bar(table, header):
        """Rename j, the mean intensity, to j_bar.

        In old versions of windsave2table, the mean intensity of the field
        was named j which created a conflict for 2D models which have a j
        cell index.

        Parameters
        ----------
        table: str
            The name of the table
        header: list[str]
            Rename the header for j to j_bar.
        """
        count = header.count("j")
        if count < 1:
            return header
        if count > 2:
            raise ValueError(f"unexpected format: too many j's ({count}) in header for {table} table")

        if "z" in header:
            if count == 1:  # This avoids new windsave2table where J -> j_bar
                return header
            idx = header.index("j") + 1
            idx += header[idx:].index("j")
            header[idx] = "j_bar"
        else:
            idx = header.index("j")
            header[idx] = "j_bar"

        return header

    def _set_velocity_conversion_factor(self, units):
        """Set the velocity conversion factor.

        Parameters
        ----------
        units: str
            The velocity units.
        """
        self.velocity_units = WindVelocityUnits(units.lower())

        if self.velocity_units == WindVelocityUnits.kms:
            self.velocity_conversion_factor = CMS_TO_KMS
        elif self.velocity_units == WindVelocityUnits.cms:
            self.velocity_conversion_factor = 1
        else:
            self.velocity_conversion_factor = 1 / VLIGHT

    def _setup_coords(self):
        """Set up the various coordinate variables."""

        # Setup the x/r coordinates

        if "x" in self.parameters:
            self.x_coords = tuple(np.unique(self.wind["x"]))
            self.x_cen_coords = tuple(np.unique(self.wind["xcen"]))
        else:
            self.x_coords = tuple(np.unique(self.wind["r"]))
            self.x_cen_coords = tuple(np.unique(self.wind["rcen"]))

        # Setup the z/theta coordinates

        if "z" in self.parameters:
            self.y_coords = tuple(np.unique(self.wind["z"]))
            self.y_cen_coords = tuple(np.unique(self.wind["zcen"]))
        elif "theta" in self.parameters:
            self.y_coords = tuple(np.unique(self.wind["theta"]))
            self.y_cen_coords = tuple(np.unique(self.wind["theta_cen"]))

        # Set up the coordinate system in used and the axes names available

        if self.nz == 1:
            self.coord_system = WindCoordSystem.spherical
            self.axes = ["r", "r_cen"]
        elif "r" in self.parameters and "theta" in self.parameters:
            self.coord_system = WindCoordSystem.polar
            self.axes = ["r", "theta", "r_cen", "theta_cen"]
        else:
            self.coord_system = WindCoordSystem.cylindrical
            self.axes = ["x", "z", "x_cen", "z_cen"]

    # Methods ------------------------------------------------------------------

    def convert_cm_to_rg(self, co_mass_in_msol=None):
        """Convert the spatial units from cm to gravitational radius.

        If the mass of the central source isn't supplied, then the parameter
        file will be searched for the mass.

        Parameters
        ----------
        co_mass_in_msol: float
            The mass of the central object in solar masses.

        Returns
        -------
        rg: float
            The gravitational radius for the model.
        """

        if self.spatial_units == WindDistanceUnits.rg:
            return

        if not co_mass_in_msol:
            try:
                co_mass_in_msol = float(get_parameter_value(self.pf, "Central_object.mass(msol)"))
            except Exception as e:
                print(e)
                raise ValueError("unable to find CO mass from parameter file, please supply the mass instead")

        rg = gravitational_radius(co_mass_in_msol)

        if self.coord_system in [WindCoordSystem.spherical, WindCoordSystem.polar]:
            self.wind["r"] /= rg
        else:
            self.wind["x"] /= rg
            self.wind["z"] /= rg

        self.spatial_units = WindDistanceUnits.rg
        self.gravitational_radius = rg
        self._setup_coords()

        return rg

    def convert_rg_to_cm(self, co_mass_in_msol=None):
        """Convert the spatial units from gravitational radius to cm.

        If the mass of the central source isn't supplied, then the parameter
        file will be searched for the mass.

        Parameters
        ----------
        co_mass_in_msol: float
            The mass of the central object in solar masses.

        Returns
        -------
        rg: float
            The gravitational radius for the model.
        """

        if self.spatial_units == WindDistanceUnits.cm:
            return

        if not co_mass_in_msol:
            try:
                co_mass_in_msol = float(get_parameter_value(self.pf, "Central_object.mass(msol)"))
            except Exception as e:
                print(e)
                raise ValueError("unable to find CO mass from parameter file, please supply the mass instead")

        rg = gravitational_radius(co_mass_in_msol)

        if self.coord_system == WindCoordSystem.spherical or self.coord_system == WindCoordSystem.polar:
            self.wind["r"] *= rg
        else:
            self.wind["x"] *= rg
            self.wind["z"] *= rg

        self.spatial_units = WindDistanceUnits.cm
        self.gravitational_radius = rg
        self._setup_coords()

        return rg

    def convert_velocity_units(self, units):
        """Convert the velocity units of the outflow.

        The velocity is first converted back into the base units of the model.
        Then the velocity conversion factor is set again and the units are
        converted using this.

        Parameters
        ----------
        units: str
            The new velocity units for the outflow.
        """
        print("warning: this function is currently untested")

        if units == self.velocity_units:
            return

        # Convert the units back into the base units, i.e. cm/s

        self.wind["v_x"] /= self.velocity_conversion_factor
        self.wind["v_y"] /= self.velocity_conversion_factor
        self.wind["v_z"] /= self.velocity_conversion_factor

        if self.coord_system == WindCoordSystem.cylindrical:
            self.wind["v_l"] /= self.velocity_conversion_factor
            self.wind["v_rot"] /= self.velocity_conversion_factor
            self.wind["v_r"] /= self.velocity_conversion_factor

        # Set the new unit conversion scale

        self._set_velocity_conversion_factor(units)

        # Now convert into the new units requested

        self.wind["v_x"] *= self.velocity_conversion_factor
        self.wind["v_y"] *= self.velocity_conversion_factor
        self.wind["v_z"] *= self.velocity_conversion_factor

        if self.coord_system == WindCoordSystem.cylindrical:
            self.wind["v_l"] *= self.velocity_conversion_factor
            self.wind["v_rot"] *= self.velocity_conversion_factor
            self.wind["v_r"] *= self.velocity_conversion_factor

    def create_masked_arrays(self):
        """Mask cells which are not in the wind.

        Convert each array into a masked array using the in-wind. This
        is helpful when using pcolormesh, as matplotlib will ignore
        masked out cells so there will be no background colour to a
        color plot.
        """
        to_mask_wind = list(self.parameters)

        # Create masked array for wind parameters
        # Remove some of the columns from the standard wind parameters, as these
        # shouldn't be masked otherwise weird things will happen

        for item_to_remove in [
                "x", "z", "r", "theta", "xcen", "zcen", "rcen", "theta_cen", "i", "j", "inwind", "cell_spec"
        ]:
            try:
                to_mask_wind.remove(item_to_remove)
            except ValueError:  # sometimes a key wont exist and this catches it
                continue

        for col in to_mask_wind:
            self.wind[col] = np.ma.masked_where(self.wind["inwind"] < 0, self.wind[col])

        # Create masked arrays for the wind ions, loop over each element read in

        for element in self.elements:
            self._mask_ions_for_element(element)

        # Create masked array for cell and model spectra, but only in a special
        # mode set by a global variable lol

        if DEBUG_MASK_CELL_SPEC:
            for cell_spec_type in self.spectra:
                self.wind[cell_spec_type].spectra = np.ma.masked_where(self.wind["inwind"] < 0,
                                                                       self.wind[cell_spec_type].spectra)

    def create_wind_tables(self):
        """Force the creation of wind save tables for the model.

        This is best used when a simulation has been re-run, as the
        library is unable to detect when the currently available wind
        tables do not reflect a new simulation. This function will
        create the standard wind tables, as well as the fractional and
        density ion tables and create the xspec cell spectra files.
        """

        create_wind_save_tables(self.root, self.fp, ion_density=True, version=self.version)
        create_wind_save_tables(self.root, self.fp, ion_density=False, version=self.version)
        create_wind_save_tables(self.root, self.fp, cell_spec=True, version=self.version)

    def get(self, parameter):
        """Get a parameter array. This is just another way to access the
        dictionary self.variables and is a nice wrapper around getting ion
        fractions of densities.

        To get an ion fraction or density use, e.g., C_i04f or C_i04d
        respectively.

        Parameters
        ----------
        parameter: str
            The name of the parameter to get.
        """
        element_name = parameter[:2].replace("_", "")
        ion_name = parameter[2:].replace("_", "")

        if element_name in self.elements:
            variable = self._get_element_variable(element_name, ion_name)
        else:
            variable = self.wind[parameter]

        return variable

    def get_all_tables(self, get_cell_spec, delim):
        """Read all the wind save tables into the object.

        Parameters
        ----------
        get_cell_spec: bool
            Also read in the cell spectra. In old version of Python, these were
            not available.
        delim: str
            The file delimiter.
        """
        self.get_wind_parameters(delim)
        self.get_wind_elements(delim=delim)
        if get_cell_spec:
            try:  # In case someone asks for cell spec when we can't get them
                self.wind["cell_spec"] = CellSpectra(self.root, self.fp, self.nx, self.nz)
                self.wind["cell_model"] = ModelledCellSpectra(self.root, self.fp)
                self.spectra += ("cell_spec", "cell_model")
            except ValueError:
                print(f"unable to load cell or model spectra for model {self.fp}{self.root} as windsave2table is too "
                      f"old")

    def get_elem_number_from_ij(self, i, j):
        """Get the wind element number for a given i and j index.

        Used when indexing into a 1D array, such as in Python itself.

        Parameters
        ----------
        i: int
            The i-th index of the cell.
        j: int
            The j-th index of the cell.
        """
        return self.nz * i + j

    def get_ij_from_elem_number(self, elem):
        """Get the i and j index for a given wind element number.

        Used when converting a wind element number into two indices for use
        in this package.

        Parameters
        ----------
        elem: int
            The element number.
        """
        return np.unravel_index(elem, (self.nx, self.nz))

    def get_sight_line_coordinates(self, theta):
        """Get the vertical z coordinates for a given set of x coordinates and
        inclination angle.

        Parameters
        ----------
        theta: float
            The angle of the sight line to extract from. Given in degrees.
        """
        return np.array(self.x_coords, dtype=np.float64) * np.tan(PI / 2 - np.deg2rad(theta))

    def get_variable_along_sight_line(self, theta, parameter):
        """Extract a variable along a given sight line.

        Parameters
        ----------
        theta: float
            The angle to extract along.
        parameter: str
            The parameter to extract.
        """
        if self.coord_system == WindCoordSystem.polar:
            raise NotImplementedError("This hasn't been implemented for polar winds, lol")

        if type(theta) is not float:
            theta = float(theta)

        z_array = np.array(self.y_coords, dtype=np.float64)
        z_coords = self.get_sight_line_coordinates(theta)

        values = np.zeros_like(z_coords, dtype=np.float64)
        w_array = self.get(parameter)

        for x_index, z in enumerate(z_coords):  # todo: I wonder if this can be vectorized
            z_index = get_array_index(z_array, z)
            values[x_index] = w_array[x_index, z_index]
        values = np.nan_to_num(values)

        return np.array(self.x_coords), z_array, values

    def get_wind_elements(self, elements_to_get=None, delim=None):
        """Read in the ion parameters.

        Reads each element in and its ions into a dictionary. This function will
        try to read in both ion fractions and densities.

        Each element will have a dict of two keys, either fraction or density.
        Inside each dict with be more dicts of keys of the available ions for
        that element.

        Parameters
        ----------
        elements_to_get: List[str] or Tuple[str]
            The elements to read ions in for.
        delim: str [optional]
            The file delimiter.
        """
        if elements_to_get is None:
            elements_to_get = ("H", "He", "C", "N", "O", "Si", "Fe")
        else:
            if type(elements_to_get) not in [str, list, tuple]:
                raise TypeError("ions_to_get should be a tuple/list of strings or a string")

        # Loop over each element wind table

        n_elements_read = 0

        for element in elements_to_get:
            element = element.capitalize()
            self.wind[element] = _AttributeDict({})

            # Loop over the different ion "types", i.e. frac or den.
            # ion_type_index_name is used to give the name for the keys in the
            # dictionary as frac and den is used in python, but I prefer to use
            # fraction and density

            for ion_type, ion_type_index_name in zip(["frac", "den"], ["fraction", "density"]):

                fp = self.fp + self.root + "." + element + "." + ion_type + ".txt"

                if not path.exists(fp):
                    fp = self.fp + "tables/" + self.root + "." + element + "." + ion_type + ".txt"
                    if not path.exists(fp):
                        continue

                n_elements_read += 1

                if element not in self.elements:
                    self.elements += element,

                with open(fp, "r") as f:
                    ion_file = f.readlines()

                wind_lines = []

                for line in ion_file:
                    if delim:
                        line = line.split(delim)
                    else:
                        line = line.split()
                    if len(line) == 0 or line[0] == "#":
                        continue
                    wind_lines.append(line)

                # Now construct the dict of ions, in the layout as described
                # in the doc string. First, we have to find out where the output
                # we actually wants start

                if wind_lines[0][0].isdigit() is False:
                    columns = tuple(wind_lines[0])
                    index = columns.index("i01")
                else:
                    columns = tuple(np.arange(len(wind_lines[0]), dtype=np.dtype.str))
                    index = 0

                columns = columns[index:]
                wind_lines = np.array(wind_lines[1:], dtype=np.float64)[:, index:]

                # Now we can populate the dictionary with the different columns
                # of the file

                self.wind[element][ion_type_index_name] = _AttributeDict({})
                for index, col in enumerate(columns):
                    self.wind[element][ion_type_index_name][col] = wind_lines[:, index].reshape(self.nx, self.nz)

        if n_elements_read == 0 and len(self.columns) == 0:
            raise IOError(
                "Unable to open any parameter or ion tables: try running windsave2table with the correct version")

    def get_wind_parameters(self, delim=None):
        """Read in the wind parameters.

        This reads in the master, heat, gradient, converge and spec file into
        a dictionary. Each header of the tables is a key in the dictionary.

        Parameters
        ----------
        delim: str [optional]
            The deliminator in the wind table files.
        """
        wind_all = []
        wind_columns = []

        # Read in each file, one by one, if they exist. This makes the
        # assumption that all the tables are the same size.

        n_read = 0
        files_to_read = ["master", "heat", "gradient", "converge"]

        for table in files_to_read:
            fp = self.fp + self.root + "." + table + ".txt"
            if not path.exists(fp):
                fp = self.fp + "tables/" + self.root + "." + table + ".txt"
                if not path.exists(fp):
                    continue
            n_read += 1

            with open(fp, "r") as f:
                wind_file = f.readlines()

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
            # current file into wind_all, the list of lists, the master list and
            # Add the wind parameters to the wind_all list, but not the
            # header

            if wind_list[0][0].isdigit() is False:
                header = wind_list[0]
                if table == "heat":
                    header = self._rename_j_to_j_bar(table, header)
                wind_columns += header
            else:
                wind_columns += list(np.arange(len(wind_list[0]), dtype=np.dtype.str))

            wind_all.append(np.array(wind_list[1:], dtype=np.float64))

        if n_read == 0:
            raise IOError(f"Unable to open any wind tables for root {self.root} directory {self.fp}")

        # Determine the number of nx and nz elements. There is a basic check to
        # only check for nz if a 'j' column exists, i.e. if it is a 2d model.

        i_col = wind_columns.index("i")
        self.nx = int(np.max(wind_all[0][:, i_col]) + 1)

        if "z" in wind_columns or "theta" in wind_columns:
            j_col = wind_columns.index("j")
            self.nz = int(np.max(wind_all[0][:, j_col]) + 1)
        self.n_elem = int(self.nx * self.nz)

        # Now we join the different wind tables together and create a dictionary
        # of all of the parameters. We also can now set up the list of available
        # parameters and setup the coordinate parameters too

        wind_all = np.hstack(wind_all)

        for index, col in enumerate(wind_columns):
            if col in self.wind.keys():
                continue
            self.wind[col] = wind_all[:, index].reshape(self.nx, self.nz)

        self.parameters = tuple(self.wind.keys())
        self._setup_coords()

        # Convert velocity into desired units and also calculate the cylindrical
        # velocities.

        if self.coord_system == WindCoordSystem.cylindrical:
            self.project_cartesian_velocity_to_cylindrical()

        self.wind["v_x"] *= self.velocity_conversion_factor
        self.wind["v_y"] *= self.velocity_conversion_factor
        self.wind["v_z"] *= self.velocity_conversion_factor

    def plot(self, variable_name, use_cell_coordinates=True, scale="loglog", log_variable=True):
        """Create a plot of the wind for the given variable.

        Only one thing can be plotted at once, in their own figure window. More
        advanced plotting things can be found in pypython.plot.wind, or write
        something yourself.

        Parameters
        ----------
        variable_name: str
            The name of the variable to plot. Ions are accessed as, i.e.,
            H_i01, He_i02, etc.
        use_cell_coordinates: bool [optional]
            Plot using the cell coordinates instead of cell index numbers
        scale: str [optional]
            The type of scaling for the axes
        log_variable: bool [optional]
            Plot the variable in logarithmic units
        """
        variable = self.get(variable_name)

        if use_cell_coordinates:
            n_points, m_points = self._get_wind_coordinates()
        else:
            n_points, m_points = self._get_wind_indices()

        if self.coord_system == WindCoordSystem.spherical:
            fig, ax = plot.plot_1d_wind(n_points, variable, self.spatial_units, scale=scale)
        else:
            if log_variable:
                variable = np.log10(variable)
            fig, ax = plot.plot_2d_wind(n_points,
                                        m_points,
                                        variable,
                                        self.spatial_units,
                                        self.coord_system,
                                        scale=scale)

        if len(ax) == 1:
            ax = ax[0, 0]
            title = f"{variable_name}".replace("_", " ")
            if self.coord_system == "spherical":
                ax.set_ylabel(title)
            else:
                ax.set_title(title)

        return fig, ax

    def plot_cell_spec(self, i, j, energetic=False):
        """Plot the spectrum for a cell.

        This will plot the cell spectrum and the model of that spectrum used
        internally by Python to calculate the heating/cooling rates.

        Parameters
        ----------
        i: int
            The i-th index for the cell.
        j: int
            The j-th index for the cell.
        energetic: bool
            Plot in energetic (nu * J_nu) units.
        """
        if not self.spectra:
            raise ValueError("no cell spectra have been read in")

        fig, ax = self.wind["cell_spec"].plot(i, j, energetic)
        fig, ax = self.wind["cell_model"].plot(i, j, energetic, fig=fig, ax=ax)
        ax.legend()

        return fig, ax

    def project_cartesian_velocity_to_cylindrical(self):
        """Project cartesian velocities into cylindrical velocities.

        This makes the variables v_r, v_rot and v_l available in
        variables dictionary. Only works for cylindrical coordinates
        systems, which outputs the velocities in cartesian coordinates.
        """
        v_l = np.zeros_like(self.wind["v_x"])
        v_rot = np.zeros_like(v_l)
        v_r = np.zeros_like(v_l)
        n1, n2 = v_l.shape

        for i in range(n1):
            for j in range(n2):
                cart_point = np.array([self.wind["x"][i, j], 0, self.wind["z"][i, j]])
                # todo: don't think I need to do this check anymore
                if self.wind["inwind"][i, j] < 0:
                    v_l[i, j] = 0
                    v_rot[i, j] = 0
                    v_r[i, j] = 0
                else:
                    cart_velocity_vector = np.array(
                        [self.wind["v_x"][i, j], self.wind["v_y"][i, j], self.wind["v_z"][i, j]])
                    cyl_velocity_vector = vector.project_cartesian_to_cylindrical_coordinates(
                        cart_point, cart_velocity_vector)
                    if type(cyl_velocity_vector) is int:
                        continue
                    v_l[i, j] = np.sqrt(cyl_velocity_vector[0]**2 + cyl_velocity_vector[2]**2)
                    v_rot[i, j] = cyl_velocity_vector[1]
                    v_r[i, j] = cyl_velocity_vector[0]

        self.wind["v_l"] = v_l * self.velocity_conversion_factor
        self.wind["v_rot"] = v_rot * self.velocity_conversion_factor
        self.wind["v_r"] = v_r * self.velocity_conversion_factor

    @staticmethod
    def show(block=True):
        """Show a plot which has been created.

        Wrapper around pyplot.show().

        Parameters
        ----------
        block: bool
            Use blocking or non-blocking figure display.
        """
        plt.show(block=block)

    # Built in stuff -----------------------------------------------------------

    def __getattr__(self, key):
        return self.wind[key]

    def __getitem__(self, key):
        return self.wind[key]

    # def __setattr__(self, key, value):
    #     self.wind[key] = value

    def __setitem__(self, key, value):
        self.wind[key] = value

    def __str__(self):
        txt = f"root: {self.root}\nfilepath: {self.fp}\ncoordinate system: {self.coord_system}\n" \
              f"parameters: {self.parameters}\nelements: {self.elements}\nspectra: {self.spectra}"

        return textwrap.dedent(txt)


# This is placed here due to a circular dependency

from pypython.wind import plot
