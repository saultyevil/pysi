#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sub-class containing plotting functions.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy

from pypython import plot
from pypython.wind import util, enum


class WindPlot(util.WindUtil):
    """An extension to the WindGrid base class which adds various plotting
    functionality.

    # TODO: add units for things like velocity, obvious shit
    # TODO: add general look up unit table for common quantities
    """

    DISTANCE_AXIS_LABEL_LOOKUP = {
        enum.DistanceUnits.CENTIMETRES: "[cm]",
        enum.DistanceUnits.METRES: "[m]",
        enum.DistanceUnits.KILOMETRES: "[km]",
        enum.DistanceUnits.GRAVITATIONAL_RADIUS: r"$ / R_{g}$",
    }

    def __init__(self, root: str, directory: str, **kwargs):
        """Initialize the class.

        Parameters
        ----------
        root: str
            The root name of the simulation.
        directory: str
            The directory containing the simulation.
        """
        super().__init__(root, directory, **kwargs)

    # pylint: disable=too-many-arguments
    def plot_parameter(
        self,
        thing: str,
        axes_scales: str = "loglog",
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        figsize: Tuple[int, int] = (8, 6),
        a_idx: int = 0,
        a_jdx: int = 0,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a wind parameter.

        Parmeters
        ---------
        thing: str
            The name of the parameter to plot.

        Returns
        -------
        fig: plt.Figure
            The create Figure object, containing the axes.
        ax: plt.Axes
            The axes object for the plot.
        """
        if not fig and not ax:
            if self.coord_type == enum.CoordSystem.POLAR:
                subplot_kw = {"projection": "polar"}
            else:
                subplot_kw = None
            fig, ax = plt.subplots(figsize=figsize, squeeze=False, subplot_kw=subplot_kw)
        elif not fig and ax or fig and not ax:
            raise ValueError("fig and ax need to be supplied together")

        if self.coord_type == enum.CoordSystem.SPHERICAL:
            fig, ax = self.__wind1d(thing, axes_scales, fig, ax, a_idx, a_jdx, **kwargs)
        else:
            fig, ax = self.__wind2d(thing, axes_scales, fig, ax, a_idx, a_jdx, **kwargs)

        return fig, ax

    def plot_parameter_along_sightline(self):
        """Plot a variable along an given inclination angle."""
        pass

    # pylint: disable=too-many-arguments
    def plot_cell_spectrum(
        self,
        idx: int,
        jdx: int = 0,
        axes_scales: str = "loglog",
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        figsize: Tuple[int, int] = (12, 6),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a spectrum for a wind cell.

        Creates (and returns) a figure

        Parameters
        ----------
        i: int
            The i-th cell index
        j: int [optional]
            The j-th cell index
        axes_scales: str
            The scale types for each axis.

        Returns
        -------
        fig: plt.Figure
            The create Figure object, containing the axes.
        ax: plt.Axes
            The axes object for the plot.
        """
        if self.parameters["spec_flux"] is None:
            raise ValueError("There are no cell spectra for this simulation.")

        if not fig and not ax:
            fig, ax = plt.subplots(figsize=figsize)
        elif not fig and ax or fig and not ax:
            raise ValueError("fig and ax need to be supplied together")

        if self.coord_type == "spherical":
            spectrum = self.parameters["spec_flux"][idx]
            freq = self.parameters["spec_freq"][idx]
        else:
            spectrum = self.parameters["spec_flux"][idx, jdx]
            freq = self.parameters["spec_freq"][idx, jdx]

        ax.plot(freq, freq * spectrum, label="Spectrum")
        ax.set_xlabel(r"Rest-frame Frequency [$\nu$]")
        ax.set_ylabel(r"$\nu ~ J_{\nu}$ [ergs s$^{-1}$ cm$^{-2}$]")

        ax = plot.set_axes_scales(ax, axes_scales)
        fig = plot.finish_figure(fig, f"Cell ({idx}, {jdx}) spectrum")

        return fig, ax

    # pylint: disable=too-many-arguments
    def plot_cell_model(
        self,
        idx: int,
        jdx: int = 0,
        axes_scales: str = "loglog",
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        figsize: Tuple[int, int] = (12, 6),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a spectrum for a wind cell.

        Creates (and returns) a figure

        Parameters
        ----------
        i: int
            The i-th cell index
        j: int [optional]
            The j-th cell index
        axes_scales: str
            The scale types for each axis.

        Returns
        -------
        fig: plt.Figure
            The create Figure object, containing the axes.
        ax: plt.Axes
            The axes object for the plot.
        """
        if self.parameters["model_flux"] is None:
            raise ValueError("There are no cell models for this simulation.")

        if not fig and not ax:
            fig, ax = plt.subplots(figsize=figsize)
        elif not fig and ax or fig and not ax:
            raise ValueError("fig and ax need to be supplied together")

        print(self.coord_type)

        if self.coord_type == enum.CoordSystem.SPHERICAL:
            spectrum = self.parameters["model_flux"][idx]
            freq = self.parameters["model_freq"][idx]
        else:
            spectrum = self.parameters["model_flux"][idx, jdx]
            freq = self.parameters["model_freq"][idx, jdx]

        ax.plot(freq, freq * spectrum, label="Spectrum")
        ax.set_xlabel(r"Rest-frame Frequency [$\nu$]")
        ax.set_ylabel(r"$\nu ~ J_{\nu}$ [ergs s$^{-1}$ cm$^{-2}$]")

        ax = plot.set_axes_scales(ax, axes_scales)
        fig = plot.finish_figure(fig, f"Model ({idx}, {jdx}) spectrum")

        return fig, ax

    def show_figures(self) -> None:
        """Show any plot windows."""
        plt.show()

    # Private methods ----------------------------------------------------------

    def __add_inclination_sight_lines(self, angles, x_points, z_points, fig, ax, **kwargs):
        """Add lines to show what various inclination observers

        Parameters
        ----------
        angles: List[float]

        x_points: numpy.ndarray

        z_points: numpy.ndarray

        fig: plt.Figure

        ax. plt.Axes


        Returns
        -------
        fig: plt.Figure
            The (updated) Figure object for the plot.
        ax: plt.Axes
            The (updated) axes array for the plot.
        """

        n_coords = numpy.unique(x_points)
        for angle in angles:
            if self.coord_type == enum.CoordSystem.CYLINDRICAL:
                m_coords = n_coords * numpy.tan(0.5 * numpy.pi - numpy.deg2rad(float(angle)))
            else:
                x_coords = numpy.logspace(numpy.log10(0), numpy.max(z_points))
                m_coords = x_coords * numpy.tan(0.5 * numpy.pi - numpy.deg2rad(90 - float(angle)))
                m_coords = numpy.sqrt(x_coords**2 + m_coords**2)

            ax[0, 0].plot(n_coords, m_coords, label=angle + r"$^{\circ}$")

        ax[0, 0].legend(loc=kwargs.get("legend_loc", "upper_right"))

        return fig, ax

    def __set_wind2d_axes_labels_limits(self, ax, scale, x_points, z_points, a_idx, a_jdx):
        """Set the axes labels and limits for a 2D wind.

        Parameters
        ----------
        ax

        scale

        x_points

        z_points

        a_idx

        a_jdx
        """
        if self.coord_type == enum.CoordSystem.CYLINDRICAL:
            ax[a_idx, a_jdx].set_xlabel(f"$x$ {self.DISTANCE_AXIS_LABEL_LOOKUP[self.distance_units]}")
            ax[a_idx, a_jdx].set_ylabel(f"$z$ {self.DISTANCE_AXIS_LABEL_LOOKUP[self.distance_units]}")
            ax[a_idx, a_jdx].set_xlim(numpy.min(x_points[x_points > 0]), numpy.max(x_points))
            ax[a_idx, a_jdx] = plot.set_axes_scales(ax[a_idx, a_jdx], scale)
        else:
            ax[a_idx, a_jdx].set_theta_zero_location("N")
            ax[a_idx, a_jdx].set_theta_direction(-1)
            ax[a_idx, a_jdx].set_thetamin(0)
            ax[a_idx, a_jdx].set_thetamax(90)
            ax[a_idx, a_jdx].set_rlabel_position(90)
            ax[a_idx, a_jdx].set_ylabel(r"$\log_{10}(r)$" + f" {self.DISTANCE_AXIS_LABEL_LOOKUP[self.distance_units]}")

        ax[a_idx, a_jdx].set_ylim(numpy.min(z_points[z_points > 0]), numpy.max(z_points))

        return ax

    def __wind1d(
        self,
        thing: str,
        axes_scales: str = "logx",
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        a_idx: int = 0,
        a_jdx: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        """Plot a 1D wind.

        Parameters
        ----------
        thing: str
            The name of the parameter to plot.
        scale: str [optional]
            The scaling of the axes: [logx, logy, loglog, linlin]
        fig: plt.Figure [optional]
            A Figure object to update, otherwise a new one will be created.
        ax: plt.Axes [optional]
            An axes array to update, otherwise a new one will be created.
        a_idx: int [optional]
            The i index for the sub panel to plot onto.
        a_jdx: int [optional]
            The j index for the sub panel to plot onto.

        Returns
        -------
        fig: plt.Figure
            The (updated) Figure object for the plot.
        ax: plt.Axes
            The (updated) axes array for the plot.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 6), squeeze=False)
        if thing not in self.parameter_keys:
            raise ValueError(f"Unknown parameter {thing}: {thing} not in wind tables")

        ax[a_idx, a_jdx].plot(self.parameters["r"], self.parameters[thing])
        ax[a_idx, a_jdx].set_xlabel(f"$R$ {self.DISTANCE_AXIS_LABEL_LOOKUP[self.distance_units]}")
        ax[a_idx, a_jdx].set_ylabel(f"{thing}")
        ax[a_idx, a_jdx] = plot.set_axes_scales(ax[a_idx, a_jdx], axes_scales)
        fig = plot.finish_figure(fig)

        return fig, ax

    def __wind2d(
        self,
        thing: str,
        scale: str = "loglog",
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        i: int = 0,
        j: int = 0,
        **kwargs,
    ):
        """Plot a 2D wind using a contour plot.

        Parameters
        ----------
        thing: str
            The name of the parameter to plot.
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
        if fig is None or ax is None:
            if self.coord_type == enum.CoordSystem.CYLINDRICAL:
                fig, ax = plt.subplots(figsize=(8, 6), squeeze=False)
            elif self.coord_type == enum.CoordSystem.POLAR:
                fig, ax = plt.subplots(figsize=(8, 6), squeeze=False, subplot_kw={"projection": "polar"})
            else:
                raise ValueError(
                    f"Unknown projection, expected {enum.CoordSystem.CYLINDRICAL} or {enum.CoordSystem.POLAR}"
                )

        vmin = kwargs.get("vmin", None)
        vmax = kwargs.get("vmax", None)
        inclinations_to_plot = kwargs.get("inclinations_to_plot", None)

        if self.coord_type == enum.CoordSystem.CYLINDRICAL:
            x_points, z_points = self.parameters["x"], self.parameters["z"]
        else:
            x_points, z_points = numpy.deg2rad(self.parameters["theta"]), numpy.log10(self.parameters["r"])

        if thing not in self.parameter_keys:
            raise ValueError(f"Unknown parameter {thing}: {thing} not in wind tables")
        parameter_points = self.parameters[thing]

        im = ax[i, j].pcolormesh(
            x_points,
            z_points,
            numpy.log10(parameter_points),
            shading="auto",
            vmin=vmin,
            vmax=vmax,
            linewidth=0,
            rasterized=True,
        )

        fig.colorbar(im, ax=ax[i, j])

        if inclinations_to_plot:
            fig, ax = self.__add_inclination_sight_lines(inclinations_to_plot, x_points, z_points, fig, ax, **kwargs)

        ax = self.__set_wind2d_axes_labels_limits(ax, scale, x_points, z_points, i, j)
        fig = plot.finish_figure(fig)

        return fig, ax
