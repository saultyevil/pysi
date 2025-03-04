"""Sub-class containing plotting functions."""

from __future__ import annotations

import pathlib
import typing

import matplotlib.pyplot as plt
import numpy

from pysi.util import plot
from pysi.wind import enum
from pysi.wind.model import util


class WindPlot(util.WindUtil):
    """An extension to the WindGrid base class adds plotting utilities.

    TODO: add general look up unit table for common quantities
    """

    DISTANCE_AXIS_LABEL_LOOKUP: typing.ClassVar = {
        enum.DistanceUnits.CENTIMETRES: "[cm]",
        enum.DistanceUnits.METRES: "[m]",
        enum.DistanceUnits.KILOMETRES: "[km]",
        enum.DistanceUnits.GRAVITATIONAL_RADIUS: r"$ / R_{g}$",
    }

    def __init__(self, root: str, directory: str = pathlib.Path(), **kwargs) -> None:
        """Initialize the class.

        Parameters
        ----------
        root:  str
            The root name of the simulation.
        directory : str
            The directory containing the simulation.
        kwargs : dict
            Various other keywords arguments.

        """
        super().__init__(root, directory, **kwargs)

    def plot_parameter(  # noqa: PLR0913
        self,
        thing: str | numpy.ndarray,
        axes_scales: str = "loglog",
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        figsize: tuple[int, int] = (8, 6),
        a_idx: int = 0,
        a_jdx: int = 0,
        **kwargs: dict,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot a wind parameter.

        Parmeters
        ---------
        thing: str | numpy.ndarray
            The name of the parameter to plot, or a numpy array of the thing
            to plot.
        axes_scales: str [optional]
            The scale types for each axis.
        fig: plt.Figure [optional]
            A Figure object to update, otherwise a new one will be created.
        ax: plt.Axes [optional]
            An axes array to update, otherwise a new one will be created.
        figsize: tuple[int, int] [optional]
            The size of the figure.
        a_idx: int [optional]
            The i index for the sub panel to plot onto.
        a_jdx: int [optional]
            The j index for the sub panel to plot onto.
        kwargs: dict [optional]
            Various other keyword arguments.

        Returns
        -------
        fig: plt.Figure
            The create Figure object, containing the axes.
        ax: plt.Axes
            The axes object for the plot.

        """
        if fig is None and ax is None:
            subplot_kw = {"projection": "polar"} if self.coord_type == enum.CoordSystem.POLAR else None
            fig, ax = plt.subplots(figsize=figsize, squeeze=False, subplot_kw=subplot_kw)
        elif fig is None and ax is not None or fig is None and ax is not None:
            msg = "fig and ax need to be supplied together"
            raise ValueError(msg)

        if self.coord_type == enum.CoordSystem.SPHERICAL:
            fig, ax = self._plot_wind1d(thing, axes_scales, fig, ax, a_idx, a_jdx, **kwargs)
        else:
            fig, ax = self._plot_wind2d(thing, axes_scales, fig, ax, a_idx, a_jdx, **kwargs)

        return fig, ax

    def plot_parameter_along_sightline(self) -> None:
        """Plot a variable along an given inclination angle."""
        msg = "Method is not implemented yet."
        raise NotImplementedError(msg)

    def plot_cell_spectrum(  # noqa: PLR0913
        self,
        idx: int,
        jdx: int = 0,
        axes_scales: str = "loglog",
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        figsize: tuple[int, int] = (12, 6),
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot a spectrum for a wind cell.

        Creates (and returns) a figure

        Parameters
        ----------
        idx: int
            The i-th cell index
        jdx: int [optional]
            The j-th cell index
        axes_scales: str
            The scale types for each axis.
        fig: plt.Figure [optional]
            A Figure object to update, otherwise a new one will be created.
        ax: plt.Axes [optional]
            An axes array to update, otherwise a new one will be created.
        figsize: tuple[int, int] [optional]
            The size of the figure.

        Returns
        -------
        fig: plt.Figure
            The create Figure object, containing the axes.
        ax: plt.Axes
            The axes object for the plot.

        """
        if self.parameters["spec_flux"] is None:
            msg = "There are no cell spectra for this simulation."
            raise ValueError(msg)

        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        elif fig is None and ax is not None or fig is not None and ax is None:
            msg = "fig and ax need to be supplied together"
            raise ValueError(msg)

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

    def plot_cell_model(  # noqa: PLR0913
        self,
        idx: int,
        jdx: int = 0,
        axes_scales: str = "loglog",
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        figsize: tuple[int, int] = (12, 6),
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot a spectrum for a wind cell.

        Creates (and returns) a figure

        Parameters
        ----------
        idx: int
            The i-th cell index
        jdx: int [optional]
            The j-th cell index
        axes_scales: str
            The scale types for each axis.
        fig: plt.Figure [optional]
            A Figure object to update, otherwise a new one will be created.
        ax: plt.Axes [optional]
            An axes array to update, otherwise a new one will be created.
        figsize: tuple[int, int] [optional]
            The size of the figure.

        Returns
        -------
        fig: plt.Figure
            The create Figure object, containing the axes.
        ax: plt.Axes
            The axes object for the plot.

        """
        if self.parameters["model_flux"] is None:
            msg = "There are no cell models for this simulation."
            raise ValueError(msg)

        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        elif fig is None and ax is not None or fig is not None and ax is None:
            msg = "fig and ax need to be supplied together"
            raise ValueError(msg)

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

    def close_figures(self, fig: plt.Figure = None) -> None:
        """Clsoe open figure windows.

        Parameters
        ----------
        fig : plt.Figure, optional
            The specific figure to close, by default None which closes all
            figure windows.

        """
        if fig is not None:
            plt.close(fig)
        else:
            plt.close("all")

    def show_figures(self) -> None:
        """Show any plot windows."""
        plt.show()

    # Private methods ----------------------------------------------------------

    def _add_inclination_sight_lines(
        self,
        angles: numpy.array | list,
        x_points: numpy.array | list,
        z_points: numpy.array | list,
        fig: plt.Figure,
        ax: plt.Axes,
        **kwargs: dict,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Add lines to show what various inclination observers.

        Parameters
        ----------
        angles: List[float]
            The inclination angles to plot
        x_points: numpy.ndarray
            The X coordinates to use (???)
        z_points: numpy.ndarray
            the Z coordinates to use (???)
        fig: plt.Figure
            The Figure object to update
        ax: plt.Axes
            The Axes object to update
        kwargs: dict
            Various other keyword arguments

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

    def _set_wind2d_axes_labels_limits(  # noqa: PLR0913
        self,
        ax: plt.Axes,
        scale: str,
        x_points: numpy.array | list,
        z_points: numpy.array | list,
        a_idx: int,
        a_jdx: int,
    ) -> plt.Axes:
        """Set the axes labels and limits for a 2D wind.

        Parameters
        ----------
        ax: plt.Axes
            The Axes object to update
        scale: str
            The scaling of the axes: [logx, logy, loglog, linlin]
        x_points
            The x points of the wind, used to determine the limits of the x axis
        z_points
            The z points of the wind, used to determine the limits of the z axis
        a_idx
            The i index for the sub panel to plot onto
        a_jdx
            The j index for the sub panel to plot onto

        Returns
        -------
        ax: plt.Axes
            The updated Axes object

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

    def _plot_wind1d(  # noqa: PLR0913
        self,
        thing: str | numpy.ndarray,
        axes_scales: str = "logx",
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        a_idx: int = 0,
        a_jdx: int = 0,
        **kwargs: dict,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot a 1D wind.

        Parameters
        ----------
        thing: str | numpy.ndarray
            The name of the parameter to plot, or a numpy array of the thing
            to plot.
        axes_scales: str [optional]
            The scaling of the axes: [logx, logy, loglog, linlin]
        fig: plt.Figure [optional]
            A Figure object to update, otherwise a new one will be created.
        ax: plt.Axes [optional]
            An axes array to update, otherwise a new one will be created.
        a_idx: int [optional]
            The i index for the sub panel to plot onto.
        a_jdx: int [optional]
            The j index for the sub panel to plot onto.
        kwargs: dict
            Various other keyword arguments

        Returns
        -------
        fig: plt.Figure
            The (updated) Figure object for the plot.
        ax: plt.Axes
            The (updated) axes array for the plot.

        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 6), squeeze=False)

        log_parameter = kwargs.get("log_parameter", True)

        if isinstance(thing, numpy.ndarray):
            thing = ""
            parameter_points = thing
        elif isinstance(thing, str):
            title = thing
            parameter_points = self[thing]
        else:
            raise TypeError(f"Unsupported type {type(self.parameters)} for plotting parameter")

        if log_parameter:
            with numpy.errstate(over="ignore", divide="ignore"):
                parameter_points = numpy.log10(parameter_points)
            ax[a_idx, a_jdx].set_ylabel(r"$\log_{10}(" + f"{title})$")
        else:
            ax[a_idx, a_jdx].set_ylabel(f"{title}")

        ax[a_idx, a_jdx].plot(self.parameters["r"], parameter_points)
        ax[a_idx, a_jdx].set_xlabel(f"$R$ {self.DISTANCE_AXIS_LABEL_LOOKUP[self.distance_units]}")
        ax[a_idx, a_jdx] = plot.set_axes_scales(ax[a_idx, a_jdx], axes_scales)
        fig = plot.finish_figure(fig)

        return fig, ax

    def _plot_wind2d(  # noqa: PLR0913
        self,
        thing: str | numpy.ndarray,
        scale: str = "loglog",
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        a_idx: int = 0,
        a_jdx: int = 0,
        **kwargs: dict,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot a 2D wind using a contour plot.

        Parameters
        ----------
        thing: str | numpy.ndarray
            The name of the parameter to plot, or a numpy array of the thing
            to plot.
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
        kwargs: dict
            Various other keyword arguments

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

        vmin = kwargs.get("vmin")
        vmax = kwargs.get("vmax")
        log_parameter = kwargs.get("log_parameter", True)
        inclinations_to_plot = kwargs.get("inclinations_to_plot")

        if self.coord_type == enum.CoordSystem.CYLINDRICAL:
            x_points, z_points = self.parameters["x"], self.parameters["z"]
        else:
            x_points, z_points = numpy.deg2rad(self.parameters["theta"]), numpy.log10(self.parameters["r"])

        if isinstance(thing, numpy.ndarray):
            title = ""
            parameter_points = thing
        elif isinstance(thing, str):
            title = thing.replace("_", r" ")
            parameter_points = self[thing]
        else:
            raise TypeError(f"Unsupported type {type(self.parameters)} for plotting parameter")

        if log_parameter:
            with numpy.errstate(over="ignore", divide="ignore"):
                parameter_points = numpy.log10(parameter_points)

        if title:
            if log_parameter:
                ax[a_idx, a_jdx].set_title(r"$\log_{10}(" + rf"{title})$")
            else:
                ax[a_idx, a_jdx].set_title(rf"{title}")

        im = ax[a_idx, a_jdx].pcolormesh(
            x_points,
            z_points,
            parameter_points,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
            linewidth=0,
            rasterized=True,
        )

        fig.colorbar(im, ax=ax[a_idx, a_jdx])

        if inclinations_to_plot:
            fig, ax = self._add_inclination_sight_lines(inclinations_to_plot, x_points, z_points, fig, ax, **kwargs)

        ax = self._set_wind2d_axes_labels_limits(ax, scale, x_points, z_points, a_idx, a_jdx)
        fig = plot.finish_figure(fig)

        return fig, ax
