#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, List, Tuple
from matplotlib import pyplot as plt
import numpy as np

from . import normalize_figure_style
# from pypython import Wind
from pypython.physics.constants import PI


def plot_wind(wind,
              parameter: Union[str, np.ndarray],
              use_indices: bool = False,
              log_parameter: bool = True,
              ion_fraction: bool = True,
              inclinations_to_plot: List[str] = None,
              scale: str = "loglog",
              vmin: float = None,
              vmax: float = None,
              fig: plt.Figure = None,
              ax: plt.Axes = None,
              i: int = 0,
              j: int = 0) -> Tuple[plt.Figure, Union[np.ndarray, plt.Axes]]:
    """Wrapper function for plotting a wind. Will decide if a wind is 1D or 2D
    and call the appropriate function.
    Parameters
    ----------
    wind: Wind
        The Wind object.
    parameter: np.ndarray
        The wind parameter to be plotted, in the same shape as the coordinate
        arrays. Can also be the name of the variable.
    log_parameter: bool [optional]
        Plot the wind parameter as log_10(parameters).
    ion_fraction: bool [optional]
        Plot ions as a fraction [True] or a density.
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
    if type(parameter) is str:
        parameter_points = wind.get(parameter, ion_fraction)
        if log_parameter:
            parameter_points = np.log10(parameter_points)
    elif type(parameter) in [np.ndarray, np.ma.core.MaskedArray]:
        parameter_points = parameter
    else:
        print(f"Incompatible type {type(parameter)} for parameter")
        return fig, ax

    # Finally plot the variable depending on the coordinate type

    if wind.coord_system == "spherical":
        if use_indices:
            n = wind["i"]
        else:
            n = wind["r"]
        fig, ax = plot_1d_wind(n, parameter_points, "logx", fig, ax, i, j)
    elif wind.coord_system == "polar":
        if use_indices:
            raise ValueError("use_indices cannot be used with polar winds")
        fig, ax = plot_2d_wind(np.deg2rad(wind["theta"]), np.log10(wind["r"]),
                               parameter_points, wind.coord_system,
                               inclinations_to_plot, scale, vmin, vmax, fig,
                               ax, i, j)
    else:
        if use_indices:
            n = wind["i"]
            m = wind["j"]
        else:
            n = wind["x"]
            m = wind["z"]
        fig, ax = plot_2d_wind(n, m, parameter_points, wind.coord_system,
                               inclinations_to_plot, scale, vmin, vmax, fig, ax,
                               i, j)

    return fig, ax


def plot_1d_wind(m_points: np.ndarray,
                 parameter_points: np.ndarray,
                 scale: str = "logx",
                 fig: plt.Figure = None,
                 ax: plt.Axes = None,
                 i: int = 0,
                 j: int = 0) -> Tuple[plt.Figure, Union[np.ndarray, plt.Axes]]:
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
        The (updated) axes array for the plot.
    """
    normalize_figure_style()

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), squeeze=False)

    if scale not in ["logx", "logy", "loglog", "linlin"]:
        print(
            f"unknown axes scaling {scale}. Allowed: [logx, logy, loglog, linlin]"
        )
        exit(1)

    ax[i, j].plot(m_points, parameter_points)

    ax[i, j].set_xlabel(r"$r$ [cm]")
    ax[i, j].set_xlim(np.min(m_points[m_points > 0]), np.max(m_points))
    if scale == "loglog" or scale == "logx":
        ax[i, j].set_xscale("log")
    if scale == "loglog" or scale == "logy":
        ax[i, j].set_yscale("log")

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    return fig, ax


def plot_2d_wind(m_points: np.ndarray,
                 n_points: np.ndarray,
                 parameter_points: np.ndarray,
                 coordinate_system: str = "rectilinear",
                 inclinations_to_plot: List[str] = None,
                 scale: str = "loglog",
                 vmin: float = None,
                 vmax: float = None,
                 fig: plt.Figure = None,
                 ax: plt.Axes = None,
                 i: int = 0,
                 j: int = 0) -> Tuple[plt.Figure, Union[np.ndarray, plt.Axes]]:
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
        The (updated) axes array for the plot.
    """
    normalize_figure_style()

    if fig is None or ax is None:
        if coordinate_system == "rectilinear":
            fig, ax = plt.subplots(figsize=(7, 5), squeeze=False)
        elif coordinate_system == "polar":
            fig, ax = plt.subplots(figsize=(7, 5),
                                   squeeze=False,
                                   subplot_kw={"projection": "polar"})
        else:
            print(
                f"unknown wind projection {coordinate_system}. Expected rectilinear or polar"
            )
            exit(1)

    if scale not in ["logx", "logy", "loglog", "linlin"]:
        print(
            f"unknown axes scaling {scale}. Allowed: [logx, logy, loglog, linlin]"
        )
        exit(1)

    im = ax[i, j].pcolormesh(m_points,
                             n_points,
                             parameter_points,
                             shading="auto",
                             vmin=vmin,
                             vmax=vmax)
    fig.colorbar(im, ax=ax[i, j])

    # this plots lines representing sight lines for different observers of
    # different inclinations

    if inclinations_to_plot:
        n_coords = np.unique(m_points)
        for inclination in inclinations_to_plot:
            if coordinate_system == "rectilinear":
                m_coords = n_coords * np.tan(0.5 * PI -
                                             np.deg2rad(float(inclination)))
            else:
                x_coords = np.logspace(np.log10(0), np.max(n_points))
                m_coords = x_coords * np.tan(
                    0.5 * PI - np.deg2rad(90 - float(inclination)))
                m_coords = np.sqrt(x_coords**2 + m_coords**2)
            ax[0, 0].plot(n_coords,
                          m_coords,
                          label=inclination + r"$^{\circ}$")
        ax[0, 0].legend(loc="lower left")

    # Clean up the axes with labs and set up scales, limits etc

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
