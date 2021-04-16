#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description of file.
"""

from .physics.constants import PI
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Tuple


plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 14


def plot_1d_wind():
    """Description of function."""
    raise NotImplementedError


def plot_2d_wind(
    n_points: np.ndarray, m_points: np.ndarray, parameter_points: np.ndarray, coordinate_system: str = "rectilinear",
    inclinations_to_plot: List[str] = None, scale: str = "loglog", vmin: float = None, vmax: float = None,
    fig: plt.Figure = None, ax: plt.Axes = None, i: int = 0, j: int = 0
) -> Tuple[plt.Figure, np.ndarray]:
    """Description of function.

    Parameters
    ----------
    n_points: np.ndarray
        The 1st axis points, either x or angular (in degrees) bins.
    m_points: np.ndarray
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

    if fig is None or ax is None:
        if coordinate_system == "rectilinear":
            fig, ax = plt.subplots(figsize=(5, 5), squeeze=False)
        elif coordinate_system == "polar":
            fig, ax = plt.subplots(figsize=(5, 5), squeeze=False, subplot_kw={"projection": "polar"})
        else:
            print("unknown wind projection", coordinate_system)
            print("allowed: ['rectilinear', 'polar']")
            exit(1)  # todo: error code

    if scale not in ["logx", "logy", "loglog", "linlin"]:
        print("unknown axes scaling", scale)
        print("allwoed:", ["logx", "logy", "loglog", "linlin"])
        exit(1)  # todo: error code

    im = ax[i, j].pcolormesh(n_points, m_points, parameter_points, shading="auto", vmin=vmin, vmax=vmax)

    if inclinations_to_plot:
        n_coords = np.unique(n_points)
        for inclination in inclinations_to_plot:
            if coordinate_system == "rectilinear":
                m_coords = n_coords * np.tan(0.5 * PI - np.deg2rad(float(inclination)))
            else:
                x_coords = np.logspace(np.log10(0), np.max(m_points))
                m_coords = x_coords * np.tan(0.5 * PI - np.deg2rad(90 - float(inclination)))
                m_coords = np.sqrt(x_coords ** 2 + m_coords ** 2)
            ax[0, 0].plot(n_coords, m_coords, label=inclination + r"$^{\circ}$")
        ax[0, 0].legend(loc="lower left")

    fig.colorbar(im, ax=ax[i, j])
    if coordinate_system == "rectilinear":
        ax[i, j].set_xlabel("x [cm]")
        ax[i, j].set_ylabel("z [cm]")
        ax[i, j].set_xlim(np.min(n_points[n_points > 0]), np.max(n_points))
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
        ax[i, j].set_ylabel("R [cm]")
    ax[i, j].set_ylim(np.min(m_points[m_points > 0]), np.max(m_points))

    return fig, ax
