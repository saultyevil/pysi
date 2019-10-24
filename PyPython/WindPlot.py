#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various functions which should allow plotting wind variables.
The design of the function is to return the figure and axes objects so the
user can then add axes labels and etc themselves.
"""

from .Error import InvalidParameter

import numpy as np
from matplotlib import pyplot as plt
from typing import List, Tuple


def sightline_coords(x: np.ndarray, theta: float):
    """
    Return the vertical coordinates for a sightline given the x coordinates
    and the inclination of the sightline.

    Parameters
    ----------
    x: np.ndarray[float]
        The x-coordinates of the sightline
    theta: float
        The opening angle of the sightline

    Returns
    -------
    z: np.ndarray[float]
        The z-coordinates of the sightline
    """

    return x * np.tan(np.pi / 2 - theta)


def create_rectilinear_wind_plot(x: np.ndarray, z: np.ndarray, w: np.ndarray, wtype: str, wname: str,
                                 fig: plt.Figure = None, ax: plt.Axes = None, i: int = None, j: int = None,
                                 scale: str = "loglog", obs_los: List[float] = None,
                                 figsize: Tuple[int, int] = (5, 5)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Creates a wind plot in rectilinear coordinates. If fig or ax is supplied,
    then fig, ax, i and j must also be supplied. Note that ax should also be a
    2d numpy array, i.e. squeeze = False.

    Parameters
    ----------
    x: np.ndarray[float]
        The x-coordinates of the model.
    z: np.ndarry[float]
        The z-coordinates of the model.
    w: np.ndarray[float]
        The value of the wind variable for each grid cell.
    wtype: str
        The type of wind variable, this can either be "wind" or "ion".
    wname: str
        The name of the wind variable.
    fig: plt.Figure [optional]
        A plt.Figure object to modify.
    ax: plt.Axes [optional]
        A plt.Axes object to modify.
    i: int [optional]
        The row index to create the plot in the plt.Axes object.
    j: int [optional]
        The column index to create the plot in the plt.Axes object.
    scale: str [optional]
        The scaling of the x and y axes, this can either be "logx", "logy" or
        "loglog".
    obs_los: List[float] [optional]
        A list of inclination angles to overplot observer line of sights.
    figsize: Tuple[int, int] [optional]
        The size of the figure in inches (width, height).

    Returns
    -------
    fig: plt.Figure
        The plt.Figure object.
    ax: plt.Axes
        The plt.Axes object.
    """

    n = create_rectilinear_wind_plot.__name__

    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=False)
        i = 0
        j = 0
    else:
        if fig is None:
            raise InvalidParameter("{}: fig was not provided when it was expected".format(n))
        if ax is None:
            raise InvalidParameter("{}: ax was not provided when it was expected".format(n))
        if i is None:
            raise InvalidParameter("{}: fig was not provided when it was expected".format(n))
        if j is None:
            raise InvalidParameter("{}: fig was not provided when it was expected".format(n))

    with np.errstate(divide="ignore"):
        if wname == "converge" or wname == "convergence":
            im = ax[i, j].pcolor(x, z, w)
        elif wtype == "ion":
            im = ax[i, j].pcolor(x, z, np.log10(w), vmin=-5, vmax=0)
        elif wtype == "wind":
            im = ax[i, j].pcolor(x, z, np.log10(w))
        else:
            raise InvalidParameter("{}: unknown wind variable type {}".format(n, wtype))

    if obs_los:
        xsight = np.linspace(0, np.max(x), int(1e5))
        for inc in obs_los:
            zsight = sightline_coords(xsight, np.deg2rad(inc))
            ax[i, j].plot(xsight, zsight, label="i = {}".format(inc) + r"$^{\circ}$ sightline")

    fig.colorbar(im, ax=ax[i, j])

    # TODO: maybe I should leave this to the user as well?
    if wname == "converge" or wname == "convergence":
        ax[i, j].set_title(r"convergence")
    else:
        ax[i, j].set_title(r"$\log_{10}(" + wname + ")")
    ax[i, j].set_xlabel("x [cm]")
    ax[i, j].set_ylabel("z [cm]")

    if scale == "loglog" or scale == "logx":
        ax[i, j].set_xscale("log")
        ax[i, j].set_xlabel(r"\log_{10}(x) [cm]")
    if scale == "loglog" or scale == "logy":
        ax[i, j].set_yscale("log")
        ax[i, j].set_ylabel(r"\log_{10}(y) [cm]")

    return fig, ax


# def create_polar_wind_plot(r: np.ndarray, theta: np.ndarray, w: np.ndarray, wtype: str, wname: str,
#                            ) -> Tuple[plt.Figure, plt.Axes]:
#
#
#     return fig, ax
