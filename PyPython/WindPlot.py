#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various functions which should allow plotting wind variables.
The design of the function is to return the figure and axes objects so the
user can then add axes labels and etc themselves.
"""

from .WindUtils import sightline_coords
from .Error import InvalidParameter

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from typing import List, Tuple


plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.labelsize'] = 15


# TODO rename to wind_1d or something
def spherical_wind(
    r: np.ndarray, w: np.ndarray, w_name: str, w_type: str, fig: plt.Figure = None, ax: plt.Axes = None, i: int = None,
    j: int = None, scale: str = "loglog", figsize: Tuple[int, int] = (5, 5)
) -> Tuple[plt.Figure, plt.Axes]:
    """Creates wind plots for 1D spherically symmetric winds."""

    n = spherical_wind.__name__

    if fig is None and ax is None:
        i = 0
        j = 0
        fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=False)
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
        if w_name == "converge" or w_name == "convergence" or w_name == "converging":
            ax[i, j].plot(r, w)
        elif w_type == "ion" or w_type == "wind" or w_type == "ion_density":
            ax[i, j].plot(r, np.log10(w))
        else:
            raise InvalidParameter("{}: unknown wind variable type {}".format(n, w_type))

    if w_name == "converge" or w_name == "convergence" or w_name == "converging":
        ax[i, j].set_title(w_name)
    else:
        ax[i, j].set_title(r"$\log_{10}$(" + w_name + ")")
    ax[i, j].set_xlabel("r [cm]")
    ax[i, j].set_xlim(np.min(r[r != 0]), np.max(r))
    ax[i, j].set_ylim(np.min(w[w != 0]), np.max(w))
    if scale == "loglog" or scale == "logx":
        ax[i, j].set_xscale("log")
    if scale == "loglog" or scale == "logy":
        ax[i, j].set_yscale("log")

    return fig, ax


def rectilinear_wind(
    x: np.ndarray, z: np.ndarray, w: np.ndarray, w_name: str, w_type: str, fig: plt.Figure = None, ax: plt.Axes = None,
    i: int = None, j: int = None, scale: str = "loglog", obs_los: List[float] = None, figsize: Tuple[int, int] = (5, 5)
) -> Tuple[plt.Figure, plt.Axes]:
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
    w_type: str
        The type of wind variable, this can either be "wind" or "ion".
    w_name: str
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

    n = rectilinear_wind.__name__

    if fig is None and ax is None:
        i = 0
        j = 0
        fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=False)
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
        if w_name == "converge" or w_name == "convergence" or w_name == "converging":
            im = ax[i, j].pcolormesh(x, z, w, vmin=0, vmax=3)
        elif w_name.find("units = c") != -1:
            w = np.ma.masked_where(w == 0, w)
            im = ax[i, j].pcolormesh(x, z, w)
        elif w_type == "ion":
            im = ax[i, j].pcolormesh(x, z, np.log10(w), norm=colors.Normalize(vmin=-10, vmax=0))
        elif w_type == "ion_density":
            im = ax[i, j].pcolormesh(x, z, np.log10(w))
        elif w_type == "wind":
            im = ax[i, j].pcolormesh(x, z, np.log10(w))
        else:
            raise InvalidParameter("{}: unknown wind variable type {}".format(n, w_type))

    if obs_los:
        xsight = np.linspace(0, np.max(x), int(1e5))
        for inc in obs_los:
            zsight = sightline_coords(xsight, np.deg2rad(float(inc)))
            ax[i, j].plot_simple(xsight, zsight, label="i = {}".format(inc) + r"$^{\circ}$ sightline")

    fig.colorbar(im, ax=ax[i, j])

    if w_name == "converge" or w_name == "convergence" or w_name == "converging":
        ax[i, j].set_title(w_name)
    else:
        ax[i, j].set_title(r"$\log_{10}$(" + w_name + ")")
    ax[i, j].set_xlabel("x [cm]")
    ax[i, j].set_ylabel("z [cm]")
    ax[i, j].set_xlim(np.min(x[x != 0]), np.max(x))
    ax[i, j].set_ylim(np.min(z[z != 0]), np.max(z))
    if scale == "loglog" or scale == "logx":
        ax[i, j].set_xscale("log")
    if scale == "loglog" or scale == "logy":
        ax[i, j].set_yscale("log")

    if obs_los and i == 0 and j == 0:
        ax[i, j].legend(loc="lower left")

    return fig, ax


def polar_wind(
    r: np.ndarray, theta: np.ndarray, w: np.ndarray, w_name: str, w_type: str, ax: plt.Axes = None, index: int = None,
    obs_los: List[float] = None, scale: str = "log"
) -> plt.Axes:
    """
    Creates a wind plot in polar coordinates. If ax is supplied then index must
    also be supplied. Note that ax should also be single plt.Axes object.

    Parameters
    ----------
    r: np.ndarray[float]
        The r-coordinates of the model.
    theta: np.ndarrayp[float]
        The theta-coordinates of the model.
    w: np.ndarray[float]
        The value of the wind variable for each grid cell.
    w_type: str
        The type of wind variable, this can either be "wind" or "ion".
    w_name: str
        The name of the wind variable.
    ax: plt.Axes [optional]
        A plt.Axes object to modify.
    index: int [optional]
        The subplot index.
    obs_los: List[float] [optional]
        A list of inclination angles to overplot observer line of sights.
    scale: str [optional]
        The scaling of the x and y axes, this can either be "logx", "logy" or
        "loglog".
    figsize: Tuple[int, int] [optional]
        The size of the figure in inches (width, height).

    Returns
    -------
    ax: plt.Axes
        The plt.Axes object.
    """

    n = polar_wind.__name__

    if ax:
        if index is None:
            raise InvalidParameter("{}: index was expected by not provided".format(n))
    else:
        ax = plt.subplot(1, 1, 1, projection="polar")

    if scale == "loglog":
        scale = "log"

    if scale == "log":
        r = np.log10(r)

    with np.errstate(divide="ignore"):
        if w_name == "converge" or w_name == "convergence" or w_name == "converging":
            im = ax.pcolormesh(theta, r, w, vmin=0, vmax=3)
        elif w_name == "inwind":
            im = ax.pcolormesh(theta, r, w)
        elif w_type == "wind":
            im = ax.pcolormesh(theta, r, np.log10(w))
        elif w_type == "ion":
            im = ax.pcolormesh(theta, r, np.log10(w), norm=colors.Normalize(vmin=-10, vmax=0))
        else:
            raise InvalidParameter("{}: unknown wind variable type {}".format(n, w_type))

    if obs_los:
        xsight = np.linspace(0, np.max(r), int(1e5))
        for inc in obs_los:
            zsight = sightline_coords(xsight, np.deg2rad(90 - inc))
            rsight = np.sqrt(xsight ** 2 + zsight ** 2)
            thetasight = np.arctan2(zsight, xsight)
            if scale == "log":
                rsight = np.log10(rsight)
            ax.plot(thetasight, rsight, label="i = {}".format(inc) + r"$^{\circ}$ sightline")

    plt.colorbar(im, ax=ax)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_rlabel_position(90)
    ax.set_rlim(np.min(r), np.max(r))
    if scale == "log":
        ax.set_ylabel(r"$\log_{10}(R)$ [cm]")
    else:
        ax.set_ylabel("R [cm]")

    if w_name == "converge" or w_name == "convergence" or w_name == "converging":
        ax.set_title(w_name)
    else:
        ax.set_title(r"$\log_{10}$(" + w_name + ")")

    return ax
