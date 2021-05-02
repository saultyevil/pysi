#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The purpose of this script is to provide quick plotting of the wind for a Python
simulation. As such, it is not very flexible with input to modify the output.
The script will create a figure of the "important" wind quantities, such as
the electron temperature and density, as well figures for the ion fractions
for H, He, C, N, O and Si.
"""

import argparse as ap
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from pypython import plotutil
from pypython import wind

default_wind_parameters = ("t_e", "t_r", "ne", "rho", "c4", "ip")

default_wind_velocities = ("v_x", "v_y", "v_z", "v_l", "v_rot", "v_r")

default_wind_ions = (
    ["H", 2],
    ["He", 3],
    ["C", 6],
    ["N", 8],
    ["O", 9],
    ["Si", 15],
)

default_ion_fig_dims = ((1, 2), (1, 3), (3, 2), (4, 2), (4, 2), (5, 3))

default_ion_figsize = ((7.5, 4.46), (19.25, 6.46), (13, 14.01), (13, 18.68),
                       (13, 18.68), (19.25, 23.25))


def setup_script() -> tuple:
    """Parse the different modes this script can be run from the command line.

    Returns
    -------
    setup: tuple
        A list containing all of the different setup of parameters for
        plotting."""

    p = ap.ArgumentParser(description=__doc__)

    p.add_argument("root", help="The root name of the simulation.")
    p.add_argument("-wd",
                   "--working_directory",
                   default=".",
                   help="The directory containing the simulation.")
    p.add_argument("-d",
                   "--ion_density",
                   action="store_true",
                   default=False,
                   help="Use ion densities instead of ion fractions.")
    p.add_argument("-u",
                   "--velocity_units",
                   default="kms",
                   choices=["kms", "cms", "c"],
                   help="The velocity units.")
    p.add_argument("-s",
                   "--scale",
                   default="loglog",
                   choices=["logx", "logy", "loglog", "linlin"],
                   help="The axes scaling to use.")
    p.add_argument("-c",
                   "--cells",
                   action="store_true",
                   default=False,
                   help="Plot using cell indices rather than spatial scales.")
    p.add_argument("--display",
                   action="store_true",
                   default=False,
                   help="Display the plot before exiting the script.")

    args = p.parse_args()

    setup = (args.root, args.working_directory, args.ion_density,
             args.velocity_units, args.scale, args.cells, args.display)

    return setup





def plot_wind_parameters(
        w: wind.Wind,
        parameters_to_plot: Tuple[str] = default_wind_parameters,
        axes_scales: str = "loglog",
        subplot_kw: dict = None,
        display: bool = False) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the parameters for the wind.
    Parameters
    ----------
    todo"""

    n_rows, n_cols = plotutil.subplot_dims(len(parameters_to_plot))

    fig, ax = plt.subplots(n_rows,
                           n_cols,
                           figsize=(13, 14),
                           squeeze=False,
                           sharex="col",
                           sharey="row",
                           subplot_kw=subplot_kw)

    log_plot = True
    wind_index = 0

    for i in range(n_rows):
        for j in range(n_cols):
            if log_plot:
                with np.errstate(divide="ignore"):
                    to_plot = np.log10(w[parameters_to_plot[wind_index]])
                ax[i,
                   j].set_title("log(" + parameters_to_plot[wind_index] + ")")
            else:
                to_plot = w[parameters_to_plot[wind_index]]
                ax[i, j].set_title(parameters_to_plot[wind_index])

            fig, ax = wind.plot_wind(w, to_plot, False, False, None,
                                     axes_scales, None, None, fig, ax, i, j)

            wind_index += 1

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig(w.cd + "/" + w.root + "_wind_parameters.png", dpi=300)

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_wind_velocity(w: wind.Wind,
                       wind_velocities_to_plot: Tuple[str],
                       velocity_units: str,
                       subplot_kw: dict = None,
                       axes_scales: str = "loglog",
                       display: bool = False) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the velocity for the wind.
    Parameters
    ----------
    todo"""

    n_rows, n_cols = plotutil.subplot_dims(len(wind_velocities_to_plot))
    fig, ax = plt.subplots(n_rows,
                           n_cols,
                           figsize=(13, 14),
                           squeeze=False,
                           sharex="col",
                           sharey="row",
                           subplot_kw=subplot_kw)

    if velocity_units == "c":
        logplot = True
    else:
        logplot = True

    wind_index = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if logplot:  # todo: ignore division warning
                with np.errstate(divide="ignore"):
                    toplot = np.log10(w[wind_velocities_to_plot[wind_index]])
                ax[i, j].set_title(
                    "log(" +
                    wind_velocities_to_plot[wind_index].replace("_", r"\_") +
                    ")" + " [" + w.velocity_units + "]")
            else:
                toplot = w[wind_velocities_to_plot[wind_index]]
                ax[i, j].set_title(
                    wind_velocities_to_plot[wind_index].replace("_", r"\_") +
                    " [" + w.velocity_units + "]")

            fig, ax = wind.plot_wind(w, toplot, False, False, None,
                                     axes_scales, None, None, fig, ax, i, j)

            wind_index += 1

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig(w.cd + "/" + w.root + "_wind_velocities.png", dpi=300)

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_wind_ions(w: wind.Wind,
                   ions_to_plot: Tuple[str],
                   wind_ion_dims: Tuple[List[int]],
                   wind_ion_size: Tuple[List[int]],
                   use_ion_density: bool = False,
                   axes_scales: str = "loglog",
                   subplot_kw: dict = None,
                   display: bool = False) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the ions for the wind
    Parameters
    ----------
    todo"""

    if use_ion_density:
        title = "Ion Densities"
        ion_type_key = "density"
        vmin = vmax = None
    else:
        title = "Ion Fractions"
        ion_type_key = "fraction"
        vmin = -20
        vmax = 0

    for (element, n_ions), (n_rows, n_cols), (width, height) in zip(
            ions_to_plot, wind_ion_dims, wind_ion_size):

        fig, ax = plt.subplots(n_rows,
                               n_cols,
                               figsize=(width, height),
                               squeeze=False,
                               sharex="col",
                               sharey="row",
                               subplot_kw=subplot_kw)
        fig, ax = plotutil.remove_extra_axes(fig, ax, n_ions, n_rows * n_cols)

        ion_index = 1
        for i in range(n_rows):
            for j in range(n_cols):
                ion_key = "i{:02d}".format(ion_index)
                with np.errstate(divide="ignore"):
                    fig, ax = wind.plot_wind(
                        w, np.log10(w[element][ion_type_key][ion_key]), False,
                        False, None, axes_scales, vmin, vmax, fig, ax, i, j)

                ax[i, j].set_title("log(" + element + ion_key + ")")

                ion_index += 1
                if ion_index > n_ions:
                    break

        fig.suptitle(title)
        fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
        fig.savefig(w.cd + "/" + w.root + "_" + element + "_ions.png", dpi=300)

        if display:
            plt.show()
        else:
            plt.close()

    return fig, ax


def main():
    """The main function of the script."""

    root, cd, use_ion_density, velocity_units, axes_scales, use_cell_indices, display = setup_script(
    )

    # Read in the wind, set the wing parameters we want to plot, as well as the
    # elements of the ions we want to plot and the number of ions.

    w = wind.Wind(root, cd, velocity_units, True)

    if w.coord_system == "polar":
        subplot_kw = {"projection": "polar"}
    else:
        subplot_kw = {}

    # First, plot the wind parameters

    plot_wind_parameters(w,
                         default_wind_parameters,
                         axes_scales=axes_scales,
                         subplot_kw=subplot_kw,
                         display=display)

    # Next, plot the wind velocities, if the grid is rectilinear

    if w.coord_system == "rectilinear":
        plot_wind_velocity(w, default_wind_velocities, velocity_units,
                           subplot_kw, axes_scales, display)

    # Now, plot the wind ions. This is a bit messier...

    fig, ax = plot_wind_ions(w, default_wind_ions, default_ion_fig_dims,
                             default_ion_figsize, use_ion_density, axes_scales,
                             subplot_kw, display)

    return fig, ax


if __name__ == "__main__":
    main()
