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
import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt
from pypython import windplot
from pypython import plotutil
from pypython.wind import Wind2D

plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.labelsize'] = 15


def setup_script() -> tuple:
    """Parse the different modes this script can be run from the command line.

    Returns
    -------
    setup: tuple
        A list containing all of the different setup of parameters for
        plotting."""

    p = ap.ArgumentParser(description=__doc__)

    p.add_argument(
        "root", help="The root name of the simulation."
    )
    p.add_argument(
        "-wd", "--working_directory", default=".", help="The directory containing the simulation."
    )
    p.add_argument(
        "-d", "--ion_density", action="store_true", default=False, help="Use ion densities instead of ion fractions."
    )
    p.add_argument(
        "-p", "--polar_coords", action="store_true", default=False, help="Plot using polar projection."
    )
    p.add_argument(
        "-s", "--scale", default="loglog", choices=["logx", "logy", "loglog", "linlin"],
        help="The axes scaling to use."
    )
    p.add_argument(
        "-c", "--cells", action="store_true", default=False, help="Plot using cell indices rather than spatial scales."
    )
    p.add_argument(
        "-e", "--ext", default="png", help="The file extension for the output figure."
    )
    p.add_argument(
        "--display", action="store_true", default=False, help="Display the plot before exiting the script."
    )

    args = p.parse_args()

    setup = (
        args.root,
        args.working_directory,
        args.polar_coords,
        args.ion_density,
        args.scale,
        args.cells,
        args.ext,
        args.display
    )

    return setup


def main(
    setup: tuple = None
) -> Tuple[plt.Figure, plt.Axes]:
    """The main function of the script.

    Parameters
    ----------

    Returns
    -------"""

    if setup:
        root, cd, polar_coords, use_ion_density, axes_scales, use_cell_indices, file_ext, display = setup
    else:
        root, cd, polar_coords, use_ion_density, axes_scales, use_cell_indices, file_ext, display = setup_script()

    if polar_coords:
        coordinate_system = "polar"
        subplot_kw = {"projection": "polar"}
    else:
        coordinate_system = "rectilinear"
        subplot_kw = {}

    # Read in the wind, set the wing parameters we want to plot, as well as the
    # elements of the ions we want to plot and the number of ions.

    wind = Wind2D(root, cd, coordinate_system, "kms", True)

    wind_parameters = [
        "t_e", "t_r", "ne", "rho", "c4", "ip"
    ]

    wind_velocities = [
        "v_x", "v_y", "v_z", "v_l", "v_rot", "v_r"
    ]

    # (element, n_ions)
    wind_ions = [
        ["H", 2], ["He", 3], ["C", 6], ["N", 8], ["O", 9], ["Si", 15],
    ]

    # (n_rows, n_cols)
    wind_ion_dims = [
        (1, 2), (1, 3), (3, 2), (4, 2), (4, 2), (5, 3)
    ]

    # (width, height)
    wind_ion_size = [
        (7.5, 3.67), (19.25, 6.46), (13, 14.01), (13, 18.68), (13, 18.68), (19.25, 23.25)
    ]

    # First, plot the wind parameters

    n_rows, n_cols = plotutil.subplot_dims(len(wind_parameters))
    fig, ax = plt.subplots(
        n_rows, n_cols, figsize=(13, 14), squeeze=False, sharex="col", sharey="row", subplot_kw=subplot_kw
    )

    logplot = True  # todo: make variable input

    wind_index = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if logplot:  # todo: ignore division warning
                toplot = np.log10(wind[wind_parameters[wind_index]])
                ax[i, j].set_title("log(" + wind_parameters[wind_index] + ")")
            else:
                toplot = wind[wind_parameters[wind_index]]
                ax[i, j].set_title(wind_parameters[wind_index])
            fig, ax = windplot.plot_2d_wind(
                wind["x"], wind["z"], toplot, coordinate_system, None, axes_scales, None, None, fig, ax, i, j
            )
            wind_index += 1

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig(cd + "/" + root + "_wind_parameters.png", dpi=300)

    if display:
        plt.show()
    else:
        plt.close()

    # Next, plot the wind velocities

    n_rows, n_cols = plotutil.subplot_dims(len(wind_velocities))
    fig, ax = plt.subplots(
        n_rows, n_cols, figsize=(13, 14), squeeze=False, sharex="col", sharey="row", subplot_kw=subplot_kw
    )

    logplot = True  # todo: make variable input

    wind_index = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if logplot:  # todo: ignore division warning
                toplot = np.log10(wind[wind_velocities[wind_index]])
                ax[i, j].set_title("log(" + wind_velocities[wind_index] + ")" + " [" + wind.velocity_units + "]")
            else:
                toplot = wind[wind_velocities[wind_index]]
                ax[i, j].set_title(wind_velocities[wind_index] + " [" + wind.velocity_units + "]")
            fig, ax = windplot.plot_2d_wind(
                wind["x"], wind["z"], toplot, coordinate_system, None, axes_scales, None, None, fig, ax, i, j
            )
            wind_index += 1

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig(cd + "/" + root + "_wind_velocities.png", dpi=300)

    if display:
        plt.show()
    else:
        plt.close()

    # Now, plot the wind ions. This is a bit messier...

    if use_ion_density:
        title = "Ion Densities"
        ion_type_key = "density"
        vmin = vmax = None
    else:
        title = "Ion Fractions"
        ion_type_key = "fraction"
        vmin = -20
        vmax = 0

    for (element, n_ions), (n_rows, n_cols), (width, height) in zip(wind_ions, wind_ion_dims, wind_ion_size):
        fig, ax = plt.subplots(
            n_rows, n_cols, figsize=(width, height), squeeze=False, sharex="col", sharey="row", subplot_kw=subplot_kw
        )
        fig, ax = plotutil.remove_extra_axes(fig, ax, n_ions, n_rows * n_cols)

        ion_index = 1
        for i in range(n_rows):
            for j in range(n_cols):
                ion_key = "i{:02d}".format(ion_index)
                fig, ax = windplot.plot_2d_wind(
                    wind["x"], wind["z"], np.log10(wind[element][ion_type_key][ion_key]), coordinate_system, None,
                    axes_scales, vmin, vmax, fig, ax, i, j
                )
                ax[i, j].set_title("log(" + element + ion_key + ")")
                ion_index += 1
                if ion_index > n_ions:
                    break

        fig.suptitle(title)
        fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
        fig.savefig(cd + "/" + root + "_" + element + "_ions.png", dpi=300)

        if display:
            plt.show()
        else:
            plt.close()

    return fig, ax


if __name__ == "__main__":
    fig, ax = main()
