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
from sys import exit
from typing import List, Tuple
from matplotlib import pyplot as plt

from PyPython import WindPlot
from PyPython import WindUtils
from PyPython import PythonUtils
from PyPython.Error import EXIT_FAIL


plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.labelsize'] = 15


def plot_wind(root: str, wind_variables: List[str], wind_variable_types: List[str], output_name: str, wd: str = "./",
              projection: str = "rectilinear", axes_scales: str = "loglog", use_cell_indices: bool = False,
              panel_dims: Tuple[int, int] = (4, 2), figure_size: Tuple[int, int] = (10, 15), title: str = None,
              file_ext: str = "png") \
        -> Tuple[plt.Figure, plt.Axes]:
    """
    The purpose of this function is to oversee the creation of the different
    possible wind plots.

    Parameters
    ----------
    root: str
        The root name of the model.
    wind_variables: List[str]
        A list containing the names of the quantities to plot.
    wind_variable_types: List[str]
        A list containing the type of the wind variable. This can either be
        wind or ion.
    output_name: str
        An additional name to provide to distinguish the plot created.
    wd: str [optional]
        The directory where the simulation is stored, by default this assumes
        that it is in the calling directory.
    projection: str [optional]
        The projection required for the plot, allowed values are rectilinear or
        polar.
    axes_scales: str [optional]
        The type of scaling for the axes of the figure, allowed values are
        logx, logy, loglog or linlin.
    use_cell_indices: bool [optional]
        If True then the wind will not be plotted in spatial coordinates, but
        rather cell index coordinates.
    panel_dims: Tuple[int, int] [optional]
        The number of rows and columns of subplot panels to create.
    figure_size: Tuple[int, int] [optional]
        The width and height of the figure in inches (thanks matplotlib).
    title: str [optional]
        The title of the figure.
    file_ext: str [optional]
        The extension of the final output file.

    Returns
    -------
    fig: plt.Figure
        The matplotlib Figure object for the created plot.
    ax: plt.Axes
        The matplotlib Axes objects for the plot panels.
    """

    # Check the same amount of wind variables and their types have been passed

    if len(wind_variables) != len(wind_variable_types):
        print("The size of the wind_variables and the correspond types lists are unequal in length.")
        exit(EXIT_FAIL)

    # Check the projection requested and set up fig and ax depending on the 
    # projection

    allowed_projections = ["rectilinear", "polar"]
    if projection not in allowed_projections:
        print("The provided projection {} is not an allowed projection.")
        print("Allowed projections are: ", end="")
        for p in allowed_projections:
            print("{} ".format(p), end="")
        print(".")
        exit(EXIT_FAIL)
    
    if projection == "rectilinear":
        fig, ax = plt.subplots(panel_dims[0], panel_dims[1], figsize=figure_size, squeeze=False)
    else:
        ax = []
        fig = plt.figure(figsize=figure_size)

    # Set the scale to linear-linear when plotting with cell indices

    if use_cell_indices:
        axes_scales = "linlin"

    # Now construct the plot
    
    index = 0
    nsize = len(wind_variables) - 1

    for i in range(panel_dims[0]):
        for j in range(panel_dims[1]):

            if index > nsize:
                break

            quantity = wind_variables[index]
            quantity_type = wind_variable_types[index]

            try:
                the_type = quantity_type
                if quantity_type == "ion_density":
                    the_type = "ion"
                x, z, w = WindUtils.get_wind_variable(root, quantity, the_type, wd, projection,
                                                      return_indices=use_cell_indices)
            except Exception as e:
                print("\nAn exception '{}' occurred for some reason.".format(e))
                print("Unable to plot quantity {} with type {}.".format(quantity, quantity_type))
                index += 1
                continue

            if projection == "rectilinear":
                fig, ax = WindPlot.rectilinear_wind(x, z, w, quantity, quantity_type, fig, ax, i, j, scale=axes_scales)
            else:
                polar_ax = plt.subplot(panel_dims[0], panel_dims[1], index + 1, projection="polar")
                axx = WindPlot.polar_wind(x, z, w, quantity, quantity_type, polar_ax, index + 1, scale=axes_scales)
                ax.append(axx)

            index += 1

    if title:
        fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig("{}/{}_{}.{}".format(wd, root, output_name, file_ext))

    return fig, ax


def setup_script() \
        -> tuple:
    """
    Parse the different modes this script can be run from the command line.

    Returns
    -------
    setup: tuple
        A list containing all of the different setup of parameters for plotting.
    
        setup = (
            args.root,
            wd,
            projection,
            axes_scales,
            cell_indices,
            file_ext,
            display
        )
    """

    p = ap.ArgumentParser(description=__doc__)

    p.add_argument("root", help="The root name of the simulation.")
    p.add_argument("-wd", action="store", help="The directory containing the simulation.")
    p.add_argument("-d", "--ion_density", action="store_true", help="Use ion densities instead of ion fractions.")
    p.add_argument("-p", "--polar", action="store_true", help="Plot using polar projection.")
    p.add_argument("-s", "--scale", action="store", help="The axes scaling to use: logx, logy, loglog, linlin.")
    p.add_argument("-c", "--cells", action="store_true", help="Plot using cell indices rather than spatial scales.")
    p.add_argument("-e", "--ext", action="store", help="The file extension for the output figure.")
    p.add_argument("--display", action="store_true", help="Display the plot before exiting the script.")

    args = p.parse_args()

    wd = "./"
    if args.wd:
        wd = args.wd

    projection = "rectilinear"
    if args.polar:
        projection = "polar"

    ion_density = False
    if args.ion_density:
        ion_density = True

    cell_indices = False
    if args.cells:
        cell_indices = True

    file_ext = "png"
    if args.ext:
        file_ext = args.ext

    axes_scales = "loglog"
    if args.scale:
        allowed = ["logx", "logy", "loglog", "linlin"]
        if args.scale not in allowed:
            print("The axes scaling {} is unknown.".format(args.scale))
            print("Allowed values are: logx, logy, loglog, linlin.")
            exit(EXIT_FAIL)
        axes_scales = args.scale

    display = False
    if args.display:
        display = True

    setup = (
        args.root,
        wd,
        projection,
        ion_density,
        axes_scales,
        cell_indices,
        file_ext,
        display
    )

    return setup


def main(setup: tuple = None) \
        -> Tuple[plt.Figure, plt.Axes]:
    """
    The main function of the script. First, the important wind quantaties are
    plotted. This is then followed by the important ions.
`
Parameters
    ----------
    setup: tuple
        A tuple containing the setup parameters to run the script. If this
        isn't provided, then the script will parse them from the command line.

        setup = (
            args.root,
            wd,
            projection,
            axes_scales,
            cell_indices,
            file_ext,
            display
        )


    Returns
    -------
    fig: plt.Figure
        The matplotlib Figure object for the created plot.
    ax: plt.Axes
        The matplotlib Axes objects for the plot panels.
    """

    div_len = 80

    if setup:
        root, wd, projection, use_ion_density, axes_scales, use_cell_indices, file_ext, display = setup
    else:
        root, wd, projection, use_ion_density, axes_scales, use_cell_indices, file_ext, display = setup_script()

    root = root.replace("/", "")
    wdd = wd
    if wd == "./":
        wdd = ""

    print("-" * div_len)
    print("\nCreating wind and ion plots for {}{}.pf".format(wdd, root))
    
    # First, we probably need to run windsave2table

    # PythonUtils.windsave2table(root, wd, ion_density=use_ion_density)

    # Plot the wind quantities first

    wind = ["t_e", "t_r", "ne", "rho", "c4", "ip", "converge", "ntot"]
    wind_types = ["wind"] * len(wind)
    
    print("\nCreating a figure containing:\n\t", end="")
    for w in wind:
        print("{} ".format(w), end="")
    print("")

    fig, ax = plot_wind(root, wind, wind_types, "wind", wd, projection, axes_scales, use_cell_indices)

    # Plot the ions

    if use_ion_density:
        title = "Ion Density"
    else:
        title = "Ion Fractions"

    dims = [(2, 2), (1, 2), (2, 2), (3, 2), (4, 2), (4, 2), (5, 3)]
    size = [(15, 10), (15, 5), (15, 10), (15, 15), (15, 20), (15, 20), (22.5, 25)]
    elements = ["KeyIons", "H", "He", "C", "N", "O", "Si"]
    ions = [
        ["H_i01", "Si_i04", "N_i05", "C_i04"],
        # ["O_i05", "Si_i04", "Si_i05", "N_i04", "N_i05", "N_i06", "C_i04", "C_i05"],
        ["H_i01", "H_i02"],
        ["He_i01", "He_i02", "He_i03"],
        ["C_i01", "C_i02", "C_i03", "C_i04", "C_i05", "C_i06"],
        ["N_i01", "N_i02", "N_i03", "N_i04", "N_i05", "N_i06", "N_i07", "N_i08"],
        ["O_i01", "O_i02", "O_i03", "O_i04", "O_i05", "O_i06", "O_i07", "O_i08"],
        ["Si_i01", "Si_i02", "Si_i03", "Si_i04", "Si_i05", "Si_i06", "Si_i07", "Si_i08", "Si_i09", "Si_i10",
         "Si_i11", "Si_i12", "Si_i13", "Si_i14", "Si_i15"]
    ]

    print("\nCreating figures containing ions for the elements:\n\t", end="")
    for el in elements:
        print("{} ".format(el), end="")
    print("\n")

    for i in range(len(elements)):
        extra_name = elements[i] + "_ions"
        if use_ion_density:
            ion_type = ["ion_density"] * len(ions[i])
        else:
            ion_type = ["ion"] * len(ions[i])
        fig, ax = plot_wind(root, ions[i], ion_type, extra_name, wd, projection, axes_scales=axes_scales,
                            use_cell_indices=use_cell_indices, panel_dims=dims[i], figure_size=size[i],
                            title=title, file_ext=file_ext)

    if display:
        plt.show()

    if __name__ == "__main__":
        print("-" * div_len)

    return fig, ax


if __name__ == "__main__":
    fig, ax = main()
