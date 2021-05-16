#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The purpose of this script is to create quick plots which show the optical
depth as a function of frequency or wavelength for multiple inclination
angles."""

import argparse as ap

from matplotlib import pyplot as plt

from pypython.plot import spectrumplot


def setup_script():
    """Parse the different modes this script can be run from the command line.

    Returns
    -------
    setup: tuple
        A list containing all of the different setup of parameters for plotting.

        setup = (
            args.root,
            wd,
            xmin,
            xmax,
            frequency_space,
            absorption_edges,
            axes_scales,
            file_ext,
            display
        )
    """

    p = ap.ArgumentParser(description=__doc__)

    p.add_argument("root", type=str, help="The root name of the simulation.")
    p.add_argument("-wd", "--working_directory", default=".", help="The directory containing the simulation.")
    p.add_argument("-xl", "--xmin", type=float, default=None, help="The lower x-axis boundary to display.")
    p.add_argument("-xu", "--xmax", type=float, default=None, help="The upper x-axis boundary to display.")
    p.add_argument("-s",
                   "--scales",
                   default="logy",
                   choices=["logx", "logy", "loglog", "linlin"],
                   help="The axes scaling to use: logx, logy, loglog, linlin.")
    p.add_argument("-a",
                   "--absorption_edges",
                   action="store_true",
                   default=False,
                   help="Plot labels for important absorption edges.")
    p.add_argument("-f",
                   "--frequency_space",
                   action="store_true",
                   default=False,
                   help="Create the figure in frequency space.")
    p.add_argument("-e", "--ext", default="png", help="The file extension for the output figure.")
    p.add_argument("--display", action="store_true", default=False, help="Display the plot before exiting the script.")

    args = p.parse_args()

    setup = (args.root, args.working_directory, args.xmin, args.xmax, args.frequency_space, args.absorption_edges,
             args.scales, args.ext, args.display)

    return setup


def plot_optical_depth_spectrum(root,
                                wd="./",
                                xmin=None,
                                xmax=None,
                                scale="loglog",
                                show_absorption_edges=False,
                                frequency_space=False,
                                file_ext="png",
                                display=False):
    """Description of function.

    Parameters
    ----------

    Returns
    -------
    """

    fig, ax = spectrumplot.optical_depth(root,
                                         wd, ["all"],
                                         xmin,
                                         xmax,
                                         scale,
                                         show_absorption_edges,
                                         frequency_space,
                                         display=display)

    fig.savefig("{}/{}_optical_depth.{}".format(wd, root, file_ext))

    return fig, ax


def main(setup=None):
    """The main function of the script. First, the important wind quantities
    are plotted. This is then followed by the important ions.

    Parameters
    ----------
    setup: tuple
        A tuple containing the setup parameters to run the script. If this
        isn't provided, then the script will parse them from the command line.

        setup = (
            root,
            wd,
            xmin,
            xmax,
            frequency_space,
            absorption_edges,
            axes_scales,
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

    if setup:
        root, wd, xmin, xmax, frequency_space, absorption_edges, axes_scales, file_ext, display = setup
    else:
        root, wd, xmin, xmax, frequency_space, absorption_edges, axes_scales, file_ext, display = setup_script()

    root = root.replace("/", "")

    fig, ax = plot_optical_depth_spectrum(root, wd, xmin, xmax, axes_scales, absorption_edges, frequency_space,
                                          file_ext, display)

    return fig, ax


if __name__ == "__main__":
    fig, ax = main()
