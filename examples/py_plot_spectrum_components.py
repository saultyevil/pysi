#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This script is a convenient wrapper to plot the components of a Python
model as well as the components stored in the log_spec_tot file.
"""


import argparse as ap
from typing import Tuple
from matplotlib import pyplot as plt

from pyPython import spectrumPlot


plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.labelsize'] = 15

import warnings
warnings.filterwarnings("ignore", module="matplotlib")


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
            xmin,
            xmax,
            smooth_amount,
            frequency_space,
            absorption_edges,
            axes_scales,
            file_ext,
            display
        )
    """

    p = ap.ArgumentParser(description=__doc__)

    # Required arguments
    p.add_argument("root",
                   type=str,
                   help="The root name of the simulation.")

    # Supplementary arguments
    p.add_argument("-wd",
                   "--working_directory",
                   default=".",
                   help="The directory containing the simulation.")

    p.add_argument("-xl",
                   "--xmin",
                   type=float,
                   default=None,
                   help="The lower x-axis boundary to display.")

    p.add_argument("-xu",
                   "--xmax",
                   type=float,
                   default=None,
                   help="The upper x-axis boundary to display.")

    p.add_argument("-s",
                   "--scales",
                   default="logy",
                   choices=["logx", "logy", "loglog", "linlin"],
                   help="The axes scaling to use: logx, logy, loglog, linlin.")

    p.add_argument("-l",
                   "--common_lines",
                   action="store_true",
                   default=False,
                   help="Plot labels for important absorption edges.")

    p.add_argument("-f",
                   "--frequency_space",
                   action="store_true",
                   default=False,
                   help="Create the figure in frequency space.")

    p.add_argument("-sm",
                   "--smooth_amount",
                   type=int,
                   default=5,
                   help="The size of the boxcar smoothing filter.")

    p.add_argument("-e",
                   "--ext",
                   default="png",
                   help="The file extension for the output figure.")

    p.add_argument("--display",
                   action="store_true",
                   default=False,
                   help="Display the plot before exiting the script.")

    args = p.parse_args()

    setup = (
        args.root,
        args.working_directory,
        args.xmin,
        args.xmax,
        args.smooth_amount,
        args.frequency_space,
        args.common_lines,
        args.scales,
        args.ext,
        args.display
    )

    return setup


def main(setup: tuple = None) \
        -> Tuple[plt.Figure, plt.Axes]:
    """
    The main function of the script. First, the important wind quantaties are
    plotted. This is then followed by the important ions.

`   Parameters
    ----------
    setup: tuple
        A tuple containing the setup parameters to run the script. If this
        isn't provided, then the script will parse them from the command line.

        setup = (
            root,
            wd,
            xmin,
            xmax,
            smooth_amount,
            frequency_space,
            common_lines,
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
        root, wd, xmin, xmax, smooth_amount, frequency_space, common_lines, axes_scales, file_ext, display = setup
    else:
        root, wd, xmin, xmax, smooth_amount, frequency_space, common_lines, axes_scales, file_ext, display = \
            setup_script()

    root = root.replace("/", "")
    alpha = 0.75

    # Spectrum Components - extracted spectrum
    fig, ax = spectrumPlot.plot_spectrum_components(
        root, wd, False, False, xmin, xmax, smooth_amount, axes_scales, alpha, frequency_space, display
    )
    fig.savefig("{}/{}_spectrum_components.{}".format(wd, root, file_ext))

    # log_spec_tot - all photons
    fig, ax = spectrumPlot.plot_spectrum_components(
        root, wd, True, False, xmin, xmax, smooth_amount, axes_scales, alpha, frequency_space, display
    )
    fig.savefig("{}/{}_spec_tot.{}".format(wd, root, file_ext))

    # log_spec_tot_wind - anything which is "inwind"
    fig, ax = spectrumPlot.plot_spectrum_components(
        root, wd, False, True, xmin, xmax, smooth_amount, axes_scales, alpha, frequency_space, display
    )
    fig.savefig("{}/{}_spec_tot_wind.{}".format(wd, root, file_ext))

    return fig, ax


if __name__ == "__main__":
    fig, ax = main()
