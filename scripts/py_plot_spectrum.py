#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create plots of the spectrum files from a Python simulation.

This script will create,
"""

import argparse as ap

from matplotlib import pyplot as plt

from pypython import Spectrum, plot


def setup_script():
    """Parse the different modes this script can be run from the command line.

    Returns
    -------
    setup: tuple
        A list containing all of the different setup of parameters for
        plotting.
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
    p.add_argument("-sm", "--smooth_amount", type=int, default=5, help="The size of the boxcar smoothing filter.")
    p.add_argument("-e", "--ext", default="png", help="The file extension for the output figure.")
    p.add_argument("--display", action="store_true", default=False, help="Display the plot before exiting the script.")

    args = p.parse_args()

    setup = (args.root, args.working_directory, args.xmin, args.xmax, args.frequency_space, args.common_lines,
             args.scales, args.smooth_amount, args.ext, args.display)

    return setup


def main():
    """The main function of the script. First, the important wind quantaties
    are plotted. This is then followed by the important ions.


    Returns
    -------
    fig: plt.Figure
        The matplotlib Figure object for the created plot.
    ax: plt.Axes
        The matplotlib Axes objects for the plot panels.
    """

    root, cd, xmin, xmax, frequency_space, common_lines, axes_scales, smooth_amount, file_ext, display = setup_script()

    spectrum = Spectrum(root, cd)

    # Observer spectra

    # Spectrum Components - extracted spectrum

    # spec_tot - all photons

    # spec_tot_wind - anything which is "inwind"

    if display:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
