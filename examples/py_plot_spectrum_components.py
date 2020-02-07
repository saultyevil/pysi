#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse as ap
from typing import Tuple
from sys import exit
from matplotlib import pyplot as plt

from PyPython import SpectrumPlot
from PyPython import SpectrumUtils
from PyPython.Error import EXIT_FAIL


def parse_input() -> tuple:
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
    p.add_argument("root", help="The root name of the simulation.")
    p.add_argument("-wd", action="store", help="The directory containing the simulation.")
    p.add_argument("-xl", "--xmin", action="store", help="The lower x-axis boundary to display.")
    p.add_argument("-xu", "--xmax", action="store", help="The upper x-axis boundary to display.")
    p.add_argument("-s", "--scales", action="store", help="The axes scaling to use: logx, logy, loglog, linlin.")
    p.add_argument("-l", "--common_lines", action="store_true", help="Plot labels for important absorption edges.")
    p.add_argument("-f", "--frequency_space", action="store_true", help="Create the figure in frequency space.")
    p.add_argument("-sm", "--smooth_amount", action="store_true", help="The size of the boxcar smoothing filter.")
    p.add_argument("-e", "--ext", action="store", help="The file extension for the output figure.")
    p.add_argument("--display", action="store_true", help="Display the plot before exiting the script.")
    args = p.parse_args()

    wd = "./"
    if args.wd:
        wd = args.wd

    xmin = None
    if args.xmin:
        xmin = args.xmin

    xmax = None
    if args.xmax:
        xmax = args.xmax

    common_lines = True
    if args.common_lines:
        common_lines = args.common_lines

    frequency_space = False
    if args.frequency_space:
        frequency_space = args.frequency_space

    file_ext = "png"
    if args.ext:
        file_ext = args.ext

    axes_scales = "logy"
    if args.scales:
        allowed = ["logx", "logy", "loglog", "linlin"]
        if args.scales not in allowed:
            print("The axes scaling {} is unknown.".format(args.scales))
            print("Allowed values are: logx, logy, loglog, linlin.")
            exit(EXIT_FAIL)
        axes_scales = args.scales

    smooth_amount = 5
    if args.smooth_amount:
        if args.smooth_amount < 1:
            print("The size of the smoothing filter must at least be 1")
            exit(EXIT_FAIL)

    display = False
    if args.display:
        display = True

    setup = (
        args.root,
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

    return setup


def main(setup: tuple = None) -> Tuple[plt.Figure, plt.Axes]:
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

    div_len = 80

    if setup:
        root, wd, xmin, xmax, smooth_amount, frequency_space, common_lines, axes_scales, file_ext, display = setup
    else:
        root, wd, xmin, xmax, smooth_amount, frequency_space, common_lines, axes_scales, file_ext, display = parse_input()

    root = root.replace("/", "")
    wdd = wd
    if wd == "./":
        wdd = ""

    print("-" * div_len)
    print("\nPlotting spectrum components for {}{}.pf".format(wdd, root))

    fig, ax = SpectrumPlot.spectrum_components(root, wd, False, xmin, xmax, smooth_amount, axes_scales, frequency_space)
    fig.savefig("{}/{}_spectrum_components.{}".format(wd, root, file_ext))

    print("\nPlotting diagnostic spectrum components for {}{}.pf".format(wdd, root))

    fig, ax = SpectrumPlot.spectrum_components(root, wd, True, xmin, xmax, smooth_amount, axes_scales, frequency_space)
    fig.savefig("{}/{}_wind_tot.{}".format(wd, root, file_ext))

    if display:
        plt.show()

    print("")
    print("-" * div_len)

    return fig, ax


if __name__ == "__main__":
    fig, ax = main()

