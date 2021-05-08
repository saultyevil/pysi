#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The purpose of this script is to easily create a figure which overplots multiple
spectra onto a single plot. Spectrum files are recursively searched from the
calling directory.
"""

import argparse as ap
from typing import Tuple

from matplotlib import pyplot as plt

from pypython import get
from pypython import spectrum
from pypython.error import EXIT_FAIL


def setup_script():
    """
    Parse the different modes this script can be run from the command line.

    Returns
    -------
    setup: tuple
        A list containing all of the different setup of parameters for plotting.
    """

    p = ap.ArgumentParser(description=__doc__)

    p.add_argument("name",
                   type=str,
                   help="The output name of the comparison plot.")
    p.add_argument("-wd",
                   "--working_directory",
                   default=".",
                   help="The directory containing the simulation.")
    p.add_argument("-i",
                   "--inclination",
                   default="all",
                   help="The inclination angles")
    p.add_argument("-r",
                   "--root",
                   default=None,
                   help="Only plots models which have the provided root name.")
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
                   default="loglog",
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

    setup = (args.name, args.working_directory, args.inclination, args.root,
             args.xmin, args.xmax, args.frequency_space, args.common_lines,
             args.scales, args.smooth_amount, args.ext, args.display)

    return setup


def main(setup: tuple = None) -> Tuple[plt.Figure, plt.Axes]:
    """The main function of the script.

    Parameters
    ----------
    setup: tuple
        A tuple containing the setup parameters to run the script. If this
        isn't provided, then the script will parse them from the command
        line."""

    if setup:
        output_name, wd, inclination, root, x_min, x_max, frequency_space, common_lines, axes_scales, smooth_amount,\
            file_extension, display = setup
    else:
        output_name, wd, inclination, root, x_min, x_max, frequency_space, common_lines, axes_scales, smooth_amount, \
            file_extension, display = setup_script()

    spectra = get("*.spec")
    if len(spectra) == 0:
        print("Unable to find any spectrum files")
        exit(EXIT_FAIL)

    fig, ax = spectrum.plot_multiple_model_spectra(
        output_name, spectra, inclination, wd, x_min, x_max, frequency_space,
        axes_scales, smooth_amount, common_lines, file_extension, display)

    return fig, ax


if __name__ == "__main__":
    main()
