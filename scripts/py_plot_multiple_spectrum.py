#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The purpose of this script is to easily create a figure which overplots
multiple spectra onto a single plot.

Spectrum files are recursively searched from the calling directory.
"""

import argparse as ap

from pypython import get_files
from pypython.plot.spectrum import multiple_spectra


def setup_script():
    """Parse the different modes this script can be run from the command line.

    Returns
    -------
    setup: tuple
        A list containing all of the different setup of parameters for plotting.
    """

    p = ap.ArgumentParser(description=__doc__)

    p.add_argument("name", type=str, help="The output name of the comparison plot.")
    p.add_argument("-wd", "--working_directory", default=".", help="The directory containing the simulation.")
    p.add_argument("-i", "--inclination", default="all", help="The inclination angles")
    p.add_argument("-xl", "--xmin", type=float, default=None, help="The lower x-axis boundary to display.")
    p.add_argument("-xu", "--xmax", type=float, default=None, help="The upper x-axis boundary to display.")
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
    p.add_argument("-sm", "--smooth_amount", type=int, default=5, help="The size of the boxcar smoothing filter.")
    p.add_argument("--display", action="store_true", default=False, help="Display the plot before exiting the script.")

    args = p.parse_args()

    return (args.name, args.working_directory, args.inclination, args.xmin, args.xmax, args.common_lines, args.scales,
            args.smooth_amount, args.display)


def main():
    """The main function of the script."""

    output_name, wd, inclination, x_min, x_max, common_lines, axes_scales, smooth_amount, display = setup_script()

    spectra_to_plot = get_files("*.spec")

    if len(spectra_to_plot) == 0:
        raise ValueError("Unable to find any spectrum files")

    fig, ax = multiple_spectra(output_name, spectra_to_plot, inclination, wd, x_min, x_max, frequency_space,
                               axes_scales, smooth_amount, common_lines, file_extension, display)

    return fig, ax


if __name__ == "__main__":
    main()
