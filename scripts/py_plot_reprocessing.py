#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The purpose of this script is to create a figure which overplots the
continuum and emitted spectrum from model.

The continuum optical depths are plotted as well. Both are plotted as a
function of frequency.
"""

import argparse as ap

import numpy as np
from matplotlib import pyplot as plt

from pypython import Spectrum
from pypython.plot.spectrum import reprocessing


def setup_script():
    """Setup the script.

    Returns
    -------
    setup: tuple
        The various setup parameters for the script

            setup = (
                args.root,
                args.working_directory,
                args.ncores,
                args.smooth_amount,
                args.display
            )
    """

    p = ap.ArgumentParser(description=__doc__)

    p.add_argument("root", help="The root name of simulation.")
    p.add_argument("-wd", "--working_directory", default=".", help="The directory containing the simulation.")
    p.add_argument("-sm",
                   "--smooth_amount",
                   type=int,
                   default=5,
                   help="The amount of smoothing to use on the spectra.")
    p.add_argument("--display", action="store_true", default=False, help="Display the figure.")

    args = p.parse_args()

    return args.root, args.working_directory, args.smooth_amount, args.display


def main():
    """Main function of the script."""

    root, wd, smooth, display = setup_script()
    reprocessing(Spectrum(root, wd, smooth=smooth), display=display)

    return


if __name__ == "__main__":
    main()
