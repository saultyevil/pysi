#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The purpose of this script is to create a figure which overplots the
continuum and emitted spectrum from model.

The continuum optical depths are plotted as well. Both are plotted as a
function of frequency.
"""

import argparse as ap
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from pypython import plot, smooth_array
from pypython.physics.constants import PARSEC, PI
from pypython.spectrum import Spectrum
from pypython.util import get_cpu_count


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
    p.add_argument("-n",
                   "--ncores",
                   type=int,
                   default=0,
                   help="The number of cores to use to create the continuum spectrum if required.")
    p.add_argument("-sm",
                   "--smooth_amount",
                   type=int,
                   default=50,
                   help="The amount of smoothing to use on the spectra.")
    p.add_argument("--display", action="store_true", default=False, help="Display the figure.")

    args = p.parse_args()

    setup = (args.root, args.working_directory, args.ncores if args.ncores > 0 else get_cpu_count(), args.smooth_amount,
             args.display)

    return setup


def create_plot(root, spectrum, optical_depth_spectrum, sm=1, bgalpha=0.50, display=False):
    """Create a figure to show how the underlying continuum is being
    reprocessed.

    Parameters
    ----------
    root: str
        The root name of the simulation.
    spectrum: Spectrum
        The spectrum file for the complete simulation.
    optical_depth_spectrum: Spectrum
        The optical depth spectrum for the simulation.
    cont_spectrum: Spectrum
        The spectrum for the continuum only model.
    sm: int [optional]
        The amount of smoothing to be used for the spectra.
    bgalpha: float [optional]
        The transparency of the spectra.
    display: bool [optional]
        If True, the plot will be shown to screen.
    """

    # Find the various sight lines of the optical depth spectra

    sightlines = optical_depth_spectrum.inclinations
    optical_depth_freq = optical_depth_spectrum["Freq."]

    # Extract the two spectrum components of interest from the spectra files and
    # convert flux to nu Lnu
    # todo: check units of spectrum before doing this

    emerg_spec_freq = spectrum["Freq."]
    emerg_spec_flux = spectrum["Emitted"] * 4 * PI * (100 * PARSEC)**2 * spectrum["Lambda"]
    cont_spec_freq = spectrum["Freq."]
    cont_spec_flux = spectrum["Created"] * 4 * PI * (100 * PARSEC)**2 * spectrum["Lambda"]

    # Plot the spectra, these spectra are plotted on ax2 to have a separates y
    # axis on loglog scale

    fig, ax = plt.subplots(figsize=(12, 7))
    ax2 = ax.twinx()
    ax2.loglog(cont_spec_freq, smooth_array(cont_spec_flux, sm), "k--", zorder=0, alpha=bgalpha)
    ax2.loglog(emerg_spec_freq, smooth_array(emerg_spec_flux, sm), "k-", zorder=1, alpha=bgalpha)
    ax2.set_ylabel(r"$\nu L_{\nu}$ [ergs s$^{-1}$]")

    # Plot the optical depths, again as a function of frequency

    for sl in sightlines:
        od = optical_depth_spectrum[sl]
        if np.count_nonzero(od) != len(od):
            # I think this is to check that we're not going to plot a sightline
            # which has no optical depth values
            continue
        ax.loglog(optical_depth_freq, od, label=r"$\tau($" + "i = {}".format(sl) + r"$^{\circ} )$")

    ax.legend(loc="lower left")
    ax.set_ylabel(r"Continuum Optical Depth $\tau$")
    ax.set_xlabel(r"Frequency $\nu$ [Hz]")
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    ax.set_xlim(np.min(optical_depth_freq), np.max(optical_depth_freq))
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax = plot.ax_add_line_ids(ax, plot.photoionization_edges(True), logx=True)

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig("{}_reprocessing.png".format(root), dpi=300)

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax, ax2


def main(setup=None):
    """Main function of the script.

    Parameters
    ----------
    setup: [optional] tuple
        A tuple containing the setup parameters.

            setup = (
                root,
                working_directory,
                ncores,
                smooth_amount,
                display
            )
    """

    if setup:
        root, wd, n_cores, sm, display = setup
    else:
        root, wd, n_cores, sm, display = setup_script()

    full_spectrum = Spectrum(root, wd)
    optical_depth = Spectrum(root, wd, "spec_tau")
    create_plot(root, full_spectrum, optical_depth, sm=sm, display=display)

    return


if __name__ == "__main__":
    main()
