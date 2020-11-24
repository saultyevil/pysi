#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The purpose of this script is to create a figure which overplots the continuum
and emitted spectrum from model. The continuum optical depths are plotted as well.
Both are plotted as a function of frequency.
"""

from shutil import copy
from os import mkdir
from sys import argv, exit
import pandas as pd
from pathlib import Path
from subprocess import Popen, PIPE
import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt
import argparse as ap

from pyPython.constants import PI, PARSEC
from pyPython import spectrumUtil
from pyPython.grid import update_single_parameter
from pyPython.pythonUtil import remove_data_sym_links, get_cpu_count


def setup_script() -> tuple:
    """
    Setup the script.

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

    p.add_argument("root",
                   help="The root name of simulation.")

    p.add_argument("-wd",
                   "--working_directory",
                   default=".",
                   help="The directory containing the simulation.")

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

    p.add_argument("--display",
                   action="store_true",
                   default=False,
                   help="Display the figure.")

    args = p.parse_args()

    setup = (
        args.root,
        args.working_directory,
        args.ncores if args.ncores > 0 else get_cpu_count(),
        args.smooth_amount,
        args.display
    )

    return setup


def get_continuum(
    root: str, wd: str = ".", ncores: int = 1
) -> pd.DataFrame:
    """
    Get the data for the underlying continuum spectrum. The script will attempt
    to read the file in from continuum/root_cont.spec, otherwise it will create
    a continuum model to run in Python.

    Parameters
    ----------
    root: str
        The root name of the simulation.
    wd: str [optional]
        The directory containing the simulation.
    ncores: int [optional]
        The number of cores to use to create the continuum spectrum if required.

    Returns
    -------
    t: pd.DataFrame
        The continuum spectrum.
    """

    name = "{}/continuum/{}_cont.spec".format(wd, root)
    if Path(name).is_file():
        t = spectrumUtil.read_spectrum(name)
        return t

    print("Unable to find {}\nRunning Python to create continuum spectrum".format(name))

    # Now we have to run Python to get the continuum, to do this we will only
    # run spectral cycles, make the wind very diffuse and turn the temperature
    # up to ensure the wind is fully ionized. This is done by making a copy of
    # the final parameter file and using change_parameter to change the various
    # parameters.

    try:
        mkdir("continuum")
    except FileExistsError:  # I don't think this is the intended method, but oh well
        pass

    name = "{}/continuum/{}_cont.pf".format(wd, root)
    copy("{}/{}.pf".format(wd, root), name)
    update_single_parameter(name, "Ionization_cycles", "0", backup=False)
    update_single_parameter(name, "Spectrum_cycles", "5", backup=False)
    update_single_parameter(name, "Photons_per_cycle", "1e6", backup=False)
    update_single_parameter(name, "Wind.mdot(msol/yr)", "1e-20", backup=False)
    update_single_parameter(name, "Wind.t.init", "1e8", backup=False)
    update_single_parameter(name, "Reverb.type(none,photon,wind,matom)", "none", backup=False)

    command = "cd {}; cd continuum; Setup_Py_Dir; mpirun -n {} py -gamma {}_cont.pf".format(wd, ncores, root)
    print(command)
    sh = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = sh.communicate()
    ndel = remove_data_sym_links()

    if stderr:
        print("There was a problem running Python to generate the continuum spectrum:\n")
        print(stderr.decode("utf-8"))
        exit(1)

    if ndel == 0:
        print("There was a problem deleteing the atomic data")

    t = spectrumUtil.read_spectrum("continuum/{}_cont.spec".format(root))

    return t


def create_plot(
    root: str, spectrum: pd.DataFrame, optical_depth_spectrum: pd.DataFrame, cont_spectrum: pd.DataFrame,
    sm: int = 50, bgalpha: float = 0.50, display: bool = False
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Create a figure to show how the underlying continuum is being reprocessed.

    Parameters
    ----------
    root: str
        The root name of the simulation.
    spectrum: pd.DataFrame
        The spectrum file for the complete simulation.
    optical_depth_spectrum: pd.DataFrame
        The optical depth spectrum for the simulation.
    cont_spectrum: pd.DataFrame
        The spectrum for the continuum only model.
    sm: int [optional]
        The amount of smoothing to be used for the spectra.
    bgalpha: float [optional]
        The transparency of the spectra.
    display: bool [optional]
        If True, the plot will be shown to screen.
    """

    # Find the various sightlines of the optical depth spectra
    sightlines = spectrumUtil.get_spectrum_inclinations(optical_depth_spectrum)
    optical_depth_freq = optical_depth_spectrum["Freq."].values

    # Extract the two spectrum components of interest from the spectra files and
    # convert flux to nu Lnu
    emerg_spec_freq = spectrum["Freq."].values
    emerg_spec_flux = spectrum["Emitted"].values * 4 * PI * (100 * PARSEC) ** 2 * spectrum["Lambda"].values
    cont_spec_freq = cont_spectrum["Freq."].values
    cont_spec_flux = cont_spectrum["Emitted"].values * 4 * PI * (100 * PARSEC) ** 2 * cont_spectrum["Lambda"].values

    # Plot the spectra, these spectra are plotted on ax2 to have a separates y
    # axis on loglog scale
    fig, ax = plt.subplots(figsize=(13, 7))
    ax2 = ax.twinx()
    ax2.loglog(
        cont_spec_freq, spectrumUtil.smooth(cont_spec_flux, sm), "k--", zorder=0, alpha=bgalpha
    )
    ax2.loglog(
        emerg_spec_freq, spectrumUtil.smooth(emerg_spec_flux, sm), "k-", zorder=1, alpha=bgalpha
    )
    ax2.set_ylabel(r"$\nu L_{\nu}$ [ergs s$^{-1}$]")

    # Plot the optical depths, again as a function of frequency
    for sl in sightlines:
        od = optical_depth_spectrum[sl].values
        if np.count_nonzero(od) != len(od):
            # I think this is to check that we're not going to plot a sightline
            # which has no optical depth values
            continue
        ax.loglog(
            optical_depth_freq, od, label=r"$\tau($" + "i = {}".format(sl) + r"$^{\circ} )$"
        )

    ax.legend(loc="lower left")
    ax.set_ylabel(r"Continuum Optical Depth $\tau$")
    ax.set_xlabel(r"Frequency $\nu$ [Hz]")
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    ax.set_xlim(np.min(optical_depth_freq), np.max(optical_depth_freq))
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax = spectrumUtil.plot_line_ids(ax, spectrumUtil.photo_edges_list(True), logx=True)

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig("{}_reprocess.png".format(root), dpi=300)

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax, ax2


def main(setup: tuple = None):
    """
    Main function of the script.

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
        root, wd, ncores, sm, display = setup
    else:
        root, wd, ncores, sm, display = setup_script()

    continuum_spectrum = get_continuum(root, wd, ncores)
    full_spectrum = spectrumUtil.read_spectrum("{}.spec".format(root))
    optical_depth = spectrumUtil.read_spectrum("diag_{}/{}.tau_spec.diag".format(root, root))
    create_plot(root, full_spectrum, optical_depth, continuum_spectrum, sm=sm, display=display)

    return


if __name__ == "__main__":
    main()
