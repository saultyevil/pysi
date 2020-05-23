#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from shutil import copy
from os import mkdir
from sys import argv, exit
from pathlib import Path
from subprocess import Popen, PIPE
import numpy as np
from matplotlib import pyplot as plt

from PyPython.Constants import PI, PARSEC
from PyPython import SpectrumUtils
from PyPython.Grid import change_parameter
from PyPython.PythonUtils import remove_data_sym_links


N_CORES = 3
SMOOTH_AMOUNT = 75


def get_continuum(root):
    """Either run Python again to create the underlying continuum or find it"""

    name = "continuum/{}_cont.spec".format(root)
    if Path(name).is_file():
        t = SpectrumUtils.read_spec(name)
        return t

    print("Unable to find {}\nRunning Python to create continuum spectrum".format(name))

    # Now we have to run Python to get the continuum, to do this we will only
    # run spectral cycles, make the wind very diffuse and turn the temperature
    # up to ensure the wind is fully ionized

    try:
        mkdir("continuum")
    except FileExistsError:  # This is a horrid hack and I don't care
        pass

    copy("{}.pf".format(root), "continuum/{}_cont.pf".format(root))
    change_parameter("continuum/{}_cont.pf".format(root), "Ionization_cycles", "0", backup=False)
    change_parameter("continuum/{}_cont.pf".format(root), "Spectrum_cycles", "2", backup=False)
    change_parameter("continuum/{}_cont.pf".format(root), "Wind.mdot(msol/yr)", "1e-20", backup=False)
    change_parameter("continuum/{}_cont.pf".format(root), "Wind.t.init", "1e8", backup=False)

    # ????

    change_parameter("continuum/{}.pf".format(root), "Photons_per_cycle", "1e5", backup=False)
    change_parameter("continuum/{}.pf".format(root), "Photons_per_spectrum_cycle", "1e5", backup=False)

    # Now we can run Python - or attempt to!

    command = "cd continuum; Setup_Py_Dir; mpirun -n {} py {}_cont.pf".format(N_CORES, root)
    print(command)
    sh = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = sh.communicate()

    remove_data_sym_links()

    if stderr:
        print("There was a problem getting the continuum :-(")
        print(stderr.decode("utf-8"))
        exit(1)

    t = SpectrumUtils.read_spec("continuum/{}_cont.spec".format(root))

    return t


def create_plot(root, spectrum, optical_depth_spectrum, continuum_spectrum):
    """Create the reprocessing plot."""

    fig, ax = plt.subplots(1, 1, figsize=(13, 7))
    ax2 = ax.twinx()

    sightlines = SpectrumUtils.spec_inclinations(optical_depth_spectrum)[1:]
    tau_freq = optical_depth_spectrum["Freq."].values

    # Extract the emitted spectrum for the actual spectrum

    emergent_wavelength = spectrum["Lambda"].values
    emergent_frequency = spectrum["Freq."].values
    emergent_flux = spectrum["Emitted"].values * 4 * PI * (100 * PARSEC) ** 2 * emergent_wavelength

    # Extract the emitted spectrum for the underlying continuum

    continuum_wavelength = continuum_spectrum["Lambda"].values
    continuum_frequency = continuum_spectrum["Freq."].values
    continuum_flux = continuum_spectrum["Emitted"].values * 4 * PI * (100 * PARSEC) ** 2 * continuum_wavelength

    # Plot the spectra

    ax2.loglog(continuum_frequency, SpectrumUtils.smooth(continuum_flux, SMOOTH_AMOUNT), "k--", zorder=0, alpha=0.5)
    ax2.loglog(emergent_frequency, SpectrumUtils.smooth(emergent_flux, SMOOTH_AMOUNT), "k-", zorder=1, alpha=0.5)
    ax2.set_ylabel(r"$\nu L_{\nu}$ [ergs s$^{-1}$]")
    ax2.set_ylim(1e41, 1e43)

    # Plot the optical depths

    sightlines=["60"]

    for sl in sightlines:
        t = optical_depth_spectrum[sl].values
        if np.count_nonzero(t) != len(t):
            print("!!")
            continue
        ax.loglog(tau_freq, t, label="i = {}".format(sl) + r"$^{\circ}$")

    ax.legend()
    ax.set_ylabel(r"Continuum Optical Depth $\tau$")
    ax.set_xlabel(r"Frequency $\nu$ [Hz]")
    ax.tick_params(axis="x")
    ax.tick_params(axis="y")
    ax.set_xlim(np.min(tau_freq), np.max(tau_freq))

    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    SpectrumUtils.plot_line_ids(ax, SpectrumUtils.absorption_edges(True), logx=True)

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    plt.savefig("{}_reprocess.png".format(root))
    plt.show()

    return fig, ax, ax2


def main():
    """Main function of the script."""

    root = argv[1]

    cont = get_continuum(root)
    spectrum = SpectrumUtils.read_spec("{}.spec".format(root))
    optical_depth = SpectrumUtils.read_spec("diag_{}/{}.tau_spec.diag".format(root, root))

    fig, ax, ax2 = create_plot(root, spectrum, optical_depth, cont)

    return fig, ax, ax2


if __name__ == "__main__":
    fig, ax, ax2 = main()
