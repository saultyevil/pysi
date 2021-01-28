#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The purpose of this script is to parse the diag file of Python and to compare
the photon luminosity before and after trans_phot.
FOR NOW, THIS WILL ONLY WORK WELL WITH MACRO ATOMS WHERE WE DO NOT GENERATE
WIND PHOTONS AS WE ASSUME THAT THE TOTAL LUMINOSITY BEFORE TRANS_PHOT FOR EACH
CYCLE IS THE SAME, WHICH I DO NOT BELIEVE TO BE TRUE IN SIMPLE ATOM MODE.
THIS WILL ALSO ONLY CHECK THE ROOT DIAGNOSTIC FILE. IN FUTURE THIS COULD,
FOR EXAMPLE ALSO CHECK THE OTHER DIAG FILES AND AVERAGE OVER THEM OR CREATE A
PLOT FOR EACH PROCESS.
"""

import numpy as np
import argparse as ap
from matplotlib import pyplot as plt
from pypython.util import find_parameter_files, get_root
from pypython.simulation import check_model_convergence


def get_input():
    """

    Returns
    -------
    args.root: str
        The root name of the Python simulation.
    """

    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("root", help="The root name of the simulation to check")
    args = p.parse_args()

    return args.root


def check_luminosity_balance(root: str, wd: str = "./"):
    """
    Check the luminosity before and after trans_phot for a Python simulation.
    This function will also create a plot of the relative change.

    Parameters
    ----------
    root: str
        The root name of the Python simulation.
    wd: str [optional]
        The directory containing the Python simulation.
    """

    luminosity_before = []
    luminosity_after = []
    absorbed_lost = []

    fname = "{}/diag_{}/{}_0.diag".format(wd, root, root)
    with open(fname) as f:
        diag = f.readlines()

    for line in diag:
        if line.find("!!python: Total photon luminosity before transphot") != -1:
            try:
                luminosity_before.append(float(line.split()[-1]))
            except IndexError:
                luminosity_before.append(-1)
        if line.find("!!python: Total photon luminosity after transphot") != -1:
            try:
                luminosity_after.append(float(line.split()[6]))
            except IndexError:
                luminosity_after.append(-1)
            try:
                absorbed_lost.append(float(line.split()[10][:-2]))
            except IndexError:
                absorbed_lost.append(-1)

    if len(luminosity_before) == 0:
        return

    print("Root                  = ", root)
    print("Convergence           = ", check_model_convergence(root, wd))
    print("Luminosity before     = ", luminosity_before[-1])
    print("Luminosity after      = ", luminosity_after[-1])
    print("Absorbed/lost         = ", absorbed_lost[-1])
    print("Ratio: after / before = ", luminosity_after[-1] / luminosity_before[-1])
    print()

    cycles = np.arange(1, len(luminosity_after) + 1)
    # TODO assumes luminosity is always the same before, when in simple atom it probably isn't
    plt.plot(cycles, np.array(luminosity_after) / luminosity_before[0], label="After / Before", linewidth=3)
    plt.axhline(1, color="k", linestyle="--", label="No Change", linewidth=3)
    plt.xlim(1, len(luminosity_after) + 1)
    plt.xlabel("Cycle")
    plt.ylabel("Luminosity Ratio")
    plt.legend()
    plt.savefig("{}/{}_luminosity_balance.png".format(wd, root), dpi=300)
    plt.close()

    return


def main():
    """
    Main function of the script.
    """

    pfs = find_parameter_files()
    for pf in pfs:
        if pf.find("continuum") != -1:
            continue
        root, wd = get_root(pf)
        print(pf, root, wd)
        check_luminosity_balance(root, wd)

    return


if __name__ == "__main__":
    main()
