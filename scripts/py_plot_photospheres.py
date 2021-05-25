#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Get the photosphere."""

import argparse as ap
import subprocess

import numpy as np
from matplotlib import pyplot as plt

import pypython


def setup():
    """Setup the script."""

    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("root", help="Root name of the simulation")
    p.add_argument("-fp", "--filepath", default=".", help="The directory containing the simulation")

    args = p.parse_args()

    return args.root, args.filepath


def run_py_optical_depth(root, fp, tau):
    """Run py_optical_depth to get the photosphere."""

    sh = subprocess.run(f"cd {fp}; py_optical_depth -p {tau} {root}",
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

    return sh.returncode


def read_in_photosphere_locations(root, fp):
    """Read in the photosphere file."""

    with open(f"{fp}/{root}.photosphere", "r") as f:
        lines = f.readlines()

    tau_es = 0
    locations = []

    for line in lines:
        if line.startswith("# Electron scatter photosphere"):
            tau_es = float(line.split()[-1])

        if line.startswith("#"):
            continue

        locations.append(line.split())

    locations = np.array(locations[1:], dtype=np.float64)

    return tau_es, locations


def plot_2d(wind, variable, display):
    """2D version."""

    # fig, ax = wind.plot(variable)

    # get the photosphere locations

    pypython.plot.normalize_figure_style()
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    rg = pypython.physics.blackhole.gravitational_radius(5e6)
    # rg = 1

    im = ax.pcolormesh(wind["x"] / rg, wind["z"] / rg, np.log10(wind["rho"]), shading="auto", alpha=0.8)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(r"$\log_{10}(\rm " + f"{variable}" + ")$")

    for tau in [1, 5, 10, 25, 50, 100]:
        err = run_py_optical_depth(wind.root, wind.fp, tau)
        if err:
            print(f"error return of {err} from py_optical_depth for tau_es = {tau}")
            continue
        tau_es, locations = read_in_photosphere_locations(wind.root, wind.fp)
        ax.plot(locations[:, 0] / rg, locations[:, 2] / rg, label=r"$\tau_{\rm es} =" + f"{tau_es:.2f}" + "$")

    ax.set_xlabel("$x / R_{g}$")
    ax.set_ylabel("$z / R_{g}$")

    # ax.set_xlim(0, 1e11)
    # ax.set_ylim(0, 1e11)

    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)

    ax.legend(fontsize=11)

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig(f"{wind.fp}/{wind.root}_photosphere.png")

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_1d(wind, variable, display):
    """1D version."""

    # fig, ax = wind.plot(variable)

    # get the photosphere locations

    pypython.plot.normalize_figure_style()

    rg = wind.convert_cm_to_rg(5e6)
    fig, ax = wind.plot(variable)

    for n, tau in enumerate([1, 5, 10, 25, 50, 100]):
        err = run_py_optical_depth(wind.root, wind.fp, tau)
        if err:
            print(f"error return of {err} from py_optical_depth for tau_es = {tau}")
            continue
        tau_es, location = read_in_photosphere_locations(wind.root, wind.fp)
        ax.axvline(location / rg, color=f"C{n}", linestyle="--", label=r"$\tau_{\rm es} =" + f"{tau_es:.2f}" + "$")

    ax.legend(fontsize=11)

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig(f"{wind.fp}/{wind.root}_photosphere.png")

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def main():
    """Main function of the script."""

    root, fp = setup()
    variable = "ne"
    display = True

    # read in wind, plot variable get matplotlib objects

    wind = pypython.Wind(root, fp)

    if wind.coord_system == pypython.WIND_COORD_TYPE_SPHERICAL:
        fig, ax = plot_1d(wind, variable, display)
    else:
        fig, ax = plot_2d(wind, variable, display)

    return fig, ax


if __name__ == "__main__":
    main()
