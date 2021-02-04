#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The purpose of this script is to determine if a simulation has converged or not,
and to create any plots which are related to the convergence of a simulation.
"""

import numpy as np
from matplotlib import pyplot as plt
from pypython import simulation
from pypython import util
from typing import List

COL_WIDTH = 80


def plot_convergence(
    root: str, convergence: List[float], converging: List[float] = None, tr: List[float] = None,
    te: List[float] = None, te_max: List[float] = None, hc: List[float] = None, wd: str = "."
):
    """Create a detailed plot of the convergence of a Python simulation, including,
    if provided, a breakdown of the different convergence criteria.

    Parameters
    ----------
    root: str
        The root name of the Python simulation.
    convergence: List[float]
        The convergence fraction of the simulation for each cycle.
    converging: List[float] [optional]
        The converging fraction of the simulation for each cycle.
    tr: List[float] [optional]
        The fraction of cells which have converged radiation temperature.
    te: List[float] [optional]
        The fraction of cells which have converged electron temperature.
    te_max: List[float] [optional]
        The fraction of cells which hit the electron temperature limit.
    hc: List[float] [optional]
        The fraction of cells which have converged heating and cooling rates.
    wd: str [optional]
        The directory containing the Python simulation."""

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    n_cycles = len(convergence)
    cycles = np.arange(1, n_cycles + 1, 1)
    ax.set_xlim(1, n_cycles)
    ax.set_ylim(0, 1)
    ax.set_xticks(cycles[::2])

    ax.plot(cycles, convergence, label="Convergence")

    # As the bare minimum, we need the convergence per cycle but if the other
    # convergence stats are passed as well, plot those too

    if converging:
        ax.plot(cycles, converging, label="Converging")
    if tr:
        ax.plot(cycles, tr, "--", label="Radiation temperature")
    if te:
        ax.plot(cycles, te, "--", label="Electron temperature")
    if te_max:
        ax.plot(cycles, te_max, "--", label="Electron temperature max")
    if hc:
        ax.plot(cycles, hc, "--", label="Heating/Cooling")

    ax.legend()
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Fraction of Cells Passed")
    ax.set_title("Final Convergence = {:4.2f}%".format(float(convergence[-1]) * 100))
    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    plt.savefig("{}/{}_convergence.png".format(wd, root))
    plt.close()

    return


def get_convergence(
    root: str, wd: str = "./"
) -> None:
    """Print out the convergence of a Python simulation and then create a detailed
    plot of the convergence and convergence break down of the simulation.

    Parameters
    ----------
    root: str
        The root name of the Python simulation.
    wd: str [optional]
        The directory containing the Python simulation."""

    convergence = simulation.check_model_convergence(root, wd, return_per_cycle=True)
    converging = simulation.check_model_convergence(root, wd, return_per_cycle=True, return_converging=True)
    tr, te, te_max, hc = simulation.model_convergence_components(root, wd)

    n_cycles = len(convergence)
    if n_cycles == 0:
        print("Unable to find any convergence information for this model :-(\n")
        return

    for i in range(n_cycles):
        print("Cycle {:2d} / {:2d}: {:5.2f}% of cells converged and {:5.2f}% of cells are still converging"
              .format(i + 1, n_cycles, convergence[i] * 100, converging[i] * 100))
    print("")

    try:
        plot_convergence(root, convergence, converging, tr, te, te_max, hc, wd)
    except Exception as e:
        print("Unable to create convergence plot.")
        print(e)
        print("")

    return


def main():
    """Main function of the script."""

    print("-" * COL_WIDTH, "\n")

    parameter_files = util.get_parameter_files()
    for pf in parameter_files:
        root, cd = util.get_root_from_filepath(pf)
        if cd.find("continuum") != -1:
            continue
        print("-" * COL_WIDTH)
        print("\nGetting the convergence for {} in directory {}\n".format(root, cd[:-1]))
        get_convergence(root, cd)

    print("-" * COL_WIDTH)

    return


if __name__ == "__main__":
    main()
