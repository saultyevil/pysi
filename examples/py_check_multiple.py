#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
The purpose of this script is to check the convergence of multiple Python
simulations at once. It does this by searching recursively for Python
simulations for the calling directory. However, this script can also be used
where there is only one Python simulation.
"""


from typing import List
from PyPython import Simulation
from PyPython import PythonUtils as Utils


def check_multiple(wdpf: List[str]) -> None:
    """
    Get the convergence for multiple simulations given by the list wdpf.

    Parameters
    ----------
    wdpf: List[str]
        A list containing the directories of multiple pf files.
    """

    convergence = []

    for i in range(len(wdpf)):
        root, path = Utils.split_root_directory(wdpf[i])
        c = Simulation.check_convergence(root, path)
        convergence.append(c)

    some_convergence = []
    for i in range(len(wdpf)):
        c = convergence[i]
        if c > 0:
            some_convergence.append([wdpf[i], c])

    no_convergence = []
    for i in range(len(wdpf)):
        c = convergence[i]
        if c <= 0:
            no_convergence.append([wdpf[i], c])

    if some_convergence:
        print("The following simulations have convergence:\n"
              "-------------------------------------------")
        for i in range(len(some_convergence)):
            print(some_convergence[i][0], some_convergence[i][1])

    if no_convergence:
        print("\nThe following simulations have no convergence:\n"
              "----------------------------------------------")
        for i in range(len(no_convergence)):
            print(no_convergence[i][0], no_convergence[i][1])

    return


if __name__ == "__main__":
    check_multiple(Utils.get_pfs())
