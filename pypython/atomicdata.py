#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The point of this part of pypython is to manipulating the atomic data used in
Python, to i.e. remove various transitions from the data..
"""


from sys import exit
from os import getenv

from .error import EXIT_FAIL


def remove_photoionization_edge(
    data: str, atomic_number: int, ionization_state: int, new_value: float = 9e99
):
    """
    Remove a transition or element from some atomic data. Creates a new atomic
    data file which is placed in the current working or given directory.

    To remove a photionization edge from the data, the frequency threshold is,
    by default, set to something large. It is also possible to just change this
    threshold instead to something else (for experimentation reasons?).

    Parameters
    ----------
    data: str
        The type of atomic data to be edited: outershell or innershell.
    atomic_number: int
        The atomic number of the element related to the edge to be
        removed.
    ionization_state: int
        The ionization state corresponding to the edge to be removed.
    new_value: [optional] float
        The value of the new frequency threshold for the edge.
    """

    n = remove_photoionization_edge.__name__

    data = data.lower()
    allowed_data = [
        "outershell",
        "innershell",
    ]

    if data not in allowed_data:
        print("{}: atomic data {} is unknown, known types are {}".format(n, data, allowed_data))
        exit(EXIT_FAIL)

    filename = getenv("PYTHON") + "/xdata/atomic/"

    stop = ""
    data_name = ""

    if data == "outershell":
        stop = "PhotVfkyS"
        data_name = "vfky_outershell_tab.dat"
    elif data == "innershell":
        stop = "InnerVYS"
        data_name = "vy_innershell_tab.dat"

    filename += data_name

    atomic_number = str(atomic_number)
    ionization_state = str(ionization_state)
    new_value = str(new_value)

    new = []

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    while i < (len(lines)):
        line = lines[i].split() + ["\n"]

        if line[0] == stop and line[1] == atomic_number and line[2] == ionization_state:
            line[5] = new_value
            new.append(" ".join(line))

            n_points = int(line[6])
            for j in range(n_points):
                edit_line = lines[i + j + 1].split() + ["\n"]
                edit_line[1] = new_value
                new.append(" ".join(edit_line))
            i += n_points + 1
        else:
            i += 1
            new.append(" ".join(line))

    with open(data_name, "w") as f:
        f.writelines(new)

    return


def remove_bound_bound_transitions_ion(
    atomic_number: int, ionization_state: int
):
    """
    Remove all bound-bound transitions for a single ion from the atomic data.
    This is achieved by setting the oscillator strengths of the transition, f,
    to f = 0, effectively removing the transition.

    Parameters
    ----------
    atomic_number: int
        The atomic number for the ion/atom the line is associated with.
    ionization_state: int
        The ionization state of the ion/atom the line is associated with.
    """

    n = remove_bound_bound_transitions_ion.__name__

    filename = getenv("PYTHON") + "/xdata/atomic/lines_linked_ver_2.dat"

    atomic_number = str(atomic_number)
    ionization_state = str(ionization_state)

    with open(filename, "r") as f:
        lines = f.readlines()

    for i in range(len(lines)):
        line = lines[i].split() + ["\n"]
        if line[1] == atomic_number and line[2] == ionization_state:
            line[4] = "0.000000"
        lines[i] = " ".join(line)

    with open("lines_linked_ver_2.dat", "w") as f:
        f.writelines(lines)

    return

