#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions dedicated to manipulating the atomic data in Python.
"""


from os import getenv


def remove_photoion_transition_from_data(
    data: str, atomic: int, istate: int, new_value: float = 9e99
):
    """
    Remove a transition or element from some atomic data. Creates a new atomic
    data file which is placed in the current working or given directory.

    TODO: include Topbase
    TODO: include lines

    Parameters
    ----------
    data: str

    atomic: int

    istate: int

    new_value: [optional] float

    """

    n = remove_photoion_transition_from_data.__name__

    data = data.lower()

    allowed_data = [
        "outershell",
        "innershell",
    ]

    if data not in allowed_data:
        print("{}: atomic data {} is unknown, known types are {}".format(n, data, allowed_data))
        return

    filename = getenv("PYTHON") + "/xdata/atomic/"

    if data == "outershell":
        stop = "PhotVfkyS"
        data_name = "vfky_outershell_tab.dat"
    elif data == "innershell":
        stop = "InnerVYS"
        data_name = "vy_innershell_tab.dat"
    else:
        return

    filename += data_name

    atomic = str(atomic)
    istate = str(istate)
    new_value = str(new_value)

    new = []

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    while i < (len(lines)):
        line = lines[i].split() + ["\n"]

        if line[0] == stop and line[1] == atomic and line[2] == istate:
            line[5] = new_value
            new.append(" ".join(line))

            npoints = int(line[6])
            for j in range(npoints):
                edit_line = lines[i + j + 1].split() + ["\n"]
                edit_line[1] = new_value
                new.append(" ".join(edit_line))
            i += npoints + 1
        else:
            i += 1
            new.append(" ".join(line))

    with open(data_name, "w") as f:
        f.writelines(new)

    return


def remove_bound_lines_for_ion(
    z: int, istate: int
):
    """
    Remove all bound-bound transitions for a single ion from the atomic data.
    This is achieved by setting the oscillator strengths of the transition, f,
    to f = 0, effectively removing the transition.

    Parameters
    ----------
    z: int
        The atomic number for the ion.
    istate: int
        The ionization state of the ion.
    """

    n = remove_bound_lines_for_ion.__name__

    filename = getenv("PYTHON") + "/xdata/atomic/lines_linked_ver_2.dat"

    z = str(z)
    istate = str(istate)

    with open(filename, "r") as f:
        lines = f.readlines()

    for i in range(len(lines)):
        line = lines[i].split() + ["\n"]
        if line[1] == z and line[2] == istate:
            line[4] = "0.000000"
        lines[i] = " ".join(line)

    with open("lines_linked_ver_2.dat", "w") as f:
        f.writelines(lines)

    return

