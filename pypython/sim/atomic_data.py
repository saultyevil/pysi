#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove certain interactions form the atomic data.

Sometimes it's desirable to remove certain resonance line or
photoioniation cross-sections from the atomic data, so photons do not
interact with them.
"""

from os import getenv


def remove_bound_bound_transitions_ion(atomic_number, ionization_state):
    """Remove all bound-bound transitions for a single ion.

    This is achieved by setting the oscillator strengths of the transition, f,
    to f = 0, effectively removing the transition.

    Parameters
    ----------
    atomic_number: int
        The atomic number for the ion/atom the line is associated with.
    ionization_state: int
        The ionization state of the ion/atom the line is associated with.
    """

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
