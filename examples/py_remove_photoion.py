#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Remove an element or transition from a data set. This was created because it is
often tedious to remove a single transition from the photoionization data.
"""

import argparse as ap
from PyPython.PythonUtils import remove_photoion_transition_from_data


def command_line():
    """Get input from the command line"""

    p = ap.ArgumentParser(description=__doc__)

    p.add_argument("data", type=str)
    p.add_argument("atomic_number", type=int)
    p.add_argument("ionisation_state", type=int)

    args = p.parse_args()

    data = args.data
    atomic_number = args.atomic_number
    ionisation_state = args.ionisation_state

    return data, atomic_number, ionisation_state


def main():
    """Main function of script"""

    data, atomic_number, ionisation_state = command_line()
    remove_photoion_transition_from_data(data, atomic_number, ionisation_state)

    return


if __name__ == "__main__":
    main()
