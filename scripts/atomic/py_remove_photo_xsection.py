#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove an element or transition from a data set. This was created because it is
often tedious to remove a single transition from the photoionization data.
"""

import argparse as ap

from pypython.simulation.atomicdata import remove_photoionization_edge


def command_line():
    """Get input from the command line"""

    p = ap.ArgumentParser(description=__doc__)

    p.add_argument("data",
                   choices=["outershell", "innershell", "topbase"],
                   help="The name of the atomic data to modify.")

    p.add_argument("atomic_number",
                   type=int,
                   help="The atomic number of the ion to remove.")

    p.add_argument("ionisation_state",
                   type=int,
                   help="The ionization state of the ion to remove.")

    args = p.parse_args()

    return args.data, args.atomic_number, args.ionisation_state


def main():
    """Main function of script"""

    data, atomic_number, ionisation_state = command_line()
    remove_photoionization_edge(data, atomic_number, ionisation_state)

    return


if __name__ == "__main__":
    main()
