#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove all bound-bound transitions for a given ion in the atomic data. The
script will find the lines for a given ion, defined by its atomic number and
ionisation state, and set the oscillator strength of the transition to 0.
"""

import argparse as ap
from pypython.atomicdata import remove_bound_bound_transitions_ion


def setup_script():
    """Get input from the command line"""

    p = ap.ArgumentParser(description=__doc__)

    p.add_argument("atomic_number",
                   type=int,
                   help="The atomic number of the ion to remove.")

    p.add_argument("ionisation_state",
                   type=int,
                   help="The ionization state of the ion to remove.")

    args = p.parse_args()

    return args.atomic_number, args.ionisation_state


def main():
    """
    Main function of script
    """

    z, istate = setup_script()
    remove_bound_bound_transitions_ion(z, istate)

    return


if __name__ == "__main__":
    main()
