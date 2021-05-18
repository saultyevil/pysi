#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add a parameter to already existing parameter file(s).

The script will search recursively from the calling directory for
parameter files. If a root name is provided, however, then the script
will only operate on pf files which have the same root name.
"""

import argparse as ap
from typing import List

from pypython import get_files
from pypython.simulation import grid


def add_parameter(filepaths, parameter, value):
    """Iterate over a list of pfs, and add the parameter given to the end of
    the parameter file. This function will also print out verbose, because it
    seems the most sensible to be loud about this.

    Parameters
    ----------
    filepaths: List[str]
        A list containing the directories of multiple pf files.
    parameter: str
        The parameter name of the parameter which is being updated.
    value: str
        The updated parameter value.
    """

    for filepath in filepaths:
        grid.add_single_parameter(filepath, parameter, value)

    return


def main():
    """Main function."""

    p = ap.ArgumentParser(description=__doc__)

    p.add_argument("parameter", help="Name of the parameter to add.")
    p.add_argument("value", help="The value for the new parameter.")
    p.add_argument("--root", default=None, help="Add the parameter to parameter files with this specific root name.")

    args = p.parse_args()

    if args.root is None:
        root = ""
    else:
        root = args.root

    add_parameter(get_files(f"*/{root}.pf"), args.parameter, args.value)

    return


if __name__ == "__main__":
    main()
