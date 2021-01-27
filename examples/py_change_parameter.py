#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Update an existing parameter for some parameter file(s).
The script will search recursively from the calling directory for parameter
files. If a root name is provided, however, then the script will only operate
 on pf files which have the same root name.
"""


import argparse as ap
from pypython import grid
from pypython import util
from typing import List


def change_pfs(wdpf: List[str], parameter: str, value: str) -> None:
    """
    Iterate over a list of pfs, and update the parameter given by the variable
    parameter with the new value given by value. This function will also
    print out verbose, because it seems most sensible to be loud about this.

    Parameters
    ----------
    wdpf: List[str]
        A list containing the directories of multiple pf files.
    parameter: str
        The parameter name of the parameter which is being updated.
    value: str
        The updated parameter value.
    """

    for i in range(len(wdpf)):
        grid.update_single_parameter(wdpf[i], parameter, value, verbose=True)

    return


def main():
    """
    Main function of the script.
    """

    p = ap.ArgumentParser(description=__doc__)

    p.add_argument("parameter",
                   help="Name of the parameter to add.")

    p.add_argument("value",
                   help="The value for the new parameter.")

    p.add_argument("--root",
                   default=None,
                   help="Add the parameter to parameter files with this specific root name.")

    args = p.parse_args()

    change_pfs(util.find_parameter_files(args.root), args.parameter, args.value)

    return


if __name__ == "__main__":
    main()
