#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The purpose of this script is to collate the errors over all of the MPI processes
for a bunch of simulations.
TODO: this will only work for a simulation which didn't crash
"""


from pyPython.pythonUtil import find_parameter_files, get_root
from pyPython.simulation import error_summary


COL_LEN = 80


def main():
    """Main function of the script"""

    print("-" * COL_LEN)
    pfs = find_parameter_files()

    for i in range(len(pfs)):
        root, wd = get_root(pfs[i])
        if wd.find("continuum") != -1:
            continue
        errors = error_summary(root, wd, print_errors=True)

        print("-" * COL_LEN)

    return


if __name__ == "__main__":
    main()
