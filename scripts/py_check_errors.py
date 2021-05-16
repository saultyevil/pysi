#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The purpose of this script is to collate the errors over all of the MPI
processes for a bunch of simulations.

todo: this will only work for a simulation which didn't crash.......... according to past me at least.
"""

from pypython import get_file, get_root
from pypython.simulation import model_error_summary

COL_LEN = 80


def main():
    """Main function of the script."""

    pfs = get_file("*.pf")

    for i in range(len(pfs)):
        print("-" * 80)
        root, fp = get_root(pfs[i])
        if fp.find("continuum") != -1:
            continue
        model_error_summary(root, fp, print_errors=True)

    print("-" * 80)

    return


if __name__ == "__main__":
    main()
