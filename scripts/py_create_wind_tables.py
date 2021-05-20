#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create wind save tables for a model.

This script will overwrite any previously created wind save table. Verbose
printing is on by default, but can be suppressed.
"""

import argparse as ap

import pypython


def setup():
    """Setup the script.

    argparse is used to get parameters from the command line.
    """

    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("root", help="The root name of the simulation")
    p.add_argument("-fp", "--filepath", default=".", help="The directory containing the simulation")
    p.add_argument("-q", "--quiet", default=False, action="store_true", help="Suppress the output from the script")

    args = p.parse_args()

    return args.rootm, args.filepath, args.quiet


def main():
    """Main function of the script."""

    verbose = True
    root, fp, quiet = setup()
    if quiet:
        verbose = False

    pypython.create_wind_save_tables(root, fp, False, verbose)
    pypython.create_wind_save_tables(root, fp, True, verbose)

    return


if __name__ == "__main__":
    main()
