#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The purpose of this script is to automatically generate a *.slurm file for a
Python simulation given sensible inputs. This script can also be used to
update an already existing .slurm file, for example if one wishes to restart
a Python simulation.
"""


import argparse
from typing import Tuple

from pyPython.hpc import create_slurm_file


def parse_arguments() -> Tuple[str, int, int, int, bool, str, str]:
    """
    Parse arguments from the command line.

    Returns
    -------
    args.name: str
        The name of the slurm file
    args.root: str
        The root name of the Python simulation
    args.ncores: int
        The number of CPUs to use
    args.thours: int
        The maximum run time allowed + 1 hours
    args.flags: str
        Any flags to pass to Python
    args.vers: str
        The version of Python to use
    """

    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("name",
                   help="The name of the slurm file, i.e. name.slurm.")

    p.add_argument("root",
                   help="The root name of the model.")

    p.add_argument("ncores",
                   type=int,
                   help="The number of CPUs to use.")

    p.add_argument("thours",
                   type=int,
                   help="The number of hours of run time allowed.")

    p.add_argument("tminutes",
                   type=int,
                   help="The number of minutes of additional run time allowed.")

    p.add_argument("-f",
                   "--flags",
                   default="",
                   help="Any flags to pass to the py_run.py Python running script.")

    p.add_argument("-sc",
                   "--split_cycle",
                   action="store_true",
                   default=False,
                   help="Use the split cycle method for py_run.py")

    args = p.parse_args()

    return args.name, args.ncores, args.thours, args.tminutes, args.split_cycle, args.root, args.flags


def main() -> None:
    """
    Main function of the script. Parses the arguments from the command line and
    then executes the function to generate the slurm file.
    """

    name, ncores, thours, tminutes, split_cycle, root, flags = parse_arguments()
    flags += " -t {} ".format(int(thours * 3600 + tminutes * 60))
    create_slurm_file(name, ncores, split_cycle, thours, tminutes, root, flags)

    return


if __name__ == "__main__":
    main()
