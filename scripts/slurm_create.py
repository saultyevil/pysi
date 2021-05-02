#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a *.slurm file for a Python simulation. This script can also be used to
update an already existing .slurm file, for example if one wishes to restart
a Python simulation.
"""

import argparse
from typing import Tuple

from pypython.util import create_slurm_file


def parse_arguments() -> Tuple[str, int, int, int, bool, str, str]:
    """Parse arguments from the command line.

    Returns
    -------
    args.name: str
        The name of the slurm file
    args.ncores: int
        The number of CPUs to use
    args.thours: int
        The maximum run time allowed + 1 hours
    args.tminutes: int
        The number of minutes allowed.
    args.split_cycle: bool
        Split the ionization and spectral cycles.
    args.root: str
        The root name of the Python simulation
    args.flags: str
        Any flags to pass to Python"""

    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("name", help="The name of the slurm file, i.e. name.slurm.")
    p.add_argument("root", help="The root name of the model.")
    p.add_argument("ncores", type=int, help="The number of CPUs to use.")
    p.add_argument("thours",
                   type=int,
                   help="The number of hours of run time allowed.")
    p.add_argument(
        "tminutes",
        type=int,
        help="The number of minutes of additional run time allowed.")
    p.add_argument(
        "-f",
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
    """Main function of the script."""

    name, n_cores, t_hours, t_minutes, split_cycle, root, flags = parse_arguments(
    )
    flags += " -t {} ".format(int(t_hours * 3600 + t_minutes * 60))
    create_slurm_file(name, n_cores, split_cycle, t_hours, t_minutes, root,
                      flags)

    return


if __name__ == "__main__":
    main()
