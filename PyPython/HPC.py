#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains functions which are helpful to use in a HPC environment. It
is obviously specifically tailored for me and Iridis 5.
"""


def create_slurm_file(
    name: str, ncores: int, split_cycle: bool, thours: int, tminutes: int, root: str, flags: str, wd: str = "."
) -> None:
    """
    Create a slurm file in the directory wd with the name root.slurm. All
    of the script flags are passed using the flags variable.

    Parameters
    ----------
    name: str
        The name of the slurm file
    ncores: int
        The number of cores which to use
    thours: int
        The number of hours to allow
    tminutes: int
        The number of minutes to allow
    split_cycle: bool
        If True, then py_run will use the split_cycle method
    flags: str
        The run-time flags of which to execute Python with
    root: str
        The root name of the model
    wd: str
        The directory to write the file to
    """

    if split_cycle:
        split = "-sc"
    else:
        split = ""

    slurm = \
        """#!/bin/bash
#SBATCH --mail-user=ejp1n17@soton.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --ntasks={}
#SBATCH --time={}:{}:00
#SBATCH --partition=batch
module load openmpi/3.0.0/gcc
module load conda/py3-latest
source activate PyPython
python /home/ejp1n17/PythonScripts/py_run.py -n {} {} -f="{}"
""".format(ncores, thours, tminutes, ncores, split, flags, root)

    if wd[-1] != "/":
        wd += "/"
    fname = wd + name + ".slurm"
    with open(fname, "w") as f:
        f.write("{}".format(slurm))

    return
