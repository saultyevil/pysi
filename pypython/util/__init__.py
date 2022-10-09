#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Basic utility functions for making life easier.

Includes functions for creating slurm files, bash "run scripts", as well
as getting system info and a logging utility.
"""

import subprocess
from typing import List

from psutil import cpu_count


def get_cpu_count(enable_smt=False):
    """Return the number of CPU cores which can be used when running a Python
    simulation. By default, this will only return the number of physical cores
    and will ignore logical threads, i.e. in Intel terms, it will not count the
    hyperthreads.

    Parameters
    ----------
    enable_smt: [optional] bool
        Return the number of logical cores, which includes both physical and
        logical (SMT/hyperthreads) threads.

    Returns
    -------
    n_cores: int
        The number of available CPU cores
    """

    n_cores = 0

    try:
        n_cores = cpu_count(logical=enable_smt)
    except NotImplementedError:
        print("unable to determine number of CPU cores, psutil.cpu_count not implemented for your system")

    return n_cores


def get_file_len(filename):
    """Slowly count the number of lines in a file.
    todo: update to jit_open or some other more efficient method

    Parameters
    ----------
    filename: str
        The file name and path of the file to count the lines of.

    Returns
    -------
    The number of lines in the file.
    """
    with open(filename, "r") as f:
        for i, l in enumerate(f):
            pass

    return i + 1


def run_command(command, fp=".", verbose=False):
    """Run a shell command.

    Parameters
    ----------
    command: List[str] or str
        The shell command to run. Must either be a single string to call a
        program, or a list of the program and arguments for the program.
    fp: str [optional]
        The directory to run the command in.
    verbose: bool
        Print stdout to the screen.
    """

    sh = subprocess.run(command, capture_output=True, cwd=fp)
    if verbose:
        print(sh.stdout.decode("utf-8"))
    if sh.stderr:
        print("stderr reported errors:\n", sh.stderr.decode("utf-8"))

    return sh
