#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various utility functions which can be used to ease the
trials and tribulations of using Python and the Unix command line.
"""

import pandas as pd
from subprocess import Popen, PIPE
from platform import system
from shutil import which
from typing import Tuple


def remove_data_sym_links(search_dir: str = "./", verbose: bool = False):
    """
    Search recursively from the specified directory search_dir for all symbolic
    links named data. The purpose of this script is to clean up the symbolic
    links if a directory is being uploaded to cloud storage or transferred using
    scp.

    This script will only work on Unix systems where the find command is
    available.

    Parameters
    ----------
    search_dir: str
        The starting directory to search recursively from for symbolic links
    verbose: bool [optional]
        Enable verbose output

    Returns
    -------
    ndel: int
        The number of symbolic links which were deleted
    """

    n = remove_data_sym_links.__name__
    ndel = 0

    os = system().lower()
    if os != "darwin" and os != "linux":
        print("{}: system {} unavailable", n, os)
        return ndel

    # - type l will only search for symbolic links
    cmd = "cd {}; find . -type l -name 'data'".format(search_dir)
    stdout, stderr = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True).communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")

    if stderr:
        print("{}: stderr".format(n))
        print(stderr)
    if stdout:
        if verbose:
            print("{}: deleting data symbolic links in the following directories:\n\n{}".format(n, stdout[:-1]))
    else:
        if verbose:
            print("{}: no data symlinks to delete".format(n))
        return ndel

    dirs = stdout.split()
    for i in range(len(dirs)):
        dir = search_dir + dirs[i][1:]
        cmd = "rm {}".format(dir)
        stdout, stderr = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True).communicate()
        stdout = stdout.decode("utf-8")
        if stdout and verbose:
            print(stdout)
        stderr = stderr.decode("utf-8")
        if stderr:
            print(stderr)
        else:
            ndel += 1

    return ndel


def get_python_version(py: str = "py", verbose: bool = False) -> Tuple[str, str]:
    """
    Get the Python version and commit hash for the provided Python binary.
    This should also work with windsave2table.

    Parameters
    ----------
    py: str, optional
        The name of the Python executable in $PATH whose version will be queried
    verbose: bool, optional
        Enable verbose logging

    Returns
    --------
    version: str
        The version number of Python
    commit_hash: str
        The commit hash of Python
    """

    n = get_python_version.__name__
    version = ""
    commit_hash = ""

    path = which(py)
    if not path:
        raise OSError("{}: {} is not in $PATH".format(n, py))

    command = "{} --version".format(py)
    cmd = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = cmd.communicate()
    out = stdout.decode("utf-8").split()
    err = stderr.decode("utf-8")

    if err:
        print("{}: captured from stderr".format(n))
        print(stderr)

    for i in range(len(out)):
        if out[i] == "Version":
            version = out[i + 1]
        if out[i] == "hash":
            commit_hash = out[i + 1]

    if version == "" and verbose:
        print("{}: couldn't find version for {}".format(n, py))
    if commit_hash == "" and verbose:
        print("{}: couldn't find commit hash for {}".format(n, py))

    if verbose and version and commit_hash:
        print("{} version {}".format(py, version))
        print("Git commit hash    {}".format(commit_hash))
        print("Short commit hash  {}".format(commit_hash[:7]))

    return version, commit_hash


def windsave2table(root: str, path: str, verbose: bool = False) -> None:
    """
    Runs windsave2table in the directory path to create a bunch of data tables
    from the Python wind_save file. This function also created a
    root.ep.complete file which merges the heat and master data tables together.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    path: str
        The directory of the Python simulation where windsave2table will be run
    verbose: bool, optional
        Enable verbose logging
    """

    n = windsave2table.__name__

    version, hash = get_python_version("windsave2table", verbose)
    try:
        with open("version", "r") as f:
            lines = f.readlines()
        run_version = lines[0]
        run_hash = lines[1]
        if run_version != version and run_hash != hash:
            print("{}: unable to determine windsave2table and wind_save versions".format(n))
    except IOError:
        print("{}: unable to determine windsave2table version from py_run.py version file".format(n))

    in_path = which("windsave2table")
    if not in_path:
        raise OSError("{}: windsave2table not in $PATH and executable".format(n))

    command = "cd {}; Setup_Py_Dir; windsave2table {}".format(path, root)
    cmd = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = cmd.communicate()
    output = stdout.decode("utf-8")
    err = stderr.decode("utf-8")

    if verbose:
        print(output)
    if err:
        print("{}: the following was sent to stderr:".format(n))
        print(err)

    # Now create a "complete" file which is the master and heat put together into one csv
    heat_file = "{}/{}.0.heat.txt".format(path, root)
    master_file = "{}/{}.0.master.txt".format(path, root)

    try:
        heat = pd.read_csv(heat_file, delim_whitespace=True)
        master = pd.read_csv(master_file, delim_whitespace=True)
    except IOError:
        print("{}: could not open master or heat file for root {}".format(n, root))
        return

    # This merges the heat and master table together :-)
    append = heat.columns.values[14:]
    for i, col in enumerate(append):
        master[col] = pd.Series(heat[col])
    master.to_csv("{}/{}.ep.complete".format(path, root), sep=" ")

    return


def subplot_dims(nplots: int) -> Tuple[int, int]:
    """
    Determine the dimensions for a plot with multiple subplot panels. A design
    of two columns of subplot panels will always be used.

    TODO: 1 or 3, 4, etc column plots should be possible as well

    Parameters
    ----------
    nplots: int
        The number of subplots which will be plotted

    Returns
    -------
    dims: Tuple[int, int]
        The dimensions of the subplots returned as (nrows, ncols)
    """

    n = subplot_dims.__name__

    if nplots < 1 or type(nplots) != int:
        raise ValueError("{}: nplots should be a non-zero, positive and an integer".format(n))

    if nplots > 2:
        ncols = 2
        nrows = (1 + nplots) // ncols
    elif nplots > 9:
        ncols = 3
        nrows = (1 + nplots) // ncols
    else:
        ncols = 1
        nrows = nplots

    dims = (nrows, ncols)

    return dims
