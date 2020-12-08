#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various utility functions which can be used to ease the
trials and tribulations of using Python and the Unix command line.
"""

from .constants import LOG_BASE_10_OF_TWO

from os import remove
from pathlib import Path
import pandas as pd
from subprocess import Popen, PIPE
from platform import system
from shutil import which
from typing import Tuple, List, Union
from psutil import cpu_count
import numpy as np
from matplotlib import pyplot as plt


def get_array_index(
    x: np.ndarray, target: float
) -> int:
    """
    Return the index for an array for a given value.

    Parameters
    ----------
    x: np.ndarray

    target: float

    Returns
    -------
    The index for the target value in the array x.
    """

    if target < np.min(x):
        return 0
    if target > np.max(x):
        return -1

    index = np.abs(x - target).argmin()

    return index


def round_to_sig_figs(
    x: np.ndarray, sigfigs: int
):
    """
    Truncate values in a numpy array to the given level of significant figures.

    Written by some maniac on Stack Overflow.
    https://stackoverflow.com/questions/18915378/rounding-to-significant-figures-in-numpy

    Parameters
    ----------
    x: np.ndarray
        The array to preform the operation on.
    sigfigs: int
        The number of significant figures.

    Returns
    -------
    x: np.ndarray
        The array rounded to the provided number of significant figures.
    """

    xsgn = np.sign(x)
    absx = xsgn * x
    mantissa, binaryExponent = np.frexp(absx)

    decimalExponent = LOG_BASE_10_OF_TWO * binaryExponent
    omag = np.floor(decimalExponent)

    mantissa *= 10.0 ** (decimalExponent - omag)

    if mantissa.any() < 1.0:
        mantissa *= 10.0
        omag -= 1.0

    return xsgn * np.around(mantissa, decimals=sigfigs - 1) * 10.0 ** omag


def file_len(
    fname: str
) -> int:
    """
    Count the number of lines in a file.

    Parameters
    ----------
    fname: str

    Returns
    -------
    The number of lines in the file.
    """

    with open(fname, "r") as f:
        for i, l in enumerate(f):
            pass

    return i + 1


def remove_data_sym_links(
    search_dir: str = "./", verbose: bool = False
):
    """
    Search recursively from the specified directory search_dir for all symbolic
    links named data. The purpose of this script is to clean up the symbolic
    links if a directory is being uploaded to cloud storage or transferred using
    scp.

    This script will only work on Unix systems where the find command is
    available.

    TODO: find solution which is system agnostic -> pathlib?

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
        stderr = stderr.decode("utf-8")

        if verbose:
            print(stdout)

        if stderr:
            print(stderr)
        else:
            ndel += 1

    return ndel


def root_from_file_path(
    path: str
) -> Tuple[str, str]:
    """
    Split a path name into a directory path and root name for a Python
    simulation.

    TODO: probably better to use find() or rfind() string methods

    Parameters
    ----------
    path: str
        The directory path to a Python .pf file

    Returns
    -------
    root: str
        The root name of the Python simulation
    wd: str
        The directory path containing the provided Python .pf file
    """

    n = root_from_file_path.__name__

    if type(path) != str:
        raise TypeError("{}: expected string as input".format(n))

    dot = 0
    slash = 0
    for i in range(len(path)):
        letter = path[i]
        if letter == ".":
            dot = i
        elif letter == "/":
            slash = i + 1

    root = path[slash:dot]
    wd = path[:slash]
    if wd == "":
        wd = "./"

    return root, wd


def find_parameter_files(
    root: str = None, path: str = "./"
) -> List[str]:
    """
    Find Python .pf parameter files recursively from the directory path.

    This function will ignore py_wind.pf parameter files, as well as any
    root.out.pf files.

    Parameters
    ----------
    root: str [optional]
        If given, only .pf files with the given root will be returned.
    path: str [optional]
        The directory to search for Python .pf files from

    Returns
    -------
    pfs: List[str]
        The file path for any Python pf files founds
    """

    parameter_files = []

    for filename in Path(path).glob("**/*.pf"):
        file = str(filename)

        if file.find(".out.pf") != -1:
            continue
        elif file.find("py_wind.pf") != -1:
            continue
        elif file[0] == "/":
            file = "." + file

        t_root, wd = root_from_file_path(file)
        if root and t_root != root:
            continue

        parameter_files.append(file)

    parameter_files = sorted(parameter_files, key=str.lower)

    return parameter_files


def get_cpu_count(
    enable_smt: bool = False
):
    """
    Return the number of CPU cores which can be used when running a Python
    simulation.

    By default, this will only return the number of physics cores and will
    exclude hyperthreads.

    Parameters
    ----------
    enable_smt: [optional] bool
        Return the number of logical cores, which includes both physical and
        SMT threads (hyperthreads).

    Returns
    -------
    ncores: int
        The number of available CPU cores
    """

    n = get_cpu_count.__name__
    ncores = 0

    try:
        ncores = cpu_count(logical=enable_smt)
    except NotImplementedError:
        print("{}: unable to determine number of CPU cores, psutil.cpu_count not implemented".format(n))

    return ncores


def get_python_version(
    py: str = "py", verbose: bool = False
) -> Tuple[str, str]:
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
        print(stderr)
        return "N/A", "N/A"

    for i in range(len(out)):
        if out[i] == "Version":
            version = out[i + 1]
        if out[i] == "hash":
            commit_hash = out[i + 1]

    if verbose:
        print("{} version {}".format(py, version))
        print("Git commit hash    {}".format(commit_hash))
        print("Short commit hash  {}".format(commit_hash[:7]))

    return version, commit_hash


def windsave2table(
    root: str, wd: str = ".", ion_density: bool = False, no_all_complete: bool = False, verbose: bool = False
) -> None:
    """
    Runs windsave2table in the directory path to create a bunch of data tables
    from the Python wind_save file. This function also created a
    root.ep.complete file which merges the heat and master data tables together.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    wd: str
        The directory of the Python simulation where windsave2table will be run
    ion_density: bool [optional]
        Pump out the ion density instead of ion fractions
    no_all_complete: bool [optional]
        Return from this function before a root.ep.complete file is created.
    verbose: bool [optional]
        Enable verbose logging
    """

    n = windsave2table.__name__

    version, hash = get_python_version("windsave2table", verbose)

    try:
        with open(".py_version", "r") as f:
            lines = f.readlines()
        run_version = lines[0]
        run_hash = lines[1]
        if run_version != version and run_hash != hash:
            print("{}: windsave2table and wind_save versions are different: be careful!".format(n))
    except IOError:
        if verbose:
            print("{}: unable to determine wind_save version: be careful!".format(n))

    in_path = which("windsave2table")
    if not in_path:
        raise OSError("{}: windsave2table not in $PATH and executable".format(n))

    command = "cd {}; Setup_Py_Dir; windsave2table".format(wd)
    if ion_density:
        command += " -d"
    command += " {}".format(root)

    cmd = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = cmd.communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")

    if verbose:
        print(stdout)
    if stderr:
        print("{}: the following was sent to stderr:".format(n))
        print(stderr)

    if no_all_complete:
        return

    # Now create a "complete" file which includes all the "wind" files, excluding
    # the ion files, created by windsave2table. It attempts to remove duplicate
    # columns, but I could have made a mistake

    include_files = [
        "heat", "gradient", "converge", "spec"
    ]

    include_files_index = [  # This is which column to index from to include as to avoid duplicate columns
        14, 9, 26, 8
    ]

    # Read in the master file first...

    mdf = pd.read_csv("{}/{}.master.txt".format(wd, root), delim_whitespace=True)

    # Now append all the other tables onto the end, to create one big thing

    for i, file in enumerate(include_files):
        fname = "{}/{}.{}.txt".format(wd, root, file)
        try:
            df = pd.read_csv(fname, delim_whitespace=True)
        except IOError:
            print("{}: unable to append {} to complete file".format(n, file))
            continue

        columns_to_add = df.columns.values[include_files_index[i]:]

        for new_column in columns_to_add:
            mdf[new_column] = pd.Series(df[new_column])

    if verbose:
        print(mdf)

    # Use numpy to write it as a fixed width file. Need to add the headings
    # at the top

    shape = mdf.values.shape
    marr = mdf.columns.values
    marr = np.reshape(np.append(marr, mdf.values), (shape[0] + 1, shape[1]))
    np.savetxt("{}/{}.all.complete.txt".format(wd, root), marr, fmt="%25s")

    return


def py_wind(
    root: str, commands: List[str], wd: str = "./"
) -> List[str]:
    """
    Run py_wind using the provided commands.

    Parameters
    ----------
    root: str
        The root name of the model.
    commands: list[str]
        The commands to pass to py_wind.
    wd: [optional] str
        The directory containing the model.

    Returns
    -------
    output: list[str]
        The stdout output from py_wind.
    """

    n = py_wind.__name__

    cmd_file = "{}/.tmpcmds.txt".format(wd)

    with open(cmd_file, "w") as f:
        for i in range(len(commands)):
            f.write("{}\n".format(commands[i]))

    sh = Popen("cd {}; py_wind {} < .tmpcmds.txt".format(wd, root), stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = sh.communicate()
    if stderr:
        print(stderr.decode("utf-8"))

    remove(cmd_file)

    return stdout.decode("utf-8").split("\n")


def subplot_dims(
    nplots: int
) -> Tuple[int, int]:
    """
    Determine the dimensions for a plot with multiple subplot panels. A design
    of two columns of subplot panels will always be used.

    TODO: update for more sub-panels

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

    return nrows, ncols


def create_run_script(commands: List[str]):
    """
    Create a shell run script given a list of commands to do. This assumes that
    you want to use a bash interpreter.

    Parameters
    ----------
    commands: List[str]
        The commands which are going to be run.
    """

    # Find any python parameter file in the directory and subdirectories
    directories = []
    pfs = find_parameter_files()
    for pf in pfs:
        root, directory = root_from_file_path(pf)
        directories.append(directory)

    file = "#!/bin/bash\n\ndeclare -a directories=(\n"
    for d in directories:
        file += "\t\"{}\"\n".format(d)
    file += ")\n\ncwd=$(pwd)\nfor i in \"${directories[@]}\"\ndo\n\tcd $i\n\tpwd\n"
    if len(commands) > 1:
        for k in range(len(commands) - 1):
            file += "\t{}\n".format(commands[k + 1])
    else:
        file += "\t# commands\n"
    file += "\tcd $cwd\ndone\n"

    # Print the file and write it to the current directory
    print(file)
    with open("commands.sh", "w") as f:
        f.write(file)

    return


def remove_extra_axes(
    fig: plt.Figure, ax: Union[plt.Axes, np.ndarray], n_wanted: int, n_panel: int
):
    """
    Remove additional axes which are included in a plot. This can be used if you
    have 4 x 2 = 8 panels but only want to use 7 of tha panels. The 8th panel
    will be removed.

    Parameters
    ----------
    fig: plt.Figure
        The Figure object to modify.
    ax: plt.Axes
        The Axes objects to modify.
    n_wanted: int
        The actual number of plots/panels which are wanted.
    n_panel: int
        The number of panels which are currently in the Figure and Axes objects.

    Returns
    -------
    fig: plt.Figure
        The modified Figure.
    ax: plt.Axes
        The modified Axes.
    """

    if type(ax) != np.ndarray:
        return fig, ax
    elif len(ax) == 1:
        return fig, ax

    # Flatten the axes array to make life easier with indexing

    shape = ax.shape
    ax = ax.flatten()

    if n_panel > n_wanted:
        for i in range(n_wanted, n_panel):
            fig.delaxes(ax[i])

    # Return ax to the shape it was passed as

    ax = np.reshape(ax, (shape[0], shape[1]))

    return fig, ax
