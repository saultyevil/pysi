#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import time
from os import listdir, remove
from pathlib import Path
from platform import system
from shutil import which
from subprocess import PIPE, Popen
from typing import List, Tuple, Union

import automodinit
import numpy as np
from scipy.signal import boxcar, convolve

import spectrum
import wind

name = "pypython"


# Classes ------------------

Wind = wind.Wind
Spectrum = spectrum.Spectrum

# Functions ----------------

def cleanup_data(fp: str = ".", verbose: bool = False):
    """Search recursively from the specified directory for symbolic links named
    data. This script will only work on Unix systems where the find command is
    available.
    todo: update to a system agnostic method to find symbolic links like pathlib

    Parameters
    ----------
    fp: str
        The starting directory to search recursively from for symbolic links
    verbose: bool [optional]
        Enable verbose output

    Returns
    -------
    n_del: int
        The number of symbolic links deleted
    """
    n_del = 0

    os = system().lower()
    if os != "darwin" and os != "linux":
        raise OSError("your OS does not work with this function, sorry!")

    # - type l will only search for symbolic links
    cmd = "cd {}; find . -type l -name 'data'".format(fp)
    stdout, stderr = Popen(cmd, stdout=PIPE, stderr=PIPE,
                           shell=True).communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")

    if stderr:
        print("sent from stderr")
        print(stderr)

    if stdout and verbose:
        print(
            "deleting data symbolic links in the following directories:\n\n{}".
            format(stdout[:-1]))
    else:
        print("no data symlinks to delete")
        return n_del

    directories = stdout.split()

    for directory in directories:
        current = fp + directory[1:]
        cmd = "rm {}".format(current)
        stdout, stderr = Popen(cmd, stdout=PIPE, stderr=PIPE,
                               shell=True).communicate()
        if stderr:
            print(stderr.decode("utf-8"))
        else:
            n_del += 1

    return n_del


def get_file(pattern: str, fp: str = "."):
    """Find files of the given pattern recursively.

    Parameters
    ----------
    pattern: str
        Patterns to search recursively for, i.e. *.pf, *.spec, tde_std.pf
    fp: str [optional]
        The directory to search from, if not specified in the pattern.
    """

    files = [str(file_) for file_ in Path(f"{fp}").rglob(pattern)]
    if ".pf" in pattern:
        files = [
            file_ for file_ in files
            if "out.pf" not in file_ and "py_wind" not in file_
        ]
    files.sort(key=lambda var: [
        int(x) if x.isdigit() else x
        for x in re.findall(r'[^0-9]|[0-9]+', var)
    ])

    return files


def get_array_index(x: np.ndarray, target: float) -> int:
    """Return the index for a given value in an array. This function will not
    be happy if you pass an array with duplicate values. It will always return
    the first instance of the duplicate array.

    Parameters
    ----------
    x: np.ndarray
        The array of values.
    target: float
        The value, or closest value, to find the index of.

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


def get_root(fp: str) -> Tuple[str, str]:
    """Get the root name of a Python simulation, extracting it from a file path.

    Parameters
    ----------
    fp: str
        The directory path to a Python .pf file

    Returns
    -------
    root: str
        The root name of the Python simulation
    where: str
        The directory path containing the provided Python .pf file
    """
    if type(fp) is not str:
        raise TypeError(
            "expected a string as input for the file path, not whatever you put"
        )

    dot = fp.rfind(".")
    slash = fp.rfind("/")

    root = fp[slash + 1:dot]
    fp = fp[:slash + 1]
    if fp == "":
        fp = "./"

    return root, fp


def smooth_array(array: Union[np.ndarray, List[Union[float, int]]],
                 width: Union[int, float]) -> np.ndarray:
    """Smooth a 1D array of data using a boxcar filter.

    Parameters
    ----------
    array: np.array[float]
        The array to be smoothed.
    width: int
        The size of the boxcar filter.

    Returns
    -------
    smoothed: np.ndarray
        The smoothed array
    """
    if width is None or width == 0:
        return array

    if type(width) is not int:
        try:
            width = int(width)
        except ValueError:
            print("Unable to cast {} into an int".format(width))
            return array

    if type(array) is not np.ndarray:
        array = np.array(array)

    array = np.reshape(
        array,
        (len(array), ))  # todo: why do I have to do this? safety probably

    return convolve(array, boxcar(width) / float(width), mode="same")


def create_wind_save_tables(root: str,
                            fp: str = ".",
                            ion_density: bool = False,
                            verbose: bool = False) -> None:
    """Run windsave2table in a directory to create the standard data tables. The
    function can also create a root.all.complete.txt file which merges all the
    data tables together into one (a little big) file.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    fp: str
        The directory where windsave2table will run
    ion_density: bool [optional]
        Use windsave2table in the ion density version instead of ion fractions
    verbose: bool [optional]
        Enable verbose output
    """
    in_path = which("windsave2table")
    if not in_path:
        raise OSError("windsave2table not in $PATH and executable")

    files_before = listdir(fp)

    command = f"cd {fp};"
    if not Path(f"{fp}/data").exists():
        command += "Setup_Py_Dir;"
    command += "windsave2table"
    if ion_density:
        command += " -d"
    command += " {}".format(root)

    cmd = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = cmd.communicate()

    files_after = listdir(fp)

    if verbose:
        print(stdout.decode("utf-8"))
    if stderr:
        print("There may have been a problem running windsave2table")

    # Move the new files in fp/tables

    s = set(files_before)
    new_files = [x for x in files_after if x not in s]
    Path(f"{fp}/tables").mkdir(exist_ok=True)
    for new in new_files:
        try:
            Path(f"{fp}/{new}").rename(f"{fp}/tables/{new}")
        except PermissionError:
            time.sleep(1.5)
            Path(f"{fp}/{new}").rename(f"{fp}/tables/{new}")

    return


def run_py_wind_commands(root: str,
                         commands: List[str],
                         fp: str = ".") -> List[str]:
    """Run py_wind with the provided commands.

    Parameters
    ----------
    root: str
        The root name of the model.
    commands: list[str]
        The commands to pass to py_wind.
    fp: [optional] str
        The directory containing the model.

    Returns
    -------
    output: list[str]
        The stdout output from py_wind.
    """
    cmd_file = "{}/.tmpcmds.txt".format(fp)

    with open(cmd_file, "w") as f:
        for i in range(len(commands)):
            f.write("{}\n".format(commands[i]))

    sh = Popen("cd {}; py_wind {} < .tmpcmds.txt".format(fp, root),
               stdout=PIPE,
               stderr=PIPE,
               shell=True)
    stdout, stderr = sh.communicate()
    if stderr:
        print(stderr.decode("utf-8"))

    remove(cmd_file)

    return stdout.decode("utf-8").split("\n")


# Import all files using automodinit

__all__ = [""]
automodinit.automodinit(__name__, __file__, globals())
del automodinit
