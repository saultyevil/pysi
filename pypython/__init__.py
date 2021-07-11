#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pypython - making using Python a wee bit easier.

pypython is a companion python package to handle and analyse the data which
comes out of a Python simulation.
"""

import re
import textwrap
import time
from os import listdir, remove
from os.path import islink
from pathlib import Path
from shutil import which
from subprocess import run

import numpy as np
from scipy.signal import boxcar, convolve

from pypython.constants import (BOLTZMANN, CMS_TO_KMS, PARSEC, PI, PLANCK, VLIGHT)
from pypython.error import RunError
from pypython.math import vector
from pypython.physics.blackhole import gravitational_radius
from pypython.simulation.grid import get_parameter_value

# Dictionary class -------------------------------------------------------------


class _AttributeDict(dict):
    """A modified dictionary class.

    This class allows users to use . (dot) notation to also access the
    contents of a dictionary.
    """
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return dict.__repr__(self)


# Functions --------------------------------------------------------------------


def _check_ascending(x):
    """Check if an array is sorted in ascending order.

    Parameters
    ----------
    x: np.ndarray, list
        The array to check.
    """
    return np.all(np.diff(x) >= 0)


def check_sorted_array_ascending(x):
    """Check if an array is sorted in ascending or descending order.

    If the array is not sorted, a ValueError will be raised.

    Parameters
    ----------
    x: np.ndarray, list
        The array to check.

    Returns
    -------
        Returns True if the array is in ascending order, otherwise will return
        False if in descending order.
    """
    if not _check_ascending(x):
        if _check_ascending(x.copy()[::-1]):
            return False
        else:
            raise ValueError("check_sorted_array_ascending: the array provided is not sorted at all")

    return True


def cleanup_data(fp="."):
    """Remove data symbolic links created by Python.

    Search recursively from the specified directory for symbolic links named
    data.

    Parameters
    ----------
    fp: str
        The starting directory to search recursively from for symbolic links

    Returns
    -------
    n_del: int
        The number of symbolic links deleted
    """
    links = [d for d in find("data", fp) if islink(d)]
    links += [d for d in find("xmod*", fp) if islink(d)]

    for directory in links:
        Path(directory).unlink()

    return len(links)


def create_run_script(commands):
    """Create a shell run script given a list of commands to do using bash a
    bash script.

    Parameters
    ----------
    commands: List[str]
        The commands which are going to be run.
    """

    paths = []
    pf_fp = find("*.pf")
    for fp in pf_fp:
        root, path = get_root_name(fp)
        paths.append(path)

    file = "#!/bin/bash\n\ndeclare -a directories=(\n"
    for fp in paths:
        file += "\t\"{}\"\n".format(fp)
    file += ")\n\ncfp=$(pfp)\nfor i in \"${directories[@]}\"\ndo\n\tcd $i\n\tpfp\n"
    if len(commands) > 1:
        for k in range(len(commands) - 1):
            file += "\t{}\n".format(commands[k + 1])
    else:
        file += "\t# commands\n"
    file += "\tcd $cfp\ndone\n"

    print(file)
    with open("commands.sh", "w") as f:
        f.write(file)


def create_slurm_file(name, n_cores, n_hours, n_minutes, py_flags, py_run_flags, fp="."):
    """Create a slurm file in the directory fp with the name root.slurm.

    All of the script flags are passed using the flags variable.

    Parameters
    ----------
    name: str
        The name of the slurm file
    n_cores: int
        The number of cores which to use
    n_hours: int
        The number of hours to allow
    n_minutes: int
        The number of minutes to allow
    py_flags: str
        The run-time flags of which to execute Python with
    py_run_flags: str
        The run-time flags to pass to py_run.py
    fp: str
        The directory to write the file to
    """

    slurm = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --mail-user=ejp1n17@soton.ac.uk
        #SBATCH --mail-type=ALL
        #SBATCH --ntasks={n_cores}
        #SBATCH --time={n_hours}:{n_minutes}:00
        #SBATCH --partition=batch
        module load openmpi/3.0.0/gcc
        module load conda/py3-latest
        source activate pypython
        python /home/ejp1n17/PythonScripts/py_run.py -n {n_cores} {py_run_flags} -f='{py_flags}'
        """)

    if fp[-1] != "/":
        fp += "/"
    file_name = fp + name + ".slurm"
    with open(file_name, "w") as f:
        f.write(f"{slurm}")


def create_wind_save_tables(root, fp=".", ion_density=False, cell_spec=False, version=None, verbose=False):
    """Run windsave2table in a directory to create the standard data tables.

    The function can also create a root.all.complete.txt file which merges all
    the data tables together into one (a little big) file.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    fp: str
        The directory where windsave2table will run
    ion_density: bool [optional]
        Use windsave2table in the ion density version instead of ion fractions
    cell_spec: bool [optional]
        Use windsave2table to get the cell spectra.
    version: str [optional]
        The version number of windsave2table to use
    verbose: bool [optional]
        Enable verbose output
    """
    name = "windsave2table"
    if version:
        name += version

    in_path = which(name)
    if not in_path:
        raise OSError(f"{name} not in $PATH and executable")

    files_before = listdir(fp)

    if not Path(f"{fp}/data").exists():
        run_command("Setup_Py_Dir", fp)

    command = [name]
    if ion_density:
        command.append("-d")
    if cell_spec:
        command.append("-xall")
    command.append(root)

    cmd = run_command(command, fp, verbose)
    if cmd.returncode != 0:
        raise RunError(
            f"windsave2table has failed to run, possibly due to an incompatible version\n{cmd.stdout.decode('utf-8')}")

    files_after = listdir(fp)

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

    return cmd.returncode


def find(pattern, fp="."):
    """Find files of the given pattern recursively.

    This is used to find a number files given a globale pattern, i.e. *.spec,
    *.pf. When *.py is used, it'll ignore out.pf and py_wind.pf files. To find
    py_wind.pf files, use py_wind.pf as the pattern.

    Parameters
    ----------
    pattern: str
        Patterns to search recursively for, i.e. *.pf, *.spec, tde_std.pf
    fp: str [optional]
        The directory to search from, if not specified in the pattern.
    """

    files = [str(file_) for file_ in Path(f"{fp}").rglob(pattern)]
    if ".pf" in pattern:
        files = [this_file for this_file in files if "out.pf" not in this_file and "py_wind" not in this_file]

    try:
        files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    except TypeError:
        files.sort()

    return files


def get_array_index(x, target):
    """Return the index for a given value in an array.

    If an array with duplicate values is passed, the first instance of that
    value will be returned. The array must also be sorted, in either ascending
    or descending order.

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
    if check_sorted_array_ascending(x):
        if target < np.min(x):
            return 0
        if target > np.max(x):
            return -1
    else:
        if target < np.min(x):
            return -1
        if target > np.max(x):
            return 0

    return np.abs(x - target).argmin()


def get_python_version():
    """Check the version of Python available in $PATH.

    There are a number of features in this package which are not
    available in older versions of Python.

    Returns
    -------
    version: str
        The version string of the currently compiled Python.
    """

    command = run_command(["py", "--version"])
    stdout = command.stdout.decode("utf-8").split("\n")
    stderr = command.stderr.decode("utf-8")

    if stderr:
        raise SystemError(f"{stderr}")

    version = None
    for line in stdout:
        if line.startswith("Python Version"):
            version = line[len("Python Version") + 1:]

    return version


def get_root_name(fp):
    """Get the root name of a Python simulation.

    Extracts both the file path and the root name of the simulation.

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
        raise TypeError("expected a string as input for the file path, not whatever you put")

    dot = fp.rfind(".")
    slash = fp.rfind("/")

    root = fp[slash + 1:dot]
    fp = fp[:slash + 1]

    if fp == "":
        fp = "./"

    return root, fp


def get_xy_subset(x, y, xmin, xmax):
    """Get a subset of values from two array given xmin and xmax.

    The array must be sorted in ascending or descending order.

    Parameters
    ----------
    x: np.ndarray
        The first array to get the subset from, set by xmin and xmax.
    y: np.ndarray
        The second array to get the subset from.
    xmin: float
        The minimum x value
    xmax: float
        The maximum x value

    Returns
    -------
    x, y: np.ndarray
        The subset arrays.
    """
    assert len(x) == len(y)

    # The array has to be indexed differently depending on if it is ascending
    # or descending

    if check_sorted_array_ascending(x):
        if xmin:
            idx = get_array_index(x, xmin)
            x = x[idx:]
            y = y[idx:]
        if xmax:
            idx = get_array_index(x, xmax)
            x = x[:idx]
            y = y[:idx]
    else:
        if xmin:
            idx = get_array_index(x, xmin)
            x = x[:idx]
            y = y[:idx]
        if xmax:
            idx = get_array_index(x, xmax)
            x = x[idx:]
            y = y[idx:]

    return x, y


def smooth_array(array, width):
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

    array = np.reshape(array, (len(array), ))  # todo: why do I have to do this? safety probably

    return convolve(array, boxcar(width) / float(width), mode="same")


def run_py_optical_depth(root, photosphere=None, fp=".", verbose=False):

    command = ["py_optical_depth"]
    if photosphere:
        command.append(f"-p {float(photosphere)}")
    command.append(root)

    cmd = run_command(command, fp)
    stdout, stderr = cmd.stdout, cmd.stderr

    if verbose:
        print(stdout.decode("utf-8"))

    if stderr:
        print(stderr.decode("utf-8"))


def run_py_wind(root, commands, fp="."):
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
    cmd_file = f"{fp}/.tmpcmds.txt"

    with open(cmd_file, "w") as f:
        for i in range(len(commands)):
            f.write(f"{commands[i]}\n")

    with open(cmd_file, "r") as stdin:
        sh = run(["py_wind", root], stdin=stdin, capture_output=True, cwd=fp)

    stdout, stderr = sh.stdout, sh.stderr
    if stderr:
        print(stderr.decode("utf-8"))

    remove(cmd_file)

    return stdout.decode("utf-8").split("\n")


# Load in all the submodules ---------------------------------------------------

__all__ = [
    # functions in pypython
    "check_sorted_array_ascending",
    "cleanup_data",
    "create_run_script",
    "create_slurm_file",
    "create_wind_save_tables",
    "find",
    "get_array_index",
    "get_python_version",
    "get_root_name",
    "get_xy_subset",
    "smooth_array",
    "run_py_optical_depth",
    "run_py_wind",
    # sub-modules
    "math",
    "observations",
    "physics",
    "plot",
    "simulation",
    "spectrum",
    "util",
    "wind",
    # other things
    "constants"
]

# These are put here to solve a circular dependency ----------------------------

from pypython.plot import normalize_figure_style
from pypython.spectrum import Spectrum
from pypython.util import run_command
from pypython.wind import Wind

normalize_figure_style()
