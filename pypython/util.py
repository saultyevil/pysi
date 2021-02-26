#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions to ease the pain of using Python or a Unix environment whilst
trying to do computational astrophysics.
"""


from os import remove
from pathlib import Path
from subprocess import Popen, PIPE
from platform import system
from shutil import which
from scipy.signal import convolve, boxcar
from typing import Tuple, List, Union
from psutil import cpu_count
import numpy as np
import textwrap


def get_array_index(
    x: np.ndarray, target: float
) -> int:
    """Return the index for a given value in an array.

    This function is fairly limited in that it can't deal with arrays with
    duplicate values. It will always return the first value which is closest
    to the target value.

    Parameters
    ----------
    x: np.ndarray
        The array of values.
    target: float
        The value, or closest value, to find the index of.

    Returns
    -------
    The index for the target value in the array x."""

    if target < np.min(x):
        return 0
    if target > np.max(x):
        return -1

    index = np.abs(x - target).argmin()

    return index


def smooth_array(
    array: Union[np.ndarray, List[Union[float, int]]], width: Union[int, float]
) -> np.ndarray:
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
        The smoothed array"""

    # If smooth_amount is None or 1, then the user has indicated they didn't
    # actually want to use any smoothing, so return the original array

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

    array = np.reshape(array, (len(array),))  # todo: why do I have to do this? safety probably

    return convolve(array, boxcar(width) / float(width), mode="same")


def get_file_len(
    filename: str
) -> int:
    """Slowly count the number of lines in a file.
    todo: update to jit_open or some other more efficient method

    Parameters
    ----------
    filename: str
        The file name and path of the file to count the lines of.

    Returns
    -------
    The number of lines in the file."""

    with open(filename, "r") as f:
        for i, l in enumerate(f):
            pass

    return i + 1


def clean_up_data_sym_links(
    wd: str = ".", verbose: bool = False
):
    """Search recursively from the specified directory for symbolic links named
    data.
    This script will only work on Unix systems where the find command is
    available.
    todo: update to a system agnostic method to find symbolic links like pathlib

    Parameters
    ----------
    wd: str
        The starting directory to search recursively from for symbolic links
    verbose: bool [optional]
        Enable verbose output

    Returns
    -------
    n_del: int
        The number of symbolic links deleted"""

    n_del = 0

    os = system().lower()
    if os != "darwin" and os != "linux":
        print("your system does not work with this function", os)
        return n_del

    # - type l will only search for symbolic links
    cmd = "cd {}; find . -type l -name 'data'".format(wd)
    stdout, stderr = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True).communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")

    if stderr:
        print("sent from stderr")
        print(stderr)

    if stdout and verbose:
        print("deleting data symbolic links in the following directories:\n\n{}".format(stdout[:-1]))
    else:
        print("no data symlinks to delete")
        return n_del

    directories = stdout.split()

    for directory in directories:
        current = wd + directory[1:]
        cmd = "rm {}".format(current)
        stdout, stderr = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True).communicate()
        if stderr:
            print(stderr.decode("utf-8"))
        else:
            n_del += 1

    return n_del


def get_root_from_filepath(
    path: str, return_cd: bool = True
) -> Union[str, Tuple[str, str]]:
    """Get the root name of a Python simulation, extracting it from a file path.

    Parameters
    ----------
    path: str
        The directory path to a Python .pf file
    return_cd: str
        Returns the directory containing the .pf file.

    Returns
    -------
    root: str
        The root name of the Python simulation
    cd: str
        The directory path containing the provided Python .pf file"""

    if type(path) is not str:
        raise TypeError("expected string as input, not whatever you put")

    dot = 0
    slash = 0

    # todo: use find or rfind instead to avoid this mess

    for i in range(len(path)):
        letter = path[i]
        if letter == ".":
            dot = i
        elif letter == "/":
            slash = i + 1

    root = path[slash:dot]
    cd = path[:slash]

    if cd == "":
        cd = "."

    if return_cd:
        return root, cd
    else:
        return root


def get_parameter_files(
    root: str = None, cd: str = "."
) -> List[str]:
    """Search recursively for Python .pf files. This function will ignore
    py_wind.pf parameter files, as well as any root.out.pf files.

    Parameters
    ----------
    root: str [optional]
        If given, only .pf files with the given root will be returned.
    cd: str [optional]
        The directory to search for Python .pf files from

    Returns
    -------
    parameter_files: List[str]
        The file path for any Python pf files founds"""

    parameter_files = []

    for filepath in Path(cd).glob("**/*.pf"):
        str_filepath = str(filepath)
        if str_filepath.find(".out.pf") != -1:
            continue
        elif str_filepath.find("py_wind.pf") != -1:
            continue
        elif str_filepath[0] == "/":
            str_filepath = "." + str_filepath
        t_root, wd = get_root_from_filepath(str_filepath)
        if root and t_root != root:
            continue
        parameter_files.append(str_filepath)

    parameter_files = sorted(parameter_files, key=str.lower)

    return parameter_files


def get_cpu_count(
    enablesmt: bool = False
):
    """Return the number of CPU cores which can be used when running a Python
    simulation. By default, this will only return the number of physical cores
    and will ignore logical threads, i.e. in Intel terms, it will not count the
    hyperthreads.

    Parameters
    ----------
    enablesmt: [optional] bool
        Return the number of logical cores, which includes both physical and
        logical (SMT/hyperthreads) threads.

    Returns
    -------
    n_cores: int
        The number of available CPU cores"""

    n_cores = 0

    try:
        n_cores = cpu_count(logical=enablesmt)
    except NotImplementedError:
        print("unable to determine number of CPU cores, psutil.cpu_count not implemented for your system")

    return n_cores


def create_wind_save_tables(
    root: str, wd: str = ".", ion_density: bool = False, verbose: bool = False
) -> None:
    """Run windsave2table in a directory to create the standard data tables. The
    function can also create a root.all.complete.txt file which merges all the
    data tables together into one (a little big) file.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    wd: str
        The directory where windsave2table will run
    ion_density: bool [optional]
        Use windsave2table in the ion density version instead of ion fractions
    verbose: bool [optional]
        Enable verbose output"""

    in_path = which("windsave2table")
    if not in_path:
        raise OSError("windsave2table not in $PATH and executable")

    command = "cd {}; Setup_Py_Dir; windsave2table".format(wd)
    if ion_density:
        command += " -d"
    command += " {}".format(root)

    cmd = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = cmd.communicate()

    if verbose:
        print(stdout.decode("utf-8"))
    if stderr:
        print("the following was sent to stderr:")
        print(stderr.decode("utf-8"))

    return


def run_py_wind_commands(
    root: str, commands: List[str], wd: str = "."
) -> List[str]:
    """Run py_wind with the provided commands.

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
        The stdout output from py_wind."""

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


def create_slurm_file(
    name: str, n_cores: int, split_cycle: bool, n_hours: int, n_minutes: int, root: str, flags: str, wd: str = "."
) -> None:
    """Create a slurm file in the directory wd with the name root.slurm. All
    of the script flags are passed using the flags variable.

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
    split_cycle: bool
        If True, then py_run will use the split_cycle method
    flags: str
        The run-time flags of which to execute Python with
    root: str
        The root name of the model
    wd: str
        The directory to write the file to"""

    if split_cycle:
        split = "-sc"
    else:
        split = ""

    slurm = textwrap.dedent("""\
        #!/bin/bash
        #SBATCH --mail-user=ejp1n17@soton.ac.uk
        #SBATCH --mail-type=ALL
        #SBATCH --ntasks={}
        #SBATCH --time={}:{}:00
        #SBATCH --partition=batch
        module load openmpi/3.0.0/gcc
        module load conda/py3-latest
        source activate pypython
        python /home/ejp1n17/PythonScripts/py_run.py -n {} {} -f="{}"
        """.format(n_cores, n_hours, n_minutes, n_cores, split, flags, root)
    )

    if wd[-1] != "/":
        wd += "/"
    file_name = wd + name + ".slurm"
    with open(file_name, "w") as f:
        f.write("{}".format(slurm))

    return


def create_run_script(
    commands: List[str]
) -> None:
    """Create a shell run script given a list of commands to do. This assumes that
    you want to use a bash interpreter.

    Parameters
    ----------
    commands: List[str]
        The commands which are going to be run."""

    directories = []
    pfs = get_parameter_files()
    for pf in pfs:
        root, directory = get_root_from_filepath(pf)
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

    print(file)
    with open("commands.sh", "w") as f:
        f.write(file)

    return
