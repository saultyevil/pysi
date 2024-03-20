#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Basic utility functions.

Functions which offer utilty and are used project-wide belong in this
file.
"""

import os.path
import re
import subprocess
import time
from os import listdir, remove
from os.path import islink
from pathlib import Path
from shutil import which
from subprocess import run
from typing import List, Union

from psutil import cpu_count

from pypython import error


def count_cpu_cores(smt_allowed: bool = False) -> int:
    """Return the number of CPU cores which can be used when running a Python
    simulation. By default, this will only return the number of physical cores
    and will ignore logical threads, i.e. in Intel terms, it will not count the
    hyperthread.

    Parameters
    ----------
    smt_allowed: [optional] bool
        Return the number of logical cores, which includes both physical and
        logical (SMT/hyperthreads) threads.

    Returns
    -------
    n_cores: int
        The number of available CPU cores
    """
    n_cores = 0
    try:
        n_cores = cpu_count(logical=smt_allowed)
    except NotImplementedError:
        print("unable to determine number of CPU cores, psutil.cpu_count not implemented for your system")

    return int(n_cores)


def get_file_len(filename: str) -> int:
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
    with open(filename, "r", encoding="utf-8") as file_in:
        for i, _ in enumerate(file_in):
            pass

    return int(i + 1)  # pylint: disable=undefined-loop-variable


def run_shell_command(
    command: Union[List[str], str],
    file_path: Union[str, Path] = Path("."),
    verbose: bool = False,
) -> subprocess.CompletedProcess:
    """Run a shell command.

    Parameters
    ----------
    command: List[str] or str
        The shell command to run. Must either be a single string to call a
        program, or a list of the program and arguments for the program.
    fp: str or pathlib.Path [optional]
        The directory to run the command in.
    verbose: bool
        Print stdout to the screen.
    """
    shell_out = subprocess.run(command, capture_output=True, cwd=file_path, check=True)

    if verbose:
        print(shell_out.stdout.decode("utf-8"))
    if shell_out.stderr:
        print(
            f"Errors were reported for {' '.join(command)}:\n",
            shell_out.stderr.decode("utf-8"),
        )

    return shell_out


def cleanup_data(filepath: Union[str, Path] = ".") -> int:
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
    links = [d for d in find_files("data", filepath) if islink(d)]
    links += [d for d in find_files("xmod*", filepath) if islink(d)]
    for directory in links:
        Path(directory).unlink()

    return len(links)


def create_run_script(commands: Union[str, List[str]]) -> str:
    """Create a shell run script given a list of commands to do using bash a
    bash script.

    Parameters
    ----------
    commands: List[str]
        The commands which are going to be run.

    Returns
    -------
    file_contents: str
        The contents of the file, as a string.
    """

    paths_to_include = []
    prameter_files = find_files("*.pf")
    for file_contents in prameter_files:
        _, path = get_root_directory(file_contents)
        paths_to_include.append(path)

    file_contents = "#!/bin/bash\n\ndeclare -a directories=(\n"
    for file in paths_to_include:
        file_contents += f'\t"{file}"\n'
    file_contents += ')\n\ncwd=$(pwd)\nfor i in "${directories[@]}"\ndo\n\tcd $i\n\tpwd\n'

    if len(commands) > 1:
        for k in range(len(commands) - 1):
            file_contents += f"\t{commands[k + 1]}\n"
    else:
        file_contents += "\t# commands\n"

    file_contents += "\tcd $cwd\ndone\n"

    with open("commands.sh", "w", encoding="utf-8") as file_out:
        file_out.write(file_contents)

    return file_contents


def create_wind_save_tables(
    root, file_path=".", ion_density: bool = False, cell_spec: bool = False, version: str = None, verbose: bool = False
):
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
        with open(f"{file_path}/.py-version", "w") as file_out:
            file_out.write(f"{version}\n")

    in_path = which(name)
    if not in_path:
        raise OSError(f"{name} not in $PATH and executable")

    files_before = listdir(file_path)

    if not Path(f"{file_path}/data").exists():
        run_shell_command("Setup_Py_Dir", file_path)

    command = [name]
    if ion_density:
        command.append("-d")
    if cell_spec:
        command.append("-xall")
    command.append(root)

    cmd = run_shell_command(command, file_path, verbose)
    if cmd.returncode != 0:
        raise error.RunError(
            f"windsave2table has failed to run, possibly due to an incompatible version\n{cmd.stdout.decode('utf-8')}"
        )

    files_after = listdir(file_path)

    # Move the new files in fp/tables

    s = set(files_before)
    new_files = [x for x in files_after if x not in s]
    Path(f"{file_path}/tables").mkdir(exist_ok=True)
    for new in new_files:
        try:
            Path(f"{file_path}/{new}").rename(f"{file_path}/tables/{new}")
        except PermissionError:
            time.sleep(1.5)
            Path(f"{file_path}/{new}").rename(f"{file_path}/tables/{new}")

    return cmd.returncode


def find_files(pattern, file_path="."):
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
    file_path = os.path.expanduser(file_path)
    files = [str(file_) for file_ in Path(f"{file_path}").rglob(pattern)]
    if ".pf" in pattern:
        files = [this_file for this_file in files if "out.pf" not in this_file and "py_wind" not in this_file]

    try:
        files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])
    except TypeError:
        files.sort()

    return files


def get_python_version():
    """Check the version of Python available in $PATH.

    There are a number of features in this package which are not
    available in older versions of Python.

    Returns
    -------
    version: str
        The version string of the currently compiled Python.
    """

    command = run_shell_command(["py", "--version"])
    stdout = command.stdout.decode("utf-8").split("\n")
    stderr = command.stderr.decode("utf-8")

    if stderr:
        raise SystemError(f"{stderr}")

    version = None
    for line in stdout:
        if line.startswith("Python Version"):
            version = line[len("Python Version") + 1 :]

    return version


def get_root_directory(file_path):
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
    if not isinstance(file_path, str):
        raise TypeError("expected a string as input for the file path, not whatever you put")

    dot = file_path.rfind(".")
    slash = file_path.rfind("/")

    root = file_path[slash + 1 : dot]
    file_path = file_path[: slash + 1]

    if file_path == "":
        file_path = "./"

    return root, file_path


def run_py_optical_depth(root, photosphere=None, file_path=".", verbose=False):
    """Run py optical depth."""

    command = ["py_optical_depth"]
    if photosphere:
        command.append(f"-p {float(photosphere)}")
    command.append(root)

    cmd = run_shell_command(command, file_path)
    stdout, stderr = cmd.stdout, cmd.stderr

    if verbose:
        print(stdout.decode("utf-8"))

    if stderr:
        print(stderr.decode("utf-8"))


def run_py_wind(root, commands, file_path="."):
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
    cmd_file = f"{file_path}/.tmpcmds.txt"

    with open(cmd_file, "w", encoding="utf-8") as file_out:
        for command in commands:
            file_out.write(f"{command}\n")

    with open(cmd_file, "r", encoding="utf-8") as stdin:
        sh_out = run(["py_wind", root], stdin=stdin, capture_output=True, cwd=file_path, check=True)
    remove(cmd_file)

    stdout, stderr = sh_out.stdout, sh_out.stderr
    if stderr:
        print(stderr.decode("utf-8"))

    return stdout.decode("utf-8").split("\n")
