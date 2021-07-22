#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create or modify a grid of parameter files."""

from shutil import copyfile


def add_single_parameter(fp, name, new_value, backup=True):
    """Add a parameter which doesn't already exist.

    The parameter will be appended to the end of the parameter file but will be
    cleaned up in the root.out.pf file once the model is run.

    Parameters
    ----------
    fp: str
        The path to the parameter file
    name: str
        The name of the parameter to be added
    new_value: str
        The value of the parameter
    backup: bool [optional]
        Create a back up of the original parameter file
    """

    if fp.find(".pf") == -1:
        raise IOError(f"provided file path {fp} is not a .pf parameter file")

    if backup:
        copyfile(fp, fp + ".bak")

    with open(fp, "r") as f:
        pf = f.readlines()

    pf.append("{:40s} {}\n".format(name, new_value))

    with open(fp, "w") as f:
        f.writelines(pf)

    return


def create_grid(fp, name, values, extra_name=None, backup=True, verbose=False):
    """Create a bunch of new parameter files with the choice of values for a
    given parameter.

    This will only work for one parameter at a time and one parameter file.
    By default, a back up of the original parameter file is made as a safety
    precaution.

    Parameters
    ----------
    fp: str
        The path to the base parameter file to construct the grid from
    name: str
        The name of the parameter to create a grid of
    values: List[str]
        A list of values for the simulation grid for the parameter
    extra_name: str [optional]
        Adds an util name to the output grid parameter file names
    backup: bool [optional]
        Create a back up of the original parameter file
    verbose: bool [optional]
        Enable verbose output to the screen

    Returns
    -------
    grid: List[str]
        The paths to the newly generated parameter files for the grid
    """

    grid = []
    n_grid = len(values)

    if backup:
        copyfile(fp, fp + ".bak")

    ext = fp.find(".pf")
    if ext == -1:
        raise IOError(f"provided file path {fp} is not a .pf parameter file")

    for i in range(n_grid):
        fp_new = fp[:ext]
        if extra_name:
            fp_new += "_{}".format(extra_name)
        fp_new += "_{}".format(values[i]) + ".pf"
        copyfile(fp, fp_new)
        update_single_parameter(fp_new, name, values[i], backup=False, verbose=verbose)
        grid.append(fp_new)

    return grid


def get_parameter_value(fp, name):
    """Get the value for a parameter in a parameter file.

    The entire parameter file is searched to find the given parameter and
    returned as a string. If the parameter is not found, then a ValueError is
    raised.

    Parameters
    ----------
    fp: str
        The path to the parameter file
    name: str
        The name of the parameter

    Returns
    -------
    value: str
        The value of the parameter
    """

    if fp.find(".pf") == -1:
        raise IOError(f"provided file path {fp} is not a .pf parameter file")

    with open(fp, "r") as f:
        lines = f.readlines()

    value = None

    for line in lines:
        if line.find(name) != -1:
            split = line.split()
            if len(split) != 2:
                raise IndexError(f"invalid syntax for {name} in parameter file {fp}")
            return split[-1]

    if value is None:
        raise ValueError(f"parameter {name} was not found in {fp}")

    return value


def update_single_parameter(fp, name, new_value, backup=True, verbose=False):
    """Change the value of a parameter in a Python parameter file.

    If the old and new parameter value are the same, the script will still
    update the parameter file.

    Parameters
    ----------
    fp: str
        The path to the parameter file
    name: str
        The name of the parameter to update
    new_value: str
        The updated value of the parameter
    backup: bool [optional]
        Create a back up of the original parameter file
    verbose: bool [optional]
        Enable verbose output to the screen
    """

    if fp.find(".pf") == -1:
        raise IOError(f"provided file path {fp} is not a .pf parameter file")

    if backup:
        copyfile(fp, fp + ".bak")

    old = ""
    new = ""

    with open(fp, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.find(name) != -1:
            old = line
            new = "{}{:20s}{}\n".format(name, " ", new_value)
            lines[i] = new
            break

    if old and new:
        if verbose:
            print("changed parameter {} from {} to {}".format(name, old.replace("\n", ""), new.replace("\n", "")))
    else:
        print("unable to update: could not find parameter {} in file {}".format(name, fp))
        return

    with open(fp, "w") as f:
        f.writelines(lines)

    return
