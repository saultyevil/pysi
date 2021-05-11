#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions used to create a grid of parameter files or to edit a grid of
parameter files.
"""

from shutil import copyfile
from typing import List


def update_single_parameter(path,
                            parameter_name,
                            new_value,
                            backup=True,
                            verbose=False):
    """Change the value of a parameter in a Python parameter file. If the old and
    new parameter value are the same, the script will still update the parameter
    file.

    Parameters
    ----------
    path: str
        The path to the parameter file
    parameter_name: str
        The name of the parameter to update
    new_value: str
        The updated value of the parameter
    backup: bool [optional]
        Create a back up of the original parameter file
    verbose: bool [optional]
        Enable verbose output to the screen"""

    if path.find(".pf") == -1:
        raise IOError(
            "provided parameter file path {} is not a .pf parameter file".
            format(path))

    if backup:
        copyfile(path, path + ".bak")

    old = ""
    new = ""

    try:
        with open(path, "r") as f:
            pf = f.readlines()
    except IOError:
        print("unable to open parameter file {}".format(path))
        return

    for i, line in enumerate(pf):
        if line.find(parameter_name) != -1:
            old = line
            new = "{}{:20s}{}\n".format(parameter_name, " ", new_value)
            pf[i] = new
            break

    if old and new:
        if verbose:
            print("changed parameter {} from {} to {}".format(
                parameter_name, old.replace("\n", ""), new.replace("\n", "")))
    else:
        print(
            "unable to update: could not find parameter {} in file {}".format(
                parameter_name, path))
        return

    with open(path, "w") as f:
        f.writelines(pf)

    return


def add_single_parameter(path, parameter_name, new_value, backup=True):
    """Add a parameter which doesn't already exist to the end of an already
    existing Python parameter file. The parameter will be appended to the
    end of the parameter file but will be cleaned up in the root.out.pf file
    once the model is run.

    Parameters
    ----------
    path: str
        The path to the parameter file
    parameter_name: str
        The name of the parameter to be added
    new_value: str
        The value of the parameter
    backup: bool [optional]
        Create a back up of the original parameter file"""

    if path.find(".pf") == -1:
        raise IOError(
            "provided parameter file path {} is not a .pf parameter file".
            format(path))

    if backup:
        copyfile(path, path + ".bak")

    try:
        with open(path, "r") as f:
            pf = f.readlines()
    except IOError:
        print("unable to open parameter file {}".format(path))
        return

    pf.append("{:40s} {}\n".format(parameter_name, new_value))

    with open(path, "w") as f:
        f.writelines(pf)

    return


def create_grid(path,
                parameter_name,
                grid_values,
                extra_name=None,
                backup=True,
                verbose=False):
    """Creates a bunch of new parameter files with the choice of values for a
    given parameter. This will only work for one parameter at a time and one
    parameter file. By default, a back up of the original parameter file is made
    as a safety precaution.

    Parameters
    ----------
    path: str
        The path to the base parameter file to construct the grid from
    parameter_name: str
        The name of the parameter to create a grid of
    grid_values: List[str]
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
        The paths to the newly generated parameter files for the grid"""

    n_grid = len(grid_values)
    grid = []

    if backup:
        copyfile(path, path + ".bak")

    ext = path.find(".pf")
    if ext == -1:
        raise IOError(
            "provided file path {} is not a .pf parameter file".format(path))

    for i in range(n_grid):
        path = path[:ext]
        if extra_name:
            path += "_{}".format(extra_name)
        path += "_{}".format(grid_values[i]) + ".pf"
        print(path)
        copyfile(path, path)
        update_single_parameter(path,
                                parameter_name,
                                grid_values[i],
                                backup=False,
                                verbose=verbose)
        grid.append(path)

    return grid
