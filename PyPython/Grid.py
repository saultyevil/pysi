#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various functions used to create a grid of parameter files
or to edit a grid of parameter files.
"""


from os import mkdir
from typing import List
from shutil import copyfile


def change_parameter(pf_path: str, parameter_name: str, new_value: str, backup: bool = True, verbose: bool = False):
    """
    Change the value of a parameter in a Python parameter file. If the old and
    new parameter value are the same, the script will still update the parameter
    file.

    Parameters
    ----------
    pf_path: str
        The absolute or relative path to the parameter file
    parameter_name: str
        The name of the parameter being updated
    new_value: str
        The updated value for the parameter
    backup: bool [optional]
        Create a back up of the original parameter file
    verbose: bool [optional]
        Enable verbose output to the screen
    """

    n = change_parameter.__name__

    assert(type(pf_path) == str), "{}: The path to the parameter file is not a string".format(n)
    assert(type(parameter_name) == str), "{}: The parameter value passed is not a string".format(n)
    assert(type(new_value) == str), "{}: The new value passed is not a string".format(n)

    if pf_path.find(".pf") == -1:
        raise IOError("{}: provided parameter file path {} is not a .pf parameter file".format(n, pf_path))
    if verbose:
        print("{}: updating parameter file {}".format(n, pf_path))
    if backup:
        copyfile(pf_path, pf_path + ".bak")

    old = ""
    new = ""

    try:
        with open(pf_path, "r") as f:
            pf = f.readlines()
    except IOError:
        print("{}: unable to open parameter file {}".format(n, pf_path))
        return
    for i, line in enumerate(pf):
        if line.find(parameter_name) != -1:
            old = line
            new = "{}{:20s}{}\n".format(parameter_name, " ", new_value)
            pf[i] = new
            break
    if old and new:
        if verbose:
            print("{}: changed parameter {} from {} to {}".format(n, parameter_name, old.replace("\n", ""),
                                                                  new.replace("\n", "")))
    else:
        print("{}: couldn't find parameter {} in parameter file {}".format(n, parameter_name, pf_path))
        return

    with open(pf_path, "w") as f:
        f.writelines(pf)

    return


def add_parameter(pf_path: str, parameter_name: str, new_value: str, backup: bool = True, verbose: bool = False):
    """
    Add a parameter which doesn't already exist to the end of an already
    existing Python parameter file. The parameter will be appended to the
    end of the parameter file but will be cleaned up in the root.out.pf file
    once the model is run.

    Parameters
    ----------
    pf_path: str
        The absolute or relative path to the parameter file
    parameter_name: str
        The name of the parameter being updated
    new_value: str
        The updated value for the parameter
    backup: bool [optional]
        Create a back up of the original parameter file
    verbose: bool [optional]
        Enable verbose output to the screen
    """

    n = add_parameter.__name__

    assert(type(pf_path) == str), "{}: The path to the parameter file is not a string".format(n)
    assert(type(parameter_name) == str), "{}: The parameter value passed is not a string".format(n)
    assert(type(new_value) == str), "{}: The new value passed is not a string".format(n)

    if pf_path.find(".pf") == -1:
        raise IOError("{}: provided parameter file path {} is not a .pf parameter file".format(n, pf_path))
    if verbose:
        print("{}: updating parameter file {}".format(n, pf_path))
    if backup:
        copyfile(pf_path, pf_path + ".bak")

    try:
        with open(pf_path, "r") as f:
            pf = f.readlines()
    except IOError:
        print("{}: unable to open parameter file {}".format(n, pf_path))
        return

    pf.append("{:40s} {}\n".format(parameter_name, new_value))

    with open(pf_path, "w") as f:
        f.writelines(pf)

    return


def create_grid(pf_path: str, parameter_name: str, grid_values: List[str], extra_name: str = None, backup: bool = True,
                verbose: bool = False) -> List[str]:
    """
    Creates a bunch of new parameter files with the choice of values for a
    given parameter. This will only work for one parameter at a time and one
    parameter file. By default, a back up of the original parameter file is made
    as a safety precaution.

    Parameters
    ----------
    pf_path: str
        An absolute or relative path to the base parameter file to construct
        the grid from
    parameter_name: str
        The name of the parameter to construct a grid for
    grid_values: List[str]
        A list of values for the simulation grid for the parameter
    extra_name: str [optional]
        Adds an extra name to the output grid parameter file names
    backup: bool [optional]
        Create a back up of the original parameter file
    verbose: bool [optional]
        Enable verbose output to the screen

    Returns
    -------
    grid_pf: List[str]
        The paths to the newly generated parameter files for the grid
    """

    n = create_grid.__name__
    ngrid = len(grid_values)
    grid_pf = []

    if backup:
        copyfile(pf_path, pf_path + ".bak")

    ext = pf_path.find(".pf")
    if ext == -1:
        raise IOError("{}: provided parameter file path {} is not a .pf parameter file".format(n, pf_path))

    for i in range(ngrid):
        path = pf_path[:ext]
        if extra_name:
            path += "_{}".format(extra_name)
        path += "_{}".format(grid_values[i]) + ".pf"
        print(path)
        copyfile(pf_path, path)
        change_parameter(path, parameter_name, grid_values[i], backup=False, verbose=verbose)
        grid_pf.append(path)

    return grid_pf
