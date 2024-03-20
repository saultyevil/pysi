#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create or modify a grid of parameter files."""

from shutil import copyfile


def add_parameter(fp, name, value, insert=None, backup=True, verbose=False):
    """Add a parameter which doesn't already exist.

    The parameter will either be appended to the end of the parameter file,
    or will be inserted after the parameter contained in insert.

    Parameters
    ----------
    fp: str
        The path to the parameter file
    name: str
        The name of the parameter to be added
    value: str
        The value of the parameter
    insert: str [optional]
        Insert the new parameter after this parameter
    backup: bool [optional]
        Create a back up of the original parameter file
    verbose: bool [optional]
        Enable verbose output.
    """

    if fp.find(".pf") == -1:
        raise IOError(f"provided file path {fp} is not a .pf parameter file")

    if backup:
        copyfile(fp, fp + ".bak")

    with open(fp, "r") as f:
        lines = f.readlines()

    # Get the parameters and values into a list. Removes blank lines and
    # comment lines

    lines = [line.split() for line in lines if line.split() and line.startswith("###") == False]
    names = [line[0] for line in lines]
    values = [line[1] for line in lines]

    # Check if the parameter is already in there and use update instead.
    # Otherwise, insert the new parameter somewhere and write the new
    # parameter file out

    if name in names:
        update_parameter(fp, name, value, backup)
    else:
        if insert:
            where = names.index(insert) + 1
        else:
            where = len(names)
        names.insert(where, name)
        values.insert(where, value)

        # Print update if verbose and write to file

        if verbose:
            header = f"- {fp} "
            while len(header) < 80:
                header += "-"
            print(header)
            print(f"   -> {name} {value}")

        with open(fp, "w") as f:
            for name, value in zip(names, values):
                f.write(f"{name:40s} {value}\n")

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
        update_parameter(fp_new, name, values[i], backup=False, verbose=verbose)
        grid.append(fp_new)

    return grid


def get_parameter(fp, name):
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
        raise IOError(f"Provided file path {fp} is not to a parameter file")

    with open(fp, "r", encoding="utf-8") as file_in:
        lines = file_in.readlines()

    value = None

    for line in lines:
        if line.find(name) != -1:
            split = line.split()
            if len(split) != 2:
                raise IndexError(f"Invalid syntax for {name} in {fp}")
            return split[-1]

    if value is None:
        raise ValueError(f"Could not find the parameter {name} in {fp}")

    return value


def update_parameter(fp, name, new_value, backup=True, verbose=False):
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
        raise IOError(f"The provided file path {fp} is not to a parameter file")

    if backup:
        copyfile(fp, fp + ".bak")

    old = ""
    new = ""

    with open(fp, "r", encoding="utf-8") as file_in:
        lines = file_in.readlines()

    for i, line in enumerate(lines):
        if line.find(name) != -1:
            old = line
            new = f"{name:40s} {new_value}\n"
            lines[i] = new
            break

    if old and new:
        if verbose:
            header = f"- {fp} "
            while len(header) < 80:
                header += "-"
            print(header)
            print(f"   {name}: {old.split()[-1]} -> {new_value}")
    else:
        raise ValueError(f"Could not find the parameter {name} in {fp}")

    with open(fp, "w", encoding="utf-8") as file_out:
        file_out.writelines(lines)

    return
