#!/usr/bin/env python3

"""Modify parameter files and construct grids of parameter files."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile


def add_parameter(
    filepath: str,
    parameter_name: str,
    parameter_value: str | float,
    *,
    insert_after: str | None = None,
    backup_original: bool | None = True,
) -> None:
    """Add a parameter which doesn't already exist.

    The parameter will either be appended to the end of the parameter file,
    or will be inserted after the parameter contained in insert.

    Parameters
    ----------
    filepath : str
        The path to the parameter file
    parameter_name : str
        The name of the parameter to be added
    parameter_value : str
        The value of the parameter
    insert_after : str [optional]
        Insert the new parameter after this parameter
    backup_original : bool [optional]
        Create a back up of the original parameter file

    """
    if filepath.find(".pf") == -1:
        raise OSError(f"provided file path {filepath} is not a .pf parameter file")
    if backup_original:
        copyfile(filepath, filepath + ".bak")
    with Path.open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    # Get the parameters and values into a list. Removes blank lines and
    # comment lines
    lines = [line.split() for line in lines if line.split() and not line.startswith("###")]
    names = [line[0] for line in lines]
    values = [line[1] for line in lines]

    # Check if the parameter is already in there and use update instead.
    # Otherwise, insert the new parameter somewhere and write the new
    # parameter file out
    if parameter_name in names:
        update_parameter_value(filepath, parameter_name, parameter_value, backup_original)
    else:
        where = names.index(insert_after) + 1 if insert_after else len(names)
        names.insert(where, parameter_name)
        values.insert(where, parameter_value)
        with Path.open(filepath, "w", encoding="utf-8") as f:
            for name, value in zip(names, values, strict=True):
                f.write(f"{name:40s} {value}\n")


def get_parameter_value(filepath: str, parameter_name: str) -> str:
    """Get the value for a parameter in a parameter file.

    The entire parameter file is searched to find the given parameter and
    returned as a string. If the parameter is not found, then a ValueError is
    raised.

    Parameters
    ----------
    filepath : str
        The path to the parameter file
    parameter_name : str
        The name of the parameter

    Returns
    -------
    value : str
        The value of the parameter, as a string.

    """
    if filepath.find(".pf") == -1:
        raise OSError(f"Provided file path {filepath} is not to a parameter file")
    with Path.open(filepath, encoding="utf-8") as file_in:
        lines = file_in.readlines()

    value = None
    for line in lines:
        if line.find(parameter_name) != -1:
            split = line.split()
            if len(split) != 2:  # noqa: PLR2004
                raise IndexError(f"Invalid syntax for {parameter_name} in {filepath}")
            return split[-1]
    if value is None:
        raise ValueError(f"Could not find the parameter {parameter_name} in {filepath}")

    return value


def update_parameter_value(
    filepath: str, parameter_name: str, parameter_value: str | float, *, backup_original: bool = True
) -> None:
    """Change the value of a parameter in a Python parameter file.

    If the old and new parameter value are the same, the script will still
    update the parameter file.

    Parameters
    ----------
    filepath : str
        The path to the parameter file
    parameter_name: str
        The name of the parameter to update
    parameter_value : str
        The updated value of the parameter
    backup_original : bool [optional]
        Create a back up of the original parameter file

    """
    if filepath.find(".pf") == -1:
        raise OSError(f"The provided file path {filepath} is not to a parameter file")
    if backup_original:
        copyfile(filepath, filepath + ".bak")
    with Path.open(filepath, encoding="utf-8") as file_in:
        lines = file_in.readlines()

    old = ""
    new = ""
    for i, line in enumerate(lines):
        if line.find(parameter_name) != -1:
            old = line
            new = f"{parameter_name:40s} {parameter_value}\n"
            lines[i] = new
            break
    if not old and not new:
        raise ValueError(f"Could not find the parameter {parameter_name} in {filepath}")

    with Path.open(filepath, mode="w", encoding="utf-8") as file_out:
        file_out.writelines(lines)


def create_grid(
    filepath: str,
    parameter_name: str,
    parameter_values: list[str | float],
    *,
    grid_name: str | None = None,
    backup_original: bool = True,
) -> None:
    """Create a grid of parameter files for a given parameter.

    This creates a grid of parameter files for the given garameter for the
    provided values. Note that this will work for only a single parameter at a
    time and will use an existing parameter file as input for the rest of the
    parameters.

    By default, a backup of the original parameter file is made.

    Parameters
    ----------
    filepath : str
        The path to the base parameter file to construct the grid from
    parameter_name : str
        The name of the parameter to create a grid of
    parameter_values : List[str | float]
        A list of values for the simulation grid for the parameter
    grid_name : str [optional]
        Adds an extra name to the output grid parameter file names, to associate
        the parameter file to the grid.
    backup_original : bool [optional]
        Create a back up of the original parameter file

    Returns
    -------
    grid_parameter_files : list[str]
        The paths to the newly generated parameter files for the grid

    """
    grid_parameter_files = []
    n_grid = len(parameter_values)
    if backup_original:
        copyfile(filepath, filepath + ".bak")
    ext = filepath.find(".pf")
    if ext == -1:
        raise OSError(f"provided file path {filepath} is not a .pf parameter file")

    for i in range(n_grid):
        fp_new = filepath[:ext]
        if grid_name:
            fp_new += f"_{grid_name}"
        fp_new += f"_{parameter_values[i]}" + ".pf"
        copyfile(filepath, fp_new)
        update_parameter_value(fp_new, parameter_name, parameter_values[i], backup_original=False)
        grid_parameter_files.append(fp_new)

    return grid_parameter_files
