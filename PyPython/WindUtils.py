#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various functions used to create visualisations of the
wind and ionisation structure of the wind from Python. This includes utility
functions from creating data tables, as well as functions to create plots of the
wind.
"""


from .Error import CoordError, InvalidParameter

import os
from typing import Tuple
import numpy as np
import pandas as pd


def extract_wind_var(root: str, var_name: str, var_type: str, path: str = "./", coord: str = "rectilinear",
                     input_file: str = None) -> Tuple[np.array, np.array, np.array]:
    """
    Read in variables contained within a root.ep.complete or ion file generated
    by windsave2table.

    This requires the user to have already run windsave2table so the data is in
    the directory. It is also assumed that a root.ep.complete file exists which
    contains both the heat and master data.

    This function will also only work for 2d models :^^^).
    TODO: handle 1d data

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    var_name: str
        The name of the quantity from the Python simulation
    var_type: str
        The type of quantity to extract, this can either be wind or ion
    path: str [optional]
        The directory containing the Python simulation
    coord: str [optional]
        The coordinate system in use. Currently this only works for polar
        and rectilinear coordinate systems
    input_file: str [optional]
        If this is provided, then the wind quantity will be searched from in
        the file provided

    Returns
    -------
    x: np.array[float]
        The x coordinates of the wind in the Python simulation
    z: np.array[float]
        The z coordinates of the wind in the Python simulation
    var_mask: np.array[float]
        A numpy masked array for the quantity defined in var
    """

    n = extract_wind_var.__name__
    err_return = np.zeros(1)

    assert(type(root) == str), "{}: root must be a string".format(n)
    assert(type(var_name) == str), "{}: var_name must be a string".format(n)
    assert(type(var_type) == str), "{}: var_type must be a string".format(n)

    # Create string containing the name of the data file required to be loaded
    key = var_name
    if input_file:
        if type(input_file) is not str:
            print("{}: input_file should be provided as a string".format(n))
        file = input_file
    elif var_type.lower() == "ion":
        ele_idx = var_name.find("_")
        element = var_name[:ele_idx]
        key = var_name[ele_idx + 1:]
        file = "{}/{}.0.{}.txt".format(path, root, element)
    elif var_type.lower() == "wind":
        file = "{}/{}.ep.complete".format(path, root)
    else:
        raise InvalidParameter("{}: var type {} not recognised for var {}".format(n, var_type, var_name))

    file_exists = os.path.isfile(file)
    if not file_exists:
        raise IOError("{}: file {} doesn't exist var {} var type {}".format(n, file, var_name, var_type))

    try:
        data = pd.read_csv(file, delim_whitespace=True)
        if coord == "polar" and var_type != "ion":
            data = data[~(data["theta"] > 90)]
    except IOError:
        print("{}: could not open file {} for some reason".format(n, file))
        return err_return, err_return, err_return

    # Now try and read the data from the wind file
    try:
        # Get the i, j indices for the grid
        xi = data["i"]
        zj = data["j"]
        nx_cells = int(np.max(xi) + 1)
        nz_cells = int(np.max(zj) + 1)
        # Get the x, z or r, theta cells depending on the coord system
        if coord == "rectilinear":
            x = data["x"].values.reshape(nx_cells, nz_cells)
            z = data["z"].values.reshape(nx_cells, nz_cells)
        elif coord == "polar":
            # Transform from cartesian to polar coordinates manually for ion tables
            if var_type.lower() == "ion":
                x = data["x"].values.reshape(nx_cells, nz_cells)
                z = data["z"].values.reshape(nx_cells, nz_cells)
                r = np.sqrt(x ** 2 + z ** 2)  # r
                theta = np.arctan2(z, x)      # theta
                x = r
                z = theta
            else:
                x = data["r"].values.reshape(nx_cells, nz_cells)
                z = np.deg2rad(data["theta"].values.reshape(nx_cells, nz_cells))
        else:
            raise CoordError("{}: unknown projection {}: use rectilinear or polar".format(n, coord))
    except KeyError:
        print("{}: could not find var {} or another key".format(n, var_name))
        return err_return, err_return, err_return

    # Construct mask for variable
    var = data[key].values.reshape(nx_cells, nz_cells)
    inwind = data["inwind"].values.reshape(nx_cells, nz_cells)
    mask = (inwind < 0)
    var_mask = np.ma.masked_where(mask, var)

    return x, z, var_mask


def get_wind_elem_number(nx: int, nz: int, i: int, j: int) -> int:
    """
    Return the wind element number for the given grid indices (i, j) for a grid
    of size (nx, nz).

    Parameters
    ----------
    nx: int
        The number of grid cells in the x-direction.
    nz: int
        The number of grid cells in the z-direction.
    i: int
        The i (x) index for the grid cell in question
    j: int
        The j (z) index for the grid cell in question


    Returns
    -------
    elem: int
        The element number in the wind array
    """

    return nz * i + j
