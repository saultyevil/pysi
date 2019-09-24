#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various functions used to create visualisations of the
wind and ionisation structure of the wind from Python. This includes utility
functions from creating data tables, as well as functions to create plots of the
wind.
"""

from .Error import CoordError

import os
from typing import Tuple
import numpy as np
import pandas as pd


def extract_wind_var(root: str, var_name: str, var_type: str, path: str = "./", coord: str = "rectilinear") \
        -> Tuple[np.array, np.array, np.array]:
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

    assert(type(root) == str), "{}: root must be a string".format(n)
    assert(type(var_name) == str), "{}: var_name must be a string".format(n)
    assert(type(var_type) == str), "{}: var_type must be a string".format(n)

    # Create string containing the name of the data file required to be loaded
    nx_cells = 0
    nz_cells = 0
    key = var_name
    if var_type.lower() == "ion":
        ele_idx = var_name.find("_")
        element = var_name[:ele_idx]
        key = var_name[ele_idx + 1:]
        file = "{}/{}.0.{}.txt".format(path, root, element)
    elif var_type.lower() == "wind":
        file = "{}/{}.ep.complete".format(path, root)
    else:
        raise KeyError("{}: var type {} not recognised for var {}".format(n, var_type, var_name))

    file_exists = os.path.isfile(file)
    if not file_exists:
        raise IOError("{}: file {} doesn't exist. var {} var type {}".format(n, file, var_name, var_type))

    # Open the file and remove any garbage cells for theta grids
    try:
        data = pd.read_csv(file, delim_whitespace=True)
        if coord == "polar" and var_type != "ion":
            data = data[~(data["theta"] > 90)]
    except IOError:
        raise IOError("{}: could not open file {} for some reason".format(n, file))

    # Now we can try and read the data out of the file...
    # TODO: we should handle the two coordinates systems better..
    try:
        xi = data["i"]
        zj = data["j"]
        nx_cells = int(np.max(xi) + 1)
        nz_cells = int(np.max(zj) + 1)
        if coord == "rectilinear":
            x = data["x"].values.reshape(nx_cells, nz_cells)
            z = data["z"].values.reshape(nx_cells, nz_cells)
        elif coord == "polar":
            # Transform from cartesian to polar coordinates manually for ion tables
            if var_type.lower() == "ion":
                x = data["x"].values.reshape(nx_cells, nz_cells)
                z = data["z"].values.reshape(nx_cells, nz_cells)
                r = np.sqrt(x ** 2 + z ** 2)
                theta = np.rad2deg(np.arctan(z / x))
                x = r
                z = theta
            else:
                try:
                    x = data["r"].values.reshape(nx_cells, nz_cells)
                    z = data["theta"].values.reshape(nx_cells, nz_cells)
                except KeyError:
                    print("{}: trying to read r theta  points in non-polar model".format(n))
        else:
            raise CoordError("{}: unknown projection {}: use rectilinear or polar".format(n, coord))
    except KeyError:
        print("{}: could not find var {} or another key".format(n, var_name))

    # Construct mask for variable
    var = data[key].values.reshape(nx_cells, nz_cells)
    inwind = data["inwind"].values.reshape(nx_cells, nz_cells)
    mask = (inwind < 0)
    var_mask = np.ma.masked_where(mask, var)

    return x, z, var_mask
