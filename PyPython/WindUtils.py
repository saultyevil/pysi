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
from typing import Tuple, Union
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.table import Table


def get_wind_variable(
    root: str, var_name: str, var_type: str, path: str = ".", coord: str = "rectilinear", input_file: str = None,
    return_indices: bool = False
) -> Tuple[np.array, np.array, np.array]:
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
    return_indices: bool [optional]
        Return the cell i, j indicies instead of the x, z coordinates

    Returns
    -------
    x: np.array[float]
        The x coordinates of the wind in the Python simulation
    z: np.array[float]
        The z coordinates of the wind in the Python simulation
    var_mask: np.array[float]
        A numpy masked array for the quantity defined in var
    """

    n = get_wind_variable.__name__
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
        file = "{}/{}.{}.frac.txt".format(path, root, element)
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
                r = np.sqrt(x ** 2 + z ** 2)
                theta = np.arctan2(z, x)
                x = r
                z = theta
            else:
                x = data["r"].values.reshape(nx_cells, nz_cells)
                z = np.deg2rad(data["theta"].values.reshape(nx_cells, nz_cells))
        else:
            raise CoordError("{}: unknown projection {}: use rectilinear or polar".format(n, coord))
    except KeyError as e:
        print("{}: could not find key {} for var {}".format(n, e, var_name))
        return err_return, err_return, err_return

    # Reshape the variable and remove 0's or NaNs
    var = data[key].values.reshape(nx_cells, nz_cells)
    inwind = data["inwind"].values.reshape(nx_cells, nz_cells)
    var = np.ma.masked_where(inwind < 0, var)

    if return_indices:
        x = xi.values.reshape(nx_cells, nz_cells)
        z = zj.values.reshape(nx_cells, nz_cells)

    return x, z, var


def get_wind_elem_number(
    nx: int, nz: int, i: int, j: int
) -> int:
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


def sightline_coords(
    x: np.ndarray, theta: float
):
    """
    Return the vertical coordinates for a sightline given the x coordinates
    and the inclination of the sightline.

    Parameters
    ----------
    x: np.ndarray[float]
        The x-coordinates of the sightline
    theta: float
        The opening angle of the sightline

    Returns
    -------
    z: np.ndarray[float]
        The z-coordinates of the sightline
    """

    return x * np.tan(np.pi / 2 - np.deg2rad(theta))


def extract_variable_along_sightline(
    inclination: float, variable: str, root: str = None, wd: str = ".", grid: Table = None, legacy: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Extract a wind variable along a given sightline.

    TODO this needs to be properly documented and return correctly on error

    Parameters
    ----------
    inclination: float

    variable: str

    root: [optional] str

    wd: [optional] str

    grid: [optional] astropy.table.Table

    legacy: [optional] bool

    Returns
    -------
    xcoords: np.ndarray

    zcoords: np.ndarray

    extracted: np.ndarray

    """

    n = extract_variable_along_sightline.__name__

    if grid is None:
        if root is None:
            print("{}: neither a root name or grid have been provided".format(n))
            return
        try:
            t = ascii.read("{}/{}.master.txt".format(wd, root), format="basic", data_start=1)
        except IOError:
            print("{}: unable to open {}/{}.master.txt".format(n, wd, root))
            return
    else:
        t = grid

    columns = list(t.columns)
    if variable not in columns:
        print("{}: {} is not a variable in the grid".format(n, variable))
        return

    if type(inclination) is not float:
        try:
            inclination = float(inclination)
        except ValueError:
            print("{}: could not convert inclination to a number".format(n))
            return

    stride = np.max(t["j"]) + 1
    x_coords = np.array(t["x"][::stride])
    z_coords = sightline_coords(x_coords, inclination)
    extracted = np.zeros_like(x_coords)

    assert (len(x_coords) == len(z_coords))

    index = 0

    for i in range(len(x_coords)):
        j = 0
        while t["x"][j] < x_coords[i]:
            j += 1
            if j > len(t["x"]):
                j = -1
                break
        if j == -1:
            continue  # We haven't found the coordinate :-(
        k = 0
        tt = t[j:j + stride]
        while tt["z"][k] < z_coords[i]:
            k += 1
            if k > stride - 1:
                k = -1
                break
        if k == -1:
            continue  # We haven't found the coordinate :-(
        index = j + k
        extracted[i] = t[variable][index]

    output = np.zeros((len(x_coords), 3))
    output[:, 0] = x_coords
    output[:, 1] = z_coords
    output[:, 2] = extracted

    if legacy:
        return x_coords, z_coords, extracted
    else:
        return output
