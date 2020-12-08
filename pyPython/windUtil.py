#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various functions used to create visualisations of the
wind and ionisation structure of the wind from Python. This includes utility
functions from creating data tables, as well as functions to create plots of the
wind.
"""


from .error import CoordError, InvalidParameter

import os
from typing import Tuple, Union
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.table import Table


def get_wind_variable(
    root: str, v: str, vtype: str, path: str = ".", coord: str = "rectilinear", input_dataframe: pd.DataFrame = None,
    input_fname: str = None, return_indices: bool = False
) -> Tuple[np.array, np.array, np.array]:
    """
    Read in a given variable for a model in Python. For this to work, the user
    needs to have used windsave2table to have generated the tables. Ideally,
    the user needs to have used pythonUtil.windsave2table to create the
    .all.complete file.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    v: str
        The name of the quantity from the Python simulation
    vtype: str
        The type of quantity to extract, this can either be wind or ion
    path: str [optional]
        The directory containing the Python simulation
    coord: str [optional]
        The coordinate system in use. Currently this only works for polar
        and rectilinear coordinate systems
    input_dataframe: pd.DataFrame [bool]
        If this is provided, then this DataFrame will be used to search for
        the wind variable.
    input_fname: str [optional]
        If this is provided, then the wind quantity will be searched from in
        the file provided
    return_indices: bool [optional]
        Return the cell i, j indices instead of the x, z coordinates

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

    if type(root) != str:
        raise TypeError("{}: root name given is not a string".format(n))
    if type(v) != str:
        raise TypeError("{}: variable name given is not a string".format(n))
    if type(vtype) != str:
        raise TypeError("{}: variable type given is not a string".format(n))

    key = v.lower()
    vtype = vtype.lower()

    allowed_types = [
        "ion", "ion_density", "heat", "master", "wind", "gradient", "converge", "spec"
    ]

    # Open the file containing the key, this is determined by the variable
    # type. For input_file, any file can be provided

    if input_dataframe is not None:
        data = input_dataframe
    else:
        if input_fname is not None:
            if type(input_fname) != str:
                raise TypeError("{}: input_file should be provided as a string".format(n))
            fname = input_fname
        else:
            # Ion file
            if vtype== "ion" or vtype == "ion_density":
                ele_idx = v.find("_")
                element = v[:ele_idx]
                key = v[ele_idx + 1:]
                if vtype == "ion":
                    fname = "{}/{}.{}.frac.txt".format(path, root, element)
                else:
                    fname = "{}/{}.{}.den.txt".format(path, root, element)
            # Wind file - this uses the .all.complete file
            elif vtype == "wind":
                fname = "{}/{}.all.complete.txt".format(path, root)
            # This catches all the other cases
            elif vtype in allowed_types:
                fname = "{}/{}.{}.txt".format(path, root, vtype)
            else:
                raise InvalidParameter(
                    "{}: v type {} not recognised with v {}. Allowed types: {}".format(n, vtype, v, allowed_types)
                )

        file_exists = os.path.isfile(fname)
        if not file_exists:
            raise IOError("{}: the file {} does not exist!!".format(n, fname))

        data = pd.read_csv(fname, delim_whitespace=True)

    # For polar coordinates, ignore anything > 90 degrees, expect for ion files
    # for some reason which are fine

    if coord == "polar" and (vtype != "ion" or vtype != "ion_density"):
        data = data[~(data["theta"] > 90)]

    # Now we shall try and read the data from the file...

    xi = data["i"]
    zj = data["j"]
    nx_cells = int(np.max(xi) + 1)
    nz_cells = int(np.max(zj) + 1)

    # Get the x, z or r, theta cells depending on the coord system
    # For the polar coordinate system, we have to transform the coordinates into
    # polar coordinates when reading the ion tables
    # TODO check if we need to transform to polar for gradient, heat, converge etc.

    if coord == "rectilinear":
        x = data["x"].values.reshape(nx_cells, nz_cells)
        z = data["z"].values.reshape(nx_cells, nz_cells)
    elif coord == "polar":
        if vtype == "ion":
            x = data["x"].values.reshape(nx_cells, nz_cells)
            z = data["z"].values.reshape(nx_cells, nz_cells)
            r = np.sqrt(x ** 2 + z ** 2)
            theta = np.arctan2(z, x)
            # This fixes a previous bug where the wrong thing was returned
            x = r
            z = theta
        else:
            x = data["r"].values.reshape(nx_cells, nz_cells)
            z = np.deg2rad(data["theta"].values.reshape(nx_cells, nz_cells))
    else:
        raise CoordError("{}: unknown projection {}. Allowed projections: rectilinear or polar".format(n, coord))

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
    Return the vertical coordinates for a sight line given the x coordinates
    and the inclination of the sight line.

    Parameters
    ----------
    x: np.ndarray[float]
        The x-coordinates of the sight line
    theta: float
        The opening angle of the sight line

    Returns
    -------
    z: np.ndarray[float]
        The z-coordinates of the sight line
    """

    return x * np.tan(np.pi / 2 - np.deg2rad(theta))


def extract_variable_along_sightline(
    inclination: float, variable: str, root: str = None, wd: str = ".", grid: Table = None, legacy: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Extract a wind variable along a given sight line.

    Parameters
    ----------
    inclination: float
        The inclination to extract upon
    variable: str
        The variable to be extracted
    root: [optional] str
        The root name of the simulation.
    wd: [optional] str
        The working directory if the simulation
    grid: [optional] astropy.table.Table
        Use this provided grid of variables instead of reading one in.
    legacy: [optional] bool
        Uses legacy return of a tuple.

    Returns
    -------
    output: np.ndarray
        The x, z and variable extracted along the inclination.
    xcoords: np.ndarray
        The x coordiantes along the inclination.
    zcoords: np.ndarray
        The z coordinates along the inclination.
    extracted: np.ndarray
        The extracted variable along the inclination.
    """

    n = extract_variable_along_sightline.__name__

    if grid is None:
        if root is None:
            raise InvalidParameter("{}: neither a root name or grid have been provided".format(n))
        else:
            t = ascii.read("{}/{}.all.complete.txt".format(wd, root), format="basic", data_start=1)
    else:
        t = grid

    # Check that the key is in the columns list

    columns = list(t.columns)
    if variable not in columns:
        raise KeyError("{}: {} is not a variable in the grid".format(n, variable))

    # Attempt to convert inclination to a float if it is not one, I do not
    # try to avoid an error if this doesn't work and just let the script crash

    if type(inclination) is not float:
        inclination = float(inclination)

    stride = np.max(t["j"]) + 1
    x_coords = np.array(t["x"][::stride])
    z_coords = sightline_coords(x_coords, inclination)
    if len(x_coords) != len(z_coords):
        raise ValueError("{}: there are an unequal amount of x and z coordinate points provided".format(n))
    extracted = np.zeros_like(x_coords)

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
