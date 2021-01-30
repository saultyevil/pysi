#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions for extracting wind quantities, as well as just ease of life things to
make plotting the wind easier.
"""


from .extrautil.error import CoordError, InvalidParameter
import os
from typing import Tuple, Union
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.table import Table


def extract_variable_along_sightline(
    inclination: float, variable: str, root: str = None, wd: str = ".", grid: Table = None, legacy: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract a wind variable along a given sight line.

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
        The extracted variable along the inclination."""

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
