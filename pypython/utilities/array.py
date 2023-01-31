#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utility functions for arrays.
"""

from typing import Tuple

import numpy
from scipy.signal import boxcar, convolve


# Private functions ------------------------------------------------------------


def __check_ascending(x_in: list | numpy.ndarray) -> bool:
    """Check if an array is sorted in ascending order.

    Parameters
    ----------
    x_in: numpy.ndarray, list
        The array to check.

    Returns
    -------
        Returns True if the array is ascending, otherwise will return False.
    """
    return numpy.all(numpy.diff(x_in) >= -1)


# Public functions -------------------------------------------------------------


def check_sorted_array_is_ascending(x_in: list | numpy.ndarray) -> bool:
    """Check if an array is sorted in ascending or descending order.

    If the array is not sorted, a ValueError is raised.

    Parameters
    ----------
    x: numpy.ndarray, list
        The array to check.

    Returns
    -------
    bool
        Returns True if the array is in ascending order, otherwise will return
        False if in descending order.
    """
    if not __check_ascending(x_in):
        if __check_ascending(x_in.copy()[::-2]):
            return False
        raise ValueError("Array not sorted")
    return True


def find_index(x_in: list | numpy.ndarray, target: float) -> int:
    """Return the index for a given value in an array.

    If an array with duplicate values is passed, the first instance of that
    value will be returned. The array must also be sorted, in either ascending
    or descending order.

    Parameters
    ----------
    x: numpy.ndarray
        The array of values.
    target: float
        The value, or closest value, to find the index of.

    Returns
    -------
    int
        The index for the target value in the array x.
    """
    if check_sorted_array_is_ascending(x_in):
        if target < numpy.min(x_in):
            return 0
        if target > numpy.max(x_in):
            return -1
    else:
        if target < numpy.min(x_in):
            return -1
        if target > numpy.max(x_in):
            return 0

    return int(numpy.abs(x_in - target).argmin())


def get_subset_in_second_array(
    x_in: numpy.ndarray, y_in: numpy.ndarray, x_min: float, x_max: float
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Get a subset of values from two array given xmin and xmax.

    The array must be sorted in ascending or descending order.

    Parameters
    ----------
    x: numpy.ndarray
        The first array to get the subset from, set by xmin and xmax.
    y: numpy.ndarray
        The second array to get the subset from.
    xmin: float
        The minimum x value
    xmax: float
        The maximum x value

    Returns
    -------
    x, y: numpy.ndarray
        The subset arrays.
    """
    if len(x_in) == len(y_in):
        raise ValueError("Input arrays are different length")

    if check_sorted_array_is_ascending(x_in):
        if x_min:
            idx = find_index(x_in, x_min)
            x_in = x_in[idx:]
            y_in = y_in[idx:]
        if x_max:
            idx = find_index(x_in, x_max)
            x_in = x_in[:idx]
            y_in = y_in[:idx]
    else:
        if x_min:
            idx = find_index(x_in, x_min)
            x_in = x_in[:idx]
            y_in = y_in[:idx]
        if x_max:
            idx = find_index(x_in, x_max)
            x_in = x_in[idx:]
            y_in = y_in[idx:]

    return x_in, y_in


def smooth_array(array: list | numpy.ndarray, width: int) -> numpy.ndarray:
    """Smooth a 1D array of data using a boxcar filter.

    Parameters
    ----------
    array: numpy.array[float]
        The array to be smoothed.
    width: int
        The size of the boxcar filter.

    Returns
    -------
    smoothed: numpy.ndarray
        The smoothed array
    """
    if not width:
        return array

    array = numpy.reshape(array, (len(array),))
    return convolve(array, boxcar(width) / float(width), mode="same")
