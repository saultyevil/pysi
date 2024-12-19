"""Utility functions for arrays."""

import numpy
from scipy.signal import convolve
from scipy.signal.windows import boxcar


def check_array_is_ascending_order(x_in: list | numpy.ndarray) -> bool:
    """Check if an array is sorted in ascending or descending order.

    If the array is not sorted, a ValueError is raised.

    Parameters
    ----------
    x_in : numpy.ndarray, list
        The array to check.

    Returns
    -------
    bool
        Returns True if the array is in ascending order, otherwise will return
        False if in descending order.

    """
    check = lambda x: numpy.all(numpy.diff(x) >= -1)  # noqa: E731
    if not check(x_in):
        if check(x_in.copy()[::-2]):
            return False
        msg = f"x_in is not sorted in ascending or descending order: {x_in}"
        raise ValueError(msg)
    return True


def find_where_target_in_array(x_in: list | numpy.ndarray, target: float) -> int:
    """Return the index for a given value in an array.

    If an array with duplicate values is passed, the first instance of that
    value will be returned. The array must also be sorted, in either ascending
    or descending order.

    Parameters
    ----------
    x_in : numpy.ndarray
        The array of values.
    target : float
        The value, or closest value, to find the index of.

    Returns
    -------
    int
        The index for the target value in the array x.

    """
    if check_array_is_ascending_order(x_in):
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
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Get a subset of values from two array given xmin and xmax.

    The array must be sorted in ascending or descending order. Since we are
    slicing in Numpy, the arrays return will be a copy of x_in and y_in so
    the original arrays will not be modified.

    Parameters
    ----------
    x_in : numpy.ndarray
        The first array to get the subset from, set by xmin and xmax.
    y_in : numpy.ndarray
        The second array to get the subset from.
    x_max : float
        The minimum x value
    x_min : float
        The maximum x value

    Returns
    -------
    x, y: numpy.ndarray
        The subset arrays.

    """
    if len(x_in) != len(y_in):
        msg = "Input arrays are different length"
        raise ValueError(msg)

    if check_array_is_ascending_order(x_in):
        if x_min:
            idx = find_where_target_in_array(x_in, x_min)
            x_in = x_in[idx:]
            y_in = y_in[idx:]
        if x_max:
            idx = find_where_target_in_array(x_in, x_max)
            x_in = x_in[:idx]
            y_in = y_in[:idx]
    else:
        if x_min:
            idx = find_where_target_in_array(x_in, x_min)
            x_in = x_in[:idx]
            y_in = y_in[:idx]
        if x_max:
            idx = find_where_target_in_array(x_in, x_max)
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
