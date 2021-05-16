#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Basic vector functions which are not part of numpy.

These are taken directly from python, hence can be a little cursed.
"""

import numpy as np

from pypython.error import EXIT_FAIL


def renormalize_vector(a, scalar):
    """This function is used to renormalise a 3-vector quantity.

    Parameters
    ----------
    a:  np.ndarray
        The 3-vector to renormalise.
    scalar: Union[float, int]
        The desired length of the renormalised 3-vector.

    Returns
    -------
    a: np.ndarray
        The renormalized 3-vector quantity.
    """

    eps = 1e-10

    x = np.dot(a, a)

    if x < eps:
        print("Cannot renormalize a vector of magnitude 0")
        return EXIT_FAIL

    x = scalar / np.sqrt(x)
    a[0] *= x
    a[1] *= x
    a[2] *= x

    return a


def project_cartesian_to_cylindrical_coordinates(a, b):
    """Attempt to project a vector from cartesian into cylindrical coordinates.

    Parameters
    ----------
    a: np.ndarray
        The position of the vector in cartesian coordinates.
    b: np.ndarray
        The vector to project into cylindrical coordinates (also in cartesian
        coordinates).

    Returns
    -------
    result: np.ndarray
        The input vector b which is now projected into cylindrical
        coordinates.
    """

    result = np.zeros(3)
    n_rho = np.zeros(3)
    n_z = np.zeros(3)

    n_rho[0] = a[0]
    n_rho[1] = a[1]
    n_rho[2] = 0

    rc = renormalize_vector(n_rho, 1.0)
    if type(rc) == int:
        return rc

    n_z[0] = n_z[1] = 0
    n_z[2] = 1

    n_phi = np.cross(n_z, n_rho)

    result[0] = np.dot(b, n_rho)
    result[1] = np.dot(b, n_phi)
    result[2] = b[2]

    return result
