#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Basic vector functions used in Python and PyPython.

These are taken directly from Python, so may be a little cursed and unpythonic.
"""

import numpy


def renorm_vec(in_vec: numpy.ndarray, renorm_length: float, epsilon: float = 1e-10) -> numpy.ndarray:
    """This function is used to renormalise a 3-vector quantity.

    Parameters
    ----------
    a:  numpy.ndarray
        The 3-vector to renormalise.
    renorm_length: float
        The desired length of the renormalised 3-vector.

    Returns
    -------
    a: numpy.ndarray
        The renormalised 3-vector quantity.
    """
    x_vec = numpy.dot(in_vec, in_vec)
    if x_vec < epsilon:
        raise ValueError("Trying to renormalise a vector with magnitude 0")

    return in_vec * (renorm_length / numpy.sqrt(x_vec))


def project_cartesian_vec_to_cylindrical_vec(pos_vec: numpy.ndarray, vec: numpy.ndarray) -> numpy.ndarray:
    """Project a Cartesian vector into cylindrical coordinates.

    Parameters
    ----------
    pos_vec: numpy.ndarray
        The position of the vector in cartesian coordinates.
    vec: numpy.ndarray
        The vector to project into cylindrical coordinates (also in cartesian
        coordinates).

    Returns
    -------
    result_vec: numpy.ndarray
        The input vector b which is now projected into cylindrical
        coordinates.
    """

    result_vec = numpy.zeros(3)
    n_rho = numpy.zeros(3)
    n_z = numpy.zeros(3)

    n_rho[0] = pos_vec[0]
    n_rho[1] = pos_vec[1]
    n_rho[2] = 0  # this is zero due to "2.5d" nature of python
    n_rho = renorm_vec(n_rho, 1.0)

    n_z[0] = n_z[1] = 0
    n_z[2] = 1
    n_phi = numpy.cross(n_z, n_rho)

    result_vec[0] = numpy.dot(vec, n_rho)
    result_vec[1] = numpy.dot(vec, n_phi)
    result_vec[2] = vec[2]

    return result_vec
