#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""""

import numpy as np
from numba import jit

import pypython
import pypython.dump as dump


@jit(nopython=True)
def wind_bin_photon_weights(n_photons, nres, photon_x, photon_y, photon_z, photon_nres, photon_weight, x_points,
                            z_points, nx, nz):
    """Bin photon weights by extract location."""

    hist2d_weight = np.zeros((nx, nz))
    hist2d_count = np.zeros((nx, nz))

    for i in range(n_photons):
        if photon_nres[i] != nres:
            continue
        rho = np.sqrt(photon_x[i]**2 + photon_y[i]**2)
        # get array index for rho point
        if rho < np.min(x_points):
            ix = 0
        elif rho > np.max(x_points):
            ix = -1
        else:
            ix = np.abs(x_points - rho).argmin()
        # get array index for z point
        z = np.abs(photon_z[i])
        if z < np.min(z_points):
            iz = 0
        elif z > np.max(z_points):
            iz = -1
        else:
            iz = np.abs(z_points - z).argmin()
        hist2d_weight[ix, iz] += photon_weight[i]
        hist2d_count[ix, iz] += 1

    return hist2d_weight, hist2d_count


def wind_bin_interaction_weight(root, nres, cd=".", n_cores=1):
    """Bin photon weights by extract location.

    Parameters
    ----------
    root: str
        The root name of the model.
    nres: int
        The resonance number of the photon to bin.
    cd: str [optional]
        The directory containing the simulation.
    n_cores: int [optional]
        The number of cores to normalize the binning by.

    Returns
    -------
    hist2d_weight: np.ndarray
        The photon weights. Each element of the array corresponds to a cell on
        the grid.
    """

    w = pypython.Wind(root, cd, masked=False)
    x_points = np.array(w.x)
    z_points = np.array(w.z)

    photons = dump.read_dump(root, cd, False)
    if photons.empty:
        print("photon dataframe is empty")
        exit(1)

    hist2d_weight, hist2d_count = wind_bin_photon_weights(len(photons), nres, photons["LastX"].values,
                                                          photons["LastY"].values, photons["LastZ"].values,
                                                          photons["Res."].values, photons["Weight"].values, x_points,
                                                          z_points, w.nx, w.nz)

    hist2d_weight /= n_cores

    name = "{}/{}_wind_Res{}_".format(cd, root, nres)
    np.savetxt(name + "weight.txt", hist2d_weight)
    np.savetxt(name + "count.txt", hist2d_count)

    return hist2d_weight, hist2d_count
