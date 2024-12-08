import numpy as np
from numba import njit


@njit
def wind_bin_photon_weights(
    n_photons, nres, photon_x, photon_y, photon_z, photon_nres, photon_weight, x_points, z_points, nx, nz
):
    """Bin photon weights by extract location."""
    hist2d_weight = np.zeros((nx, nz))
    hist2d_count = np.zeros((nx, nz))

    xmin = np.min(x_points)
    xmax = np.max(x_points)

    zmin = np.min(z_points)
    zmax = np.max(z_points)

    for i in range(n_photons):
        if photon_nres[i] != nres:
            continue

        # get array index for rho point

        rho = np.sqrt(photon_x[i] ** 2 + photon_y[i] ** 2)

        if rho < xmin:
            ix = 0
        elif rho > xmax:
            ix = -1
        else:
            ix = np.abs(x_points - rho).argmin()

        # get array index for z point

        z = np.abs(photon_z[i])

        if z < zmin:
            iz = 0
        elif z > zmax:
            iz = -1
        else:
            iz = np.abs(z_points - z).argmin()

        hist2d_weight[ix, iz] += photon_weight[i]
        hist2d_count[ix, iz] += 1

    return hist2d_weight, hist2d_count
