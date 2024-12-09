import numpy
from numba import njit


@njit
def create_interaction_histogram(  # noqa: PLR0913
    target_interation: int,
    photon_positions: numpy.ndarray,
    photon_interations: numpy.ndarray,
    photon_weights: numpy.ndarray,
    x_coords: numpy.ndarray,
    z_coords: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Create a 2D histogram of where photons interacted in the wind.

    Parameters
    ----------
    target_interation : int
        _description_
    photon_positions : numpy.ndarray
        _description_
    photon_interations : numpy.ndarray
        _description_
    photon_weights : numpy.ndarray
        _description_
    x_coords : numpy.ndarray
        _description_
    z_coords : numpy.ndarray
        _description_

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        _description_

    """
    n_x = len(x_coords)
    n_z = len(z_coords)
    hist_photon_weight = numpy.zeros((n_x, n_z))
    hist_photon_count = numpy.zeros((n_x, n_z))
    xmin = numpy.min(x_coords)
    xmax = numpy.max(x_coords)
    zmin = numpy.min(z_coords)
    zmax = numpy.max(z_coords)

    for i in range(len(photon_positions)):
        if photon_interations[i] != target_interation:
            continue

        # get array index for rho point
        rho = numpy.sqrt(photon_positions[i, 0] ** 2 + photon_positions[i, 1] ** 2)
        if rho < xmin:
            ix = 0
        elif rho > xmax:
            ix = -1
        else:
            ix = numpy.abs(x_coords - rho).argmin()

        # get array index for z point
        z = numpy.abs(photon_positions[i, 2])
        if z < zmin:
            iz = 0
        elif z > zmax:
            iz = -1
        else:
            iz = numpy.abs(z_coords - z).argmin()

        hist_photon_weight[ix, iz] += photon_weights[i]
        hist_photon_count[ix, iz] += 1

    return hist_photon_weight, hist_photon_count
