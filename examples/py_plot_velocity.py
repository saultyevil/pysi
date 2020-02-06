#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This purpose of this script is to provide a quick way to plot the different
velocity components of a Python wind model. It assumes that the output is in
cartesian coordinates, hence the transformation back into cylindrical
coordinates later is fine to calculate the polodial and rotational velocity.

NOTE: THIS ONLY WORKS WITH RECTILINEAR PLOTS FOR NOW - NO POLAR STUFF!!!
"""


import numpy as np
import argparse as ap
from typing import Tuple, Union, List
from matplotlib import pyplot as plt

from PyPython import WindPlot
from PyPython import WindUtils
from PyPython import PythonUtils
from PyPython.Constants import CMS_TO_KMS
from PyPython.Error import EXIT_FAIL

plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15


def renorm_vector(a: np.ndarray, scalar: Union[float, int]):
    """
    This function is used to renormalise a 3-vector quantity.

    Parameters
    ----------
    a:  np.ndarray
        The 3-vector to renormalise.
    scalar: Union[float, int]
        The desired length of the renormalised 3-vector.

    Returns
    -------
    a: np.ndarray
        The renormalised 3-vector quantity.
    """

    EPS = 1e-10
    n = renorm_vector.__name__

    x = np.dot(a, a)

    if x < EPS:
        # print("{}: Cannot renormalise a vector of length 0".format(n))
        # print("{}: a = {}".format(n, a))
        return -1

    x = scalar / np.sqrt(x)
    a[0] *= x
    a[1] *= x
    a[2] *= x

    return a


def project_cart_to_cyl_coords(a: Union[np.ndarray, List[float]], b: Union[np.ndarray, List[float]]):
    """
    Attempt to a vector from cartesian into cylindrical coordinates.

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
        The input vector b which is now projected into cylindrical coordinates.
    """

    result = np.zeros(3)
    n_rho = np.zeros(3)
    n_z = np.zeros(3)

    n_rho[0] = a[0]
    n_rho[1] = a[1]
    n_rho[2] = 0

    rc = renorm_vector(n_rho, 1.0)
    if type(rc) == int:
        return rc

    n_z[0] = n_z[1] = 0
    n_z[2] = 1

    n_phi = np.cross(n_z, n_rho)

    result[0] = np.dot(b, n_rho)
    result[1] = np.dot(b, n_phi)
    result[2] = b[2]

    return result


def velocity_plot(root: str, wd: str = "./", axes_scales: str = "loglog", use_cell_indices: bool = False,
                  file_ext: str = "png") \
        -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a figure which shows the Cartesian velocities, as well as the
    polodial and the rotational velocity. The cylindrical coordinates will be
    evaluated assuming y = 0.

    Parameters
    ----------
    root: str
        The root name of the model.
    wd: str [optional]
        The directory where the simulation is stored, by default this assumes
        that it is in the calling directory.
    axes_scales: str [optional]
        The type of scaling for the axes of the figure, allowed values are
        logx, logy, loglog or linlin.
    use_cell_indices: bool [optional]
        If True then the wind will not be plotted in spatial coordinates, but
        rather cell index coordinates.
    file_ext: str [optional]
        The extension of the final output file.

    Returns
    -------
    fig: plt.Figure
        The matplotlib Figure object for the created plot.
    ax: plt.Axes
        The matplotlib Axes objects for the plot panels.
    """

    nrows = 3
    ncols = 2

    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 15), squeeze=False)

    # Set the scale to linear-linear when plotting with cell indices

    if use_cell_indices:
        axes_scales = "linlin"

    # First get the velocity in cartesian coordinates and then project this into
    # cylindrical coordinates. We will put everything into km/s.

    vx_x, vx_z, vx = WindUtils.get_wind_variable(root, "v_x", "wind", wd, "rectilinear",
                                                 return_indices=use_cell_indices)
    vy_x, vy_z, vy = WindUtils.get_wind_variable(root, "v_y", "wind", wd, "rectilinear",
                                                 return_indices=use_cell_indices)
    vz_x, vz_z, vz = WindUtils.get_wind_variable(root, "v_z", "wind", wd, "rectilinear",
                                                 return_indices=use_cell_indices)
    vl = np.zeros_like(vx)
    vrot = np.zeros_like(vx)
    vr = np.zeros_like(vx)
    n1, n2 = vx.shape

    for i in range(n1):
        for j in range(n2):
            r = [vx_x[i, j], 0, vx_z[i, j]]
            # This is some horrible hack to make sure there are no NaNs :^)
            if type(vx[i, j]) != np.float64 or type(vy[i, j]) != np.float64 or type(vz[i, j]) != np.float64:
                vl[i, j] = np.nan
                vrot[i, j] = np.nan
                vr[i, j] = np.nan
            else:
                v = [vx[i, j], vy[i, j], vz[i, j]]
                v_cyl = project_cart_to_cyl_coords(r, v)
                # If the return is an int, something has gone wrong!
                if type(v_cyl) == int:
                    continue
                vl[i, j] = np.sqrt(v_cyl[0] ** 2 + v_cyl[2] ** 2) * CMS_TO_KMS
                vrot[i, j] = v_cyl[1] * CMS_TO_KMS
                vr[i, j] = v_cyl[0] * CMS_TO_KMS

    vx *= CMS_TO_KMS
    vy *= CMS_TO_KMS
    vz *= CMS_TO_KMS

    # Now we can finally create the plot of the different velocities

    index = 0
    vels = [vx, vy, vz, vr, vrot, vl]
    velocities_to_plot = ["v_x", "v_y", "v_z", "v_r", "v_rot", "v_l"]
    nsize = len(velocities_to_plot) - 1

    for i in range(nrows):
        for j in range(ncols):

            if index > nsize:
                break

            vel = vels[index]
            name = velocities_to_plot[index]

            fig, ax = WindPlot.rectilinear_wind(vx_x, vx_z, vel, name, "wind", fig, ax, i, j, scale=axes_scales)

            index += 1

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig("{}/{}_velocities.{}".format(wd, root, file_ext))

    return fig, ax


def parse_input() -> tuple:
    """
    Parse the different modes this script can be run from the command line.

    Returns
    -------
    setup: tuple
        A list containing all of the different setup of parameters for plotting.

        setup = (
            args.root,
            wd,
            axes_scales,
            cell_indices,
            file_ext,
            display
        )
    """

    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("root", help="The root name of the simulation.")
    p.add_argument("-wd", action="store", help="The directory containing the simulation.")
    p.add_argument("-s", "--scales", action="store", help="The axes scaling to use: logx, logy, loglog, linlin.")
    p.add_argument("-c", "--cells", action="store_true", help="Plot using cell indices rather than spatial scales.")
    p.add_argument("-e", "--ext", action="store", help="The file extension for the output figure.")
    p.add_argument("--display", action="store_true", help="Display the plot before exiting the script.")
    args = p.parse_args()

    wd = "./"
    cell_indices = False
    file_ext = "png"
    axes_scales = "loglog"
    display = False

    if args.wd:
        wd = args.wd
    if args.cells:
        cell_indices = True
    if args.ext:
        file_ext = args.ext
    if args.scales:
        allowed = ["logx", "logy", "loglog", "linlin"]
        if args.scales not in allowed:
            print("The axes scaling {} is unknown.".format(args.scales))
            print("Allowed values are: logx, logy, loglog, linlin.")
            exit(EXIT_FAIL)
        axes_scales = args.scales
    if args.display:
        display = True

    setup = (
        args.root,
        wd,
        axes_scales,
        cell_indices,
        file_ext,
        display
    )

    return setup


def main() -> Tuple[plt.Figure, plt.Axes]:
    """
    The main function of the script. First, the important wind quantaties are
    plotted. This is then followed by the important ions.

    Returns
    -------
    fig: plt.Figure
        The matplotlib Figure object for the created plot.
    ax: plt.Axes
        The matplotlib Axes objects for the plot panels.
    """

    div_len = 80
    root, wd, axes_scales, cell_indices, file_ext, display = parse_input()

    root = root.replace("/", "")
    wdd = wd
    if wd == "./":
        wdd = ""

    print("-" * div_len)
    print("\nCreating velocity plots for {}{}.pf".format(wdd, root))

    # First, we probably need to run windsave2table

    PythonUtils.windsave2table(root, wd)

    # Now we can plot the stuff

    fig, ax = velocity_plot(root, wd, axes_scales, cell_indices, file_ext)

    if display:
        plt.show()

    print("")
    print("-" * div_len)

    return fig, ax


if __name__ == "__main__":
    fig, ax = main()
