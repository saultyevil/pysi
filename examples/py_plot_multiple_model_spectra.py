#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The purpose of this script is to easily create a figure which overplots multiple
spectra onto a single plot. Spectrum files are recursively searched from the
calling directory.
"""

import argparse as ap
from typing import Tuple
from matplotlib import pyplot as plt

from pyPython import spectrumUtil
from pyPython import pythonUtil


def setup_script():
    """
    Parse the different modes this script can be run from the command line.

    Returns
    -------
    setup: tuple
        A list containing all of the different setup of parameters for plotting.
    """

    p = ap.ArgumentParser(description=__doc__)

    # Required arguments
    p.add_argument("name",
                   type=str,
                   help="The output name of the comparison plot.")

    # Supplementary arguments
    p.add_argument("-wd",
                   "--working_directory",
                   default=".",
                   help="The directory containing the simulation.")

    p.add_argument("-i",
                   "--inclination",
                   default="all",
                   help="The inclination angles")

    p.add_argument("-r",
                   "--root",
                   default=None,
                   help="Only plots models which have the provided root name.")

    p.add_argument("-xl",
                   "--xmin",
                   type=float,
                   default=None,
                   help="The lower x-axis boundary to display.")

    p.add_argument("-xu",
                   "--xmax",
                   type=float,
                   default=None,
                   help="The upper x-axis boundary to display.")

    p.add_argument("-s",
                   "--scales",
                   default="loglog",
                   choices=["logx", "logy", "loglog", "linlin"],
                   help="The axes scaling to use: logx, logy, loglog, linlin.")

    p.add_argument("-l",
                   "--common_lines",
                   action="store_true",
                   default=False,
                   help="Plot labels for important absorption edges.")

    p.add_argument("-f",
                   "--frequency_space",
                   action="store_true",
                   default=False,
                   help="Create the figure in frequency space.")

    p.add_argument("-sm",
                   "--smooth_amount",
                   type=int,
                   default=5,
                   help="The size of the boxcar smoothing filter.")

    p.add_argument("-e",
                   "--ext",
                   default="png",
                   help="The file extension for the output figure.")

    p.add_argument("--display",
                   action="store_true",
                   default=False,
                   help="Display the plot before exiting the script.")

    args = p.parse_args()

    setup = (
        args.name,
        args.working_directory,
        args.inclination,
        args.root,
        args.xmin,
        args.xmax,
        args.frequency_space,
        args.common_lines,
        args.scales,
        args.smooth_amount,
        args.ext,
        args.display
    )

    return setup


def plot(
    spectra: list, output_name: str, inclination: str, wd: str = ".", xmin: float = None, xmax: float = None,
    frequency_space: bool = False, axes_scales: str = "logy", smooth_amount: int = 5, plot_common_lines: bool = False,
    file_ext: str = "png", display: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plotting function.
    TODO put this function into SpectrumPlot

    When plotting in frequency space, the y axis will be lambda Flambda = nu Fnu
    instead of regular ol' Flux.
    """

    alpha = 0.75

    if inclination == "all":
        inclinations = []
        for s in spectra:
            inclinations += spectrumUtil.get_spectrum_inclinations(s)
        inclinations = sorted(list(dict.fromkeys(inclinations)))  # Removes duplicate values
        figure_size = (12, 12)
    else:
        inclinations = [inclination]
        figure_size = (12, 5)

    ninc = len(inclinations)
    nrows, ncols = pythonUtil.subplot_dims(ninc)

    fig, ax = plt.subplots(nrows, ncols, figsize=figure_size, squeeze=False)
    ax = ax.flatten()  # Allows looping over 1 dimension of plt.Axes instead

    # TODO put into a separate function in PythonUtils or something - delete_extra_axes
    if nrows * ncols > ninc:
        for i in range(ninc, nrows * ncols):
            fig.delaxes(ax[i])

    ymin = +1e99
    ymax = -1e99

    # Loop over each inclination in the spectrum
    for i, inc in enumerate(inclinations):

        # Plot each spectrum for each subplot
        for f in spectra:
            if f.find("continuum") != -1:
                continue
            s = spectrumUtil.read_spectrum(f)
            if frequency_space:
                x = s["Freq."].values
            else:
                x = s["Lambda"].values

            try:
                if frequency_space:
                    y = s["Lambda"].values * s[inc].values
                else:
                    y = s[inc].values
                y = spectrumUtil.smooth(y, smooth_amount)
            except KeyError:
                continue

            ax[i].plot(x, y, label=f, alpha=alpha)

            # An attempt to try to keep the y-scale correct when the x range is
            # limited
            if not xmin:
                xmin = x.min()
            if not xmax:
                xmax = x.max()

            t_ymin, t_ymax = spectrumUtil.calculate_axis_y_limits(x, y, xmin, xmax)
            if t_ymin < ymin:
                ymin = t_ymin
            if t_ymax > ymax:
                ymax = t_ymax

        if ymin == +1e99:
            ymin = None
        if ymax == -1e99:
            ymax = None

        ax[i].set_ylim(ymin, ymax)

        # Format things now

        ax[i].set_title(r"$i$ " + "= {}".format(inclinations[i]) + r"$^{\circ}$")

        if axes_scales == "loglog" or axes_scales == "logx":
            ax[i].set_xscale("log")
        if axes_scales == "loglog" or axes_scales == "logy":
            ax[i].set_yscale("log")

        if frequency_space:
            ax[i].set_xlabel(r"Frequency [Hz]")
            ax[i].set_ylabel(r"$\nu F_{\nu}$ (erg s$^{-1}$ cm$^{-2}$")
        else:
            ax[i].set_xlabel(r"Wavelength [$\AA$]")
            ax[i].set_ylabel(r"$F_{\lambda}$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)")

        lims = list(ax[i].get_xlim())
        if not xmin:
            xmin = lims[0]
        if not xmax:
            xmax = lims[1]
        ax[i].set_xlim(xmin, xmax)

        if plot_common_lines:
            if axes_scales == "logx" or axes_scales == "loglog":
                logx = True
            else:
                logx = False
            ax[i] = spectrumUtil.plot_line_ids(ax[i], spectrumUtil.common_lines_list(), logx)

    ax[0].legend(loc="lower left")

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    if inclination != "all":
        name = "{}/{}_i{}".format(wd, output_name, inclination)
    else:
        name = "{}/{}".format(wd, output_name)
    fig.savefig("{}.{}".format(name, file_ext))
    if file_ext == "pdf" or file_ext == "eps":
        fig.savefig("{}.png".format(name))

    if display:
        plt.show()

    return fig, ax


def main(setup: tuple = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    The main function of the script.

    Parameters
    ----------
    setup: tuple
        A tuple containing the setup parameters to run the script. If this
        isn't provided, then the script will parse them from the command line.
    """

    if setup:
        output_name, wd, inclination, root, xmin, xmax, frequency_space, common_lines, axes_scales, smooth_amount, file_ext, display = setup
    else:
        output_name, wd, inclination, root, xmin, xmax, frequency_space, common_lines, axes_scales, smooth_amount, file_ext, display = \
            setup_script()

    spectra = spectrumUtil.find_spec_files(root)
    if len(spectra) == 0:
        print("Unable to find any spectrum files")
        return
    if root:
        spectratemp = []
        for s in spectra:
            if s.find("{}.spec".format(root)) != -1:
                spectratemp.append(s)
        spectra = spectratemp

    fig, ax = plot(
        spectra, output_name, inclination, wd, xmin, xmax, frequency_space, axes_scales, smooth_amount, common_lines,
        file_ext, display
    )

    return fig, ax


if __name__ == "__main__":
    main()
