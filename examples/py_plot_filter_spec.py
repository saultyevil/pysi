#!/usr/bin/env python

"""
The purpose of this script is to take a root.delay_dump file in from a Python
simulation and to transform that into a spectrum.
"""


import argparse as ap
import numpy as np
from matplotlib import pyplot as plt

from PyPython import SpectrumUtils
from PyPython.Constants import C
from PyPython import FilteredSpectrum


plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.labelsize'] = 15


def setup_script() -> tuple:
    """
    Parse the different modes this script can be run from the command line.

    Returns
    -------
    setup: tuple
        A list containing all of the different setup of parameters for plotting.
    """

    p = ap.ArgumentParser(description=__doc__)

    # Required arguments
    p.add_argument("mode",
                   choices=["create", "plot"],
                   help="The mode of the script: allowed values are 'create' or 'plot'.")

    p.add_argument("root",
                   help="The root name of the simulation.")

    p.add_argument("spec_norm",
                   type=float,
                   help="The normalization constant, usually spec_cycles / current_spec_cycle.")

    p.add_argument("ncores_norm",
                   type=int,
                   help="The number of MPI processes used to create the root.delay_dump file.")

    # Supplementary arguments
    p.add_argument("-dn",
                   "--distance_norm",
                   type=float,
                   default=100,
                   help="The distance normalization in units of parsec.")

    p.add_argument("-nj",
                   "--jit",
                   action="store_false",
                   default=True,
                   help="Disable the use of JIT to speed up photon binning.")

    p.add_argument("-el",
                   "--extract_line",
                   type=int,
                   default=-1,
                   help="The line number to only extract.")

    p.add_argument("-n",
                   "--nbins",
                   type=int,
                   default=10000,
                   help="The number of frequency or wavelength bins for the spectrum.")

    p.add_argument("-wd",
                   "--working_directory",
                   default=".",
                   help="The directory containing the simulation.")

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
        args.mode,
        args.root,
        args.spec_norm,
        args.ncores_norm,
        args.distance_norm,
        args.jit,
        args.extract_line,
        args.nbins,
        args.working_directory,
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
    root: str, wd: str, filtered_spectrum: np.ndarray, extract_line: int, sm: int = 1, dnorm: float = 100,
    scale: str = "loglog", frequency_space: bool = False, plot_lines: bool = False, file_ext: str = ".png",
    display: bool = False
):
    """
    Plotting function
    """

    try:
        plot_full = True
        full_spectrum = SpectrumUtils.read_spec("{}/{}.log_spec".format(wd, root))
        inclinations = SpectrumUtils.spec_inclinations(full_spectrum)
    except IOError:
        plot_full = False
        inclinations = np.linspace(1, filtered_spectrum.shape[1] - 1, filtered_spectrum.shape[1] - 1).tolist()

    for e, inc in enumerate(inclinations):

        fig, ax = plt.subplots(figsize=(12, 5))

        if plot_full:
            ax.plot(full_spectrum["Lambda"], SpectrumUtils.smooth(full_spectrum[inc], sm), linewidth=1.4, alpha=0.75,
                    label="Full Spectrum")

        ax.plot(C * 1e8 / filtered_spectrum[:-1, 0], SpectrumUtils.smooth(filtered_spectrum[:-1, e + 1], sm),
                linewidth=1.4, alpha=0.75, label="Filtered Spectrum")

        ax.legend(fontsize=15)
        ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=15)
        ax.set_ylabel(r"Flux at " + str(dnorm) + r" Pc [ergs s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=15)

        if plot_lines:
            if scale == "loglog" or scale == "logx":
                logx = True
            else:
                logx = False
            ax = SpectrumUtils.plot_line_ids(ax, SpectrumUtils.common_lines(), logx)

        if scale == "loglog" or scale == "logx":
            ax.set_xscale("log")
        if scale == "loglog" or scale == "logy":
            ax.set_yscale("log")

        fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
        if extract_line > -1:
            name = "{}/{}_i{}_line{}.delay_dump_spectrum".format(wd, root, inc, extract_line)
        else:
            name = "{}/{}_i{}.delay_dump_spectrum".format(wd, root, inc)
        fig.savefig("{}.{}".format(name, file_ext), dpi=300)
        if file_ext == "pdf":  # Save both pdf and png versions
            fig.savefig("{}.png".format(name), dpi=300)

    if display:
        plt.show()

    return


def main(setup: tuple = None):
    """
    Main function of the script.

    Parameters
    ----------
    setup: tuple
        A tuple containing all of the setup parameters for the script.

        setup = (
            mode,
            root,
            spec_norm,
            ncores_norm,
            distance_norm,
            jit,
            extract_line,
            nbins,
            wd,
            xmin,
            xmax,
            frequency_space,
            common_lines,
            scales,
            smooth_amount,
            ext,
            display
        )

    """

    if setup:
        mode, root, spec_norm, ncores_norm, distance_norm, jit, extract_line, nbins, wd, xmin, xmax, frequency_space, \
            common_lines, axes_scales, smooth_amount, file_ext, display = setup
    else:
        mode, root, spec_norm, ncores_norm, distance_norm, jit, extract_line, nbins, wd, xmin, xmax, frequency_space, \
            common_lines, axes_scales, smooth_amount, file_ext, display = setup_script()

    if mode == "create":
        filtered_spectrum = FilteredSpectrum.create_spectra_from_delay_dump(
            root, wd, extract_line, xmin, xmax, nbins, distance_norm, spec_norm, ncores_norm, True, jit
        )
        plot(
            root, wd, filtered_spectrum, extract_line, smooth_amount, distance_norm, axes_scales, frequency_space, common_lines,
            file_ext, display
        )
        return
    elif mode == "plot":
        try:
            # TODO can this be replaced by SpectrumUtils.read_spec?
            if extract_line > -1:
                iname = "{}/{}_line{}.delay_dump.spec".format(wd, root, extract_line)
            else:
                iname = "{}/{}.delay_dump.spec".format(wd, root)
            filtered_spectrum = np.loadtxt(iname)
        except IOError:
            print("Unable to load filtered spectrum")
            return
        plot(
            root, wd, filtered_spectrum, extract_line, smooth_amount, distance_norm, axes_scales, frequency_space, common_lines,
            file_ext, display
        )

    return


if __name__ == "__main__":
    main()
