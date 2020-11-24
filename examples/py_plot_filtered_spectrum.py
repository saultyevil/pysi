#!/usr/bin/env python

"""
The purpose of this script is to take a root.delay_dump file in from a Python
simulation and to transform that into a spectrum.
"""


import argparse as ap
import numpy as np
from matplotlib import pyplot as plt

from pyPython import spectrumUtil
from pyPython import filteredSpectrum


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
    sp = p.add_subparsers()

    # Sub-parser for creation of the spectrum

    create_p = sp.add_parser("create",
                             help="Create a filtered spectrum")

    create_p.add_argument("root",
                          help="Root name of the model")

    create_p.add_argument("spec_norm",
                          type=float,
                          help="The normalization constant, usually spec_cycles / current_spec_cycle and > 1.")

    create_p.add_argument("ncores_norm",
                          type=int,
                          help="The number of MPI processed used to create the root.delay_dump file.")

    create_p.add_argument("-dn",
                          "--distance_norm",
                          type=float,
                          default=100,
                          help="The distance normalization in units of parsec.")

    create_p.add_argument("-nj",
                          "--jit",
                          action="store_false",
                          default=True,
                          help="Disable the use of JIT to speed up photon binning.")

    create_p.add_argument("-el",
                          "--extract_line",
                          nargs="+",
                          type=int,
                          default=(filteredSpectrum.UNFILTERED_SPECTRUM,),
                          help="The line number to only extract.")

    create_p.add_argument("-n",
                          "--nbins",
                          type=int,
                          default=10000,
                          help="The number of frequency or wavelength bins for the spectrum.")

    create_p.add_argument("-wd",
                          "--working_directory",
                          default=".",
                          help="The directory containing the simulation.")

    create_p.set_defaults(which="create")

    # Sub-parser for plotting

    plot_p = sp.add_parser("plot",
                           help="Plot the filtered spectrum")

    plot_p.add_argument("root",
                        help="Root name of the model")

    plot_p.add_argument("-sm",
                        "--smooth_amount",
                        type=float,
                        default=5,
                        help="The size of the smoothing window in pixels")

    plot_p.add_argument("-wd",
                        "--working_directory",
                        default=".",
                        help="The directory containing the simulation.")

    plot_p.add_argument("-el",
                        "--extract_line",
                        nargs="+",
                        type=int,
                        default=(filteredSpectrum.UNFILTERED_SPECTRUM,),
                        help="The line number to only extract.")

    plot_p.add_argument("-xl",
                        "--xmin",
                        type=float,
                        default=None,
                        help="The lower x-axis boundary to display.")

    plot_p.add_argument("-xu",
                        "--xmax",
                        type=float,
                        default=None,
                        help="The upper x-axis boundary to display.")

    plot_p.add_argument("-s",
                        "--scales",
                        default="loglog",
                        choices=["logx", "logy", "loglog", "linlin"],
                        help="The axes scaling to use: logx, logy, loglog, linlin.")

    plot_p.add_argument("-l",
                        "--common_lines",
                        action="store_true",
                        default=False,
                        help="Plot labels for important absorption edges.")

    plot_p.add_argument("-f",
                        "--frequency_space",
                        action="store_true",
                        default=False,
                        help="Create the figure in frequency space.")

    plot_p.add_argument("-e",
                        "--ext",
                        default="png",
                        help="The file extension for the output figure.")

    plot_p.add_argument("--display",
                        action="store_true",
                        default=False,
                        help="Display the plot before exiting the script.")

    plot_p.set_defaults(which="plot")

    # Parse the arguments now

    args = p.parse_args()

    mode = args.which
    if mode == "create":
        setup = (
            mode,
            args.root,
            args.spec_norm,
            args.ncores_norm,
            args.distance_norm,
            args.jit,
            args.extract_line,
            args.nbins,
            args.working_directory,
            None,
            None,
            True,
            False,
            "loglog",
            5,
            "png",
            False,
        )
    else:
        setup = (
            mode,
            args.root,
            1,
            1,
            100,
            True,
            args.extract_line,
            10000,
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
    root: str, wd: str, filtered_spectrum: np.ndarray, extract_line: tuple = (-1,), sm: int = 1, d_norm_pc: float = 100,
    xmin: float = None, xmax: float = None, scale: str = "loglog", frequency_space: bool = False, plot_lines: bool = False, file_ext: str = ".png",
    display: bool = False
):
    """
    Plotting function
    """

    try:
        full_spectrum = spectrumUtil.read_spectrum("{}/{}.log_spec".format(wd, root))
        inclinations = spectrumUtil.get_spectrum_inclinations(full_spectrum)
        include_full_spectrum = True
    except IOError:
        include_full_spectrum = False
        full_spectrum = None
        inclinations = np.linspace(1, filtered_spectrum.shape[1] - 1, filtered_spectrum.shape[1] - 1).tolist()

    for e, inc in enumerate(inclinations):

        fig, ax = plt.subplots(figsize=(12, 5))

        if include_full_spectrum:
            ax.plot(
                full_spectrum["Lambda"], spectrumUtil.smooth(full_spectrum[inc], sm), linewidth=1.4, alpha=0.75,
                label="Full Spectrum"
            )

        if frequency_space:
            index = 0
        else:
            index = 1

        ax.plot(
            filtered_spectrum[:-1, index], spectrumUtil.smooth(filtered_spectrum[:-1, e + 2], sm), linewidth=1.4,
            alpha=0.75, label="Filtered Spectrum"
        )

        if scale == "loglog" or scale == "logx":
            ax.set_xscale("log")
        if scale == "loglog" or scale == "logy":
            ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(spectrumUtil.calculate_axis_y_limits(
                filtered_spectrum[:-1, index], spectrumUtil.smooth(filtered_spectrum[:-1, e + 2], sm), xmin, xmax
            )
        )

        ax.legend(loc="lower right", fontsize=15)
        ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=15)
        ax.set_ylabel(
            r"Flux at " + str(d_norm_pc) + r" Pc, F$_{\lambda}$ [ergs s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", fontsize=15
        )

        if plot_lines:
            if scale == "loglog" or scale == "logx":
                logx = True
            else:
                logx = False
            ax = spectrumUtil.plot_line_ids(ax, spectrumUtil.common_lines_list(freq=frequency_space), logx)

        fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

        if extract_line[0] != filteredSpectrum.UNFILTERED_SPECTRUM:
            name = "{}/{}_line".format(wd, root)
            for line in extract_line:
                name += "_{}".format(line)
            name += "_i{}_delay_dump_spec".format(inc)
        else:
            name = "{}/{}_i{}_delay_dump_spec".format(wd, root, inc)

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

    # Get the run time variables to set up the script... there are a lot of
    # variables required and this is my fault :-)

    if setup:
        mode, root, spec_norm, ncores_norm, distance_norm, jit, extract_nres, nbins, wd, xmin, xmax, frequency_space, \
            common_lines, axes_scales, smooth_amount, file_ext, display = setup
    else:
        mode, root, spec_norm, ncores_norm, distance_norm, jit, extract_nres, nbins, wd, xmin, xmax, frequency_space, \
            common_lines, axes_scales, smooth_amount, file_ext, display = setup_script()

    # Now we either create, or plot the filtered spectrum if it has already been created

    # TODO: determine the run mode first, i.e. create or plot
    # TODO: force re-creation of spectrum

    extract_nres = tuple(extract_nres)

    if mode == "create":
        filteredSpectrum.create_filtered_spectrum(
            root, wd, extract_nres, xmin, xmax, nbins, distance_norm, spec_norm, ncores_norm, True, jit
        )
    else:
        name = ""
        try:
            if extract_nres[0] != filteredSpectrum.UNFILTERED_SPECTRUM:
                name = "{}/{}_line".format(wd, root)
                for line in extract_nres:
                    name += "_{}".format(line)
                name += ".delay_dump.spec"
            else:
                name = "{}/{}.delay_dump.spec".format(wd, root)
            print(name)
            filtered_spectrum = np.loadtxt(name, skiprows=2)  # TODO: could be replaced by something in pyPython?
        except IOError:
            print("Unable to load filtered spectrum", name, "to plot anything")
            return
        plot(
            root, wd, filtered_spectrum, extract_nres, smooth_amount, distance_norm, xmin, xmax, axes_scales,
            frequency_space, common_lines, file_ext, display
        )

    return


if __name__ == "__main__":
    main()
