#!/usr/bin/env python

"""
The purpose of this script is to take a root.delay_dump file in from a Python
simulation and to transform that into a spectrum.
"""

import argparse as ap
import numpy as np
from matplotlib import pyplot as plt
from pypython import plotutil
from pypython import util
from pypython.spectrum import Spectrum
from pypython import spectrumcreate
from pypython import conversion

plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.labelsize'] = 15


def setup_script() -> tuple:
    """Parse the different modes this script can be run from the command line.
    todo: clean up this sub-parser mess. Probably don't need to be able to plot it in this script...

    Returns
    -------
    setup: tuple
        A list containing all of the different setup of parameters for
        plotting."""

    p = ap.ArgumentParser(description=__doc__)
    sp = p.add_subparsers()

    # Sub-parser for creation of the spectrum

    create_p = sp.add_parser(
        "create", help="Create a filtered spectrum"
    )
    create_p.add_argument(
        "root", help="Root name of the model"
    )
    create_p.add_argument(
        "spec_norm", type=float, help="The normalization constant, usually spec_cycles / current_spec_cycle and > 1."
    )
    create_p.add_argument(
        "ncores_norm", type=int, help="The number of MPI processed used to create the root.delay_dump file."
    )
    create_p.add_argument(
        "-dn", "--distance_norm", type=float, default=100, help="The distance normalization in units of parsec."
    )
    create_p.add_argument(
        "-xl", "--xmin", type=float, default=None, help="The lower x-axis boundary to display."
    )
    create_p.add_argument(
        "-xu", "--xmax", type=float, default=None, help="The upper x-axis boundary to display."
    )
    create_p.add_argument(
        "-el", "--extract_line", nargs="+", type=int, default=(spectrumcreate.UNFILTERED_SPECTRUM,),
        help="The line number to only extract."
    )
    create_p.add_argument(
        "-n", "--nbins", type=int, default=10000, help="The number of frequency or wavelength bins for the spectrum."
    )
    create_p.add_argument(
        "-lb", "--logbins", action="store_true", default=False,
        help="Create the spectrum using log scaling for the wavelength/frequency bins."
    )
    create_p.add_argument(
        "-wd", "--working_directory", default=".", help="The directory containing the simulation."
    )
    create_p.set_defaults(which="create")

    # Sub-parser for plotting

    plot_p = sp.add_parser(
        "plot", help="Plot the filtered spectrum"
    )
    plot_p.add_argument(
        "root", help="Root name of the model"
    )
    plot_p.add_argument(
        "-sm", "--smooth_amount", type=float, default=5, help="The size of the smoothing window in pixels"
    )
    plot_p.add_argument(
        "-wd", "--working_directory", default=".", help="The directory containing the simulation."
    )
    plot_p.add_argument(
        "-el", "--extract_line", nargs="+", type=int, default=(spectrumcreate.UNFILTERED_SPECTRUM,),
        help="The line number to only extract."
    )
    plot_p.add_argument(
        "-xl", "--xmin", type=float, default=None, help="The lower x-axis boundary to display."
    )
    plot_p.add_argument(
        "-xu", "--xmax", type=float, default=None, help="The upper x-axis boundary to display."
    )
    plot_p.add_argument(
        "-s", "--scales", default="loglog", choices=["logx", "logy", "loglog", "linlin"],
        help="The axes scaling to use: logx, logy, loglog, linlin."
    )
    plot_p.add_argument(
        "-l", "--common_lines", action="store_true", default=False,
                        help="Plot labels for important absorption edges.")

    plot_p.add_argument(
        "-f", "--frequency_space", action="store_true", default=False, help="Create the figure in frequency space."
    )
    plot_p.add_argument(
        "-lb", "--logbins", action="store_true", default=False,
        help="Create the figure using log scaling for the wavelength/frequency bins."
    )
    plot_p.add_argument(
        "-e", "--ext", default="png", help="The file extension for the output figure."
    )
    plot_p.add_argument(
        "--display", action="store_true", default=False, help="Display the plot before exiting the script."
    )
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
            args.extract_line,
            args.nbins,
            args.working_directory,
            args.xmin,
            args.xmax,
            True,
            False,
            "loglog",
            5,
            "png",
            args.logbins,
            False,
        )
    else:
        setup = (
            mode,
            args.root,
            1,
            1,
            100,
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
            args.logbins,
            args.display
        )

    return setup


def plot(
    root: str, wd: str, filtered_spectrum: np.ndarray, extract_line: tuple = (-1,), sm: int = 1, d_norm_pc: float = 100,
    xmin: float = None, xmax: float = None, scale: str = "loglog", frequency_space: bool = False, logbins: bool = True,
    plot_lines: bool = False, file_ext: str = ".png", display: bool = False
):
    """Description of the function.
    todo: create a new script for this

    Parameters
    ----------
    root
    wd
    filtered_spectrum
    extract_line
    sm
    d_norm_pc
    xmin
    xmax
    scale
    frequency_space
    logbins
    plot_lines
    file_ext
    display"""

    try:
        if logbins:
            full_spectrum = Spectrum(root, wd)
        else:
            full_spectrum = Spectrum(root, wd)
        full_spectrum.smooth(sm)
        inclinations = full_spectrum.inclinations
        include_full_spectrum = True
    except IOError:
        include_full_spectrum = False
        full_spectrum = None
        inclinations = np.linspace(1, filtered_spectrum.shape[1] - 1, filtered_spectrum.shape[1] - 1).tolist()

    for i, inclination in enumerate(inclinations):

        fig, ax = plt.subplots(figsize=(12, 5))

        if include_full_spectrum:
            ax.plot(
                full_spectrum["Lambda"], full_spectrum[inclination], linewidth=1.4, alpha=0.75, label="Full Spectrum"
            )

        if frequency_space:
            index = 0
        else:
            index = 1

        ax.plot(
            filtered_spectrum[:-1, index], util.smooth_array(filtered_spectrum[:-1, i + 2], sm), linewidth=1.4,
            alpha=0.75, label="Filtered Spectrum"
        )

        if scale == "loglog" or scale == "logx":
            ax.set_xscale("log")
        if scale == "loglog" or scale == "logy":
            ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(plotutil.get_y_lims_for_x_lims(
                filtered_spectrum[:-1, index], util.smooth_array(filtered_spectrum[:-1, i + 2], sm), xmin, xmax
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
            ax = plotutil.ax_add_line_ids(ax, plotutil.common_lines(freq=frequency_space), logx)

        fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

        if extract_line[0] != spectrumcreate.UNFILTERED_SPECTRUM:
            name = "{}/{}_line".format(wd, root)
            for line in extract_line:
                name += "_{}".format(line)
            name += "_i{}_delay_dump_spec".format(inclination)
        else:
            name = "{}/{}_i{}_delay_dump_spec".format(wd, root, inclination)

        fig.savefig("{}.{}".format(name, file_ext), dpi=300)
        if file_ext != "png":  # Save both pdf and png versions
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
        mode, root, spec_cycle_norm, n_cores_norm, d_norm_pc, extract_nres, n_bins, wd, w_xmin, w_xmax, \
            frequency_space, common_lines, axes_scales, smooth_amount, file_ext, logbins, display = setup
    else:
        mode, root, spec_cycle_norm, n_cores_norm, d_norm_pc, extract_nres, n_bins, wd, w_xmin, w_xmax, \
            frequency_space, common_lines, axes_scales, smooth_amount, file_ext, logbins, display = setup_script()

    # Now we either create, or plot the filtered spectrum if it has already been created

    extract_nres = tuple(extract_nres)

    xmin = w_xmin
    xmax = w_xmax
    if w_xmax and frequency_space:
        xmin = conversion.angstrom_to_hz(w_xmax)
    if w_xmin and frequency_space:
        xmax = conversion.angstrom_to_hz(w_xmin)

    if mode == "create":
        spectrumcreate.create_spectrum(
            root, wd, extract_nres, freq_min=xmin, freq_max=xmax, n_bins=n_bins, d_norm_pc=d_norm_pc,
            spec_cycle_norm=spec_cycle_norm, n_cores_norm=n_cores_norm
        )
    else:
        if extract_nres[0] != spectrumcreate.UNFILTERED_SPECTRUM:
            name = "{}/{}_line".format(wd, root)
            for line in extract_nres:
                name += "_{}".format(line)
            name += ".delay_dump.spec"
        else:
            name = "{}/{}.delay_dump.spec".format(wd, root)
        # todo: see if this can be read in using a Spectrum object instead as an ascii
        filtered_spectrum = np.loadtxt(name, skiprows=2)
        plot(
            root, wd, filtered_spectrum, extract_nres, smooth_amount, d_norm_pc, w_xmin, w_xmax, axes_scales,
            frequency_space, True, common_lines, file_ext, display
        )

    return


if __name__ == "__main__":
    main()
