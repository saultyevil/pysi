#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from pypython import (SPECTRUM_UNITS_FLM, SPECTRUM_UNITS_FNU, SPECTRUM_UNITS_LNU, Spectrum, get_root, smooth_array)
from pypython.constants import PARSEC
from pypython.error import InvalidParameter
from pypython.plot import (ax_add_line_ids, common_lines, get_y_lims_for_x_lims, normalize_figure_style,
                           photoionization_edges, remove_extra_axes, subplot_dims)

MIN_SPEC_COMP_FLUX = 1e-15
DEFAULT_PYTHON_DISTANCE = 100 * PARSEC


def _plot_panel_subplot(ax, x_values, spectrum, units, things_to_plot, x_limits, sm, alpha, scale, frequency_space,
                        skip_sparse):
    """Create a subplot panel for a figure given the spectrum components names
    in the list dname.
    todo: switch x_limits to xmin and xmax inputs

    Parameters
    ----------
    ax: plt.Axes
        The plt.Axes object for the subplot
    x_values: np.array[float]
        The x-axis data, i.e. wavelength or frequency
    spectrum: pd.DataFrame
        The spectrum data file as a pandas DataFrame
    units: str
        The units of the spectrum
    things_to_plot: list[str]
        The name of the spectrum components to add to the subplot panel
    x_limits: Tuple[float, float]
        The lower and upper x-axis boundaries (xlower, xupper)
    sm: int
        The size of the boxcar filter to smooth the spectrum components
    alpha: float
        The alpha value of the spectrum to be plotted.
    scale: bool
        Set the scale for the plot axes
    frequency_space: bool
        Create the figure in frequency space instead of wavelength space
    skip_sparse: bool
        If True, then sparse spectra will not be plotted
    n: str
        The name of the calling function

    Returns
    -------
    ax: pyplot.Axes
        The pyplot.Axes object for the subplot"""

    if type(things_to_plot) == str:
        things_to_plot = [things_to_plot]

    n_skip = 0

    for thing in things_to_plot:
        try:
            fl = smooth_array(spectrum[thing], sm)
        except KeyError:
            print("unable to find data column with label {}".format(thing))
            continue

        # Skip sparse spec components to make prettier plot

        if skip_sparse and len(fl[fl < MIN_SPEC_COMP_FLUX]) > 0.7 * len(fl):
            n_skip += 1
            continue

        # If plotting in frequency space, of if the units then the flux needs
        # to be converted in nu F nu

        if frequency_space and units == SPECTRUM_UNITS_FLM:
            fl *= spectrum["Lambda"]
        elif frequency_space and units == SPECTRUM_UNITS_FNU:
            fl *= spectrum["Freq."]

        # If the spectrum units are Lnu then plot nu Lnu

        if units == SPECTRUM_UNITS_LNU:
            fl *= spectrum["Freq."]

        ax.plot(x_values, fl, label=thing, alpha=alpha)

        if scale == "logx" or scale == "loglog":
            ax.set_xscale("log")
        if scale == "logy" or scale == "loglog":
            ax.set_yscale("log")

    if n_skip == len(things_to_plot):
        return ax

    ax.set_xlim(x_limits[0], x_limits[1])
    if frequency_space:
        ax.set_xlabel(r"Frequency (Hz)")
        if units == SPECTRUM_UNITS_LNU:
            ax.set_ylabel(r"$\nu L_{\nu}$ (erg s$^{-1}$ Hz$^{-1})$")
        else:
            ax.set_ylabel(r"$\nu F_{\nu}$ (erg s$^{-1}$ cm$^{-2})$")
    else:
        ax.set_xlabel(r"Wavelength ($\AA$)")
        ax.set_ylabel(r"$F_{\lambda}$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)")

    ax.legend(loc="lower left")

    return ax


def plot(x,
         y,
         xmin=None,
         xmax=None,
         xlabel=None,
         ylabel=None,
         scale="logy",
         fig=None,
         ax=None,
         label=None,
         alpha=1.0,
         display=False):
    """This is a simple plotting function designed to give you the bare
    minimum. It will create a figure and axes object for a single panel and
    that is it. It is mostly designed for quick plotting of models and real
    data.

    Parameters
    ----------
    x: np.ndarray
        The wavelength or x-axis data to plot.
    y: np.ndarray
        The flux or y-axis data to plot.
    xmin: float [optional]
        The smallest number to display on the x-axis
    xmax: float [optional]
        The largest number to display on the x-axis
    xlabel: str [optional]
        The data label for the x-axis.
    ylabel: str [optional]
        The data label for the y-axis.
    scale: str [optional]
        The scale of the axes for the plot.
    fig: plt.Figure [optional]
        A matplotlib Figure object of which to use to create the plot.
    ax: plt.Axes [optional]
        A matplotlib Axes object of which to use to create the plot.
    label: str [optional]
        A label for the data being plotted.
    alpha: float [optional]
        The alpha value for the data to be plotted.
    display: bool [optional]
        If set to True, then the plot will be displayed.

    Returns
    -------
    fig: plt.Figure
        The figure object for the plot.
    ax: plt.Axes
        The axes object containing the plot.
    """

    # It doesn't make sense to provide only fig and not ax, or ax and not fig
    # so at this point we will throw an error message and return

    normalize_figure_style()

    if fig and not ax:
        raise InvalidParameter("fig has been provided, but ax has not. Both are required.")
    if not fig and ax:
        raise InvalidParameter("fig has not been provided, but ax has. Both are required.")
    if not fig and not ax:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    if label is None:
        label = ""

    ax.plot(x, y, label=label, alpha=alpha)

    # Set the scales of the aes

    if scale == "loglog" or scale == "logx":
        ax.set_xscale("log")
    if scale == "loglog" or scale == "logy":
        ax.set_yscale("log")

    # If axis labels are provided, then set them

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Set the x and y axis limits. For the y axis, we use a function to try and
    # figure out appropriate values for the axis limits to display the data
    # sensibly

    xlims = [x.min(), x.max()]
    if not xmin:
        xmin = xlims[0]
    if not xmax:
        xmax = xlims[1]
    xlims = (xmin, xmax)
    ax.set_xlim(xlims[0], xlims[1])

    ymin, ymax = get_y_lims_for_x_lims(x, y, xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_optical_depth(root,
                       wd,
                       inclinations="all",
                       xmin=None,
                       xmax=None,
                       scale="loglog",
                       show_absorption_edge_labels=True,
                       frequency_space=True,
                       display=False):
    """Create an optical depth spectrum for a given Python models. This figure
    can be created in both wavelength or frequency space and with various
    choices of axes scaling.

    This function will return the Figure and Axes object used to create the
    plot.

    Parameters
    ----------
    root: str
        The root name of the Python models
    wd: str
        The absolute or relative path containing the Python models
    inclinations: List[str] [optional]
        A list of inclination angles to plot
    xmin: float [optional]
        The lower x boundary for the figure
    xmax: float [optional]
        The upper x boundary for the figure
    scale: str [optional]
        The scale of the axes for the plot.
    show_absorption_edge_labels: bool [optional]
        Label common absorption edges of interest onto the figure
    frequency_space: bool [optional]
        Create the figure in frequency space instead of wavelength space
    display: bool [optional]
        Display the final plot if True.

    Returns
    -------
    fig: pyplot.Figure
        The pyplot.Figure object for the created figure
    ax: pyplot.Axes
        The pyplot.Axes object for the created figure
    """

    normalize_figure_style()

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    if type(inclinations) == str:
        inclinations = [inclinations]
    spectrum = Spectrum(root, wd, "spec_tau")
    spec_angles = spectrum.inclinations
    n_angles = len(spec_angles)
    n_plots = len(inclinations)  # Really have no clue what this does in hindsight...

    # Ignore all if other inclinations are passed - assume it was a mistake to pass all

    if inclinations[0] == "all" and len(inclinations) > 1:
        inclinations = inclinations[1:]
        n_plots = len(inclinations)
    if inclinations[0] == "all":
        inclinations = spec_angles
        n_plots = n_angles

    # Set wavelength or frequency boundaries

    xlabel = "Lambda"
    if frequency_space:
        xlabel = "Freq."

    x = spectrum[xlabel]
    if not xmin:
        xmin = np.min(spectrum[xlabel])
    if not xmax:
        xmax = np.max(spectrum[xlabel])

    # This loop will plot the inclinations provided by the user

    for i in range(n_plots):
        if inclinations[0] != "all" and inclinations[i] not in spec_angles:  # Skip inclinations which don't exist
            continue
        ii = str(inclinations[i])
        label = ii + r"$^{\circ}$"
        n_non_zero = np.count_nonzero(spectrum[ii])

        # Skip inclinations which look through vacuum

        if n_non_zero == 0:
            continue

        ax.plot(x, spectrum[ii], linewidth=2, label=label)
        if scale == "logx" or scale == "loglog":
            ax.set_xscale("log")
        if scale == "logy" or scale == "loglog":
            ax.set_yscale("log")

    ax.set_ylabel(r"Optical Depth, $\tau$")
    if frequency_space:
        ax.set_xlabel(r"Frequency, [Hz]")
    else:
        ax.set_xlabel(r"Wavelength, [$\AA$]")
    ax.set_xlim(xmin, xmax)
    ax.legend(loc="lower left")

    if show_absorption_edge_labels:
        if scale == "loglog" or scale == "logx":
            logx = True
        else:
            logx = False
        ax_add_line_ids(ax, photoionization_edges(frequency_space), logx=logx)

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_spectrum_physics_process_contributions(contribution_spectra,
                                                inclination,
                                                root,
                                                wd=".",
                                                xmin=None,
                                                xmax=None,
                                                ymin=None,
                                                ymax=None,
                                                scale="logy",
                                                line_labels=True,
                                                sm=5,
                                                lw=2,
                                                alpha=0.75,
                                                file_ext="png",
                                                display=False):
    """Description of the function.
    todo: some of these things really need re-naming..... it seems very confusing

    Parameters
    ----------

    Returns
    -------
    fig: plt.Figure
        The plt.Figure object for the created figure
    ax: plt.Axes
        The plt.Axes object for the created figure"""

    normalize_figure_style()

    fig, ax = plt.subplots(figsize=(12, 8))

    for name, spectrum in contribution_spectra.items():
        ax.plot(spectrum["Lambda"], smooth_array(spectrum[inclination], sm), label=name, linewidth=lw, alpha=alpha)

    if scale == "logx" or scale == "loglog":
        ax.set_xscale("log")
    if scale == "logy" or scale == "loglog":
        ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(loc="upper center", ncol=len(contribution_spectra))
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel(r"Flux F$_{\lambda}$ [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")

    if line_labels:
        if scale == "logx" or scale == "loglog":
            logx = True
        else:
            logx = False
        ax = ax_add_line_ids(ax, common_lines(), logx=logx)

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig("{}/{}_spec_processes.{}".format(wd, root, file_ext), dpi=300)
    if file_ext != "png":
        fig.savefig("{}/{}_spec_processes.png".format(wd, root), dpi=300)

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_spectrum_components(root,
                             wd,
                             spec_tot=False,
                             wind_tot=False,
                             xmin=None,
                             xmax=None,
                             smooth_amount=5,
                             scale="loglog",
                             alpha=0.6,
                             frequency_space=False,
                             display=False):
    """Create a figure of the different spectrum components of a Python
    spectrum file. Note that all of the spectrum components added together DO
    NOT have to equal the output spectrum or the emitted spectrum (don't ask).

    Parameters
    ----------
    root: str
        The root name of the Python models
    wd: str
        The absolute or relative path containing the Python models
    spec_tot: bool [optional]
        If True, the root.log_spec_tot file will be plotted
    wind_tot: bool [optional]
        If True, the root.log_wind_tot file will be plotted
    xmin: float [optional]
        The lower x boundary for the figure
    xmax: float [optional]
        The upper x boundary for the figure
    smooth_amount: int [optional]
        The size of the boxcar filter to smooth the spectrum components
    scale: bool [optional]
        The scale to use for the axes. Allowed values are linlin, logx, logy and
        loglog.
    alpha: float [optional]
        The alpha value used for plotting the spectra.
    frequency_space: bool [optional]
        Create the figure in frequency space instead of wavelength space
    display: bool [optional]
        Display the final plot if True.

    Returns
    -------
    fig: pyplot.Figure
        The pyplot.Figure object for the created figure
    ax: pyplot.Axes
        The pyplot.Axes object for the created figure
    """

    normalize_figure_style()

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    # Determine the type of spectrum to read in

    if spec_tot:
        scale = "loglog"
        frequency_space = True
        logspec = True
        spectype = "spec_tot"
    elif wind_tot:
        scale = "loglog"
        frequency_space = True
        logspec = True
        spectype = "spec_tot_wind"
    else:
        spectype = None
        logspec = False

    spectrum = Spectrum(root, wd, spectype, logspec)
    if frequency_space:
        x = spectrum["Freq."]
    else:
        x = spectrum["Lambda"]
    xlims = (None, None)

    ax[0] = _plot_panel_subplot(ax[0], x, spectrum, spectrum.units, ["Created", "WCreated", "Emitted"], xlims,
                                smooth_amount, alpha, scale, frequency_space, True)
    ax[1] = _plot_panel_subplot(ax[1], x, spectrum, spectrum.units, ["CenSrc", "Disk", "Wind", "HitSurf"], xlims,
                                smooth_amount, alpha, scale, frequency_space, True)

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_spectrum_inclinations_in_subpanels(root,
                                            fp,
                                            xmin=None,
                                            xmax=None,
                                            smooth_amount=5,
                                            add_line_ids=True,
                                            frequency_space=False,
                                            scale="logy",
                                            figsize=None,
                                            display=False):
    """Creates a figure which plots all of the different inclination angles in
    different panels.

    Parameters
    ----------
    root: str
        The root name of the Python models
    fp: str
        The absolute or relative path containing the Python models
    xmin: float [optional]
        The lower x boundary for the figure
    xmax: float [optional]
        The upper x boundary for the figure
    smooth_amount: int [optional]
        The size of the boxcar filter to smooth the spectrum components.
    add_line_ids: bool [optional]
        Plot labels for common line transitions.
    frequency_space: bool [optional]
        Create the figure in frequency space instead of wavelength space
    scale: bool [optional]
        Set the scales for the axes in the plot
    figsize: Tuple[float, float] [optional]
        The size of the Figure in matplotlib units (inches?)
    display: bool [optional]
        Display the final plot if True.

    Returns
    -------
    fig: pyplot.Figure
        The pyplot.Figure object for the created figure
    ax: pyplot.Axes
        The pyplot.Axes object for the created figure
    """

    spectrum = Spectrum(root, fp)
    spectrum_units = spectrum.units
    spectrum_inclinations = spectrum.inclinations
    n_inclinations = spectrum.n_inclinations
    plot_dimensions = subplot_dims(n_inclinations)

    if figsize:
        size = figsize
    else:
        size = (12, 10)

    normalize_figure_style()

    fig, ax = plt.subplots(plot_dimensions[0], plot_dimensions[1], figsize=size, squeeze=False)
    fig, ax = remove_extra_axes(fig, ax, n_inclinations, plot_dimensions[0] * plot_dimensions[1])

    # Use either frequency or wavelength and set the plot limits respectively

    if frequency_space:
        x = spectrum["Freq."]
    else:
        x = spectrum["Lambda"]
    xlims = [x.min(), x.max()]
    if not xmin:
        xmin = xlims[0]
    if not xmax:
        xmax = xlims[1]
    xlims = (xmin, xmax)

    inclination_index = 0
    for i in range(plot_dimensions[0]):
        for j in range(plot_dimensions[1]):
            if inclination_index > n_inclinations - 1:
                break
            name = str(spectrum_inclinations[inclination_index])
            ax[i, j] = _plot_panel_subplot(ax[i, j], x, spectrum, spectrum_units, [name], xlims, smooth_amount, 1,
                                           scale, frequency_space, False)
            ymin, ymax = get_y_lims_for_x_lims(x, spectrum[name], xmin, xmax)
            ax[i, j].set_ylim(ymin, ymax)

            if add_line_ids:
                if scale == "loglog" or scale == "logx":
                    logx = True
                else:
                    logx = False
                ax[i, j] = ax_add_line_ids(ax[i, j], common_lines(frequency_space), logx=logx)
            inclination_index += 1

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_single_spectrum_inclination(root,
                                     fp,
                                     inclination,
                                     xmin=None,
                                     xmax=None,
                                     smooth_amount=5,
                                     scale="logy",
                                     frequency_space=False,
                                     display=False):
    """Create a plot of an individual spectrum for the provided inclination
    angle.

    Parameters
    ----------
    root: str
        The root name of the Python models
    fp: str
        The absolute or relative path containing the Python models
    inclination: str, float, int
        The specific inclination angle to plot for
    xmin: float [optional]
        The lower x boundary for the figure
    xmax: float [optional]
        The upper x boundary for the figure
    smooth_amount: int [optional]
        The size of the boxcar filter to smooth the spectrum components
    scale: str [optional]
        The scale of the axes for the plot.
    frequency_space: bool [optional]
        Create the figure in frequency space instead of wavelength space
    display: bool [optional]
        Display the final plot if True.

    Returns
    -------
    fig: pyplot.Figure
        The pyplot.Figure object for the created figure
    ax: pyplot.Axes
        The pyplot.Axes object for the created figure
    """

    normalize_figure_style()

    s = Spectrum(root, fp, smooth=smooth_amount)

    if frequency_space:
        x = s["Freq."]
    else:
        x = s["Lambda"]
    xlims = [x.min(), x.max()]
    if not xmin:
        xmin = xlims[0]
    if not xmax:
        xmax = xlims[1]
    xlims = (xmin, xmax)

    if type(inclination) != str:
        try:
            inclination = str(inclination)
        except ValueError:
            print("unable to convert into string")
            return
    y = s[inclination]
    if frequency_space:
        xax = r"Frequency [Hz]"
        yax = r"$\nu F_{\nu}$ (erg s$^{-1}$ cm$^{-2}$)"
        y *= s["Lambda"]
    else:
        xax = r"Wavelength [$\AA$]"
        yax = r"$F_{\lambda}$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)"

    fig, ax = plot(x, y, xlims[0], xlims[1], xax, yax, scale)

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_multiple_model_spectra(output_name,
                                spectra_filepaths,
                                inclination_angle,
                                fp=".",
                                xmin=None,
                                xmax=None,
                                frequency_space=False,
                                axes_scales="logy",
                                smooth_amount=5,
                                plot_common_lines=False,
                                file_ext="png",
                                display=False):
    """Plot multiple spectra, from multiple models, given in the list of spectra
    provided.
    todo: when using "all", create separate plot for each inclination

    Parameters
    ----------
    output_name: str
        The name to use for the created plot.
    spectra_filepaths: List[str]
        A list of spectrum file paths.
    inclination_angle: str
        The inclination angle(s) to plot
    fp: [optional] str
        The working directory containing the Python models
    xmin: [optional] float
        The smallest value on the x axis.
    xmax: [optional] float
        The largest value on the x axis.
    frequency_space: [optional] bool
        Create the plot in frequency space and use nu F_nu instead.
    axes_scales: [optional] str
        The scaling of the x and y axis. Allowed logx, logy, linlin, loglog
    smooth_amount: [optional] int
        The amount of smoothing to use.
    plot_common_lines: [optional] bool
        Add line labels to the figure.
    file_ext: [optional] str
        The file extension of the output plot.
    display: [optional] bool
        Show the plot when finished

    Returns
    -------
    fig: plt.Figure
        Figure object.
    ax: plt.Axes
        Axes object."""

    normalize_figure_style()

    spectrum_objects = []
    for spectrum in spectra_filepaths:
        root, cd = get_root(spectrum)
        spectrum_objects.append(Spectrum(root, cd, smooth=smooth_amount))

    if inclination_angle == "all":
        inclinations = []
        for spectrum in spectrum_objects:
            inclinations += spectrum.inclinations
        inclinations = sorted(list(dict.fromkeys(inclinations)))
        figsize = (12, 12)
    else:
        inclinations = [inclination_angle]
        figsize = (12, 5)

    n_inclinations = len(inclinations)
    n_rows, n_cols = subplot_dims(n_inclinations)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig, ax = remove_extra_axes(fig, ax, n_inclinations, n_rows * n_cols)
    ax = ax.flatten()  # for safety...

    y_min = +1e99
    y_max = -1e99

    for i, inclination in enumerate(inclinations):

        for spectrum in spectrum_objects:

            # Ignore spectra which are from continuum only models...

            if spectrum.fp.find("continuum") != -1:
                continue

            if frequency_space:
                x = spectrum["Freq."]
            else:
                x = spectrum["Lambda"]
            try:
                if frequency_space:
                    y = spectrum["Lambda"] * spectrum[inclination]
                else:
                    y = spectrum[inclination]
            except KeyError:
                continue

            ax[i].plot(x, y, label=spectrum.fp.replace("_", r"\_"), alpha=0.75)

            # Calculate the y-axis limits to keep all spectra within the
            # plot area

            if not xmin:
                xmin = x.min()
            if not xmax:
                xmax = x.max()
            this_y_min, this_y_max = get_y_lims_for_x_lims(x, y, xmin, xmax)
            if this_y_min < y_min:
                y_min = this_y_min
            if this_y_max > y_max:
                y_max = this_y_max

        if y_min == +1e99:
            y_min = None
        if y_max == -1e99:
            y_max = None

        ax[i].set_title(f"{inclinations[i]}" + r"$^{\circ}$")

        x_lims = list(ax[i].get_xlim())
        if not xmin:
            xmin = x_lims[0]
        if not xmax:
            xmax = x_lims[1]
        ax[i].set_xlim(xmin, xmax)
        ax[i].set_ylim(y_min, y_max)

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

        if plot_common_lines:
            if axes_scales == "logx" or axes_scales == "loglog":
                logx = True
            else:
                logx = False
            ax[i] = ax_add_line_ids(ax[i], common_lines(), logx=logx)

    ax[0].legend(loc="upper left")
    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    if inclination_angle != "all":
        name = "{}/{}_i{}".format(fp, output_name, inclination_angle)
    else:
        name = "{}/{}".format(fp, output_name)

    fig.savefig("{}.{}".format(name, file_ext))
    if file_ext == "pdf" or file_ext == "eps":
        fig.savefig("{}.png".format(name))

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax
