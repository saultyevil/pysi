#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from pypython import (SPECTRUM_UNITS_FLM, SPECTRUM_UNITS_FNU, SPECTRUM_UNITS_LNU, Spectrum, get_root)
from pypython.plot import (ax_add_line_ids, common_lines, get_y_lims_for_x_lims, normalize_figure_style,
                           photoionization_edges, remove_extra_axes, subplot_dims)

MIN_SPEC_COMP_FLUX = 1e-15

# Helper functions -------------------------------------------------------------


def _add_line_labels(ax, labels, scale, offset=0.0):
    """Add labels of lines or absorption edges to ax.

    Parameters
    ----------
    ax: plt.Axes
        The subplot to add labels to.
    labels: tuple or list
        The labels to add.
    scale: str
        The scaling of the axes.
    """
    if scale == "loglog" or scale == "logx":
        logx = True
    else:
        logx = False
    ax = ax_add_line_ids(ax, labels, logx=logx)

    return ax


def _get_inclinations(spectrum, inclinations):
    """Get the inclination angles which will be used.

    If "all" is passed, then the inclinations from the spectrum object will
    be used. Otherwise, this will simply create a list of strings of the
    inclinations.

    Parameters
    ----------
    spectrum: pypython.Spectrum
        The spectrum object.
    inclinations: list
        A list of inclination angles wanted to be plotted.
    """

    if type(inclinations) == str:
        inclinations = [inclinations]

    if len(inclinations) > 1:
        if inclinations[0] == "all":  # ignore "all" if other inclinations are passed
            inclinations = inclinations[1:]
    else:
        if inclinations[0] == "all":
            inclinations = spectrum.inclinations

    return [str(inclination) for inclination in inclinations]


def _plot_subplot(ax, spectrum, things_to_plot, xmin, xmax, alpha, scale, use_flux, skip_sparse):
    """Create a subplot panel.

    todo: handle the situation when restircted x range may have bad y limits

    Parameters
    ----------

    Returns
    -------
    ax: pyplot.Axes
        The pyplot.Axes object for the subplot
    """

    if type(things_to_plot) is str:
        things_to_plot = [things_to_plot]

    n_skipped = 0

    for thing in things_to_plot:

        y = spectrum[thing]
        if skip_sparse and len(y[y < MIN_SPEC_COMP_FLUX]) > 0.7 * len(y):
            n_skipped += 1
            continue

        # If plotting in frequency space, of if the units then the flux needs
        # to be converted in nu F nu

        if use_flux:
            if spectrum.units == SPECTRUM_UNITS_FLM:
                y *= spectrum["Lambda"]
            elif spectrum.units == SPECTRUM_UNITS_FNU:
                y *= spectrum["Freq."]
            else:
                y *= spectrum["Freq."]

        if spectrum.units == SPECTRUM_UNITS_FLM:
            x = spectrum["Lambda"]
        else:
            x = spectrum["Freq."]

        ax.plot(x, y, label=thing, alpha=alpha)

    if n_skipped == len(things_to_plot):
        print("Nothing was plotted due to all the spectra being too sparse")
        return ax

    if scale == "logx" or scale == "loglog":
        ax.set_xscale("log")
    if scale == "logy" or scale == "loglog":
        ax.set_yscale("log")

    ax.set_xlim(xmin, xmax)
    ax = _set_axes_labels(ax, spectrum.units, use_flux)
    ax.legend(loc="lower left")

    return ax


def _set_axes_labels(ax, units, use_flux):
    """Set the units of a given matplotlib axes.

    Parameters
    ----------
    ax: plt.Axes
        The axes object to update.
    units: str
        The units of the spectrum.
    use_flux: bool
        If flux/nu Lnu is being plotted instead of flux density or
        luminosity.
    """

    if use_flux:
        if units == SPECTRUM_UNITS_LNU:
            ax.set_xlabel(r"Rest-frame Frequency [Hz]")
            ax.set_ylabel(r"$\nu L_{\nu}$ [erg s$^{-1}$]")
        elif units == SPECTRUM_UNITS_FLM:
            ax.set_xlabel(r"Rest-frame Wavelength [\AA]")
            ax.set_ylabel(r"$\lambda F_{\lambda}$ [erg s$^{-1}$]")
        else:
            ax.set_xlabel(r"Rest-frame Frequency [Hz]")
            ax.set_ylabel(r"$\nu F_{\nu}$ [erg s$^{-1}$ cm$^{-2}$]")
    else:
        if units == SPECTRUM_UNITS_LNU:
            ax.set_xlabel(r"Rest-frame Frequency [Hz]")
            ax.set_ylabel(r"$L_{\nu}$ [erg s$^{-1}$ Hz$^{-1}$]")
        elif units == SPECTRUM_UNITS_FLM:
            ax.set_xlabel(r"Rest-frame Wavelength [\AA]")
            ax.set_ylabel(r"$F_{\lambda}$ [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]")
        else:
            ax.set_xlabel(r"Rest-frame Frequency [Hz]")
            ax.set_ylabel(r"$F_{\nu}$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")

    return ax


# Plotting functions -----------------------------------------------------------


def optical_depth(spectrum,
                  inclinations="all",
                  xmin=None,
                  xmax=None,
                  scale="loglog",
                  label_edges=True,
                  frequency_space=True,
                  display=False):
    """Plot the continuum optical depth spectrum.

    Create a plot of the continuum optical depth against either frequency or
    wavelength. Frequency space is the default and preferred. This function
    returns the Figure and Axes object.
    todo: handle the situation when restricted x range may have bad y limits

    Parameters
    ----------
    spectrum: pypython.Spectrum
        The spectrum object.
    inclinations: list [optional]
        A list of inclination angles to plot, but "all" is also an acceptable
        choice if all inclinations are to be plotted.
    xmin: float [optional]
        The lower x boundary for the figure
    xmax: float [optional]
        The upper x boundary for the figure
    scale: str [optional]
        The scale of the axes for the plot.
    label_edges: bool [optional]
        Label common absorption edges of interest onto the figure
    frequency_space: bool [optional]
        Create the figure in frequency space instead of wavelength space
    display: bool [optional]
        Display the final plot if True.

    Returns
    -------
    fig: plt.Figure
        matplotlib Figure object.
    ax: plt.Axes
        matplotlib Axes object.
    """
    if spectrum.current != "spec_tau":
        spectrum.set("spec_tau")

    normalize_figure_style()
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Determine if we're plotting in frequency or wavelength space and then
    # determine the inclinations we want to plot

    if frequency_space:
        xlabel = "Freq."
    else:
        xlabel = "Lambda"

    if not xmin:
        xmin = np.min(spectrum[xlabel])
    if not xmax:
        xmax = np.max(spectrum[xlabel])

    inclinations = _get_inclinations(spectrum, inclinations)

    # Now loop over the inclinations and plot each one, skipping ones which are
    # completely empty

    for inclination in inclinations:
        if inclination != "all":
            if inclination not in spectrum.inclinations:  # Skip inclinations which don't exist
                continue
        label = f"{inclination}" + r"$^{\circ}$"

        if np.count_nonzero(spectrum[inclination]) == 0:
            continue

        ax.plot(spectrum[xlabel], spectrum[inclination], linewidth=2, label=label)

        if scale == "logx" or scale == "loglog":
            ax.set_xscale("log")
        if scale == "logy" or scale == "loglog":
            ax.set_yscale("log")

    ax.set_ylabel(r"Continuum Optical Depth")

    if frequency_space:
        ax.set_xlabel(r"Rest-frame Frequency [Hz]")
    else:
        ax.set_xlabel(r"Rest-frame Wavelength [$\AA$]")

    ax.set_xlim(xmin, xmax)
    ax.legend(loc="upper left")

    if label_edges:
        ax = _add_line_labels(ax, photoionization_edges(frequency_space), scale)

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig(f"{spectrum.fp}/{spectrum.root}_spec_optical_depth.png")

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def reprocessing(spectrum,
                 xmin=None,
                 xmax=None,
                 scale="loglog",
                 display=False):
    raise NotImplemented()


def spectrum_components(spectrum,
                        xmin=None,
                        xmax=None,
                        scale="loglog",
                        alpha=0.65,
                        use_flux=False,
                        display=False):
    """Plot the different "components" of the spectrum.

    The components are the columns labelled with words, rather than inclination
    angles. These columns are not supposed to add together, i.e. all of them
    together shouldn't equal the "Emitted" spectrum, which is the angle
    averaged escaping flux/luminosity.

    Parameters
    ----–-----
    spectrum: pypython.Spectrum
        The spectrum object.
    xmin: float [optional]
        The minimum x boundary to plot.
    xmax: float [optional]
        The maximum x boundary to plot.
    scale: str [optional]
        The scaling of the axes.
    alpha: float [optional]
        The line transparency on the plot.
    use_flux: bool [optional]
        Plot in flux or nu Lnu instead of flux density or luminosity.
    display: bool [optional]
        Display the object once plotted.

    Returns
    -------
    fig: plt.Figure
        matplotlib Figure object.
    ax: plt.Axes
        matplotlib Axes object.
    """

    normalize_figure_style()

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    ax[0] = _plot_subplot(ax[0], spectrum, ["CenSrc", "Disk", "Wind"], xmin, xmax, alpha, scale, use_flux, True)
    ax[1] = _plot_subplot(ax[1], spectrum, ["Created", "WCreated", "Emitted", "HitSurf"], xmin, xmax, alpha, scale,
                          use_flux, True)

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig(f"{spectrum.fp}/{spectrum.root}_spec_components.png")

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def spectrum_observer(spectrum,
                      inclinations,
                      xmin=None,
                      xmax=None,
                      scale="logy",
                      use_flux=False,
                      label_lines=True,
                      display=False):
    """Plot the request observer spectrum.

    If "all" is passed to inclinations, then all the observer angles will be
    plotted on a single figure.

    Parameters
    ----------
    spectrum: pypython.Spectrum
        The spectrum being plotted, "spec" or "log_spec" should be the active
        spectrum.
    inclinations: list or str
        The inclination angles to plot.
    xmin: float
        The lower x boundary of the plot.
    xmax: float
        The upper x boundary of the plot.
    scale: str
        The scale of the axes.
    use_flux: bool
        Plot the flux instead of flux density.
    label_lines: bool
        Label common spectrum lines.
    display: bool
        Display the figure once plotted.

    Returns
    -------
    fig: plt.Figure
        matplotlib Figure object.
    ax: plt.Axes
        matplotlib Axes object.
    """

    if spectrum.current not in ["spec", "log_spec"]:
        spectrum.set("spec")

    normalize_figure_style()
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    inclinations = _get_inclinations(spectrum, inclinations)

    wrong_input = []
    for inclination in inclinations:
        if inclination not in spectrum.inclinations:
            wrong_input.append(inclination)

    if len(wrong_input) > 0:
        print(f"The following inclinations provided are not in the spectrum inclinations and will be skipped:"
              f" {', '.join(wrong_input)}")
        for to_remove in wrong_input:
            inclinations.remove(to_remove)

    if len(inclinations) == 0:
        print(f"Returning empty figure without creating plot as there is nothing to plot")
        return fig, ax

    ax = _plot_subplot(ax, spectrum, inclinations, xmin, xmax, 1.0, scale, use_flux, False)

    if label_lines:
        ax = _add_line_labels(ax, common_lines(spectrum=spectrum), scale)

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


def multiple_spectra(output_name,
                     filepaths,
                     inclinations,
                     s=".",
                     xmin=None,
                     xmax=None,
                     frequency_space=False,
                     axes_scales="logy",
                     smooth_amount=5,
                     plot_common_lines=False,
                     file_ext="png",
                     display=False):
    """Plot multiple spectra, from multiple models, given in the list of
    spectra provided.

    Parameters
    ----------
    output_name: str
        The name to use for the created plot.
    filepaths: List[str]
        A list of spectrum file paths.
    inclinations: str
        The inclination angle(s) to plot
    s: [optional] str
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
        Axes object.
    """

    normalize_figure_style()

    spectra = []

    for s in filepaths:
        root, s = get_root(s)
        spectra.append(
            Spectrum(root, s, smooth=smooth_amount)
        )

    # Now, this is some convoluted shit to get the inclination angles
    # get only the unique, sorted, values if inclination == "all"

    figsize = (12, 5)

    if inclinations == "all":
        inclinations = sorted(list(dict.fromkeys([s.inclinations for s in spectra])))
    else:
        inclinations = inclinations

    if type(inclinations) is not list:
        inclinations = list(inclinations)

    n_inclinations = len(inclinations)

    n_rows, n_cols = subplot_dims(n_inclinations)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig, ax = remove_extra_axes(fig, ax, n_inclinations, n_rows * n_cols)
    ax = ax.flatten()

    y_min = +1e99
    y_max = -1e99

    for i, inclination in enumerate(inclinations):

        for s in spectra:

            if frequency_space:
                x = s["Freq."]
            else:
                x = s["Lambda"]
            try:
                if frequency_space:
                    y = s["Lambda"] * s[inclination]
                else:
                    y = s[inclination]
            except KeyError:
                continue

            ax[i].plot(x, y, label=s.fp.replace("_", r"\_"), alpha=0.75)

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

    if inclinations != "all":
        name = "{}/{}_i{}".format(s, output_name, inclinations)
    else:
        name = "{}/{}".format(s, output_name)

    fig.savefig("{}.{}".format(name, file_ext))
    if file_ext == "pdf" or file_ext == "eps":
        fig.savefig("{}.png".format(name))

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax
