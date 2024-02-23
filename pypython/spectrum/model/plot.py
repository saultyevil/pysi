#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Class extension for plotting spectra."""

from __future__ import annotations
from typing import Iterable, Tuple

from matplotlib import pyplot

import pypython.spectrum.enum
from pypython.utility import plot
from pypython.utility import array
from pypython.spectrum.model.base import SpectrumBase


def _add_flux_ax_labels(ax, units, distance):
    """Add spectrum labels for flux, or luminosity multiplied by the spatial
    unit.

    Parameters
    ----------
    units: pypython.spectrum.SpectrumUnits
        The units of the spectrum
    distance: float
        The distance of the spectrum

    Returns
    -------
    ax: plt.Axes
        The updated Axes object with axes labels.
    """
    if units == pypython.spectrum.enum.SpectrumUnits.L_NU:
        ax.set_xlabel(r"Rest-frame frequency [Hz]")
        ax.set_ylabel(r"$\nu L_{\nu}$ [erg s$^{-1}$]")
    elif units == pypython.spectrum.enum.SpectrumUnits.L_LAM:
        ax.set_xlabel(r"Rest-frame wavelength [\AA]")
        ax.set_ylabel(r"$\lambda L_{\lambda}$ [erg s$^{-1}$]")
    elif units == pypython.spectrum.enum.SpectrumUnits.F_LAM:
        ax.set_xlabel(r"Rest-frame wavelength [\AA]")
        ax.set_ylabel(r"$\lambda F_{\lambda}$ at " + f"{distance:g} pc " + r"[erg s$^{-1}$]")
    else:
        ax.set_xlabel(r"Rest-frame frequency [Hz]")
        ax.set_ylabel(r"$\nu F_{\nu}$ at " + f"{distance:g} pc " + r"[erg s$^{-1}$ cm$^{-2}$]")

    return ax


def _add_flux_density_ax_labels(ax, units, distance):
    """Add spectrum labels for a flux density, or luminosity.

    Parameters
    ----------
    units: pypython.spectrum.SpectrumUnits
        The units of the spectrum
    distance: float
        The distance of the spectrum

    Returns
    -------
    ax: plt.Axes
        The updated Axes object with axes labels.
    """
    if units == pypython.spectrum.enum.SpectrumUnits.L_NU:
        ax.set_xlabel(r"Rest-frame frequency [Hz]")
        ax.set_ylabel(r"$L_{\nu}$ [erg s$^{-1}$ Hz$^{-1}$]")
    elif units == pypython.spectrum.enum.SpectrumUnits.L_LAM:
        ax.set_xlabel(r"Rest-frame wavelength [\AA]")
        ax.set_ylabel(r"$L_{\lambda}$ [erg s$^{-1}$ \AA$^{-1}$]")
    elif units == pypython.spectrum.enum.SpectrumUnits.F_LAM:
        ax.set_xlabel(r"Rest-frame wavelength [\AA]")
        ax.set_ylabel(r"$F_{\lambda}$ at " + f"{distance:g} pc " + r"[erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]")
    else:
        ax.set_xlabel(r"Rest-frame frequency [Hz]")
        ax.set_ylabel(r"$F_{\nu}$ at " + f"{distance:g} pc " + r"[erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")

    return ax


def _set_axes_labels(ax, spectrum=None, units=None, distance=None, multiply_by_spatial_units=False):
    """Set the units of a given matplotlib axes.
    todo: should have an else if the units are unknown, not for f_nu

    Parameters
    ----------
    ax: plt.Axes
        The axes object to update.
    spectrum: pypython.Spectrum
        The spectrum being plotted. Used to determine the axes labels.
    units: pypython.spectrum.SpectrumUnits
        The units of the spectrum
    distance: float
        The distance of the spectrum
    multiply_by_spatial_units: bool
        If flux/nu Lnu is being plotted instead of flux density or
        luminosity.

    Returns
    -------
    ax: plt.Axes
        The updated axes object.
    """
    if spectrum is None and units is None and distance is None:
        raise ValueError("either the spectrum or the units and distance needs to be provided")

    if units and distance is None or distance and units is None:
        raise ValueError("the units and distance have to be provided together")

    if units is None and distance is None:
        units = spectrum.units
        distance = spectrum.distance

    if multiply_by_spatial_units:
        ax = _add_flux_ax_labels(ax, units, distance)
    else:
        ax = _add_flux_density_ax_labels(ax, units, distance)

    return ax


def _create_plot(
    ax: pyplot.Axes,
    spectrum: pypython.spectrum.Spectrum,
    things_to_plot: Iterable[str] | str,
    xmin: float,
    xmax: float,
    alpha: float,
    scale: str,
    use_flux: bool,
):
    """Plot some things to a provided matplotlib ax object.

    This function is used to do a lot of the plotting heavy lifting in this
    sub-module. It's still fairly flexible to be used outside of the main
    plotting functions, however. You are just required to pass an axes to
    plot onto.

    Parameters
    ----------
    ax: plt.Axes
        A matplotlib axes object to plot onto.
    spectrum: pypython.Spectrum
        The spectrum object to plot. The current spectrum wishing to be set
        must be correct, otherwise the wrong thing may be plotted.
    things_to_plot: str or list or tuple of str
        A collection of names of things to plot to iterate over.
    xmin: float
        The lower x boundary of the plot.
    xmax: float
        The upper x boundary of the plot.
    alpha: float
        The transparency of the spectra plotted.
    scale: str
        The scaling of the plot axes.
    use_flux: bool
        Plot the spectrum as a flux or nu Lnu instead of flux density or
        luminosity.

    Returns
    -------
    ax: pyplot.Axes
        The modified matplotlib Axes object.
    """
    if isinstance(things_to_plot, str):
        things_to_plot = (things_to_plot,)

    for thing in things_to_plot:
        y_data = spectrum[thing]

        # If plotting in frequency space, of if the units then the flux needs
        # to be converted in nu F nu

        if use_flux:
            if spectrum[spectrum.current]["spectral_axis"] == pypython.spectrum.enum.SpectrumSpectralAxis.WAVELENGTH:
                y_data *= spectrum["Lambda"]
            else:
                y_data *= spectrum["Freq."]

        if spectrum[spectrum.current]["spectral_axis"] == pypython.spectrum.enum.SpectrumSpectralAxis.WAVELENGTH:
            x_data = spectrum["Lambda"]
        else:
            x_data = spectrum["Freq."]

        x_data, y_data = array.get_subset_in_second_array(x_data, y_data, xmin, xmax)

        ax.plot(x_data, y_data, label=thing, alpha=alpha)

    ax.legend(loc="lower left")
    ax = plot.set_axes_scales(ax, scale)

    ax = _set_axes_labels(
        ax,
        units=spectrum[spectrum.current]["units"],
        distance=spectrum[spectrum.current]["distance"],
        multiply_by_spatial_units=use_flux,
    )

    return ax


class SpectrumPlot(SpectrumBase):
    """Class for plotting spectra."""

    def __init__(self, root: str, directory, **kwargs):
        """Initialize the class."""
        super().__init__(root, directory, **kwargs)

    def plot(
        self, label: str, fig: pyplot.Figure = None, ax: pyplot.Axes = None, ax_scale: str = "loglog"
    ) -> Tuple[pyplot.Figure, pyplot.Axes]:
        """Plot a labelled spectrum for the current spectrum file.

        Parameters
        ----------
        label : str
            The label given to the spectrum to plot.
        fig : pyplot.Figure
            A pyplot.Figure to update with the spectrum
        ax : pyplot.Axes
            A pyplot.Axes to add the spectrum to
        ax_scale: str
            The scaling choice for the axes: lin, logy, logx, loglog

        Returns
        -------
        fig : pyplot.Figure
            An updated figure
        ax : pyplot.Axes
            An updated axes

        Notes
        -----
        If fig and ax are not provided, they will be created.
        """
        if not fig and not ax:
            fig, ax = pyplot.subplots(1, 1, figsize=(12, 5))
        elif not fig and ax or fig and not ax:
            raise ValueError("fig and ax need to be supplied together")

        ax = _create_plot(ax, self, label, None, None, 1.0, ax_scale, False)

        # if label_lines:
        #     ax = add_line_ids(ax, common_lines(spectrum=spectrum), "none")

        fig = plot.finish_figure(fig, label)

        return fig, ax

    def plot_diagnostic_spectra(self) -> Tuple[pyplot.Figure, pyplot.Axes]:
        """Plot the diagnostic spec_tot files.

        Returns
        -------
        fig : pyplot.Figure
            An updated figure
        ax : pyplot.Axes
            An updated axes

        Notes
        -----
        If fig and ax are not provided, they will be created.
        """
        fig, ax = pyplot.subplots(2, 1, figsize=(12, 9), sharex=True)
        current = self.current
        try:
            self.set_spectrum("spec_tot")  # temporarily change the spectrum target
        except KeyError:
            print("Unable to open `spec_tot` spectrum")
            return fig, ax

        ax[0] = _create_plot(
            ax[0],
            self,
            ("CenSrc", "Disk", "WCreated"),
            None,
            None,
            0.75,
            "loglog",
            False,
        )
        ax[1] = _create_plot(
            ax[1],
            self,
            ("Created", "Emitted", "Wind", "HitSurf"),
            None,
            None,
            0.75,
            "loglog",
            False,
        )

        ax[0].set_xlabel(None)
        fig = plot.finish_figure(fig, "Diagnostic spectra", hspace=0)
        self.set_spectrum(current)

        return fig, ax

    def plot_optical_depth(self) -> Tuple[pyplot.Figure, pyplot.Axes]:
        """Plot the optical depth.

        Returns
        -------
        fig : pyplot.Figure
            An updated figure
        ax : pyplot.Axes
            An updated axes

        Notes
        -----
        If fig and ax are not provided, they will be created.
        """
        fig, ax = pyplot.subplots(1, 1, figsize=(12, 5))
        current = self.current
        try:
            self.set_spectrum("spec_tau")  # temporarily change the spectrum target
        except KeyError:
            print("Unable to open `spec_tau` spectrum")
            return fig, ax

        ax = _create_plot(
            ax,
            self,
            self["inclinations"],
            None,
            None,
            0.75,
            "loglog",
            False,
        )

        fig = plot.finish_figure(fig, "Continuum Optical Depth")
        self.current = current

        return fig, ax

    @staticmethod
    def show_figures() -> None:
        """Show plotted figures.

        Notes
        ------
        This is a wrapper around pyplot.show().
        """
        pyplot.show()
