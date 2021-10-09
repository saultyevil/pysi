#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions for working with delay dump files.
"""

from copy import deepcopy
from sys import exit

import numpy as np
import pandas as pd
from matplotlib import animation as ani
from matplotlib import pyplot as plt

import pypython
import pypython.constants as c
import pypython.dump.spectrum
import pypython.dump.wind

BOUND_FREE_NRES = 20000
UNFILTERED_SPECTRUM = -3


start = 0


def photon_transport_movie(root, dump, fp=".", wind=None, wind_variable="rho", scale="loglog", figsize=(7, 6)):
    """Create a movie of a photon moving through the wind.

    Plots each photon path, one by one, to create an animation which should
    represent the random walk photons take through the wind.

    todo: make this work

    Parameters
    ----------
    root: str
        The root name of the simulation.
    dump: pd.DataFrame or similar
        The interactions of each photon, generated using modes.save_animation
    fp: str [optional]
        The file path containin the simulation.
    wind: pypython.Wind [optional]
        The Wind object of the wind to animated photon transport through.
    wind_variable: str [optional]
        The name of the wind variable to plot.
    scale: str [optional]
        The scales of the x and y axis, i.e. loglog, logx.
    figsize: tuple (int)
        The size of the figure in inches (width, height)

    Returns
    -------
    anim:
        The animation object.
    """
    if wind is None:
        wind = pypython.Wind(root, fp)

    if type(dump) is str:
        dump = read_dump_pd(root, fp)
    elif type(dump) != pd.DataFrame:
        raise TypeError("interactions has to be str or pd.DataFrame")

    nframes = len(dump)

    # First, create the figure and axes. Add the wind and set the axes units
    # and scales

    fig, ax = plt.subplots(figsize=figsize)
    ax = pypython.plot.set_axes_scales(ax, scale)

    ax.pcolormesh(wind.x, wind.z, wind.get(wind_variable), alpha=1.0)
    ax.set_xlabel(r"$\rho$" + f" [{wind.distance_units.value}]")
    ax.set_ylabel(r"$z$" + f" [{wind.distance_units.value}]")

    xpoints = wind["x"][wind["x"] > 0]
    zpoints = wind["z"][wind["z"] > 0]
    xmin, xmax = xpoints.min(), xpoints.max()
    zmin, zmax = zpoints.min(), zpoints.max()

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)

    np_numbers = np.array(dump["Np"])
    x_points = np.array(dump["LastX"])
    y_points = np.array(dump["LastY"])
    z_points = np.array(dump["LastZ"])

    line, = ax.plot([], [], lw=3)
    # start = 0

    def animate(nframe):
        """The actual animation function. Has to be here, because it relies on
        ax being defined.

        Parameters
        ----------
        nframe: int
            The frame being rendered.
        """
        global start
        if nframe == 0:
            return

        z = np.abs(z_points[start:nframe])
        rho = np.sqrt(x_points[start:nframe] ** 2 + y_points[start:nframe] ** 2)

        # if rho[-1] < xmin:
        #     return ax,
        # if z[-1] < zmin:
        #     return ax,

        if np_numbers[nframe] != np_numbers[nframe + 1]:
            start = nframe
            line.set_data([], [])

        line.set_data(rho, z)
        line.set_color(f"C{np_numbers[nframe]}")

        return line,

    anim = ani.FuncAnimation(fig,
                             animate,
                             frames=len(dump),
                             interval=1 / len(dump))

    return anim


def read_dump_pd(root, fp="."):
    """Process the photons which have been dumped to the delay_dump file.

    Parameters
    ----------
    root: str
        The root name of the simulation.
    fp: str
        The directory containing the simulation.

    Returns
    -------
    dumped_photons: pd.DataFrame
        An array containing the dumped photons with the quantities specified
        by the extract dict.
    """
    names = {
        "Np": np.int32,
        "Freq.": np.float64,
        "Lambda": np.float64,
        "Weight": np.float64,
        "LastX": np.float64,
        "LastY": np.float64,
        "LastZ": np.float64,
        "Scat.": np.int32,
        "RScat.": np.int32,
        "Delay": np.float64,
        "Spec.": np.int32,
        "Orig.": np.int32,
        "Res.": np.int32,
        "LineRes.": np.int32
    }

    return pd.read_csv(f"{fp}/{root}.delay_dump",
                       names=list(names.keys()),
                       dtype=names,
                       delim_whitespace=True,
                       comment="#")


def create_spectrum(root,
                    fp=".",
                    extract=(UNFILTERED_SPECTRUM, ),
                    dumped_photons=None,
                    freq_bins=None,
                    freq_min=None,
                    freq_max=None,
                    n_bins=10000,
                    d_norm_pc=100,
                    spec_cycle_norm=1,
                    n_cores_norm=1,
                    log_bins=True,
                    output_numpy=False):
    """Create a spectrum for each inclination angle using the photons which
    have been dumped to the root.delay_dump file.

    Spectrum frequency bins are rounded to 7 significant figures as this makes
    them the same values as what is output from the Python spectrum.

    Parameters
    ----------
    root: str
        The root name of the simulation.
    fp: [optional] str
        The directory containing the simulation.
    extract: [optional] int
        The internal line number for a line to extract.
    dumped_photons: [optional] pd.DataFrame
        The delay dump photons in a Pandas DataFrame. If this is not provided,
        then it will be read in.
    freq_bins: [optional] np.ndarray
        Frequency bins to use to bin photons.
    freq_min: [optional] float
        The smallest frequency bin.
    freq_max: [optional] float
        The largest frequency bin
    n_bins: [optional] int
        The number of frequency bins.
    d_norm_pc: [optional] float
        The distance normalization of the spectrum.
    spec_cycle_norm: float
        A normalization constant for the spectrum, is > 1.
    n_cores_norm: [optional] int
        The number of cores which were used to generate the delay_dump file
    log_bins: [optional] bool
        If True, the frequency bins are spaced equally in log space. Otherwise
        the bins are in linear space.
    output_numpy: [optional] bool
        If True, the spectrum will be a numpy array instead of a pandas data
        frame

    Returns
    -------
    filtered_spectrum: np.ndarray or pd.DataFrame
        A 2D array containing the frequency in the first column and the
        fluxes for each inclination angle in the other columns.
    """

    if type(extract) != tuple:
        print("extract_nres is not a tuple but is of type {}".format(type(extract)))
        exit(pypython.error.EXIT_FAIL)

    # If the frequency bins have been provided, we need to do some checks to make
    # sure they're sane

    if freq_bins is not None:
        if type(freq_bins) != np.ndarray:
            freq_bins = np.array(freq_bins, dtype=np.float64)
        is_increasing = np.all(np.diff(freq_bins) > 0)
        if not is_increasing:
            raise ValueError("the values for the frequency bins provided are not increasing")
        n_bins = len(freq_bins)

    if dumped_photons is None:
        dumped_photons = read_dump_pd(root, fp)

    line_res = dumped_photons["LineRes."].values.astype(int)
    n_spec = int(np.max(dumped_photons["Spec."].values)) + 1
    dump_spectrum = np.zeros((n_bins, 1 + n_spec))

    if freq_bins is not None:
        dump_spectrum[:, 0] = freq_bins
    else:
        if not freq_min:
            freq_min = np.min(dumped_photons["Freq."])
        if not freq_max:
            freq_max = np.max(dumped_photons["Freq."])
        if log_bins:
            dump_spectrum[:, 0] = np.logspace(np.log10(freq_min), np.log10(freq_max), n_bins, endpoint=True)
        else:
            dump_spectrum[:, 0] = np.linspace(freq_min, freq_max, n_bins, endpoint=True)

    # This function now constructs a spectrum given the photon frequencies and
    # weights as well as any other normalization constants

    freq_max = np.max(dump_spectrum[:, 0])
    freq_min = np.min(dump_spectrum[:, 0])

    dump_spectrum = pypython.dump.create_spectrum.bin_photon_weights(dump_spectrum, freq_min, freq_max,
                                                                     dumped_photons["Freq."].values,
                                                                     dumped_photons["Weight"].values,
                                                                     dumped_photons["Spec."].values.astype(int) + 1,
                                                                     dumped_photons["Res."].values.astype(int), line_res,
                                                                     extract, log_bins)

    dump_spectrum[:, 1:] /= n_cores_norm

    dump_spectrum = pypython.dump.create_spectrum.convert_weight_to_flux(dump_spectrum, spec_cycle_norm, d_norm_pc)

    # Remove the first and last bin, consistent with Python

    n_bins -= 2
    dump_spectrum = dump_spectrum[1:-1, :]

    dump_spectrum, inclinations = pypython.dump.create_spectrum.write_delay_dump_spectrum_to_file(root,
                                                                                                  fp,
                                                                                                  dump_spectrum,
                                                                                                  extract,
                                                                                                  n_spec,
                                                                                                  n_bins,
                                                                                                  d_norm_pc,
                                                                                                  return_inclinations=True)

    if output_numpy:
        return dump_spectrum
    else:
        lamda = np.reshape(c.C / dump_spectrum[:, 0] * 1e8, (n_bins, 1))
        dump_spectrum = np.append(lamda, dump_spectrum, axis=1)
        df = pd.DataFrame(dump_spectrum, columns=["Lambda", "Freq."] + inclinations)
        return df


def create_spectrum_breakdown(root, wl_min, wl_max, n_cores_norm=1, spec_cycle_norm=1, fp=".", nres=None, mode_line_res=True):
    """Get the spectra for the different physical processes which contribute to
    a spectrum. If nres is provided, then only a specific interaction will be
    extracted, otherwise all resonance interactions will.

    Parameters
    ----------
    root: str
        The root name of the simulation.
    wl_min: float
        The lower wavelength bound in Angstroms.
    wl_max: float
        The upper wavelength bound in Angstroms.
    n_cores_norm: int [optional]
        The number of cores normalization constant, i.e. the number of cores used
        to generate the delay_dump file.
    spec_cycle_norm: float [optional]
        The spectral cycle normalization, this is equal to 1 if all spectral
        cycles were run.
    fp: str [optional]
        The directory containing the simulation.
    nres: int [optional]
        A specific interaction to extract, is the nres number from Python.
    mode_line_res: bool [optional]
        Set as True if the delay_dump has the LineRes. value.

    Returns
    -------
    spectra: dict
        A dictionary where the keys are the name of the spectra and the values
        are pd.DataFrames of that corresponding spectrum.
    """

    dump = read_dump_pd(root, fp)
    s = pypython.Spectrum(root, fp)

    # create dataframes for each physical process, what you can actually get
    # depends on mode_line_res, i.e. if LineRes. is included or not. Store these
    # data frame in a list

    contributions = []
    contribution_names = ["Extracted"]

    # Extract either a specific interaction, or all the interactions. If LineRes.
    # is enabled, then extract the LineRes version of it too

    if nres:
        if type(nres) != int:
            nres = int(nres)
        contributions.append(dump[dump["Res."] == nres])
        contribution_names.append("Res." + str(nres))
        if mode_line_res:
            contributions.append(dump[dump["LineRes."] == nres])
            contribution_names.append("LineRes." + str(nres))
    else:
        tmp = dump[dump["Res."] <= 20000]
        contributions.append(tmp[tmp["Res."] >= 0])
        contribution_names.append("Res.")
        if mode_line_res:
            tmp = dump[dump["LineRes."] <= 20000]
            contributions.append(tmp[tmp["LineRes."] >= 0])
            contribution_names.append("LineRes.")

    # Extract the scattered spectrum, which is every continuum scatter

    contributions.append(dump[dump["Res."] == -1])
    contribution_names.append("Scattered")

    # Extract pure BF, FF and ES events, unless we're in Res. mode, which extracts
    # last scatters

    if mode_line_res:
        contributions.append(dump[dump["LineRes."] == -1])
        contributions.append(dump[dump["LineRes."] == -2])
        contributions.append(dump[dump["LineRes."] > 20000])
        contribution_names.append("ES only")
        contribution_names.append("FF only")
        contribution_names.append("BF only")
    else:
        contributions.append(dump[dump["Res."] == -2])
        contributions.append(dump[dump["Res."] > 20000])
        contribution_names.append("FF only")
        contribution_names.append("BF only")

    # Create each individual spectrum

    created_spectra = [s]
    for contribution in contributions:
        created_spectra.append(
            create_spectrum(root,
                            fp,
                            dumped_photons=contribution,
                            freq_min=pypython.physics.angstrom_to_hz(wl_max),
                            freq_max=pypython.physics.angstrom_to_hz(wl_min),
                            n_cores_norm=n_cores_norm,
                            spec_cycle_norm=spec_cycle_norm))
    n_spec = len(created_spectra)

    # dict comprehension to use contribution_names as the keys and the spectra
    # as the values

    return {contribution_names[i]: created_spectra[i] for i in range(n_spec)}


def create_wind_weight_contours(root, resonance, wind=None, fp=".", n_cores=1):
    """Bin photon interactions into the cells in which they happen.

    Returns the weight and count the interaction happens, binned onto the 2D
    coordinate grid of the wind.

    Parameters
    ----------
    root: str
        The root name of the model.
    resonance: int
        The resonance number of the photon to bin.
    wind: pypython.Wind [optional]
        The Wind object of the wind. If not provided, it will attempt to read
        it in anyway.
    fp: str [optional]
        The directory containing the simulation.
    n_cores: int [optional]
        The number of cores to normalize the binning by.

    Returns
    -------
    weight2d: np.ndarray
        The resonance weight binned per cell.
    count2d: np.ndarry
        The resonance count binned per cell.
    """
    if wind is None:
        wind = pypython.Wind(root, fp, masked=False)

    dump = read_dump_pd(root, fp)
    if dump.empty:
        print("photon dataframe is empty")
        exit(1)

    weight2d, count2d = pypython.dump.wind.wind_bin_photon_weights(len(dump), resonance, dump["LastX"].values,
                                                                   dump["LastY"].values, dump["LastZ"].values,
                                                                   dump["Res."].values, dump["Weight"].values, wind.x,
                                                                   wind.z, wind.nx, wind.nz)

    weight2d /= n_cores

    np.savetxt(f"{fp}/{root}_wind_res_{resonance}_" + "weight.txt", weight2d)
    np.savetxt(f"{fp}/{root}_wind_res_{resonance}_" + "count.txt", count2d)

    return weight2d, count2d
