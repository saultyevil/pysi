#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various utility functions for use with the reverberation
mapping part of Python. It seems to mostly house functions designed to create
a spectrum from the delay_dump output.
"""

from .Constants import PARSEC, C
from .PythonUtils import file_len, array_index, round_to_sig_figs

import numpy as np
from numba import jit
from tqdm import tqdm


@jit(nopython=True)
def jit_bin_loop(spectrum: np.ndarray, freq_min: float, freq_max: float, freqs: np.ndarray, weights: np.ndarray,
                 specs: np.ndarray):
    """
    Bin the photons into frequency bins using jit to attempt to speed everything
    up.

    Parameters
    ----------
    spectrum: np.ndarray
        The spectrum array containing the frequency bins.
    freq_min: float
        The minimum frequency to bin.
    freq_max: float
        The maximum frequency to bin.
    freqs: np.ndarray
        The photon frequencies.
    weights: np.ndarray
        The photon weights.
    specs: np.ndarray
        The index for the spectrum the photons belong to.

    Returns
    -------
    spectrum: np.ndarray
        The spectrum where photon weights have been binned.
    """

    assert(len(freqs) == len(weights))
    assert(len(freqs) == len(specs))

    nphotons = freqs.shape[0]
    output = nphotons // 10

    for p in range(nphotons):
        if p % output == 0:
            print(" - JIT: Binning photons by frequency: ", np.round(p / nphotons * 100.0), "% photons binned")
        if freqs[p] < freq_min or freqs[p] > freq_max:
            continue
        freq_index = np.abs(spectrum[:, 0] - freqs[p]).argmin()  # Assumes all values are unique in spectrum[:, 0]
        spectrum[freq_index, specs[p]] += weights[p]

    return spectrum


def construct_spectrum_from_weights(photons: np.ndarray, spectrum: np.ndarray, spec_norm: float, column_names: dict,
                                    nphotons: int = None, nspec: int = None, nbins: int = None, dnorm: float = 100,
                                    use_jit: bool = True) -> np.ndarray:
    """
    Construct a spectrum from the weights of the provided photons. If nphotons,
    nspec or nbins are not provided, then the function will automatically detect
    what these numbers are. There should be no NaN values in any of the arrays
    which are provided.

    Parameters
    ----------
    photons: np.ndarray (nphotons, 3)
        An array containing the photon frequency, weight and spectrum number.
    spectrum: np.ndarray (nbins, nspec)
        An array containing the frequency bins and empty bins for each
        inclination angle.
    spec_norm: float
        The spectrum normalization amount - usually the number of spectrum
        cycles.
    column_names: dict
        A dict containing the name of photons columns and their respective index
        into that array.
    nphotons: [optional] int
        The number of photons in the photons array.
    nspec: [optional] int
        The number of inclination angles to bin.
    nbins: [optional] int
        The number of frequency bins in the spectrum.
    dnorm: [optional] float
        The distance normalization for the flux calculation in parsecs. By
        default this is 100 parsecs.
    use_jit: [optional] bool
        If True, JIT will be used to speed up the photon binning

    Returns
    -------
    spectrum: np.ndarray (nbins, nspec)
        The constructed spectrum in units of F_lambda erg/s/cm/cm/A.
    """

    n = construct_spectrum_from_weights.__name__

    assert(photons.shape[0] != 0), "No photons"
    assert(spectrum.shape[0] != 0), "No frequency bins provided"

    if np.isnan(photons).any():
        print("{}: There are NaN values in photon array!".format(n))
        raise ValueError
    if column_names["Freq."] != 0:
        print("{}: expected frequency to column index 0 of spectrum. Cannot continue!".format(n))
        raise ValueError

    if not nphotons:
        nphotons = photons.shape[0]
    if not nbins:
        nbins = spectrum.shape[0]
    if not nspec:
        nspec = np.max(photons[:, column_names["Spec."]]) + 1

    distance_normalisation = 4 * np.pi * (dnorm * PARSEC) ** 2

    # These are the vital quantities required for binning photons
    frequencies = photons[:, column_names["Freq."]]
    weights = photons[:, column_names["Weight"]]
    spectrum_indices = photons[:, column_names["Spec."]].astype(int) + 1

    # Now bin the photons into the spectrum array
    if use_jit:
        freq_min = np.min(spectrum[:, column_names["Freq."]])
        freq_max = np.max(spectrum[:, column_names["Freq."]])
        spectrum = jit_bin_loop(spectrum, freq_min, freq_max, frequencies, weights, spectrum_indices)
    else:
        for p in tqdm(range(nphotons), desc="Binning photons by frequency", unit=" photons", smoothing=0):
            freq_index = array_index(spectrum[:, column_names["Freq."]], frequencies[p])
            if freq_index != -1:
                spectrum[freq_index, spectrum_indices[p]] += weights[p]

    # Now convert the binned weights into a flux - not jit'd as it should be fast
    # The flux is FLAMBDA -- ergs/s/cm/cm/A
    for i in range(nbins - 1):
        for j in range(nspec):
            freq = spectrum[i, column_names["Freq."]]
            dfreq = spectrum[i + 1, column_names["Freq."]] - freq
            spectrum[i, j + 1] *= (freq ** 2 * 1e-8) / (dfreq * distance_normalisation * C)
        spectrum[i, 1:] *= spec_norm

    return spectrum


def read_delay_dump(root: str, wd: str, extract: dict) -> np.ndarray:
    """
    Process the photons which have been dumped to the delay_dump file. tqdm is
    used to display a progress bar of the current progress.

    Parameters
    ----------
    root: str
        The root name of the simulation.
    wd: str
        The directory containing the simulation.
    extract: dict
        A dictionary containing the names of the quantities to extract and their
        index to place them into dumped_photons array.

    Returns
    -------
    dumped_photons: np.ndarray
        An array containing the dumped photons with the quantities specified
        by the extract dict.
    """

    n = read_delay_dump.__name__

    file = "{}/{}.delay_dump".format(wd, root)
    nlines = file_len(file)

    file_columns = {
        "Np": 0, "Freq.": 1, "Lambda": 2, "Weight": 3, "Last X": 4, "Last Y": 5,
        "Last Z": 6, "Scat.": 7, "RScat.": 8, "Delay": 9, "Spec.": 10,
        "Orig.": 11, "Res.": 12, "LineRes.": 13
    }

    ncols = len(file_columns)
    nextract = len(extract)
    dumped_photons = np.zeros((nlines, nextract))
    dumped_photons[:, :] = np.nan   # because some lines will not be photons mark them as NaN

    f = open(file, "r")

    for i in tqdm(range(nlines), desc="Reading {}.delay_dump".format(root), unit=" lines", smoothing=0):
        line = f.readline().split()
        if len(line) == ncols:
            for j, e in enumerate(extract):
                dumped_photons[i, j] = float(line[file_columns[e]])

    f.close()

    # Remove the nan lines from the photons array using bitshift magic
    dumped_photons = dumped_photons[~np.isnan(dumped_photons).any(axis=1)]

    return dumped_photons


def create_spectra_from_delay_dump(root: str, wd: str = ".", fmin: float = None, fmax: float = None, nbins: int = None,
                                   spec_norm: float = 1, logbins: bool = True, use_jit: bool = True) -> np.ndarray:
    """
    Create a spectrum for each inclination angle using the photons which have
    been dumped to the root.delay_dump file.

    Spectrum frequency bins are rounded to 7 significant figures as this makes
    them the same values as what is output from the Python spectrum.

    Parameters
    ----------
    root: str
        The root name of the simulation.
    wd: [optional] str
        The directory containing the simulation.
    fmin: [optional] float
        The smallest frequency bin.
    fmax: [optional] float
        The largest frequency bin
    nbins: [optional] int
        The number of frequency bins.
    spec_norm: float
        A normalization constant for the spectrum. Usually the number of photon
        cycles.
    logbins: [optional] bool
        If True, the frequency bins are spaced equally in log space. Otherwise
        the bins are in linear space.
    use_jit: [optional] bool
        Enable using jit to try and speed up the photon binning.

    Returns
    -------
    filtered_spectrum: np.ndarray
        A 2D array containing the frequency in the first column and the
        fluxes for each inclination angle in the other columns.
    """

    # extract is the quantities which we want to extract, and their index in
    # the dumped_photons array
    extract = {
        "Freq.": 0, "Weight": 1, "Spec.": 2, "Scat.": 3, "RScat.": 4, "Orig.": 5,
        "Res.": 6, "LineRes.": 7
    }

    dumped_photons = read_delay_dump(root, wd, extract)
    nphotons = dumped_photons.shape[0]
    nspec = int(np.max(dumped_photons[:, 2])) + 1

    if not fmin:
        fmin = np.min(dumped_photons[:, 0])
    if not fmax:
        fmax = np.max(dumped_photons[:, 0])
    if not nbins:
        nbins = int(1e4)

    # Now bin the photons to create the filtered spectrum
    filtered_spectrum = np.zeros((nbins, 1 + nspec))
    if logbins:
        bins = np.logspace(np.log10(fmin), np.log10(fmax), nbins, endpoint=True)
    else:
        bins = np.linspace(fmin, fmax, nbins, endpoint=True)
    filtered_spectrum[:, 0] = round_to_sig_figs(bins, 7)

    filtered_spectrum = construct_spectrum_from_weights(dumped_photons, filtered_spectrum, spec_norm, extract,
                                                        nphotons, nspec, nbins, use_jit=use_jit)

    # TODO: improve writing out spectrum, e.g. add headers, output lambda etc.
    np.savetxt("{}/{}.delay_dump.spec".format(wd, root), filtered_spectrum)

    return filtered_spectrum
