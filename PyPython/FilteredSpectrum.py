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


def read_delay_dump_file(
    root: str, wd: str, extract: dict
) -> np.ndarray:
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

    n = read_delay_dump_file.__name__

    file = "{}/{}.delay_dump".format(wd, root)
    nlines = file_len(file)
    print("{} has {} photons to be read".format(file, nlines))

    file_columns = {
        "Np": 0, "Freq.": 1, "Lambda": 2, "Weight": 3, "Last X": 4, "Last Y": 5, "Last Z": 6, "Scat.": 7, "RScat.": 8,
        "Delay": 9, "Spec.": 10, "Orig.": 11, "Res.": 12, "LineRes.": 13
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


@jit(nopython=True)
def jit_bin_weights(
    spectrum: np.ndarray, spec_fmin: float, spec_fmax: float, phot_freq: np.ndarray, phot_weight: np.ndarray,
    phot_spec: np.ndarray, phot_nres: np.ndarray, phot_line_nres: np.ndarray, extract_nres: int
):
    """
    Bin the photons into frequency bins using jit to attempt to speed everything
    up.

    Parameters
    ----------
    spectrum: np.ndarray
        The spectrum array containing the frequency bins.
    spec_fmin: float
        The minimum frequency to bin.
    spec_fmax: float
        The maximum frequency to bin.
    phot_freq: np.ndarray
        The photon frequencies.
    phot_weight: np.ndarray
        The photon weights.
    phot_spec: np.ndarray
        The index for the spectrum the photons belong to.
    phot_nres: np.ndarry:
        The Res. values for the photons.
    phot_line_nres: np.ndarray
        The LineRes values for the photons.
    extract_nres: int
        The line number for the line to extract

    Returns
    -------
    spectrum: np.ndarray
        The spectrum where photon weights have been binned.
    """

    assert(len(phot_freq) == len(phot_weight))
    assert(len(phot_freq) == len(phot_spec))

    nphotons = phot_freq.shape[0]
    output = nphotons // 10

    for p in range(nphotons):
        if p % output == 0:
            print(" - Photon binning in progress: ", np.round(p / nphotons * 100.0), "% done")

        # Ignore photons not in frequency range

        if phot_freq[p] < spec_fmin or phot_freq[p] > spec_fmax:
            continue

        # TODO clean up
        # If a single transition is to be extracted, then we do that here. The
        # logic is kind of a mess..

        if extract_nres > -1:
            if phot_nres[p] == extract_nres or (phot_line_nres[p] == extract_nres and phot_nres[p] < 0):
                freq_index = np.abs(spectrum[:, 0] - phot_freq[p]).argmin()
                spectrum[freq_index, phot_spec[p]] += phot_weight[p]
            else:
                continue
        else:
            freq_index = np.abs(spectrum[:, 0] - phot_freq[p]).argmin()
            spectrum[freq_index, phot_spec[p]] += phot_weight[p]

    return spectrum


def create_spectrum_from_photon_weights(
    photons: np.ndarray, spectrum: np.ndarray, spec_norm: float, ncores: int, column_names: dict, nphotons: int = None,
    nspec: int = None, nbins: int = None, extract_nres: int = -1, dnorm: float = 100, use_jit: bool = True
) -> np.ndarray:
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
    ncores: [optional] int
        The number of cores which were used to generate the delay_dump filecycles.
    column_names: dict
        A dict containing the name of photons columns and their respective index
        into that array.
    nphotons: [optional] int
        The number of photons in the photons array.
    nspec: [optional] int
        The number of inclination angles to bin.
    nbins: [optional] int
        The number of frequency bins in the spectrum.
    extract_nres: [optional] int
        The line number for a specific line to be extracted.
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

    n = create_spectrum_from_photon_weights.__name__

    # Check the inputs and set default values in some cases

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

    # These are the vital quantities which are required for binning photons
    # depending on what the user wants

    phot_freq = photons[:, column_names["Freq."]]
    phot_weight = photons[:, column_names["Weight"]]
    phot_spec_index = photons[:, column_names["Spec."]].astype(int) + 1
    phot_nres = photons[:, column_names["Res."]].astype(int)
    phot_line_nres = photons[:, column_names["LineRes."]].astype(int)

    # Now bin the photons into the spectrum array - either using jit if requested
    # or using slow Python but with a pretty progress bar

    if use_jit:
        freq_min = np.min(spectrum[:, column_names["Freq."]])
        freq_max = np.max(spectrum[:, column_names["Freq."]])
        spectrum = jit_bin_weights(
            spectrum, freq_min, freq_max, phot_freq, phot_weight, phot_spec_index, phot_nres, phot_line_nres,
            extract_nres
        )
    else:
        raise NotImplemented("Please use the jit method for now.")

    # Now convert the binned weights into a flux
    # The flux is FLAMBDA -- ergs/s/cm/cm/A
    # The spectrum is normalized by the number of cores, as each core produces
    # NPHOTONS to contribute to the spectra, hence we take the "mean" value by
    # dividing through by the number of cores

    spectrum[:, 1:] /= ncores

    # For each frequency/wavelength bin
    for i in range(nbins - 1):
        # For each inclination for this frequency/wavelength bin
        for j in range(nspec):
            freq = spectrum[i, column_names["Freq."]]
            dfreq = spectrum[i + 1, column_names["Freq."]] - freq
            spectrum[i, j + 1] *= (freq ** 2 * 1e-8) / (dfreq * distance_normalisation * C)
        spectrum[i, 1:] *= spec_norm

    return spectrum


def create_filtered_spectrum(
    root: str, wd: str = ".", extract_nres: int = -1, fmin: float = None, fmax: float = None, nbins: int = 10000,
    dnorm: float = 100, spec_norm: float = 1, ncores: int = 1, logbins: bool = True, use_jit: bool = True
) -> np.ndarray:
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
    extract_nres: [optional] int
        The internal line number for a line to extract.
    fmin: [optional] float
        The smallest frequency bin.
    fmax: [optional] float
        The largest frequency bin
    nbins: [optional] int
        The number of frequency bins.
    spec_norm: float
        A normalization constant for the spectrum. Usually the number of photon
        cycles.
    ncores: [optional] int
        The number of cores which were used to generate the delay_dump file
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

    # extract are the quantities which we want to extract from the delay_dump
    # file -- i.e. it's the headings in the delay_dump file. Also included is
    # their index in the final dumped_photons array to be created

    extract = {
        "Freq.": 0, "Weight": 1, "Spec.": 2, "Scat.": 3, "RScat.": 4, "Orig.": 5, "Res.": 6, "LineRes.": 7
    }

    dumped_photons = read_delay_dump_file(root, wd, extract)
    nphotons = dumped_photons.shape[0]
    nspec = int(np.max(dumped_photons[:, 2])) + 1

    if not fmin:
        fmin = np.min(dumped_photons[:, 0])
    if not fmax:
        fmax = np.max(dumped_photons[:, 0])

    # Create the spectrum array and the frequency/wavelength bins

    filtered_spectrum = np.zeros((nbins, 1 + nspec))
    if logbins:
        bins = np.logspace(np.log10(fmin), np.log10(fmax), nbins, endpoint=True)
    else:
        bins = np.linspace(fmin, fmax, nbins, endpoint=True)
    filtered_spectrum[:, 0] = round_to_sig_figs(bins, 7)

    # This function now constructs a spectrum given the photon frequencies and
    # weights as well as any other normalization constants

    filtered_spectrum = create_spectrum_from_photon_weights(
        dumped_photons, filtered_spectrum, spec_norm, ncores, extract, nphotons, nspec, nbins,
        extract_nres=extract_nres, use_jit=use_jit, dnorm=dnorm
    )

    # Write out the spectrum to file
    # TODO: improve writing out spectrum, e.g. add headers, output lambda etc.

    if extract_nres > -1:
        oname = "{}/{}_el{}.delay_dump.spec".format(wd, root, extract_nres)
    else:
        oname = "{}/{}.delay_dump.spec".format(wd, root)

    np.savetxt(oname, filtered_spectrum)

    return filtered_spectrum
