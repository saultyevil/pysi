#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various utility functions for use with the reverberation
mapping part of Python. It seems to mostly house functions designed to create
a spectrum from the delay_dump output.
"""

from .error import EXIT_FAIL
from .constants import PARSEC, C
from .conversion import hz_to_angstrom
from .pythonUtil import file_len
from .spectrumUtil import read_spectrum, get_spectrum_inclinations

import pandas as pd
from copy import deepcopy
import numpy as np
from numba import jit
from typing import Union, Tuple


UNFILTERED_SPECTRUM = -999

spectrum_columns_dict_line_res = dumped_photon_columns_dict_lres = {
    "Freq.": 0, "Weight": 1, "Spec.": 2, "Scat.": 3, "RScat.": 4, "Orig.": 5, "Res.": 6, "LineRes.": 7
}

spectrum_columns_dict_nres = dumped_photon_columns_dict_nres = {
    "Freq.": 0, "Weight": 1, "Spec.": 2, "Scat.": 3, "RScat.": 4, "Orig.": 5, "Res.": 6
}


def write_delay_dump_spectrum_to_file(
    root: str, wd: str, spectrum: np.ndarray, extract_nres: tuple, n_spec:int, n_bins: int, d_norm_pc: float,
    return_inclinations: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Write the generated delay dump spectrum to file

    Parameters
    ----------
    root: str
        The root name of the model.
    wd: str
        The directory containing the model.
    spectrum: np.ndarray
        The delay dump spectrum.
    extract_nres: tuple
        The internal line number for a line to extract.
    n_spec: int
        The number of spectral inclination angles
    n_bins: int
        The number of frequency bins.
    d_norm_pc: float
        The distance normalization of the spectrum.
    return_inclinations: [optional] bool
        Return the inclination angles (if any) of the spectrum

    Returns
    -------
    spectrum: np.ndarray
        The delay dump spectrum.
    inclinations: [optional] np.ndarray
        An array of the inclination angles of the spectrum.
    """

    if extract_nres[0] != UNFILTERED_SPECTRUM:
        fname = "{}/{}_line".format(wd, root)
        for line in extract_nres:
            fname += "_{}".format(line)
        fname += ".delay_dump.spec"
    else:
        fname = "{}/{}.delay_dump.spec".format(wd, root)

    f = open(fname, "w")

    f.write("# Flux Flambda [erg / s / cm^2 / A at {} pc\n".format(d_norm_pc))

    try:
        full_spec = read_spectrum("{}/{}.spec".format(wd, root))
        inclinations = get_spectrum_inclinations(full_spec)
    except IOError:
        inclinations = np.arange(0, n_spec)

    header = deepcopy(inclinations)

    # Write out the header of the output file

    f.write("{:12s} {:12s}".format("Freq.", "Lambda"))
    for h in header:
        f.write(" {:12s}".format(h))
    f.write("\n")

    # Now write out the spectrum

    for i in range(n_bins):
        freq = spectrum[i, 0]
        wl_angstrom = hz_to_angstrom(freq)
        f.write("{:12e} {:12e}".format(freq, wl_angstrom))
        for j in range(spectrum.shape[1] - 1):
            f.write(" {:12e}".format(spectrum[i, j + 1]))
        f.write("\n")

    f.close()

    # If some lines are being extracted, then we can calculate their luminosities
    # and write these out to file too
    # TODO: update to allow multiple lines to be written out at once

    if extract_nres[0] != UNFILTERED_SPECTRUM and len(extract_nres) == 1:
        output_fname = "{}/{}_line".format(wd, root)
        for line in extract_nres:
            output_fname += "_{}".format(line)
        output_fname += ".line_luminosity.diag"
        f = open(output_fname, "w")
        f.write("Line luminosities -- units [erg / s]\n")
        for i in range(spectrum.shape[1] - 2):
            flux = np.sum(spectrum[:, i + 1])
            lum = 4 * np.pi * (d_norm_pc * PARSEC) ** 2 * flux
            f.write("Spectrum {} : L = {} erg / s\n".format(header[i], lum))
        f.close()

    if return_inclinations:
        return spectrum, inclinations
    else:
        return spectrum


def read_delay_dump(
    root: str, extract_dict: dict, wd: str = ".", mode_line_res: bool = True
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
    extract_dict: dict
        A dictionary containing the names of the quantities to extract and their
        index to place them into dumped_photons array.
    mode_line_res: bool [optional]
        If True, then LineRes. will be included when being read in

    Returns
    -------
    dumped_photons: np.ndarray
        An array containing the dumped photons with the quantities specified
        by the extract dict.
    """

    n = read_delay_dump.__name__

    file = "{}/{}.delay_dump".format(wd, root)
    n_lines = file_len(file)

    # There are cases where LineRes. is not defined within the delay dump file,
    # i.e. in the regular dev version

    if mode_line_res:
        file_columns = {
            "Np": 0, "Freq.": 1, "Lambda": 2, "Weight": 3, "LastX": 4, "LastY": 5, "LastZ": 6, "Scat.": 7,
            "RScat.": 8, "Delay": 9, "Spec.": 10, "Orig.": 11, "Res.": 12, "LineRes.": 13
        }
    else:
        file_columns = {
            "Np": 0, "Freq.": 1, "Lambda": 2, "Weight": 3, "LastX": 4, "LastY": 5, "LastZ": 6, "Scat.": 7,
            "RScat.": 8, "Delay": 9, "Spec.": 10, "Orig.": 11, "Res.": 12
        }

    n_cols = len(file_columns)
    n_extract = len(extract_dict)
    output = np.zeros((n_lines, n_extract))
    output[:, :] = np.nan   # because some lines will not be photons mark them as NaN

    f = open(file, "r")

    for i in range(n_lines):
        line = f.readline().split()
        if len(line) == n_cols:
            for j, e in enumerate(extract_dict):
                output[i, j] = float(line[file_columns[e]])

    f.close()

    # Remove the NaN lines from the photons array using bitshift magic

    output = output[~np.isnan(output).any(axis=1)]

    return output


@jit(nopython=True)
def jit_bin_photon_weights(
    spectrum: np.ndarray, freq_min: float, freq_max: float, photon_freqs: np.ndarray, photon_weights: np.ndarray,
    photon_spc_i: np.ndarray, photon_nres: np.ndarray, photon_line_nres: np.ndarray, extract_nres: tuple, logbins: bool
):
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
    photon_freqs: np.ndarray
        The photon frequencies.
    photon_weights: np.ndarray
        The photon weights.
    photon_spc_i: np.ndarray
        The index for the spectrum the photons belong to.
    photon_nres: np.ndarry:
        The Res. values for the photons.
    photon_line_nres: np.ndarray
        The LineRes values for the photons.
    extract_nres: int
        The line number for the line to extract

    Returns
    -------
    spectrum: np.ndarray
        The spectrum where photon weights have been binned.
    """

    n_extract = len(extract_nres)
    n_photons = photon_freqs.shape[0]
    n_bins = spectrum.shape[0]

    if logbins:
        d_freq = (np.log10(freq_max) - np.log10(freq_min)) / n_bins
    else:
        d_freq = (freq_max - freq_min) / n_bins


    for p in range(n_photons):

        # Ignore photons not in frequency range

        if photon_freqs[p] < freq_min or photon_freqs[p] > freq_max:
            continue

        if logbins:
            k = int((np.log10(photon_freqs[p]) - np.log10(freq_min)) / d_freq)
        else:
            k = int((photon_freqs[p] - freq_min) / d_freq)

        # If a single transition is to be extracted, then we do that here. Note
        # that if nres < 0, then it was a continuum scattering event

        if extract_nres[0] != UNFILTERED_SPECTRUM:
            # Loop over each nres which we want to extract
            for i in range(n_extract):
                # If it's last interaction is the nres we want, then extract
                if photon_nres[p] == extract_nres[i]:
                    spectrum[k, photon_spc_i[p]] += photon_weights[p]
                    break
                # Or if it's "belongs" to the nres we want and it's last interaction
                # was a continuum scatter, then extract
                elif photon_line_nres[p] == extract_nres[i] and photon_nres[p] < 0:
                    spectrum[k, photon_spc_i[p]] += photon_weights[p]
                    break
        else:
            spectrum[k, photon_spc_i[p]] += photon_weights[p]

    return spectrum


def normalize_spectrum(
    spectrum: np.ndarray, column_index_dict: dict, spec_cycle_norm: float, d_norm_pc: float
):
    """
    Re-normalize the photon weight bins into a Flux per unit wavelength.

    Parameters
    ----------
    spectrum: np.ndarray
        The spectrum array containing the frequency and weight bins.
    column_index_dict: dict
        A dict containing the name of photons columns and their respective index
        into that array.
    spec_cycle_norm: float
        The spectrum normalization amount - usually the number of spectrum
    d_norm_pc: [optional] float
        The distance normalization for the flux calculation in parsecs. By
        default this is 100 parsecs.

    Returns
    -------
    spectrum: np.ndarray
        The renormalized spectrum.
    """

    n_bins = spectrum.shape[0]
    n_spec = spectrum.shape[1] - 1  # We do -1 as the 1st column is the frequency of the bin
    d_norm_cm = 4 * np.pi * (d_norm_pc * PARSEC) ** 2

    # For each frequency/wavelength bin

    for i in range(n_bins - 1):

        # For each inclination for this frequency/wavelength bin

        for j in range(n_spec):
            freq = spectrum[i, column_index_dict["Freq."]]
            dfreq = spectrum[i + 1, column_index_dict["Freq."]] - freq
            spectrum[i, j + 1] *= (freq ** 2 * 1e-8) / (dfreq * d_norm_cm * C)

        # Fixes the case where less than the specified number of spectral cycles
        # were run, which would mean not all of the flux has been generated
        # just yet: spec_norm >= 1.

        spectrum[i, 1:] *= spec_cycle_norm

    return spectrum


def construct_filtered_spectrum(
    delay_dump_photons: np.ndarray, spectrum: np.ndarray, spec_norm: float, ncores: int, column_index_dict: dict,
    extract_nres: tuple = (UNFILTERED_SPECTRUM,), d_norm_pc: float = 100, use_jit: bool = True, logbins: bool = True
) -> np.ndarray:
    """
    Construct a spectrum from the weights of the provided photons. If nphotons,
    then the function will automatically detect what this should be (I hope).
    There should be no NaN values in any of the arrays which are provided.

    Parameters
    ----------
    delay_dump_photons: np.ndarray (nphotons, 3)
        An array containing the photon frequency, weight and spectrum number.
    spectrum: np.ndarray (nbins, nspec)
        An array containing the frequency bins and empty bins for each
        inclination angle.
    spec_norm: float
        The spectrum normalization amount - usually the number of spectrum
    ncores: [optional] int
        The number of cores which were used to generate the delay_dump filecycles.
    column_index_dict: dict
        A dict containing the name of photons columns and their respective index
        into that array.
    extract_nres: [optional] int
        The line number for a specific line to be extracted.
    d_norm_pc: [optional] float
        The distance normalization for the flux calculation in parsecs. By
        default this is 100 parsecs.
    use_jit: [optional] bool
        If True, JIT will be used to speed up the photon binning

    Returns
    -------
    spectrum: np.ndarray (nbins, nspec)
        The constructed spectrum in units of F_lambda erg/s/cm/cm/A.
    """

    n = construct_filtered_spectrum.__name__

    # Check the inputs and set default values in some cases

    if np.isnan(delay_dump_photons).any():
        print("{}: There are NaN values in photon array which is not good".format(n))
        raise ValueError

    # These are the vital quantities which are required for binning photons
    # depending on what the user wants

    photon_freqs = delay_dump_photons[:, column_index_dict["Freq."]]
    photon_weights = delay_dump_photons[:, column_index_dict["Weight"]]
    photon_spec_index = delay_dump_photons[:, column_index_dict["Spec."]].astype(int) + 1
    photon_nres  = delay_dump_photons[:, column_index_dict["Res."]].astype(int)
    photon_line_nres = delay_dump_photons[:, column_index_dict["LineRes."]].astype(int)

    # Now bin the photons into the spectrum array - either using jit if requested
    # or using slow Python but with a pretty progress bar

    if use_jit:
        freq_min = np.min(spectrum[:, column_index_dict["Freq."]])
        freq_max = np.max(spectrum[:, column_index_dict["Freq."]])
        spectrum = jit_bin_photon_weights(
            spectrum, freq_min, freq_max, photon_freqs, photon_weights, photon_spec_index, photon_nres, photon_line_nres,
            extract_nres, logbins
        )
    else:
        raise NotImplemented("Please use the jit method for now.")

    # The spectrum needs to be normalized by the number of processes used to
    # generate the photons as each process produces and transports
    # NPHOTONS / np_mpi_global which contribute to the spectrum. Hence, we
    # really are taking the mean value of each processes spectrum generation by
    # dividing through by the number of cores

    spectrum[:, 1:] /= ncores

    # Now we need to convert the binned weights into flux
    # The flux is in units flambda ergs/s/cm/cm/A

    spectrum = normalize_spectrum(spectrum, column_index_dict, spec_norm, d_norm_pc)

    return spectrum


def create_filtered_spectrum(
    root: str, wd: str = ".", extract_nres: tuple = (UNFILTERED_SPECTRUM,), freq_min: float = None, freq_max: float = None,
    n_bins: int = 10000, d_norm_pc: float = 100, spec_cycle_norm: float = 1, n_cores: int = 1, logbins: bool = True,
    mode_line_res: bool = True, output_numpy: bool = False, use_jit: bool = True
) -> Union[np.ndarray, pd.DataFrame]:
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
    n_cores: [optional] int
        The number of cores which were used to generate the delay_dump file
    logbins: [optional] bool
        If True, the frequency bins are spaced equally in log space. Otherwise
        the bins are in linear space.
    mode_line_res: bool [optional]
        If True, then LineRes. will be included when being read in
    output_numpy: [optional] bool
        If True, the spectrum will be a numpy array instead of a pandas data
        frame
    use_jit: [optional] bool
        Enable using jit to try and speed up the photon binning.

    Returns
    -------
    filtered_spectrum: np.ndarray
        A 2D array containing the frequency in the first column and the
        fluxes for each inclination angle in the other columns.
    """

    n = create_filtered_spectrum.__name__

    # Turn extract_nres into a tuple if it isn't - this is to avoid pain later
    # after reading in the file

    if type(extract_nres) != tuple:
        print("{}: extract_nres is not a tuple but is of type {}".format(n, type(extract_nres)))
        exit(EXIT_FAIL)

    # extract are the quantities which we want to extract from the delay_dump
    # file -- i.e. it's the headings in the delay_dump file. Also included is
    # their index in the final dumped_photons array to be created
    # Uses the global values and deepcopy to make sure we don't modify them
    
    if mode_line_res:
        spectrum_columns_dict = deepcopy(dumped_photon_columns_dict_lres)
    else:
        spectrum_columns_dict = deepcopy(dumped_photon_columns_dict_nres)

    # Read the delay dump file and determine the minimum and maximum frequency
    # of the spectrum if it hasn't been provided so we can make the frequency
    # bins for the spectrum

    dumped_photons = read_delay_dump(root, spectrum_columns_dict, wd, mode_line_res)
    n_spec = int(np.max(dumped_photons[:, spectrum_columns_dict["Spec."]])) + 1

    if not freq_min:
        freq_min = np.min(dumped_photons[:, spectrum_columns_dict["Freq."]])
    if not freq_max:
        freq_max = np.max(dumped_photons[:, spectrum_columns_dict["Freq."]])

    # Create the spectrum array and the frequency/wavelength bins

    spectrum = np.zeros((n_bins, 1 + n_spec))
    if logbins:
        spectrum[:, 0] = np.logspace(np.log10(freq_min), np.log10(freq_max), n_bins, endpoint=True)
    else:
        spectrum[:, 0] = np.linspace(freq_min, freq_max, n_bins, endpoint=True)

    # This function now constructs a spectrum given the photon frequencies and
    # weights as well as any other normalization constants

    spectrum = construct_filtered_spectrum(
        dumped_photons, spectrum, spec_cycle_norm, n_cores, spectrum_columns_dict, extract_nres=extract_nres,
        d_norm_pc=d_norm_pc, use_jit=use_jit, logbins=logbins
    )

    # Write out the spectrum to file

    spectrum, inclinations = write_delay_dump_spectrum_to_file(
        root, wd, spectrum, extract_nres, n_spec, n_bins, d_norm_pc, return_inclinations=True
    )

    if output_numpy:
        return spectrum
    else:
        lamda = np.reshape(C / spectrum[:, 0] * 1e8, (n_bins, 1))
        spectrum = np.append(lamda, spectrum, axis=1)
        df = pd.DataFrame(spectrum, columns=["Lambda", "Freq."] + inclinations)
        return df
