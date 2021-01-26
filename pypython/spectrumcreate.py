#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various utility functions for use with the reverberation
mapping part of Python. It seems to mostly house functions designed to create
a spectrum from the delay_dump output.
"""

from .error import EXIT_FAIL
from physics.constants import PARSEC, C
from .conversion import hz_to_angstrom
from .util import get_file_len
from .spectumutil import read_spectrum, get_spectrum_inclinations
from .conversion import angstrom_to_hz

import pandas as pd
from copy import deepcopy
import numpy as np
from numba import jit
from typing import Union, Tuple


BOUND_FREE_NRES = 20000
UNFILTERED_SPECTRUM = -999

spectrum_columns_dict_line_res = dumped_photon_columns_dict_lres = {
    "Freq.": 0, "Weight": 1, "Spec.": 2, "Scat.": 3, "RScat.": 4, "Orig.": 5, "Res.": 6, "LineRes.": 7
}

spectrum_columns_dict_nres = dumped_photon_columns_dict_nres = {
    "Freq.": 0, "Weight": 1, "Spec.": 2, "Scat.": 3, "RScat.": 4, "Orig.": 5, "Res.": 6
}


def write_delay_dump_spectrum_to_file(
    root: str, wd: str, spectrum: np.ndarray, extract_nres: tuple, n_spec: int, n_bins: int, d_norm_pc: float,
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
        for i in range(spectrum.shape[1] - 1):
            flux = np.sum(spectrum[:, i + 1])
            lum = 4 * np.pi * (d_norm_pc * PARSEC) ** 2 * flux
            f.write("Spectrum {} : L = {} erg / s\n".format(header[i], lum))
        f.close()

    if return_inclinations:
        return spectrum, inclinations
    else:
        return spectrum


def read_delay_dump(
    root: str, column_names: dict, wd: str = ".", mode_line_res: bool = True
) -> pd.DataFrame:
    """
    Process the photons which have been dumped to the delay_dump file. tqdm is
    used to display a progress bar of the current progress.

    Parameters
    ----------
    root: str
        The root name of the simulation.
    wd: str
        The directory containing the simulation.
    column_names: dict
        A dictionary containing the names of the quantities to extract and their
        index to place them into dumped_photons array.
    mode_line_res: bool [optional]
        If True, then LineRes. will be included when being read in

    Returns
    -------
    dumped_photons: pd.DataFrame
        An array containing the dumped photons with the quantities specified
        by the extract dict.
    """

    n = read_delay_dump.__name__

    file = "{}/{}.delay_dump".format(wd, root)
    n_lines = get_file_len(file)

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
    n_extract = len(column_names)
    output = np.zeros((n_lines, n_extract))
    output[:, :] = np.nan   # because some lines will not be photons mark them as NaN

    f = open(file, "r")

    for i in range(n_lines):
        line = f.readline().split()
        if len(line) == n_cols:
            for j, e in enumerate(column_names):
                output[i, j] = float(line[file_columns[e]])

    f.close()

    # Remove the NaN lines from the photons array using bitshift magic

    output = output[~np.isnan(output).any(axis=1)]
    output = pd.DataFrame(output, columns=column_names.keys())

    return output


def convert_weight_to_flux(
    spectrum: np.ndarray, spec_cycle_norm: float, d_norm_pc: float
):
    """
    Re-normalize the photon weight bins into a Flux per unit wavelength.

    spec_cycle_norm fixes the case where less than the specified number of
    spectral cycles were run, which would mean not all of the flux has been
    generated just yet: spec_norm >= 1.

    Parameters
    ----------
    spectrum: np.ndarray
        The spectrum array containing the frequency and weight bins.
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
    n_spec = spectrum.shape[1] - 1
    d_norm_cm = 4 * np.pi * (d_norm_pc * PARSEC) ** 2

    for i in range(n_bins - 1):
        for j in range(n_spec):
            freq = spectrum[i, 0]
            dfreq = spectrum[i + 1, 0] - freq
            spectrum[i, j + 1] *= (freq ** 2 * 1e-8) / (dfreq * d_norm_cm * C)

        spectrum[i, 1:] *= spec_cycle_norm

    return spectrum


@jit(nopython=True)
def bin_photon_weights(
    spectrum: np.ndarray, freq_min: float, freq_max: float, photon_freqs: np.ndarray, photon_weights: np.ndarray,
    photon_spc_i: np.ndarray, photon_nres: np.ndarray, photon_line_nres: np.ndarray, extract_nres: tuple, logbins: bool
):
    """
    Bin the photons into frequency bins using jit to attempt to speed everything
    up.

    BOUND_FREE_NRES = NLINES = 20000 has been hardcoded. Any values of nres
    larger than BOUND_FREE_NRES is a bound-free continuum event. If this value
    is changed in Python, then this value needs updating.

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
    logbins: bool
        Use frequency bins spaced in log space.

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

        if photon_freqs[p] < freq_min or photon_freqs[p] > freq_max:
            continue

        if logbins:
            k = int((np.log10(photon_freqs[p]) - np.log10(freq_min)) / d_freq)
        else:
            k = int((photon_freqs[p] - freq_min) / d_freq)

        if k < 0:
            k = 0
        elif k > n_bins - 1:
            k = n_bins - 1

        # If a single transition is to be extracted, then we do that here. Note
        # that if nres < 0 or nres > NLINES, then it was a continuum scattering
        # event

        if extract_nres[0] != UNFILTERED_SPECTRUM:
            # Loop over each nres which we want to extract
            for i in range(n_extract):
                # If it's last interaction is the nres we want, then extract
                if photon_nres[p] == extract_nres[i]:
                    spectrum[k, photon_spc_i[p]] += photon_weights[p]
                    break
                # Or if it's "belongs" to the nres we want and it's last interaction
                # was a continuum scatter, then extract
                elif photon_line_nres[p] == extract_nres[i] and photon_nres[p] < 0 or photon_nres[p] > BOUND_FREE_NRES:
                    spectrum[k, photon_spc_i[p]] += photon_weights[p]
                    break
        else:
            spectrum[k, photon_spc_i[p]] += photon_weights[p]

    return spectrum


def construct_spectrum_from_weights(
    delay_dump_photons: pd.DataFrame, spectrum: np.ndarray, n_cores_norm: int,
    extract_nres: tuple = (UNFILTERED_SPECTRUM,), logbins: bool = True
) -> np.ndarray:
    """
    Construct a spectrum from the weights of the provided photons. If nphotons,
    then the function will automatically detect what this should be (I hope).
    There should be no NaN values in any of the arrays which are provided.

    The spectrum needs to be normalized by the number of processes used to
    generate the photons as each process produces and transports
    NPHOTONS / np_mpi_global which contribute to the spectrum. Hence, we
    really are taking the mean value of each processes spectrum generation by
    dividing through by the number of cores

    Parameters
    ----------
    delay_dump_photons: np.ndarray (nphotons, 3)
        An array containing the photon frequency, weight and spectrum number.
    spectrum: np.ndarray (nbins, nspec)
        An array containing the frequency bins and empty bins for each
        inclination angle.
    n_cores_norm: [optional] int
        The number of cores which were used to generate the delay_dump filecycles.
    extract_nres: [optional] int
        The line number for a specific line to be extracted.
    logbins: [optional] bool
        Use frequency bins spaced equally in log space.

    Returns
    -------
    spectrum: np.ndarray (nbins, nspec)
        The constructed spectrum in units of F_lambda erg/s/cm/cm/A.
    """

    freq_min = np.min(spectrum[:, 0])
    freq_max = np.max(spectrum[:, 0])

    spectrum = bin_photon_weights(
        spectrum,
        freq_min,
        freq_max,
        delay_dump_photons["Freq."].values,
        delay_dump_photons["Weight"].values,
        delay_dump_photons["Spec."].values.astype(int) + 1,
        delay_dump_photons["Res."].values.astype(int),
        delay_dump_photons["LineRes."].values.astype(int),
        extract_nres,
        logbins
    )

    spectrum[:, 1:] /= n_cores_norm

    return spectrum


def create_spectrum(
    root: str, wd: str = ".", extract_nres: tuple = (UNFILTERED_SPECTRUM,), dumped_photons: pd.DataFrame = None,
    freq_bins: np.ndarray = None, freq_min: float = None, freq_max: float = None, n_bins: int = 10000,
    d_norm_pc: float = 100, spec_cycle_norm: float = 1, n_cores_norm: int = 1, logbins: bool = True,
    mode_line_res: bool = True, output_numpy: bool = False
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
    logbins: [optional] bool
        If True, the frequency bins are spaced equally in log space. Otherwise
        the bins are in linear space.
    mode_line_res: bool [optional]
        If True, then LineRes. will be included when being read in
    output_numpy: [optional] bool
        If True, the spectrum will be a numpy array instead of a pandas data
        frame

    Returns
    -------
    filtered_spectrum: np.ndarray or pd.DataFrame
        A 2D array containing the frequency in the first column and the
        fluxes for each inclination angle in the other columns.
    """

    n = create_spectrum.__name__

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

    # If the frequency bins have been provided, we need to do some checks to make
    # sure they're sane

    if freq_bins is not None:
        if type(freq_bins) != np.ndarray:
            freq_bins = np.array(freq_bins, dtype=np.float64)
        is_increasing = np.all(np.diff(freq_bins) > 0)
        if not is_increasing:
            raise ValueError("{}: the values for the frequency bins provided are not increasing".format(n))
        n_bins = len(freq_bins)

    if dumped_photons is None:
        dumped_photons = read_delay_dump(root, spectrum_columns_dict, wd, mode_line_res)

    n_spec = int(np.max(dumped_photons["Spec."].values)) + 1
    spectrum = np.zeros((n_bins, 1 + n_spec))

    if freq_bins is not None:
        spectrum[:, 0] = freq_bins
    else:
        if not freq_min:
            freq_min = np.min(dumped_photons["Freq."])
        if not freq_max:
            freq_max = np.max(dumped_photons["Freq."])
        if logbins:
            spectrum[:, 0] = np.logspace(np.log10(freq_min), np.log10(freq_max), n_bins, endpoint=True)
        else:
            spectrum[:, 0] = np.linspace(freq_min, freq_max, n_bins, endpoint=True)

    # This function now constructs a spectrum given the photon frequencies and
    # weights as well as any other normalization constants

    spectrum = construct_spectrum_from_weights(
        dumped_photons, spectrum, n_cores_norm, extract_nres=extract_nres, logbins=logbins
    )

    spectrum = convert_weight_to_flux(
        spectrum, spec_cycle_norm, d_norm_pc
    )

    # Remove the first and last bin, consistent with Python

    n_bins -= 2
    spectrum = spectrum[1:-1, :]

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


def create_spectrum_process_breakdown(
    root: str, wl_min: float, wl_max: float, n_cores_norm: int = 1, spec_cycle_norm: float = 1, wd: str = ".",
    nres: int = None, mode_line_res: bool = True
) -> dict:
    """
    Get the spectra for the different physical processes which contribute to a
    spectrum. If nres is provided, then only a specific interaction will be
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
    wd: str [optional]
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

    if mode_line_res:
        ex = spectrum_columns_dict_line_res
    else:
        ex = spectrum_columns_dict_nres

    df = read_delay_dump(root, ex, wd=wd)
    s = read_spectrum(wd + "/" + root + ".spec")

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
        contributions.append(df[df["Res."] == nres])
        contribution_names.append("Res." + str(nres))
        if mode_line_res:
            contributions.append(df[df["LineRes."] == nres])
            contribution_names.append("LineRes." + str(nres))
    else:
        tmp = df[df["Res."] <= 20000]
        contributions.append(tmp[tmp["Res."] >= 0])
        contribution_names.append("Res.")
        if mode_line_res:
            tmp = df[df["LineRes."] <= 20000]
            contributions.append(tmp[tmp["LineRes."] >= 0])
            contribution_names.append("LineRes.")

    # Extract the scattered spectrum, which is every continuum scatter

    contributions.append(df[df["Res."] == -1])
    contribution_names.append("Scattered")

    # Extract pure BF, FF and ES events, unless we're in Res. mode, which extracts
    # last scatters

    if mode_line_res:
        contributions.append(df[df["LineRes."] == -1])
        contributions.append(df[df["LineRes."] == -2])
        contributions.append(df[df["LineRes."] > 20000])
        contribution_names.append("ES")
        contribution_names.append("FF")
        contribution_names.append("BF")
    else:
        contributions.append(df[df["Res."] == -2])
        contributions.append(df[df["Res."] > 20000])
        contribution_names.append("FF")
        contribution_names.append("BF")

    # Create each individual spectrum

    created_spectra = [s]
    for contribution in contributions:
        created_spectra.append(
            create_spectrum(
                root, wd, dumped_photons=contribution, freq_min=angstrom_to_hz(wl_max), freq_max=angstrom_to_hz(wl_min),
                n_cores_norm=n_cores_norm, spec_cycle_norm=spec_cycle_norm
            )
        )
    n_spec = len(created_spectra)

    # dict comprehension to use contribution_names as the keys and the spectra
    # as the values

    return {contribution_names[i]: created_spectra[i] for i in range(n_spec)}
