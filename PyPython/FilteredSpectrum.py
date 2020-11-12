#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various utility functions for use with the reverberation
mapping part of Python. It seems to mostly house functions designed to create
a spectrum from the delay_dump output.
"""

from .Constants import PARSEC, C
from .Conversion import hz_to_angstrom
from .PythonUtils import file_len
from .SpectrumUtils import read_spec_file, get_spec_inclinations

import numpy as np
from numba import jit


@jit(nopython=True)
def jit_bin_photon_weights(
    spectrum: np.ndarray, freq_min: float, freq_max: float, photon_freqs: np.ndarray, photon_wghts: np.ndarray,
    photon_spc_i: np.ndarray, photon_nres: np.ndarray, photon_lnres: np.ndarray, extract_nres: tuple
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
    photon_wghts: np.ndarray
        The photon weights.
    photon_spc_i: np.ndarray
        The index for the spectrum the photons belong to.
    photon_nres: np.ndarry:
        The Res. values for the photons.
    photon_lnres: np.ndarray
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

    for p in range(n_photons):

        # Ignore photons not in frequency range

        if photon_freqs[p] < freq_min or photon_freqs[p] > freq_max:
            continue

        # TODO clean up
        # If a single transition is to be extracted, then we do that here. Note
        # that if nres < 0, then it was a continuum scattering event

        if extract_nres[0] > -1:
            for i in range(n_extract):
                if photon_nres[p] == extract_nres[i] or (photon_lnres[p] == extract_nres[i] and photon_nres[p] < 0):
                    freq_index = np.abs(spectrum[:, 0] - photon_freqs[p]).argmin()
                    spectrum[freq_index, photon_spc_i[p]] += photon_wghts[p]
                    break
        else:
            freq_index = np.abs(spectrum[:, 0] - photon_freqs[p]).argmin()
            spectrum[freq_index, photon_spc_i[p]] += photon_wghts[p]

    return spectrum


def construct_spectrum_from_photons(
        delay_dump_photons: np.ndarray, spectrum: np.ndarray, spec_norm: float, ncores: int, column_index_dict: dict,
        nspec: int = None, nbins: int = None, extract_nres: tuple = (-1,), d_norm_pc: float = 100,
        use_jit: bool = True
) -> np.ndarray:
    """
    Construct a spectrum from the weights of the provided photons. If nphotons,
    nspec or nbins are not provided, then the function will automatically detect
    what these numbers are. There should be no NaN values in any of the arrays
    which are provided.

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
    nspec: [optional] int
        The number of inclination angles to bin.
    nbins: [optional] int
        The number of frequency bins in the spectrum.

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

    n = construct_spectrum_from_photons.__name__

    # Check the inputs and set default values in some cases

    if np.isnan(delay_dump_photons).any():
        print("{}: There are NaN values in photon array which is not good".format(n))
        raise ValueError

    if not nbins:
        nbins = spectrum.shape[0]
    if not nspec:
        nspec = np.max(delay_dump_photons[:, column_index_dict["Spec."]]) + 1

    # These are the vital quantities which are required for binning photons
    # depending on what the user wants

    photon_freqs = delay_dump_photons[:, column_index_dict["Freq."]]
    photon_wghts = delay_dump_photons[:, column_index_dict["Weight"]]
    photon_spc_i = delay_dump_photons[:, column_index_dict["Spec."]].astype(int) + 1
    photon_nres  = delay_dump_photons[:, column_index_dict["Res."]].astype(int)
    photon_lnres = delay_dump_photons[:, column_index_dict["LineRes."]].astype(int)

    # Now bin the photons into the spectrum array - either using jit if requested
    # or using slow Python but with a pretty progress bar

    if use_jit:
        freq_min = np.min(spectrum[:, column_index_dict["Freq."]])
        freq_max = np.max(spectrum[:, column_index_dict["Freq."]])
        if type(extract_nres) == list:
            extract_nres = tuple(extract_nres)
        elif type(extract_nres) != tuple:
            extract_nres = (extract_nres, )
        spectrum = jit_bin_photon_weights(
            spectrum, freq_min, freq_max, photon_freqs, photon_wghts, photon_spc_i, photon_nres, photon_lnres,
            extract_nres
        )
    else:
        raise NotImplemented("Please use the jit method for now.")

    # Now we need to convert the binned weights into flux
    # The flux is in units flambda ergs/s/cm/cm/A
    # The spectrum is normalized by the number of cores, as each core produces
    # NPHOTONS which contribute to the spectrum. Hence, we take the "mean" value
    # of each processes by dividing through by the number of cores

    spectrum[:, 1:] /= ncores

    d_norm_cm = 4 * np.pi * (d_norm_pc * PARSEC) ** 2  # TODO remind me why there's a 4 pi here.. F = L / 4pi R ?

    # For each frequency/wavelength bin

    for i in range(nbins - 1):

        # For each inclination for this frequency/wavelength bin

        for j in range(nspec):
            freq = spectrum[i, column_index_dict["Freq."]]
            dfreq = spectrum[i + 1, column_index_dict["Freq."]] - freq
            spectrum[i, j + 1] *= (freq ** 2 * 1e-8) / (dfreq * d_norm_cm * C)  # TODO check why this is 1e-8

        # Fixes the case where less than the specified number of spectral cycles
        # were run, which would mean not all of the flux has been generated
        # just yet: spec_norm >= 1.

        spectrum[i, 1:] *= spec_norm

    return spectrum


def read_delay_dump_file(
    root: str, wd: str, extract_dict: dict, mode_line_res: bool = True
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

    n = read_delay_dump_file.__name__

    file = "{}/{}.delay_dump".format(wd, root)
    n_lines = file_len(file)

    # There are cases where LineRes. is not defined within the delay dump file,
    # i.e. in the regular dev version

    if mode_line_res:
        file_columns = {
            "Np": 0, "Freq.": 1, "Lambda": 2, "Weight": 3, "Last X": 4, "Last Y": 5, "Last Z": 6, "Scat.": 7,
            "RScat.": 8, "Delay": 9, "Spec.": 10, "Orig.": 11, "Res.": 12, "LineRes.": 13
        }
    else:
        file_columns = {
            "Np": 0, "Freq.": 1, "Lambda": 2, "Weight": 3, "Last X": 4, "Last Y": 5, "Last Z": 6, "Scat.": 7,
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


def create_filtered_spectrum(
    root: str, wd: str = ".", extract_nres: tuple = (-1,), freq_min: float = None, freq_max: float = None,
    n_bins: int = 10000, d_norm_pc: float = 100, spec_cycle_norm: float = 1, n_cores: int = 1, logbins: bool = True,
    mode_line_res: bool = True, use_jit: bool = True
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

    if mode_line_res:
        to_extract_dict = {
            "Freq.": 0, "Weight": 1, "Spec.": 2, "Scat.": 3, "RScat.": 4, "Orig.": 5, "Res.": 6, "LineRes.": 7
        }
    else:
        to_extract_dict = {
            "Freq.": 0, "Weight": 1, "Spec.": 2, "Scat.": 3, "RScat.": 4, "Orig.": 5, "Res.": 6
        }

    # Read the delay dump file and determine the minimum and maximum frequency
    # of the spectrum if it hasn't been provided so we can make the frequency
    # bins for the spectrum

    dumped_photons = read_delay_dump_file(root, wd, to_extract_dict, mode_line_res)
    n_spec = int(np.max(dumped_photons[:, to_extract_dict["Spec."]])) + 1

    if not freq_min:
        freq_min = np.min(dumped_photons[:, 0])
    if not freq_max:
        freq_max = np.max(dumped_photons[:, 0])

    # Create the spectrum array and the frequency/wavelength bins

    spectrum = np.zeros((n_bins, 1 + n_spec))
    if logbins:
        spectrum[:, 0] = np.logspace(np.log10(freq_min), np.log10(freq_max), n_bins, endpoint=True)
    else:
        spectrum[:, 0] = np.linspace(freq_min, freq_max, n_bins, endpoint=True)

    # This function now constructs a spectrum given the photon frequencies and
    # weights as well as any other normalization constants

    spectrum = construct_spectrum_from_photons(
        dumped_photons, spectrum, spec_cycle_norm, n_cores, to_extract_dict, n_spec, n_bins, extract_nres=extract_nres,
        d_norm_pc=d_norm_pc, use_jit=use_jit
    )

    # Write out the spectrum to file

    if extract_nres[0] > -1:
        output_fname = "{}/{}_line".format(wd, root)
        for line in extract_nres:
            output_fname += "_{}".format(line)
        output_fname += ".delay_dump.spec"
    else:
        output_fname = "{}/{}.delay_dump.spec".format(wd, root)

    f = open(output_fname, "w")

    f.write("# Flux Flambda [erg / s / cm^2 / A at {} pc\n".format(d_norm_pc))

    try:
        full_spec = read_spec_file("{}/{}.spec".format(wd, root))
        header = get_spec_inclinations(full_spec)
    except:
        header = np.arange(0, n_spec)

    # Write out the header of the output file
    # TODO should I make this a comment?

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

    if extract_nres[0] > -1:
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

    return spectrum
