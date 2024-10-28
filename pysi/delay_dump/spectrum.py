#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""""

from copy import deepcopy

import numpy as np
from numba import jit

import pysi
import pysi.constants as c
import pysi.delay_dump as delay_dump


def write_delay_dump_spectrum_to_file(
    root, wd, spectrum, extract_nres, n_spec, n_bins, d_norm_pc, return_inclinations=False
):
    """Write the generated delay dump spectrum to file.

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

    if extract_nres[0] != delay_dump.UNFILTERED_SPECTRUM:
        fname = "{}/{}_line".format(wd, root)
        for line in extract_nres:
            fname += "_{}".format(line)
        fname += ".delay_dump.spec"
    else:
        fname = "{}/{}.delay_dump.spec".format(wd, root)

    f = open(fname, "w")

    f.write("# Flux Flambda [erg / s / cm^2 / A at {} pc\n".format(d_norm_pc))

    try:
        full_spec = pysi.Spectrum(root, wd)
        inclinations = list(full_spec.inclinations)
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
        wl_angstrom = pysi.physics.hz_to_angstrom(freq)
        f.write("{:12e} {:12e}".format(freq, wl_angstrom))
        for j in range(spectrum.shape[1] - 1):
            f.write(" {:12e}".format(spectrum[i, j + 1]))
        f.write("\n")

    f.close()

    # If some lines are being extracted, then we can calculate their luminosities
    # and write these out to file too
    # TODO: update to allow multiple lines to be written out at once

    if extract_nres[0] != delay_dump.UNFILTERED_SPECTRUM and len(extract_nres) == 1:
        output_fname = "{}/{}_line".format(wd, root)
        for line in extract_nres:
            output_fname += "_{}".format(line)
        output_fname += ".line_luminosity.diag"
        f = open(output_fname, "w")
        f.write("Line luminosities -- units [erg / s]\n")
        for i in range(spectrum.shape[1] - 1):
            flux = np.sum(spectrum[:, i + 1])
            lum = 4 * np.pi * (d_norm_pc * c.PARSEC) ** 2 * flux
            f.write("Spectrum {} : L = {} erg / s\n".format(header[i], lum))
        f.close()

    if return_inclinations:
        return spectrum, inclinations
    else:
        return spectrum


def convert_weight_to_flux(spectrum, spec_cycle_norm, d_norm_pc):
    """Re-normalize the photon weight bins into a Flux per unit wavelength.

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
    d_norm_cm = 4 * np.pi * (d_norm_pc * c.PARSEC) ** 2

    for i in range(n_bins - 1):
        for j in range(n_spec):
            freq = spectrum[i, 0]
            d_freq = spectrum[i + 1, 0] - freq
            spectrum[i, j + 1] *= (freq**2 * 1e-8) / (d_freq * d_norm_cm * c.VLIGHT)

        spectrum[i, 1:] *= spec_cycle_norm

    return spectrum


@jit(nopython=True)
def bin_photon_weights(
    spectrum,
    freq_min,
    freq_max,
    photon_freqs,
    photon_weights,
    photon_spc_i,
    photon_nres,
    photon_line_nres,
    extract_nres,
    logbins,
):
    """Bin the photons into frequency bins using jit to attempt to speed
    everything up.

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
    extract_nres: int or np.ndarray or list
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

        if extract_nres[0] != delay_dump.UNFILTERED_SPECTRUM:
            # Loop over each nres which we want to extract
            for i in range(n_extract):
                # If it's last interaction is the nres we want, then extract
                if photon_nres[p] == extract_nres[i]:
                    spectrum[k, photon_spc_i[p]] += photon_weights[p]
                    break
                # Or if it's "belongs" to the nres we want and it's last interaction
                # was a continuum scatter, then extract
                elif photon_line_nres[p] == extract_nres[i]:
                    spectrum[k, photon_spc_i[p]] += photon_weights[p]
                    break
        else:
            spectrum[k, photon_spc_i[p]] += photon_weights[p]

    return spectrum
