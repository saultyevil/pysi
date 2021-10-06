#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a spectrum from a delay_dump file.

It is possible to create spectra which certain physical processes
removed. If the delay dump file is large, then use the SQL version of
the functions, otherwise the regular version is much faster.
"""

from copy import deepcopy
from sys import exit

import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import Column, Float, Integer
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import pypython
import pypython.constants as c
import pypython.dump.spectrum
import pypython.dump.wind

BOUND_FREE_NRES = 20000
UNFILTERED_SPECTRUM = -999


def read_dump(root, cd=".", mode_dev=False):
    """Process the photons which have been dumped to the delay_dump file.

    Parameters
    ----------
    root: str
        The root name of the simulation.
    cd: str
        The directory containing the simulation.
    mode_dev: bool [optional]
        Use when using the standard format currently in the main repository.

    Returns
    -------
    dumped_photons: pd.DataFrame
        An array containing the dumped photons with the quantities specified
        by the extract dict.
    """

    filename = "{}/{}.delay_dump".format(cd, root)

    # There are cases where LineRes. is not defined within the delay dump file,
    # i.e. in the regular dev version

    if mode_dev:
        names = {
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
            "Res.": np.int32
        }
    else:
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

    output = pd.read_csv(filename, names=list(names.keys()), dtype=names, delim_whitespace=True, comment="#")

    return output


def create_spectrum(root,
                    wd=".",
                    extract_nres=(UNFILTERED_SPECTRUM, ),
                    dumped_photons=None,
                    freq_bins=None,
                    freq_min=None,
                    freq_max=None,
                    n_bins=10000,
                    d_norm_pc=100,
                    spec_cycle_norm=1,
                    n_cores_norm=1,
                    logbins=True,
                    mode_dev=False,
                    output_numpy=False):
    """Create a spectrum for each inclination angle using the photons which
    have been dumped to the root.delay_dump file.

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
    mode_dev: bool [optional]
        If True, then LineRes. and Np will NOT be included when being read in
    output_numpy: [optional] bool
        If True, the spectrum will be a numpy array instead of a pandas data
        frame

    Returns
    -------
    filtered_spectrum: np.ndarray or pd.DataFrame
        A 2D array containing the frequency in the first column and the
        fluxes for each inclination angle in the other columns.
    """

    if type(extract_nres) != tuple:
        print("extract_nres is not a tuple but is of type {}".format(type(extract_nres)))
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
        dumped_photons = read_dump(root, wd, mode_dev)

    if mode_dev:
        line_res = deepcopy(dumped_photons["Res."].values.astype(int))
    else:
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
        if logbins:
            dump_spectrum[:, 0] = np.logspace(np.log10(freq_min), np.log10(freq_max), n_bins, endpoint=True)
        else:
            dump_spectrum[:, 0] = np.linspace(freq_min, freq_max, n_bins, endpoint=True)

    # This function now constructs a spectrum given the photon frequencies and
    # weights as well as any other normalization constants

    freq_max = np.max(dump_spectrum[:, 0])
    freq_min = np.min(dump_spectrum[:, 0])

    dump_spectrum = pypython.dump.spectrum.bin_photon_weights(dump_spectrum, freq_min, freq_max,
                                                              dumped_photons["Freq."].values,
                                                              dumped_photons["Weight"].values,
                                                              dumped_photons["Spec."].values.astype(int) + 1,
                                                              dumped_photons["Res."].values.astype(int), line_res,
                                                              extract_nres, logbins)

    dump_spectrum[:, 1:] /= n_cores_norm

    dump_spectrum = pypython.dump.spectrum.convert_weight_to_flux(dump_spectrum, spec_cycle_norm, d_norm_pc)

    # Remove the first and last bin, consistent with Python

    n_bins -= 2
    dump_spectrum = dump_spectrum[1:-1, :]

    dump_spectrum, inclinations = pypython.dump.spectrum.write_delay_dump_spectrum_to_file(root,
                                                                                           wd,
                                                                                           dump_spectrum,
                                                                                           extract_nres,
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


def create_spectrum_process_breakdown(root,
                                      wl_min,
                                      wl_max,
                                      n_cores_norm=1,
                                      spec_cycle_norm=1,
                                      wd=".",
                                      nres=None,
                                      mode_line_res=True):
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

    df = read_dump(root, cd=wd)
    s = pypython.Spectrum(root, wd)

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
        contribution_names.append("ES only")
        contribution_names.append("FF only")
        contribution_names.append("BF only")
    else:
        contributions.append(df[df["Res."] == -2])
        contributions.append(df[df["Res."] > 20000])
        contribution_names.append("FF only")
        contribution_names.append("BF only")

    # Create each individual spectrum

    created_spectra = [s]
    for contribution in contributions:
        created_spectra.append(
            create_spectrum(root,
                            wd,
                            dumped_photons=contribution,
                            freq_min=pypython.physics.angstrom_to_hz(wl_max),
                            freq_max=pypython.physics.angstrom_to_hz(wl_min),
                            n_cores_norm=n_cores_norm,
                            spec_cycle_norm=spec_cycle_norm))
    n_spec = len(created_spectra)

    # dict comprehension to use contribution_names as the keys and the spectra
    # as the values

    return {contribution_names[i]: created_spectra[i] for i in range(n_spec)}


Base = declarative_base()


class Photon(Base):
    """Photon object for SQL database."""
    __tablename__ = "Photons"
    id = Column(Integer, primary_key=True, autoincrement=True)
    np = Column(Integer)
    freq = Column(Float)
    wavelength = Column(Float)
    weight = Column(Float)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    scat = Column(Integer)
    rscat = Column(Integer)
    delay = Column(Integer)
    spec = Column(Integer)
    orig = Column(Integer)
    res = Column(Integer)
    lineres = Column(Integer)

    def __repr__(self):
        return str(self.id)


def get_photon_db(root, cd=".", dd_dev=False, commitfreq=1000000):
    """Create or open a database to store the delay_dump file in an easier to
    query data structure.

    Parameters
    ----------
    root: str
        The root name of the simulation.
    cd: str [optional]
        The directory containing the simulation.
    dd_dev: bool [optional]
        Expect the delay_dump file to be in the format used in the main Python
        repository.
    commitfreq: int
        The frequency to which commit the database and avoid out-of-memory
        errors. If this number is too low, database creation will take a long
        time.

    Returns
    -------
    engine:
        The SQLalchemy engine.
    session:
        The SQLalchemy session.
    """

    engine = sqlalchemy.create_engine("sqlite:///{}.db".format(root))
    engine_session = sessionmaker(bind=engine)
    session = engine_session()

    if dd_dev:
        column_names = [
            "Freq", "Lambda", "Weight", "LastX", "LastY", "LastZ", "Scat", "RScat", "Delay", "Spec", "Orig", "Res"
        ]
    else:
        column_names = [
            "Np", "Freq", "Lambda", "Weight", "LastX", "LastY", "LastZ", "Scat", "RScat", "Delay", "Spec", "Orig",
            "Res", "LineRes"
        ]
    n_columns = len(column_names)

    try:
        session.query(Photon.weight).first()
    except SQLAlchemyError:
        print("{}.db does not exist, so creating now".format(root))
        with open(cd + "/" + root + ".delay_dump", "r") as infile:
            nadd = 0
            Base.metadata.create_all(engine)
            for n, line in enumerate(infile):
                if line.startswith("#"):
                    continue
                try:
                    values = [float(i) for i in line.split()]
                except ValueError:
                    print("Line {} has values which cannot be converted into a number".format(n))
                    continue
                if len(values) != n_columns:
                    print("Line {} has unknown format with {} columns:\n{}".format(n, len(values), line))
                    continue
                if dd_dev:
                    session.add(
                        Photon(np=int(n),
                               freq=values[0],
                               wavelength=values[1],
                               weight=values[2],
                               x=values[3],
                               y=values[4],
                               z=values[5],
                               scat=int(values[6]),
                               rscat=int(values[7]),
                               delay=int(values[8]),
                               spec=int(values[9]),
                               orig=int(values[10]),
                               res=int(values[11]),
                               lineres=int(values[11])))
                else:
                    session.add(
                        Photon(np=int(values[0]),
                               freq=values[1],
                               wavelength=values[2],
                               weight=values[3],
                               x=values[4],
                               y=values[5],
                               z=values[6],
                               scat=int(values[7]),
                               rscat=int(values[8]),
                               delay=int(values[9]),
                               spec=int(values[10]),
                               orig=int(values[11]),
                               res=int(values[12]),
                               lineres=int(values[13])))

                nadd += 1
                if nadd > commitfreq:
                    session.commit()
                    nadd = 0

        session.commit()

    return engine, session
