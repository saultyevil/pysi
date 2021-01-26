#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spectrum object
"""


import numpy as np
from pathlib import Path
from scipy.signal import convolve, boxcar
from typing import List, Union
import textwrap

from .util import get_root


UNITS_LNU = "erg/s/Hz"
UNITS_FNU = "erg/s/cm^-2/Hz"
UNITS_FLAMBDA = "erg/s/cm^-2/A"


class Spectrum:
    """
    A class to store PYTHON .spec and .log_spec files.
    The PYTHON spectrum is read in and stored within a dict, where each column
    name is a key and the data is stored as a numpy array.
    """
    def __init__(self, root: str, cd: str = ".", logspec: bool = False):
        """
        Initialise a Spectrum object. This method will construct the file path
        of the spectrum file given the root, containing directory and whether
        the logarithmic spectrum is used or not. The spectrum is then read in.
         
        Parameters
        ----------
        root: str
            The root name of the model.
        cd: str [optional]
            The directory containing the model.
        logspec: bool [optional]
            Read in the logarithmic spectrum if True, otherwise linear.
        """""

        self.root = root
        self.logspec = logspec

        if cd[-1] != "/":
            cd += "/"
        self.filepath = cd + root
        if self.logspec:
            self.filepath += ".log_spec"
        else:
            self.filepath += ".spec"

        self.spectrum = {}
        self.columns = []
        self.inclinations = []
        self.units = "unknown"

        # The next method call reads in the spectrum and initializes the above
        # member variables

        self.read_spectrum()

    def read_spectrum(self, delim: str = None):
        """
        Read in a spectrum file given in self.filepath. The spectrum is stored
        as a dictionary in self.spectrum where each key is the name of the
        columns.

        Parameters
        ----------
        delim: str [optional]
            A custom delimiter, useful for reading in files which have sometimes
            between delimited with commas instead of spaces.
        """

        try:
            with open(self.filepath, "r") as f:
                spectrum_file = f.readlines()
        except IOError:
            print("Unable to open spectrum for file path " + self.filepath)
            exit(1)  # todo: error code

        # Read in the spectrum file, ignoring empty lines and lines which have
        # been commented out by # at the beginning

        spectrum = []

        for line in spectrum_file:
            line = line.strip()
            if delim:
                line = line.split(delim)
            else:
                line = line.split()
            # todo: determine the units elsewhere
            if "Units:" in line:
                self.units = line[4][1:-1]
            if len(line) == 0 or line[0] == "#":
                continue
            spectrum.append(line)

        # Extract the header columns of the spectrum. This assumes the first
        # read line in the spectrum is the header. If no header is found, then
        # the columns are numbered instead

        header = []

        if spectrum[0][0] == "Freq." or spectrum[0][0] == "Lambda":
            for i, column_name in enumerate(spectrum[0]):
                if column_name[0] == "A":
                    j = column_name.find("P")
                    column_name = column_name[1:j]
                header.append(column_name)
            spectrum = np.array(spectrum[1:], dtype=np.float)
        else:
            header = np.arange(len(spectrum[0]))

        # Add the actual spectrum to the spectrum dictionary, the keys of the
        # dictionary are the column names as given above. Set the header and
        # also the inclination angles here as well

        self.columns = header
        for i, column_name in enumerate(header):
            self.spectrum[column_name] = spectrum[:, i]
        for col in header:
            if col.isdigit() and col not in self.inclinations:
                self.inclinations.append(col)
        self.columns = tuple(self.columns)
        self.inclinations = tuple(self.inclinations)

    def smooth_spectrum(self):
        """Smooth the spectrum flux/luminosity bins."""
        raise NotImplementedError

    def __getitem__(self, key):
        """Return an array in the spectrum dictionary when indexing."""
        return self.spectrum[key]

    def __setitem__(self, key, value):
        """Allows to modify the arrays in the spectrum dictionary."""
        self.spectrum[key] = value

    def __str__(self):
        """Print the basic details about the spectrum."""
        return textwrap.dedent("""\
            PYTHON spectrum for model {}
            File path: {}
            Headers: {}""".format(self.root, self.filepath, self.columns))


def get_spectrum_files(
    root: str = None, wd: str = ".", ignore_delay_dump_spec: bool = True
) -> List[str]:
    """
    Find root.spec files recursively in the provided directory.

    Parameters
    ----------
    root: str [optional]
        If root is set, then only .spec files with this root name will be
        returned
    wd: str [optional]
        The path to recursively search from
    ignore_delay_dump_spec: [optional] bool
        When True, root.delay_dump.spec files will be ignored

    Returns
    -------
    spec_files: List[str]
        The file paths of various .spec files
    """

    spec_files = []

    for filename in Path(wd).glob("**/*.spec"):

        fname = str(filename)

        if ignore_delay_dump_spec and fname.find(".delay_dump.spec") != -1:
            continue

        if root:
            t_root, wd = get_root(fname)
            if t_root == root:
                spec_files.append(fname)
            else:
                continue
        spec_files.append(fname)

    return spec_files
