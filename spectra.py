#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various functions used to create visualisations of the
synthetic spectra which are output from Python. This includes utility functions
from finding these spectrum files, as well as functions to create plots of the
spectra.

The following packages are used within the file,
    - typing
    - shutil
    - numpy
    - pandas
    - scipy.signal
    - pathlib
    - matplotlib
    - subprocess
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Union

def find_specs(path: str = "./") -> List[str]:
    """
    Find root.spec files recursively in provided directory path.

    Parameters
    ----------
    path: str [optional]
        The path to recursively search from

    Returns
    -------
    spec_files: List[str]
        The file paths of various .spec files
    """

    spec_files = []
    for filename in Path(path).glob("**/*.spec"):
        spec_files.append(str(filename))

    return spec_files


def read_spec(file_name: str, delim: str = None, numpy: bool = False) -> Union[np.ndarray, pd.DataFrame]:
    """
    Read in data from an external file, line by line whilst ignoring comments.
        - Comments begin with #
        - The default delimiter is assumed to be a space

    Parameters
    ----------
    file_name: str
        The directory path to the spec file to be read in
    delim: str, optional
        The delimiter between values in the file, by default a space is assumed
    numpy:bool, optional
        If True, a Numpy array of strings will be used instead :-(

    Returns
    -------
    lines: np.ndarray or pd.DataFrame
        The .spec file as a Numpy array or a Pandas DataFrame
    """

    n = read_spec.__name__

    try:
        with open(file_name, "r") as f:
            flines = f.readlines()
    except IOError:
        raise Exception("{}: cannot open spec file {}".format(n, file_name))

    lines = []
    for i in range(len(flines)):
        line = flines[i].strip()
        if delim:
            line = line.split(delim)
        else:
            line = line.split()
        if len(line) > 0:
            if line[0] == "#":
                continue
            if line[0] == "Freq.":
                for j in range(len(line)):
                    if line[j][0] == "A":
                        index = line[j].find("P")
                        line[j] = line[j][1:index]
            lines.append(line)

    if numpy:
        return np.array(lines)

    return pd.DataFrame(lines[1:], columns=lines[0])

