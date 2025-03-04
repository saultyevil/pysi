"""Basic utility functions.

Functions which offer utility and are used project-wide belong in this
file.
"""

from pathlib import Path

import numpy


def read_file_with_header(file: str) -> tuple[list[str], numpy.ndarray]:
    """Read a file and extract the header and the contents.

    The first line of the file is assumed to be a header which is split into a
    list of strings. The remaining part of the file is read into a numpy
    array.

    Parameters
    ----------
    file : str
        The name of the file to read.

    Returns
    -------
    tuple[list[str], numpy.ndarray]
        A tuple containing the header and the contents of the file.

    """
    with Path(file).open(encoding="utf-8") as file_in:
        header = file_in.readline().strip().split()

    contents = numpy.loadtxt(file, skiprows=1)

    return header, contents


def remove_suffix_from_string(string: str, suffix: str) -> str:
    """Remove the provided suffix from a string.

    The string is only updated if the suffix is at the end of the string.

    Parameters
    ----------
    string : str
        The string to remove the suffix from.
    suffix : str
        The suffix to remove.

    Returns
    -------
    str
        The updated string.

    """
    if string.endswith(suffix):
        return string[: -len(suffix)]
    return string


def split_root_and_directory(root: str | Path, directory: str | Path) -> tuple[str, Path]:
    """Split the root name from its parent directories.

    This exists for cases where the root name and parents directory are provided
    in a single string or Path instance. Typically this is can be done for
    Wind and Spectrum classes.

    If a Wind and Spectrum class are initialised using the old style of
    providing the root name and directory separately, this function is not
    required.

    Parameters
    ----------
    root : str | Path
        The root name of the simulation.
    directory : str | Path
        The directory containing the simulation.

    Returns
    -------
    root: str
        The root name of the simulation.
    directory: Path
        A Path instance of the directory containing the simulation.

    Raises
    ------
    ValueError
        When the root is not a string or Path instance.

    """
    root_path = Path(root)
    if root_path.is_file():
        root = root_path.stem
        directory = root_path.parents
    elif isinstance(root, str):
        root = remove_suffix_from_string(root, ".pf")
        directory = Path(directory)
    else:
        raise ValueError(f"Root must be a string or filepath, not {type(root)}")

    return root, directory
