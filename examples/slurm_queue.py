#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Search recursively for .slurm files and add them to the slurm queue.
"""

import argparse as ap
from subprocess import Popen, PIPE
from typing import List, Tuple
from pathlib import Path


def split_path_and_filename(
    filepath: str
) -> Tuple[str, str]:
    """Extract the slurm file name and the directory it is in from the entire
    file path.

    Parameters
    ----------
    filepath: str
        The filepath to split the slurm file name and directory from."""

    assert(type(filepath) == str)

    slash_idx = -1
    extension_idx = filepath.find(".slurm")
    for i in range(extension_idx - 1, -1, -1):
        if filepath[i] == "/":
            slash_idx = i
            break

    slurm_file = filepath[slash_idx + 1:]
    if slash_idx > -1:
        directory = filepath[:slash_idx]
    else:
        directory = "."

    return slurm_file, directory


def add_files_to_slurm_queue(
    slurm_files: List[str]
) -> None:
    """Add a bunch of slurm files to the slurm queue. Uses subprocess and so
    only works on macOS and Linux.

    Parameters
    ----------
    slurm_files: List[str]
        A list containing the slurm file paths to add to the queue."""

    rc = []

    for filepath in slurm_files:
        file, cd = split_path_and_filename(filepath)
        cmd = "cd {}; sbatch {}".format(cd, file)
        sh = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
        stdout, stderr = sh.communicate()
        if stderr:
            print(stderr.decode("utf-8"))
        rc.append(stdout.decode("utf-8").split()[-1])

    if len(rc) > 1:
        print("Submitted batch jobs" + ", ".join(rc[:-1]) + " and " + rc[-1])
    else:
        print("Submitted batch job", rc[0])

    return


def find_slurm_files(
    filepath: str = "."
) -> List[str]:
    """Searches recursively from the calling direction for files which end with
    the extension *.slurm and returns a list of the found files.

    Parameters
    ----------
    filepath: str, optional
        The directory of which to search recursively from.

    Returns
    -------
    slurm_files: List[str]
        A list containing the relative paths of the slurm files."""

    slurm_files = []

    for filename in Path(filepath).glob("**/*.slurm"):
        filename = str(filename)
        if filename[0] == "/":
            filename = filename[1:]
        slurm_files.append(filename)

    if len(slurm_files) == 0:
        print("No .slurm files were found.")
        exit(1)

    slurm_files = sorted(slurm_files, key=str.lower)

    return slurm_files


def setup() -> bool:
    """Parse the command line for run time arguments.

    Returns
    -------
    add_to_queue: bool
        Indicates whether to add the slurm files to the queue or not."""

    p = ap.ArgumentParser(description=__doc__)
    p.add_argument(
        "-a", "--add_to_queue", action="store_true", default=False, help="Add slurm files to the slurm queue."
    )

    args = p.parse_args()

    return args.add_to_queue


def main() -> None:
    """Main function of the script."""

    add_to_queue = setup()
    slurm_files = find_slurm_files()

    print("The following {} .slurm files will be added to the queue:\n".format(len(slurm_files)))
    for n, file in enumerate(slurm_files):
        print("{}\t{}".format(n + 1, file))
    print("")

    if add_to_queue:
        add_files_to_slurm_queue(slurm_files)

    return


if __name__ == "__main__":
    main()
