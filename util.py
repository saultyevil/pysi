#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various utility functions which can be used to ease the
trials and tribulations of using Python and the Unix command line.

The following packages are used within the script,
    - sys
    - subprocess
    - platform
"""

from sys import exit
from subprocess import Popen, PIPE
from platform import system

def remove_data_sym_links(search_dir: str = "./", verbose: bool = False):
    """
    Search recursively from the specified directory search_dir for all symbolic
    links named data. The purpose of this script is to clean up the symbolic
    links if a directory is being uploaded to cloud storage or transferred using
    scp.

    This script will only work on Unix systems where the find command is
    available.

    Parameters
    ----------
    search_dir: str
        The starting directory to search recursively from for symbolic links
    verbose: bool [optional]
        Enable verbose output

    Returns
    -------
    ndel: int
        The number of symbolic links which were deleted
    """

    n = remove_data_sym_links.__name__
    ndel = 0

    os = system().lower()
    if os != "darwin" and os != "linux":
        print("{}: system {} unavailable", n, os)
        return ndel

    # - type l will only search for symbolic links
    cmd = "cd {}; find . -type l -name 'data'".format(search_dir)
    stdout, stderr = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True).communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")

    if stderr:
        print("{}: stderr".format(n))
        print(stderr)
    if stdout:
        if verbose:
            print("{}: deleting data symbolic links in the following directories:\n\n{}".format(n, stdout[:-1]))
    else:
        if verbose:
            print("{}: no data symlinks to delete".format(n))
        return ndel

    dirs = stdout.split()
    for i in range(len(dirs)):
        dir = search_dir + dirs[i][1:]
        cmd = "rm {}".format(dir)
        stdout, stderr = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True).communicate()
        stdout = stdout.decode("utf-8")
        if stdout and verbose:
            print(stdout)
        stderr = stderr.decode("utf-8")
        if stderr:
            print(stderr)
        else:
            ndel += 1

    return ndel
