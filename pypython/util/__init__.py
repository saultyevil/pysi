#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Basic utility functions for making life easier.

Includes functions for creating slurm files, bash "run scripts", as well
as getting system info and a logging utility.
"""

import subprocess
import textwrap
from typing import List

from psutil import cpu_count

from pypython import find, get_root_name


def get_cpu_count(enable_smt=False):
    """Return the number of CPU cores which can be used when running a Python
    simulation. By default, this will only return the number of physical cores
    and will ignore logical threads, i.e. in Intel terms, it will not count the
    hyperthreads.

    Parameters
    ----------
    enable_smt: [optional] bool
        Return the number of logical cores, which includes both physical and
        logical (SMT/hyperthreads) threads.

    Returns
    -------
    n_cores: int
        The number of available CPU cores
    """

    n_cores = 0

    try:
        n_cores = cpu_count(logical=enable_smt)
    except NotImplementedError:
        print("unable to determine number of CPU cores, psutil.cpu_count not implemented for your system")

    return n_cores


def get_file_len(filename):
    """Slowly count the number of lines in a file.
    todo: update to jit_open or some other more efficient method

    Parameters
    ----------
    filename: str
        The file name and path of the file to count the lines of.

    Returns
    -------
    The number of lines in the file.
    """
    with open(filename, "r") as f:
        for i, l in enumerate(f):
            pass

    return i + 1


def run_command(command, fp=".", verbose=False):
    """Run a shell command.

    Parameters
    ----------
    command: List[str] or str
        The shell command to run. Must either be a single string to call a
        program, or a list of the program and arguments for the program.
    fp: str [optional]
        The directory to run the command in.
    verbose: bool
        Print stdout to the screen.
    """

    sh = subprocess.run(command, capture_output=True, cwd=fp)
    if verbose:
        print(sh.stdout.decode("utf-8"))
    if sh.stderr:
        print("stderr reported errors:\n", sh.stderr.decode("utf-8"))

    return sh


# Logging functions ------------------------------------------------------------

LOGFILE = None


def close_logfile(logfile=None):
    """Close a log file for writing - this will either use the log file provided
    or will attempt to close the global log file.

    Parameters
    ----------
    logfile: io.TextIO, optional
        An external log file object"""

    global LOGFILE

    if logfile:
        logfile.close()
    elif LOGFILE:
        LOGFILE.close()
    else:
        print("No logfile to close but somehow got here? ahhh disaster")

    return


def init_logfile(logfile_name, use_global_log=True):
    """Initialise a logfile global variable.

    Parameters
    ----------
    logfile_name: str
        The name of the logfile to initialise
    use_global_log: bool, optional
        If this is false, a object for a logfile will be returned instead.
    """

    global LOGFILE

    if use_global_log:
        if LOGFILE:
            print("logfile already initialised as {}".format(LOGFILE.name))
            return
        LOGFILE = open(logfile_name, "a")
    else:
        logfile = open(logfile_name, "a")
        return logfile

    return


def log(message, logfile=None):
    """Log a message to screen and to the log file provided or the global log
    file.

    Parameters
    ----------
    message: str
        The message to log to screen and file
    logfile: io.TextIO, optional
        An open file object which is the logfile to log to. If this is not
        provided, then the global logfile.
    """

    print(message)

    if logfile:
        logfile.write("{}\n".format(message))
    elif LOGFILE:
        LOGFILE.write("{}\n".format(message))
    else:
        return

    return


def logsilent(message, logfile=None):
    """Log a message to the logfile, but do not print it to the screen.

    Parameters
    ----------
    message: str
        The message to log to file
    logfile: io.TextIO, optional
        An open file object which is the logfile to log to. If this is not
        provided, then the global logfile
    """

    if logfile:
        logfile.write("{}\n".format(message))
    elif LOGFILE:
        LOGFILE.write("{}\n".format(message))
    else:
        return

    return
