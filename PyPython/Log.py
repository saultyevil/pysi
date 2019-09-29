#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains functions related to logging to both a log file and to
the screen.
"""

LOGFILE = None


def init_logfile(logfile_name: str, use_global_log: bool = True):
    """
    Initialise a logfile global variable

    Parameters
    ----------
    logfile_name: str
        The name of the logfile to initialise
    use_global_log: bool, optional
        If this is false, a object for a logfile will be returned instead.
    """

    global LOGFILE
    n = init_logfile.__name__

    if use_global_log:
        if LOGFILE:
            print("{}: logfile already initialised as {}".format(n, LOGFILE.name))
            return
        LOGFILE = open(logfile_name, "a")
    else:
        logfile = open(logfile_name, "a")
        return logfile

    return


def close_logfile(logfile=None) -> None:
    """
    Close a log file for writing - this will either use the log file provided
    or will attempt to close the global log file.

    Parameters
    ----------
    logfile: io.TextIO, optional
        An external log file object
    """

    global LOGFILE
    n = close_logfile.__name__

    if logfile:
        logfile.close()
    elif LOGFILE:
        LOGFILE.close()
    else:
        print("{}: No logfile to close? ahhh".format(n))

    return


def log(message: str, logfile=None) -> None:
    """
    Log a message to screen and to the log file provided or the global log file.

    Parameters
    ----------
    message: str
        The message to log to screen and file
    logfile: io.TextIO, optional
        An open file object which is the logfile to log to. If this is not
        provided, then the global logfile
    """

    print(message)

    if logfile:
        logfile.write("{}\n".format(message))
    elif LOGFILE:
        LOGFILE.write("{}\n".format(message))
    else:
        return

    return
