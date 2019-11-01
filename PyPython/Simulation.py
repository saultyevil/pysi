#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various functions used monitor, check and run Python
simulations as well as run a grid of simulations.
"""


from typing import Union
from glob import glob


def check_convergence(root: str, wd: str = "./", return_converging: bool = False) -> Union[float, int]:
    """
    Check the convergence of a Python simulation by parsing the
    !!Check_convergence line in the Python diag file.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    wd: str [optional]
        The working directory of the Python simulation
    return_converging: bool [optional]
        Return the number of cells which are still converging instead of the
        number of cells which have converged

    Returns
    -------
    convergence: float or int
        The convergence fraction in the final cycle of the simulation. If this
        is -1, then a convergence fraction was not found.
    """

    n = check_convergence.__name__
    convergence = -1
    converging = -1

    diag_path = "{}/diag_{}/{}_0.diag".format(wd, root, root)
    try:
        with open(diag_path, "r") as f:
            diag = f.readlines()
    except IOError:
        try:
            diag_path = "diag_{}/{}_0.diag".format(root, root)
            with open(diag_path, "r") as f:
                diag = f.readlines()
        except IOError:
            print("{}: unable to find {}_0.diag file".format(n, root))
            return convergence

    convergence_per_cycle = []
    for line in diag:  # Unfortunately, there are multiple names for this line ;_;
        if line.find("converged") != -1 and line.find("converging") != -1:
            # if len(line.split()) != 16:
            #     continue
            convergence_per_cycle.append(line)
            line = line.split()
            try:
                tstr = line[2].replace("(", "").replace(")", "")
                convergence = float(tstr)
            except ValueError:
                continue
            try:
                tstr = line[6].replace("(", "").replace(")", "")
                converging = float(tstr)
            except ValueError:
                continue

    if convergence == -1:
        print("{}: unable to parse convergence from diag file {}".format(n, diag_path))
    if 0 > convergence > 1:
        print("{}: convergence {} is not sane".format(n, convergence))

    if return_converging:
        return converging
    return convergence


def error_summary(root: str, wd: str = "./", ncores: int = -1, print_errors: bool = False) -> dict:
    """
    Return a dictionary containing each error found in the error summary for
    each processor for a Python simulation.

    TODO: make a dict for each processes?

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    wd: str [optional]
        The working directory of the Python simulation
    ncores: int [optional]
        If this is provided, then only the first ncores processes will be
        checked for errors
    print_errors: bool [optional]
        Print the error summary to screen

    Returns
    -------
    total_errors: dict
        A dictionary containing the total errors over all processors. The keys
        are the error messages and the values are the number of times that
        error occurred.
    """

    n = error_summary.__name__
    max_read_errors = 100
    total_errors = {}

    glob_directory = "{}/diag_{}/{}_*.diag".format(wd, root, root)
    diag_files = glob(glob_directory)

    if ncores > 0:
        ndiag = ncores
    else:
        ndiag = len(diag_files)

    if ndiag == 0:
        print("{}: no diag files found in path {}".format(n, glob_directory))
        return total_errors

    broken_diag = []

    for i in range(ndiag):
        diag = diag_files[i]
        try:
            with open(diag, "r") as f:
                lines = f.readlines()
        except IOError:
            broken_diag.append(i)
            continue

        # Find the final error summary: look over the lines list in reverse
        # TODO: may be possible to take into account multiple error summaries
        j = -1
        for k, line in reversed(list(enumerate(lines))):
            if line.find("Error summary: End of program") != -1:
                j = k
                break

        if j == -1:
            print("{}: unable to find error summary, returning empty dict ".format(n))
            return total_errors

        # Now parse out the separate errors and add them the total errors dict
        # errors = lines[j:j + max_read_errors]
        errors = lines[j:]
        for line in errors:
            if len(line) == 0:
                continue
            words = line.split()
            try:
                w0 = words[0]
            except IndexError:
                print("{}: index error when trying to process line '{}' for {}"
                      .format(n, " ".join(words), diag_files[i]))
                broken_diag.append(i)
                break
            if w0.isdigit():
                try:
                    error_count = int(words[0])
                except ValueError:
                    continue
                error_message = " ".join(words[2:])
                try:
                    total_errors[error_message] += error_count
                except KeyError:
                    total_errors[error_message] = error_count

    if len(broken_diag) > 0:
        print("{}: unable to find error summaries for the following diag files".format(n))
        for k in range(len(broken_diag)):
            print("  {}_{}.diag".format(root, broken_diag[k]))

    if print_errors:
        print("Total errors reported from {} processors for {}:\n"
              .format(len(diag_files) - len(broken_diag), root))
        for key in total_errors.keys():
            print("  {:6d} -- {}".format(total_errors[key], key))

    return total_errors
