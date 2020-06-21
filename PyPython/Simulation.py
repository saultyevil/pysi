#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various functions used monitor, check and run Python
simulations as well as run a grid of simulations.
"""

from copy import copy
from typing import Union, List, Tuple
from glob import glob


def check_convergence(
    root: str, wd: str = "./", return_per_cycle: bool = False, return_converging: bool = False
) -> List[float]:
    """
    Check the convergence of a Python simulation by parsing the
    !!Check_convergence line in the Python diag file.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    wd: str [optional]
        The working directory of the Python simulation
    return_per_cycle: bool [optional]
        Return the convergence fraction for each cycle
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
    brief_summary_len = 9
    convergence = [-1]
    converging = [-1]

    # use glob to find the first diag file
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

    prev = ""
    convergence_per_cycle = []
    converging_per_cycle = []

    for i in range(len(diag)):
        line = diag[i]
        if line.find("converged") != -1 and line.find("converging") != -1:
            # Skip if the convergence statistic is from the brief run summary
            if prev.find("Convergence statistics for the wind after the ionization calculation:") != -1:
                i += brief_summary_len
                continue

            line = line.split()

            try:
                convergence = float(line[2].replace("(", "").replace(")", ""))
                convergence_per_cycle.append(convergence)
            except ValueError:
                convergence_per_cycle.append(-1)
                continue

            try:
                converging = float(line[6].replace("(", "").replace(")", ""))
                converging_per_cycle.append(converging)
            except ValueError:
                converging_per_cycle.append(-1)
                continue

        prev = copy(line)

    if return_converging:
        if return_per_cycle:
            return converging_per_cycle
        return converging

    if return_per_cycle:
        return convergence_per_cycle
    return convergence


def check_convergence_breakdown(
    root: str, wd: str = "./"
) -> Tuple[List[Union[float, int]], List[Union[float, int]], List[Union[float, int]], List[Union[float, int]]]:
    """
    Returns a break down in terms of the number of cells which have passed
    the convergence checks on radiation temperature, electron temperature and
    heating and cooling balance.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    wd: str [optional]
        The working directory of the Python simulation
    """

    n = check_convergence_breakdown.__name__
    brief_summary_len = 7
    n_tr = []
    n_te = []
    n_hc = []
    n_te_max = []
    file_found = False

    # use glob to find the first diag file
    diag_path = "{}/diag_{}/{}_0.diag".format(wd, root, root)
    try:
        with open(diag_path, "r") as f:
            diag = f.readlines()
        file_found = True
    except IOError:
        pass

    if not file_found:
        print("{}: unable to find {}_0.diag file".format(n, root))
        return n_tr, n_te, n_hc, n_te_max

    for i in range(len(diag)):
        line = diag[i]
        if line.find("t_r") != -1 and line.find("t_e(real)") != -1 and line.find("hc(real)") != -1:
            if diag[i - 2].find("Convergence statistics for the wind after the ionization calculation:") != -1:
                i += brief_summary_len
                continue

            line = line.split()

            try:
                ncells = int(diag[i - 1].split()[9])
                n_tr.append(int(line[2]) / ncells)
                n_te.append(int(line[4]) / ncells)
                n_te_max.append(int(line[6]) / ncells)
                n_hc.append(int(line[8]) / ncells)
            except ValueError:
                n_tr.append(-1)
                n_te.append(-1)
                n_te_max.append(-1)
                n_hc.append(-1)

    return n_tr, n_te, n_te_max, n_hc


def error_summary(
    root: str, wd: str = "./", ncores: int = -1, print_errors: bool = False
) -> dict:
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
            words = line.split()
            if len(words) == 0:
                continue
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
