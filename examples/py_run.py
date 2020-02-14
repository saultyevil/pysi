#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run a batch of Python models. This script searches recursively for Python pfs
(disregarding anything which is py_wind.pf or .out.pf files) and executes a
number of commands depending on what is requested by the user using command
line flags.

The script can also be run in a directory containing only one Python pf.
"""


import argparse as ap
import time
import datetime
from copy import copy
from sys import exit
from shutil import copyfile
from typing import List
from subprocess import Popen, PIPE
from py_plot import plot

from PyPython import Grid
from PyPython import Simulation
from PyPython import PythonUtils
from PyPython.Log import log, init_logfile, close_logfile
from PyPython import Quotes
from PyPython.Error import EXIT_FAIL


CONVERGED = \
    r"""
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
                                             _
  ___ ___  _ ____   _____ _ __ __ _  ___  __| |
 / __/ _ \| '_ \ \ / / _ \ '__/ _` |/ _ \/ _` |
| (_| (_) | | | \ V /  __/ | | (_| |  __/ (_| |
 \___\___/|_| |_|\_/ \___|_|  \__, |\___|\__,_|
                              |___/
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""

NOT_CONVERGED = \
    r"""
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
             _                                                 _
 _ __   ___ | |_    ___ ___  _ ____   _____ _ __ __ _  ___  __| |
| '_ \ / _ \| __|  / __/ _ \| '_ \ \ / / _ \ '__/ _` |/ _ \/ _` |
| | | | (_) | |_  | (_| (_) | | | \ V /  __/ | | (_| |  __/ (_| |
|_| |_|\___/ \__|  \___\___/|_| |_|\_/ \___|_|  \__, |\___|\__,_|
                                                |___/
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""

ITS_A_MYSTERY = \
    r"""
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ __ _ _ _ _ _ _ _ _
                                                          _
  ___ ___  _ ____   _____ _ __ __ _  ___ _ __   ___ ___  (_)___
 / __/ _ \| '_ \ \ / / _ \ '__/ _` |/ _ \ '_ \ / __/ _ \ | / __|
| (_| (_) | | | \ V /  __/ | | (_| |  __/ | | | (_|  __/ | \__ \
 \___\___/|_| |_|\_/ \___|_|  \__, |\___|_| |_|\___\___| |_|___/
                              |___/
                                    _
         __ _   _ __ ___  _   _ ___| |_ ___ _ __ _   _
        / _` | | '_ ` _ \| | | / __| __/ _ \ '__| | | |
       | (_| | | | | | | | |_| \__ \ ||  __/ |  | |_| |
        \__,_| |_| |_| |_|\__, |___/\__\___|_|   \__, |
                          |___/                  |___/
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ __ _ _ _ _ _ _ _ _
"""


# Global variables
DATE = datetime.datetime.now()
N_CORES = 1
PYTHON_BINARY = "py"
RUNTIME_FLAGS = None
RESUME_RUN = False
CONV_LIMIT = 0.85
SPLIT_CYCLES = False
DRY_RUN = False
PLOT = False


# Verbosity levels of Python output
VERBOSE_SILENT = 0
VERBOSE_PROGRESS_REPORT = 1
VERBOSE_EXTRA_INFORMATION = 2
VERBOSE_EXTRA_INFORMATION_TRANSPORT = 3
VERBOSE_ALL = 4
VERBOSITY = VERBOSE_PROGRESS_REPORT


def print_python_output(line: str, n_cores, verbosity: int = VERBOSE_EXTRA_INFORMATION_TRANSPORT) \
        -> None:
    """
    Process the output from a Python simulation and print something to screen.

    The amount printed to screen will vary depending on the verbosity level
    chosen.

    Level                                     Result
    -----                                     ------
    0: VERBOSE_SILENT                         Nothing
    1: VERBOSE_PROGRESS_REPORT                Cycle information
    2: VERBOSE_EXTRA_INFORMATION              Convergence plus the above
    3: VERBOSE_EXTRA_INFORMATION_TRANSPORT    Transport progress plus the above
    4: VERBOSE_ALL                            Everything from Python

    Parameters
    ----------
    line: str
        The line to process
    n_cores: int
        The number of cores the simulation is being run with. This is required
        to calculate the total photon number
    verbosity: bool, optional
        If this is True, then every line will be printed to screen
    """

    oline = copy(line)
    line = line.split()

    # PRINT EVERYTHING

    if verbosity >= VERBOSE_ALL:
        log("{}".format(oline))

    # PRINT CURRENT IONISATION CYCLE

    elif oline.find("for defining wind") != -1 and verbosity >= VERBOSE_PROGRESS_REPORT:
        current_cycle = line[3]
        total_cycles = line[5]
        current_time = time.strftime("%H:%M")
        log("{}  Starting Ionisation Cycle ....... {}/{}".format(current_time, current_cycle, total_cycles))

    # PRINT CURRENT SPECTRUM CYCLE

    elif oline.find("to calculate a detailed spectrum") != -1 and verbosity >= VERBOSE_PROGRESS_REPORT:
        current_cycle = line[1]
        total_cycles = line[3]
        current_time = time.strftime("%H:%M")
        log("{}  Starting Spectrum Cycle ......... {}/{}".format(current_time, current_cycle, total_cycles))

    # PRINT COMPLETE RUN TIME

    elif oline.find("Completed entire program.") != -1 and verbosity >= VERBOSE_PROGRESS_REPORT:
        tot_run_time_seconds = float(line[-1])
        tot_run_time = datetime.timedelta(seconds=tot_run_time_seconds // 1)
        log("\nSimulation completed in: {} hrs:mins:secs".format(tot_run_time))

    # PRINT TOTAL RUN TIME ELAPSED FOR A CYCLE

    elif (oline.find("Completed ionization cycle") != -1 or oline.find("Completed spectrum cycle") != -1) and \
            verbosity >= VERBOSE_EXTRA_INFORMATION:
        elapsed_time_seconds = float(line[-1])
        elapsed_time = datetime.timedelta(seconds=elapsed_time_seconds // 1)
        log("         Elapsed run time: {} hrs:mins:secs".format(elapsed_time))

    # PRINT CONVERGENCE

    elif (oline.find("converged") != -1 and oline.find("converging") != -1) \
            and verbosity >= VERBOSE_EXTRA_INFORMATION:
        try:
            cells_converged = line[1]
            fraction_converged = line[2]
            log("         {} cells converged ({})".format(cells_converged, fraction_converged))
        except IndexError:
            log("          unable to parse convergence :-(")

    # PRINT PHOTON TRANSPORT REPORT

    elif oline.find("per cent") != -1 and oline.find("Photon") != -1 \
            and verbosity >= VERBOSE_EXTRA_INFORMATION_TRANSPORT:
        try:
            if int(line[6]) == 0:
                log("         Beginning photon transport")
        except ValueError:
            pass
        try:
            percent = round(float(line[-3]), 0)
        except ValueError:
            percent = line[-3]
        try:
            nphots = round(int(line[-5]) * n_cores, 0)
            nphots = "{:1.2e}".format(nphots)
        except ValueError:
            nphots = line[-5]
        log("           - {}% of {} photons transported".format(percent, nphots))

    # PRINT PHOTON TRANSPORT RUN TIME
    elif oline.find("photon transport completed in") != -1 and verbosity >= VERBOSE_EXTRA_INFORMATION_TRANSPORT:
        transport_time_seconds = float(line[5])
        transport_time = datetime.timedelta(seconds=transport_time_seconds // 1)
        log("         Photons transported in {} hrs:mins:secs".format(transport_time))

    return


def plot_model(root: str, wd: str):
    """
    Run py_plot.py.plot() to create a bunch of default plots for the model.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    wd: str
        The working directory containing the Python simulation
    """

    plot((root, wd, None, None, False, "rectilinear", 5, "png", False))

    return


def convergence_check(root: str, wd: str, nmodels: int) \
        -> bool:
    """
    Check the convergence of a Python simulation by parsing the master diag
    file. If more than one model is being run, then the convergence of each
    model will be appended to the convergence tracking files.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    wd: str
        The working directory containing the Python simulation
    nmodels: int
        The number of models which are scheduled to run.

    Returns
    -------
    converged: bool
        If the simulation has converged, True is returned.
    """

    converged = False

    model_convergence = Simulation.check_convergence(root, wd)
    log("Model convergence ........... {}".format(model_convergence))

    # An unknown convergence has been returned
    if 0 > model_convergence > 1:
        log(ITS_A_MYSTERY)
    # The model has not converged
    elif model_convergence < CONV_LIMIT:
        log(NOT_CONVERGED)
    # The model has converged
    elif model_convergence >= CONV_LIMIT:
        converged = True
        log(CONVERGED)

    log("")

    # If there is only one model being run, then we don't need to write out
    # the convergence to a bunch of files

    if nmodels == 1:
        return converged

    # If there are multiple models being run, then we will track the convergence
    # of each model in some master convergence files

    if converged is False:
        output = "not_converged.txt"
    else:
        output = "converged.txt"

    # Write the model name and convergence to the appropriate output file

    with open(output, "a") as f:
        f.write("{}\t{}.pf\t{}\n".format(wd, root, model_convergence))

    # Write the model name and convergence to the master convergence file

    with open("convergence_report.txt", "a") as f:
        f.write("{}\t{}.pf\t{}\n".format(wd, root, model_convergence))

    return converged


def restore_backup_pf(root: str, wd: str) \
        -> None:
    """
    Copy a backup parameter file back to the original parameter file
    destination.

    Parameters
    ----------
    root: str
        The root name of the Python simulation.
    wd: str
        The working directory to run the Python simulation in.
    """

    opf = "{}/{}.pf".format(wd, root)
    bak = opf + ".bak"
    copyfile(bak, opf)

    return


def print_errors(error: dict, root: str) \
        -> None:
    """
    Print an errors dictionary.

    Parameters
    ----------
    error: dict
        A dictionary where the keys are the error messages and the values are
        the number of times the error happened.
    root: str
        The root name of the Python simulation
    """

    log("Total errors reported for {}:\n".format(root))
    for key in error.keys():
        log("  {:6d} -- {}".format(error[key], key))

    return


def run_model(root: str, wd: str, use_mpi: bool, ncores: int, resume_model: bool = False,
              restart_from_spec_cycles: bool = False, split_cycles: bool = False) \
        -> int:
    """
    The purpose of this function is to use the Subprocess library to call
    Python. Unfortunately, to cover a wide range of situations with how one
    may want to run Python, this function has become rather complicated and
    could benefit from being modularised further.

    Parameters
    ----------
    root: str
        The root name of the Python simulation.
    wd: str
        The working directory to run the Python simulation in.
    use_mpi: bool
        If True, Python will be run using mpirun.
    ncores: int
        If use_mpi is True, then Python will be run using the number of cores
        provided.
    resume_model: bool, optional
        If True, the -r flag will be passed to Python to restart a run from the
        previous cycle
    split_cycles: bool, optional
        If True, the -r flag will be passed to Python to restart a run from the
        first spectrum cycle with a reduced photon sample
    restart_from_spec_cycles: bool, optional
        If True, Python will probably run just the spectral cycles with a reduced
        photon number.

    Returns
    -------
    rc: int
        The return code from the Python simulation
    """

    if VERBOSITY >= VERBOSE_ALL:
        verbose = True
    else:
        verbose = False

    pf = root + ".pf"

    # The purpose of this is to manage the situation where we "split" the
    # ionization and spectral cycles into TWO separate Python runs. So, we first
    # set the spectrum cycles to 0, to run only ionization. Then, we set the
    # spectrum cycles to 5 and set the photon per cycle to 1e6. The point of
    # this is because you may need 5e7 photons during the ionization cycles for
    # the model to converge, but you are unlikely to need this many to make a
    # low signal/noise spectrum. Note we make a backup of the original pf.

    try:

        if split_cycles and not restart_from_spec_cycles:
            Grid.change_parameter(wd + pf, "Spectrum_cycles", "0", backup=True, verbose=verbose)

        if split_cycles and restart_from_spec_cycles:
            Grid.change_parameter(wd + pf, "Spectrum_cycles", "5", backup=False, verbose=verbose)
            Grid.change_parameter(wd + pf, "Photons_per_cycle", "1e6", backup=False, verbose=verbose)

    except IOError:
        print("Unable to open parameter file {} to change any parameters".format(wd + pf))
        return EXIT_FAIL

    # Construct shell command to run Python and use subprocess to run

    command = "cd {}; Setup_Py_Dir; ".format(wd)

    if use_mpi:
        command += "mpirun -n {} ".format(ncores)

    command += " {} ".format(PYTHON_BINARY)

    if resume_model:
        command += " -r "

    # Add the run-time flags the user provided

    if RUNTIME_FLAGS:
        command += " {} ".format(RUNTIME_FLAGS)

    # Add the root name at the end of the call to Python

    command += " {} ".format(pf)
    log("{}\n".format(command))

    # Use Popen to create a new Python process - I do this manually for some
    # reason?

    cmd = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)

    # This next bit provides real time output of Python's output...

    model_log = "{}/{}_{}{:02d}{:02d}.log.txt".format(wd, root, DATE.year, int(DATE.month), int(DATE.day))
    model_logfile = open(model_log, "a")

    for stdout_line in iter(cmd.stdout.readline, ""):
        if not stdout_line:
            break
        line = stdout_line.decode("utf-8").replace("\n", "")
        model_logfile.write("{}\n".format(line))
        print_python_output(line, ncores, VERBOSITY)

    if not verbose:
        log("")

    model_logfile.close()

    # Sometimes with Subprocess, if the output buffer is too large then
    # subprocess breaks and causes a deadlock. To get around this, one can use
    # .communicate() to flush the buffer or s/t

    stdout, stderr = cmd.communicate()
    stderr = stderr.decode("utf-8")
    if stderr:
        log("The following message was sent to stderr:")
        log(stderr)

    rc = cmd.returncode
    if rc:
        print("Python exited with non-zero exit code: {}\n".format(rc))

    # For the ease of future me, write out the version used run the model

    version, hash = PythonUtils.get_python_version(PYTHON_BINARY, verbose)
    with open("version", "w") as f:
        f.write("{}\n{}".format(version, hash))

    # If we have modified the parameter file because we are splitting the
    # model into two runs, then restore the original backup

    if split_cycles and restart_from_spec_cycles:
        restore_backup_pf(root, wd)

    return rc


def go(roots: List[str], use_mpi: bool, n_cores: int) -> None:
    """
    Run the parts of the scripts requested to by run by the user.

    Parameters
    ----------
    roots: List[str]
        A list containing the root names of the Python simulations to run.
    use_mpi: bool
        If True, MPI will be used to run Python.
    n_cores: int
        If use_mpi is True, this will be the number of cores to run Python with.
    """

    nmodels = len(roots)

    # We only track convergence in these files if there are multiple models
    # being run - so do not touch or create them otherwise!

    if nmodels > 1:
        open("not_converged.txt", "w").close()
        open("converged.txt", "w").close()
        open("convergence_report.txt", "w").close()

    for i, path in enumerate(roots):

        root, wd = PythonUtils.split_root_directory(path)
        log("------------------------\n")
        log("        Model {}/{}".format(i + 1, nmodels))
        log("\n------------------------\n")
        log("Root ...................... {}".format(root))
        log("Directory ................. {}\n".format(wd))

        # Run Python

        model_converged = False
        rc = run_model(root, wd, use_mpi, n_cores, resume_model=RESUME_RUN, restart_from_spec_cycles=False,
                       split_cycles=SPLIT_CYCLES)
        if rc:
            log("Python exited with error code {}.".format(rc))
            log("Skipping to the next model.")
            continue

        # Check for the error report and print to the screen

        errors = Simulation.error_summary(root, wd, N_CORES)
        print_errors(errors, root)

        # Check the convergence of the model

        if not rc:
            log("\nChecking the convergence of the simulation:\n")
            model_converged = convergence_check(root, wd, nmodels)

        # If the cycles are being split, handle the logic here to do so

        if SPLIT_CYCLES and model_converged > CONV_LIMIT:
            rc = run_model(root, wd, use_mpi, n_cores, resume_model=True, restart_from_spec_cycles=True,
                           split_cycles=True)
            # Check for the error report and print to the screen
            errors = Simulation.error_summary(root, wd, N_CORES)
            print_errors(errors, root)
        elif SPLIT_CYCLES and model_converged < CONV_LIMIT:
            log("The model has not converged to the set convergence limit of {}.".format(CONV_LIMIT))
            log("Skipping spectral cycles.")

        if rc:
            log("Python exited for error code {} after spectral cycles.".format(rc))
            log("Skipping to the next model.\n")
            continue

        if PLOT:
            print("Plotting the output of the model:\n")
            plot_model(root, wd)

        log("")

    return


def setup_script() \
        -> None:
    """
    Setup the global variables which control the logic of the script.
    """

    p = ap.ArgumentParser(description=__doc__)

    p.add_argument("-sc", "--split_cycles", action="store_true",
                   help="Split the ionization and spectrum cycles into two separate Python runs.")
    p.add_argument("-r", "--restart", action="store_true", help="Restart a Python model from a previous wind_save.")
    p.add_argument("-py", "--python", type=str, action="store", help="The name of the of the Python binary to use.")
    p.add_argument("-f", "--python_flags", type=str, action="store", help="Any run-time flags to pass to Python.")
    p.add_argument("-c", "--convergence_limit", type=float, action="store",
                   help="The 'limit' for considering a model converged. This value is 0 < c_value < 1.")
    p.add_argument("-v", "--verbose", action="store", help="The level of verbosity for Python's output.")
    p.add_argument("-n", "--n_cores", action="store", help="The number of processor cores to run Python with.")
    p.add_argument("-d", "--dry_run", action="store_true", help="Print the models found to screen and then exit.")
    p.add_argument("-p", "--plot", action="store_true", help="Create plots for the models after running Python.")

    args = p.parse_args()

    global VERBOSITY
    if args.verbose:
        VERBOSITY = int(args.verbose)

    global SPLIT_CYCLES
    if args.split_cycles:
        SPLIT_CYCLES = True

    global PYTHON_BINARY
    if args.python:
        PYTHON_BINARY = args.python_version

    global RUNTIME_FLAGS
    if args.python_flags:
        RUNTIME_FLAGS = args.python_flags

    global CONV_LIMIT
    if args.convergence_limit:
        if 0 < args.convergence_limit < 1:
            CONV_LIMIT = args.convergence_limit
        else:
            log("Invalid value for convergence limit {}".format(args.clim))
            exit(EXIT_FAIL)

    global DRY_RUN
    if args.dry_run:
        DRY_RUN = True

    global N_CORES
    if args.n_cores:
        N_CORES = int(args.n_cores)

    global PLOT
    if args.plot:
        PLOT = True

    log("Python  .......................... {}".format(PYTHON_BINARY))
    log("Split cycles ..................... {}".format(SPLIT_CYCLES))
    log("Resume run ....................... {}".format(RESUME_RUN))
    log("Number of cores .................. {}".format(N_CORES))
    log("Convergence limit ................ {}".format(CONV_LIMIT))
    log("Verbosity level .................. {}".format(VERBOSITY))
    log("Plot model ....................... {}".format(PLOT))

    if RUNTIME_FLAGS:
        log("\nUsing these extra python flags:\n\t{}".format(RUNTIME_FLAGS))

    return


def main() \
        -> None:
    """
    Main control function of the script.
    """

    log("------------------------\n")

    # Setup the script run mode and initialise the log file

    setup_script()
    init_logfile("Log{}{:02d}{:02d}.log.txt".format(str(DATE.year)[-2:], int(DATE.month), int(DATE.day)))

    log("")
    Quotes.random_quote()
    log("------------------------\n")

    # Find models to run by searching recursively from the calling directory
    # for .pf files

    the_pfs = PythonUtils.find_parameter_files()
    nmodels = len(the_pfs)
    if not nmodels:
        log("No parameter files found, nothing to do!\n")
        log("------------------------")
        return

    # Check to see how many processor cores are going to be use, and set the
    # mpirun flag appropriately

    if N_CORES:
        ncores_to_use = N_CORES
    else:
        ncores_to_use = PythonUtils.get_cpu_count()
    if ncores_to_use > 1:
        mpirun = True
    else:
        mpirun = False

    # Print the models which are going to be run to the screen

    log("\nThe following {} parameter files were found:\n".format(len(the_pfs)))
    for i in range(len(the_pfs)):
        log("{}".format(the_pfs[i]))
    log("")

    # If we're doing a dry-run, then we don't go any further

    if DRY_RUN:
        log("------------------------")
        return

    # Now run Python...

    go(the_pfs, mpirun, ncores_to_use)

    log("------------------------")
    close_logfile()

    return


if __name__ == "__main__":
    main()
