#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run a selection of Python models.

This script searches recursively for parameter files and executes a
number of commands, most importantly running the model, depending on
what is requested by the user using a number of runtime flags.
"""

import argparse as ap
import datetime
import textwrap
import time
from copy import copy
from os import path
from shutil import copyfile
from subprocess import PIPE, Popen
from sys import exit

import pypython
from pypython import simulation, util
from pypython.error import EXIT_FAIL, EXIT_SUCCESS
from pypython.simulation import grid
from pypython.util import close_logfile, init_logfile, log, logsilent

CONVERGED = \
    r"""
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    It is                                    _
  ___ ___  _ ____   _____ _ __ __ _  ___  __| |
 / __/ _ \| '_ \ \ / / _ \ '__/ _` |/ _ \/ _` |
| (_| (_) | | | \ V /  __/ | | (_| |  __/ (_| |
 \___\___/|_| |_|\_/ \___|_|  \__, |\___|\__,_|
                              |___/  my dudes
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""

NOT_CONVERGED = \
    r"""
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    It is    _                                                 _
 _ __   ___ | |_    ___ ___  _ ____   _____ _ __ __ _  ___  __| |
| '_ \ / _ \| __|  / __/ _ \| '_ \ \ / / _ \ '__/ _` |/ _ \/ _` |
| | | | (_) | |_  | (_| (_) | | | \ V /  __/ | | (_| |  __/ (_| |
|_| |_|\___/ \__|  \___\___/|_| |_|\_/ \___|_|  \__, |\___|\__,_|
                                                |___/  my dudes
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

N_CORES = 0
PYTHON_BINARY = "py"
RUNTIME_FLAGS = None
RESTART_MODEL = False
AUTOMATIC_RESTART_OVERRIDE = False
CONVERGENCE_LOWER_LIMIT = 0.80
SPLIT_CYCLES = False
DRY_RUN = False

# Verbosity levels of Python output

VERBOSE_SILENT = 0
VERBOSE_PROGRESS_REPORT = 1
VERBOSE_EXTRA_INFORMATION = 2
VERBOSE_EXTRA_INFORMATION_TRANSPORT = 3
VERBOSE_ALL = 4
VERBOSITY = VERBOSE_EXTRA_INFORMATION_TRANSPORT


def setup_script():
    """Setup the global variables via command line arguments."""
    global VERBOSITY
    global SPLIT_CYCLES
    global PYTHON_BINARY
    global RESTART_MODEL
    global AUTOMATIC_RESTART_OVERRIDE
    global RUNTIME_FLAGS
    global CONVERGENCE_LOWER_LIMIT
    global DRY_RUN
    global N_CORES

    p = ap.ArgumentParser(description=__doc__)

    p.add_argument("-sc",
                   "--split_cycles",
                   action="store_true",
                   default=SPLIT_CYCLES,
                   help="Split the ionization and spectrum cycles into two separate Python runs.")
    p.add_argument("-r",
                   "--restart",
                   action="store_true",
                   default=RESTART_MODEL,
                   help="Restart a Python model from a previous wind_save.")
    p.add_argument("-ro",
                   "--restart_override",
                   action="store_true",
                   default=AUTOMATIC_RESTART_OVERRIDE,
                   help="Disable the automatic restarting run function.")
    p.add_argument("-py", "--python", default=PYTHON_BINARY, help="The name of the of the Python binary to use.")
    p.add_argument("-f", "--python_flags", default=RUNTIME_FLAGS, help="Any run-time flags to pass to Python.")
    p.add_argument("-c",
                   "--convergence_limit",
                   type=float,
                   default=CONVERGENCE_LOWER_LIMIT,
                   help="The 'limit' for considering a model converged. This value is 0 < c_value < 1.")
    p.add_argument("-v",
                   "--verbosity",
                   type=int,
                   default=VERBOSE_EXTRA_INFORMATION,
                   help="The level of verbosity for Python's output.")
    p.add_argument("-n",
                   "--n_cores",
                   type=int,
                   default=N_CORES,
                   help="The number of processor cores to run Python with.")
    p.add_argument("-d",
                   "--dry_run",
                   action="store_true",
                   default=DRY_RUN,
                   help="Print the models found to screen and then exit.")

    args = p.parse_args()
    VERBOSITY = args.verbosity
    SPLIT_CYCLES = args.split_cycles
    PYTHON_BINARY = args.python
    RESTART_MODEL = args.restart
    AUTOMATIC_RESTART_OVERRIDE = args.restart_override
    RUNTIME_FLAGS = args.python_flags
    CONVERGENCE_LOWER_LIMIT = args.convergence_limit
    DRY_RUN = args.dry_run
    N_CORES = args.n_cores

    msg = textwrap.dedent("""\
        Python  .......................... {}
        Split cycles ..................... {}
        Resume run ....................... {}
        Automatic restart override ....... {}
        Number of cores .................. {}
        Convergence limit ................ {}
        Verbosity level .................. {}
        """.format(PYTHON_BINARY, SPLIT_CYCLES, RESTART_MODEL, AUTOMATIC_RESTART_OVERRIDE, N_CORES,
                   CONVERGENCE_LOWER_LIMIT, VERBOSITY))

    log(f"------------------------\n{msg}")
    if RUNTIME_FLAGS:
        log("\nUsing these util python flags:\n\t{}".format(RUNTIME_FLAGS))

    return


def print_model_output(input_line, n_cores, verbosity=VERBOSITY):
    """Process the output from a Python simulation and print something to
    screen. The amount printed to screen will vary depending on the verbosity
    level chosen.

    0: VERBOSE_SILENT                         Nothing
    1: VERBOSE_PROGRESS_REPORT                Cycle information
    2: VERBOSE_EXTRA_INFORMATION              Convergence plus the above
    3: VERBOSE_EXTRA_INFORMATION_TRANSPORT    Transport progress plus the above
    4: VERBOSE_ALL                            Everything from Python

    Parameters
    ----------
    input_line: str
        The line to process
    n_cores: int
        The number of cores the simulation is being run with. This is required
        to calculate the total photon number
    verbosity: bool, optional
        If this is True, then every line will be printed to screen
    """
    line = copy(input_line)
    split_line = line.split()

    # PRINT EVERYTHING

    if verbosity >= VERBOSE_ALL:
        log("{}".format(line))

    # PRINT CURRENT IONISATION CYCLE

    elif line.find("for defining wind") != -1 and verbosity >= VERBOSE_PROGRESS_REPORT:
        current_cycle = split_line[3]
        total_cycles = split_line[5]
        current_time = time.strftime("%H:%M")
        log("{}  Starting Ionisation Cycle ....... {}/{}".format(current_time, current_cycle, total_cycles))

    # PRINT CURRENT SPECTRUM CYCLE

    elif line.find("to calculate a detailed spectrum") != -1 and verbosity >= VERBOSE_PROGRESS_REPORT:
        current_cycle = split_line[1]
        total_cycles = split_line[3]
        current_time = time.strftime("%H:%M")
        log("{}  Starting Spectrum Cycle ......... {}/{}".format(current_time, current_cycle, total_cycles))

    # PRINT COMPLETE RUN TIME

    elif line.find("Completed entire program.") != -1 and verbosity >= VERBOSE_PROGRESS_REPORT:
        tot_run_time_seconds = float(split_line[-1])
        tot_run_time = datetime.timedelta(seconds=tot_run_time_seconds // 1)
        log("\nSimulation completed in: {} hrs:mins:secs".format(tot_run_time))

    # PRINT TOTAL RUN TIME ELAPSED FOR A CYCLE

    elif (line.find("Completed ionization cycle") != -1 or line.find("Completed spectrum cycle") != -1) and \
            verbosity >= VERBOSE_EXTRA_INFORMATION:
        elapsed_time_seconds = float(split_line[-1])
        elapsed_time = datetime.timedelta(seconds=elapsed_time_seconds // 1)
        log("         Elapsed run time: {} hrs:mins:secs".format(elapsed_time))

    # PRINT CONVERGENCE

    elif (line.find("converged") != -1 and line.find("converging") != -1) \
            and verbosity >= VERBOSE_EXTRA_INFORMATION:
        try:
            cells_converged = split_line[1]
            fraction_converged = split_line[2]
            log("         {} cells converged {}".format(cells_converged, fraction_converged))
        except IndexError:
            log("          unable to parse convergence :-(")

    # PRINT PHOTON TRANSPORT REPORT

    elif line.find("per cent") != -1 and line.find("Photon") != -1 \
            and verbosity >= VERBOSE_EXTRA_INFORMATION_TRANSPORT:
        try:
            if int(split_line[6]) == 0:
                log("         Beginning photon transport")
        except ValueError:
            pass
        try:
            percent = round(float(split_line[-3]), 0)
        except ValueError:
            percent = split_line[-3]
        try:
            nphots = round(int(split_line[-5]) * n_cores, 0)
            nphots = "{:1.2e}".format(nphots)
        except ValueError:
            nphots = split_line[-5]
        log("           - {}% of {} photons transported".format(percent, nphots))

    # PRINT PHOTON TRANSPORT RUN TIME
    elif line.find("photon transport completed in") != -1 and verbosity >= VERBOSE_EXTRA_INFORMATION_TRANSPORT:
        transport_time_seconds = float(split_line[5])
        transport_time = datetime.timedelta(seconds=transport_time_seconds // 1)
        log("         Photons transported in {} hrs:mins:secs".format(transport_time))

    # PRINT ERROR MESSAGES
    elif line.find("Error: ") != -1 and verbosity >= VERBOSE_EXTRA_INFORMATION_TRANSPORT:
        log("         {}".format(line))


def restore_parameter_file(root, fp):
    """Copy a backup parameter file back to the original parameter file
    destination.

    Parameters
    ----------
    root: str
        The root name of the Python simulation.
    fp: str
        The working directory to run the Python simulation in.
    """
    opf = "{}/{}.pf".format(fp, root)
    bak = opf + ".bak"
    copyfile(bak, opf)

    return


def check_model_convergence(root, cd):
    """Check the convergence of a Python simulation by parsing the master diag
    file. If more than one model is being run, then the convergence of each
    model will be appended to the convergence tracking files.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    cd: str
        The working directory containing the Python simulation

    Returns
    -------
    converged: bool
        If the simulation has converged, True is returned.
    """
    cycle_convergence = simulation.check_model_convergence(root, cd)
    if type(cycle_convergence) == list:  # get the last cycle
        cycle_convergence = cycle_convergence[-1]

    converged = False
    if 0 > cycle_convergence > 1:
        log(ITS_A_MYSTERY)
    elif cycle_convergence < CONVERGENCE_LOWER_LIMIT:
        log(NOT_CONVERGED)
    elif cycle_convergence >= CONVERGENCE_LOWER_LIMIT:
        log(CONVERGED)
        converged = True

    log("")

    return converged, cycle_convergence


def print_model_errors(error, root):
    """Print an errors dictionary.

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


def run_model(root,
              fp,
              use_mpi,
              n_cores,
              resume_model=False,
              restart_from_spec_cycles=False,
              split_cycles=False):
    """The purpose of this function is to use the Subprocess library to call
    Python. Unfortunately, to cover a wide range of situations with how one may
    want to run Python, this function has become rather complicated and could
    benefit from being modularised further.

    Parameters
    ----------
    root: str
        The root name of the Python simulation.
    fp: str
        The working directory to run the Python simulation in.
    use_mpi: bool
        If True, Python will be run using mpirun.
    n_cores: int
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
    if fp == ".":
        fp += "/"
    pf = root + ".pf"

    # The purpose of this is to manage the situation where we "split" the
    # ionization and spectral cycles into TWO separate Python runs. So, we first
    # set the spectrum cycles to 0, to run only ionization. Then, we set the
    # spectrum cycles to 5 and set the photon per cycle to 1e6.

    if split_cycles and not restart_from_spec_cycles:
        grid.update_single_parameter(fp + pf, "Spectrum_cycles", "0", backup=True)
    elif split_cycles and restart_from_spec_cycles:
        grid.update_single_parameter(fp + pf, "Spectrum_cycles", "5", backup=False)
        grid.update_single_parameter(fp + pf, "Photons_per_cycle", "1e6", backup=False)

    # Construct shell command to run Python and use subprocess to run

    command = "cd {}; ".format(fp)
    if not path.exists("{}/data".format(fp)):
        command += "Setup_Py_Dir; "
    if use_mpi:
        command += "mpirun -n {} ".format(n_cores)

    command += " {} ".format(PYTHON_BINARY)

    # If a root.wind_save exists, then we assume that we want to restart

    if resume_model or (not AUTOMATIC_RESTART_OVERRIDE and path.exists("{}/{}.wind_save".format(fp, root))):
        command += " -r "

    # Add the run-time flags the user provided

    if RUNTIME_FLAGS:
        command += " {} ".format(RUNTIME_FLAGS)

    command += " {} ".format(pf)
    log("{}\n".format(command))
    cmd = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)

    # This next bit provides real time output of Python's output...

    model_log = "{}/{}.log.txt".format(fp, root)
    model_logfile = open(model_log, "a")
    model_logfile.write("{}\n".format(datetime.datetime.now()))

    for stdout_line in iter(cmd.stdout.readline, ""):
        if not stdout_line:
            break
        line = stdout_line.decode("utf-8").replace("\n", "")
        model_logfile.write("{}\n".format(line))
        print_model_output(line, n_cores)

    log("")

    model_logfile.close()

    # Sometimes with Subprocess, if the output buffer is too large then
    # subprocess breaks and causes a deadlock. To get around this, flush the
    # communicate buffer

    cmd.communicate()

    rc = cmd.returncode
    if rc:
        print("Python exited with non-zero exit code: {}\n".format(rc))

    if split_cycles and restart_from_spec_cycles:
        restore_parameter_file(root, fp)

    return rc


def run_all_models(parameter_files, use_mpi, n_cores):
    """Run the parts of the scripts requested to by run by the user.

    Parameters
    ----------
    parameter_files: List[str]
        A list containing the root names of the Python simulations to run.
    use_mpi: bool
        If True, MPI will be used to run Python.
    n_cores: int
        If use_mpi is True, this will be the number of cores to run Python with.

    Returns
    -------
    the_rc: List[int]
        The return codes of the Python models
    """

    return_codes = []
    n_models = len(parameter_files)

    for i, fp in enumerate(parameter_files):
        root, fp = pypython.get_root(fp)
        msg = textwrap.dedent("""\
            ------------------------

             Model {}/{}

            ------------------------

            Root ...................... {}
            Directory ................. {}
            """.format(i + 1, n_models, root, fp))

        log(msg)

        rc = run_model(root, fp, use_mpi, n_cores, resume_model=RESTART_MODEL, restart_from_spec_cycles=False, split_cycles=SPLIT_CYCLES)

        return_codes.append(rc)
        errors = simulation.model_error_summary(root, fp, N_CORES)
        print_model_errors(errors, root)

        if rc != 0:
            log("Python exited with return code {}\n".format(rc))
            continue

        model_converged, model_convergence = check_model_convergence(root, fp)
        log("\nModel convergence ........... {}".format(model_convergence))

        # If the cycles are being split into two separate runs to lower the
        # number of photons during a spectrum cycles, handle that situation here

        if SPLIT_CYCLES:
            if model_converged:
                rc = run_model(root, fp, use_mpi, n_cores, resume_model=True, restart_from_spec_cycles=True, split_cycles=True)
                return_codes[i] = rc
                errors = simulation.model_error_summary(root, fp, N_CORES)
                print_model_errors(errors, root)
                if rc != 0:
                    log("Python exited due to error code {} after spectral cycles.".format(rc))
            else:
                log("The model has not converged to the desired limit of {}.".format(CONVERGENCE_LOWER_LIMIT))

        log("")

    return return_codes


def main():
    """Main function of the script."""

    setup_script()
    init_logfile("log.txt")
    log("------------------------\n")
    logsilent("{}".format(datetime.datetime.now()))

    # Find models to run by searching recursively from the calling directory
    # for .pf files

    parameter_files = pypython.get_file("*.pf")
    n_models = len(parameter_files)

    if not n_models:
        log("No parameter files found, nothing to do!\n")
        log("------------------------")
        exit(EXIT_SUCCESS)

    # Check to see how many processor cores are going to be use, and set the
    # mpirun flag appropriately

    if N_CORES:
        n_cores_to_use = N_CORES
    else:
        n_cores_to_use = util.get_cpu_count()

    if n_cores_to_use > 1:
        use_mpi = True
    else:
        use_mpi = False

    # Print the models which are going to be run to the screen

    log("\nThe following {} parameter files were found:\n".format(len(parameter_files)))
    for file in parameter_files:
        log("{}".format(file))
    log("")

    # If we're doing a dry-run, then we don't go any further

    if DRY_RUN:
        log("------------------------")
        return

    return_codes = run_all_models(parameter_files, use_mpi, n_cores_to_use)

    n_crashed = 0
    for pf, rc in zip(parameter_files, return_codes):
        if rc > 0:
            log("Model {} failed with rc {}".format(pf, rc))
            n_crashed += 1

    log("------------------------")
    close_logfile()

    if n_crashed:
        exit(n_crashed)
    else:
        exit(0)


if __name__ == "__main__":
    main()
