#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Basic utility functions for making life easier.

Includes functions for creating slurm files, bash "run scripts", as well
as getting system info and a logging utility.
"""

import textwrap
from typing import List

from psutil import cpu_count

from pypython import get_files, get_root_name

LOGFILE = None


def create_run_script(commands):
    """Create a shell run script given a list of commands to do using bash a
    bash script.

    Parameters
    ----------
    commands: List[str]
        The commands which are going to be run.
    """

    paths = []
    pf_fp = get_files("*.pf")
    for fp in pf_fp:
        root, path = get_root_name(fp)
        paths.append(path)

    file = "#!/bin/bash\n\ndeclare -a directories=(\n"
    for fp in paths:
        file += "\t\"{}\"\n".format(fp)
    file += ")\n\ncfp=$(pfp)\nfor i in \"${directories[@]}\"\ndo\n\tcd $i\n\tpfp\n"
    if len(commands) > 1:
        for k in range(len(commands) - 1):
            file += "\t{}\n".format(commands[k + 1])
    else:
        file += "\t# commands\n"
    file += "\tcd $cfp\ndone\n"

    print(file)
    with open("commands.sh", "w") as f:
        f.write(file)


def create_slurm_file(name, n_cores, n_hours, n_minutes, py_flags, py_run_flags, fp="."):
    """Create a slurm file in the directory fp with the name root.slurm.

    All of the script flags are passed using the flags variable.

    Parameters
    ----------
    name: str
        The name of the slurm file
    n_cores: int
        The number of cores which to use
    n_hours: int
        The number of hours to allow
    n_minutes: int
        The number of minutes to allow
    py_flags: str
        The run-time flags of which to execute Python with
    py_run_flags: str
        The run-time flags to pass to py_run.py
    fp: str
        The directory to write the file to
    """

    slurm = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --mail-user=ejp1n17@soton.ac.uk
        #SBATCH --mail-type=ALL
        #SBATCH --ntasks={n_cores}
        #SBATCH --time={n_hours}:{n_minutes}:00
        #SBATCH --partition=batch
        module load openmpi/3.0.0/gcc
        module load conda/py3-latest
        source activate pypython
        python /home/ejp1n17/PythonScripts/py_run.py -n {n_cores} {py_run_flags} -f='{py_flags}'
        """)

    if fp[-1] != "/":
        fp += "/"
    file_name = fp + name + ".slurm"
    with open(file_name, "w") as f:
        f.write(f"{slurm}")


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


# Logging functions ------------------------------------------------------------


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
