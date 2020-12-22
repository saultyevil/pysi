Running Python using pyPython
=============================

If you have one model to run, then it is simple enough to use your favourite
terminal emulator to run Python. However, if you have an entire grid of models
to run then it can become tedious to run each model one by one, especially if
you don't know how long the model is going to run for.

There is a script as part of pyPython named :code:`py_run.py` which can be used to
automatically run multiple Python models without need from input. The script
will also send an email notification when a model has started, finished or
if the script has exited incorrectly.

:code:`py_run.py` can be found in the :code:`examples` directory,::

    py_run.py [-h] [-sc] [-r] [-ro] [-py PYTHON] [-f PYTHON_FLAGS] [-c CONVERGENCE_LIMIT] [-v VERBOSITY] [-n N_CORES] [-d]

    Run a batch of Python models. Searches recursively for Python parameter files
    (ignoring any py_wind.pf or root.out.pf files) and executes a number of
    commands, most importantly running the model, depending on what is requested by
    the user using a number of runtime flags.

    optional arguments:
      -h, --help            show this help message and exit
      -sc, --split_cycles   Split the ionization and spectrum cycles into two
                            separate Python runs.
      -r, --restart         Restart a Python model from a previous wind_save.
      -ro, --restart_override
                            Disable the automatic restarting run function.
      -py PYTHON, --python PYTHON
                            The name of the of the Python binary to use.
      -f PYTHON_FLAGS, --python_flags PYTHON_FLAGS
                            Any run-time flags to pass to Python.
      -c CONVERGENCE_LIMIT, --convergence_limit CONVERGENCE_LIMIT
                            The 'limit' for considering a model converged. This
                            value is 0 < c_value < 1.
      -v VERBOSITY, --verbosity VERBOSITY
                            The level of verbosity for Python's output.
      -n N_CORES, --n_cores N_CORES
                            The number of processor cores to run Python with.
      -d, --dry_run         Print the models found to screen and then exit.

For example, a typical usage of :code:`py_run.py` is::

    py_run.py -v 3 -ro -c 0.75 -f="-p 2" -n 12

Determining the quality of a model
----------------------------------

A model may have run to completion, but that doesn't mean that model is any
good as it may not have converged, have many dangerous errors or have made some
complete nonsense. Thankfully, pyPython includes numerous modules and scripts
to help determine the quality of the model.


