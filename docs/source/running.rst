Running models
==============

If you have one model to run, then it is simple enough to use your favourite
terminal emulator to run Python. But, if you have an entire grid of models
to run then it can become tedious to run each model one by one, especially if
you don't know how long the model is going to run or you don't want to wake up
in the middle of the night to run the next model to make sure everything is done
in time for the next day of research.

Using :code:`py_run.py` to run models
--------------------------------------

pypython includes a script named :code:`py_run.py` which can be used to automatically
run multiple Python simulations without any user input. It is used by calling it
in a directory containing Python parameter files. The models can be organized
into other directories, as the script searches recursively for parameter files
from the directory is called in.

:code:`py_run.py` is found in the :code:`scripts` directory and has the
following usage,

::

  usage: py_run.py [-h] [-sc] [-r] [-ro] [-py PYTHON] [-f PYTHON_FLAGS] [-c CONVERGENCE_LIMIT] [-v VERBOSITY]
                   [-n N_CORES] [-d]

  Run a selection of Python models. This script searches recursively for parameter files and executes a number of
  commands, most importantly running the model, depending on what is requested by the user using a number of runtime
  flags.

  optional arguments:
    -h, --help
                          Show this help message and exit
    -sc, --split_cycles
                          Split the ionization and spectrum cycles into two separate Python simulations. Default is
                          disabled.
    -r, --restart
                          Restart Python using a previous wind save. Default is disabled.
    -ro, --restart_override
                          Disable automatic restarting. Default is disabled.
    -py PYTHON, --python PYTHON
                          The Python binary to use. Default is 'py'.
    -f PYTHON_FLAGS, --python_flags PYTHON_FLAGS
                          Runtime flags to use with Python.
    -c CONVERGENCE_LIMIT, --convergence_limit CONVERGENCE_LIMIT
                          The fraction of cells for a simulation to be converged, between 0 and 1. Default is 0.8.
    -v VERBOSITY, --verbosity VERBOSITY
                          The level of verbosity for Python's output. Default is 3.
    -n N_CORES, --n_cores N_CORES
                          The number of processor cores to run Python with. Requires MPI. Default is to automatically
                          determine.
    -d, --dry_run
                          Print the models found to screen and exit. Default is disabled.

For example, a typical usage of :code:`py_run.py` could be::

    py_run.py  -n 12 -sc -ro -c 0.75 -f="-p 2 --rseed"

Splitting ionization and spectrum cycles into two runs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The split cycles mode is an incredibly helpful way to run models which require
a large number of photons, i.e. :math:`10^{8}`, to converge. Spectral cycles,
at least when extract is used, take a greater amount of time to finish due to
re-weighting photon packets and incrementing the multiple spectral estimators at
each photon interaction.

When using this the split cycle mode, the ionization and spectrum cycles are run
as two separate simulations with different photon numbers. The parameter file is
modified such that the ionization cycles will first be run. Upon completion, if
the model is deemed to be converged, then it is restarted from the spectral cycles
with a reduced number of photons (:math:`10^{6}`) and run for five spectral
cycles.

The choice of using :math:`10^{6}` photons for five spectral cycles was chosen
as this combination typically gives low noise spectra when the model is run on
multiple processes as the spectra are averaged over the processes to reduce
noise.

If the spectra are still too noisy, a smoothing filter can be used, more spectral
cycles can be run manually, or the numbers can be tweaked in :code:`py_run.py`.

Determining the quality of a model
----------------------------------

A model may have run to completion, but that doesn't mean that model is any
good. It may not have converged, have many dangerous errors or have exited
silently. pypython includes a couple of scripts to help inform your decision on
whether a model is any good or not.

Checking the convergence with :code:`py_check_convergence.py`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`py_check_convergence.py` is used to check the convergence of a simulation,
cycle by cycle. Output to screen is the percentage of cells which are converged
and still converging for each ionization cycle run. A figure of the convergence,
as well as the different convergence criteria, as a function of ionization
cycle is also created with the name :code:`root_convergence.png`.

Here is what the output typically looks like, for a single model,

::

  --------------------------------------------------------------------------------

  Getting the convergence for star in directory .

  Cycle  1 /  2:  0.00% of cells converged and  0.00% of cells are still converging
  Cycle  2 /  2: 14.80% of cells converged and  0.00% of cells are still converging

  --------------------------------------------------------------------------------

If this script is called in a directory containing multiple simulations, then
the convergence of each model is output.

Obviously, a convergence percentage of 100% is desirable but is not exactly
realistic to get. A convergence fraction of 80% or higher is, usually, adequate
as long as the line forming region is converged. It is typically the (cool)
outer cells in the wind which are not converged, but these cells typically
do not contribute to the final spectrum. Each cell in the wind has a variable
named :code:`converge`, if :code:`converge = 0` then the cell is fully
converged. A :code:`converge` value of :code:`converge = 3` means a cell has
not met any of the three convergence criteria. :code:`converge` can be plotted
using the :code:`Wind` class, i.e. :code:`wind.plot("converge")`.

Checking for errors with :code:`py_check_errors.py`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are a number of error conditions in Python (some 753 to be exact, at time
of writing). Some errors will cause Python to crash, but Python is able to
recover from some errors. Of course, if enough of a single error happens then
the simulation will crash.

At the end of a simulation, or when it crashes, the errors and the number of times
an error was triggered is printed out. :code:`py_check_errors.py` is a script
which can be used to find the errors, across all MPI processes, for a model
or collection of models.

The output will tell inform you if a model crashed or exited successfully.

As with :code:`py_run.py` and :code:`py_check_convergence.py`, simulations are
found recursively from the calling directory.

Here is an example of the output,

::

  --------------------------------------------------------------------------------
  Total errors reported from 1 of 3 processes for ./cv_vert, which was aborted:
         7 -- getatomic_data: line input f odd (may be OK if Macro): %s
         1 -- Ignored %d inner shell cross sections because no matching yields
         1 -- get_wind_params: zdom[ndom].rmax = 0 for wind type %d
         3 -- wind_div_v: div v %e negative in cell %d Domain %d. Major problem if inwind (%d) == 0
         1 -- check_grid: velocity changes by >1,000 km/s in %i cells
         1 -- check_grid: some cells have large changes. Consider modifying zlog_scale or grid dims
      3840 -- randwind_thermal trapping: dvds (%e) > dvds_max (%e) ratio %e in grid %d at %e %e %e
    100001 -- translate_in_wind: nres %5d repeat after motion of %10.3e of phot %6d in ion cycle %2d spec cycle %2d stat(%d -> %d)
         2 -- error_count: This error will no longer be logged: %s
        11 -- walls: %d The previous position %11.4e %11.4e %11.4e is inside the disk by %e
  --------------------------------------------------------------------------------
  Total errors reported from 3 of 3 processes for ./star, which exited successfully:
        21 -- getatomic_data: line input f odd (may be OK if Macro): %s
         3 -- Ignored %d inner shell cross sections because no matching yields
         3 -- wind2d.c: Not currently able to calculate mdot wind for coordtype %d in domain %d
       726 -- translate_in_wind: nres %5d repeat after motion of %10.3e of phot %6d in ion cycle %2d spec cycle %2d stat(%d -> %d)
         3 -- error_count: This error will no longer be logged: %s
         1 -- trans_phot: Trying to scatter a photon in a cell with no wind volume
         1 -- trans_phot: %d grid %3d x %8.2e %8.2e %8.2e
         1 -- trans_phot: istat %d
         1 -- trans_phot: This photon is effectively lost!
  --------------------------------------------------------------------------------

