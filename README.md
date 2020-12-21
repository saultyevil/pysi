# pyPython

A Python package to make using the Monte Carlo radiative transfer and ionization
code Python a wee bit easier. Whilst originally this package was meant to be
simple and focused on creating a tool to analyse the data from Python , it has 
become the final resting place for a lot of other code I have used to do
science over the course of my PhD.

## Usage

Provided in the `examples` directory are scripts which use various parts of
the package to, for example, plot spectra, process the atomic data or some 
general error checking for models. 

## Submodules

- accretionDisc - calculate basic accretion disc quantities
- atomicData - process some of Python's atomic data files
- blackbody - calculate basic quantities for a blackbody
- blackhole - calculate basic quantities for black holes
- constants - CGS constants
- conversion - convert between units or physical quantities
- errors - errors and exceptions used throughout the package
- grid - functions for setting up a grid or parameter files or modifying parameter files
- hpc - functions for creating .slurm files or adding models to a .slurm queue
- ionization - convert ionization quantities
- log - logging utilities
- mailNotifications - email notification utilities
- pythonUtil - utility functions for running Python or other basic utility things
- quotes - some stupid quotes
- simulation - check the errors and convergence of a simulation
- spectrumCreate - functions to create a spectrum from the raw extracted photons 
- spectrumPlot - plot the extracted spectra
- spectrumUtil - utility functions for manipulating spectrum files
- windModels - TODO
- windPlot - plot the simulation grid
- windUtil - utility functions for manipulating the wind_save files

## Dependencies

pyPython requires Python 3.5+.

There are no _"weird"_ dependencies for pyPython. When installing using the
method below (i.e. using pip), the required dependencies will be automatically
installed.

## Installation

pyPython can be easily installed using pip. The easiest way to install is to use
the following command in the root directory,

```bash
$ pip install -e .
```

You may need to use --user if installing on a shared machine where you do not
have administrator privileges, or do not want to install it for all users.
