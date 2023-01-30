# pypython

[![DOI](https://zenodo.org/badge/210153582.svg)](https://zenodo.org/badge/latestdoi/210153582)

pypython is a package designed to make using the Monte Carlo radiative 
transfer and ionization code [Python](https://github.com/agnwinds/python) a wee 
bit easier. 

The main purpose of the package is to provide tools to easily plot and analyse
the output from a Python simulation. However, a number of tools also exist to
set up and run a grid of models, as well aid the analysis in determining the
quality of a converged model.

## Requirements

A minimum Python version of 3.7 is required (i.e., something from June 2018 onwards).
There are also minimum versions required for packages used in pypython, which
are located in `requirements.txt` and can be installed using `pip install -r requirements.txt`.

## Installation

pypython is easiest used and installed by using pip to add it to your Python path.
For example, it can be  installed as follows,

```bash
$ pip install .
```

You may need to use --user if installing on a shared machine where you do not
have administrator privileges, or do not want to install it for all users. If
you are a development user, install the package in "editable" mode using,

```bash
$ pip install -e .
```

## Documentation

The barely completely documentation can be built using Sphinx and is located
in the `docs` directory. To build the HTML version of the documentation,
run the following command in the `docs` directory

```bash
$ make html
```

HTML files will be created in `docs/build/html` and the root 
file will be named `index.html`.

## Usage

Helpful scripts are in the `scripts` which are used to plot output from Python,
or to setup and run Python models and determine the quality of their convergence.

The most useful scripts are the various `py_plot_*` and `py_check_*` scripts 
and `py_run.py`.
