# PyPython

[![DOI](https://zenodo.org/badge/210153582.svg)](https://zenodo.org/badge/latestdoi/210153582)

PyPython is a package and command line tool designed to make using the Monte
Carlo radiative transfer and ionization code
[Python](https://github.com/agnwinds/python) a wee bit easier to use.

The main purpose of PyPython is to provide tools which allow you to easily
plot and analyse the output from a Python simulation for both the synthetic
spectra and the properties of the wind. PyPython also includes a number of
commands for setting up and running a grid of models, as well as tools to help
determine the quality, convergence and trustworthiness of a model.

## Installation

PyPython requires Python 3.10, although it may work on earlier versions with a
bit of work (possibly down to 3.7). You can install PyPython either by modifying
your $PYTHONPATH variable, or by using Pip.

```bash
shell$ pip install .
```

## Usage

The main interface for PyPython will be the `pypython` command. This is
documented on [ReadTheDocs](https://pypython.readthedocs.io/en/stable/). Another
valid use-case would be to import pypython (the actual Python package) into
your own scripts.

## Development

If you want to develop or modify parts of PyPython, then you can either install
it in editable mode (`pip install -e .`) or I strongly recommend using
[Poetry](https://python-poetry.org/) for handling dependency management and
tool installation.

```bash
shell$ poetry install
```

### Documentation

Documentation is hosted on
[ReadTheDocs](https://pypython.readthedocs.io/en/stable/) and stored in the
`docs` directory. The documentation is built using Sphinx and is still in
development.

To build the documentation locally, use the following command,

```bash
shell$ poetry run sphinx-build -a -j auto -b html docs/source/ docs/build/html
```

This will create a directory `docs/build/html` and you can view the
documentation by opening `docs/build/html/index.html` in your web browser.
