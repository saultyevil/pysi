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

## Installation and requirements

The minimum version of Python is probably something like Python 3.10, but the
package may work on versions as early as 3.7. All of the dependencies for the
package can be installed either through the `requirements.txt` file or if you
use Anaconda through the `environment.yml` file.

To install the package,

```bash
$ python -m venv /path/to/new/env
$ pip install -r requirements.txt
$ source /path/to/new/env/bin/activate && pip install -e .
```

```
$ conda env create -f environment.yml
$ conda activate pypython
```

In both cases above, the package will be installed in "editable" mode for
development purposes.

## Documentation

Documentation is hosted on
[ReadTheDocs](https://pypython.readthedocs.io/en/stable/) and stored in the
`docs` directory. The documentation is built using Sphinx and is still in
development.

To build the documentation locally, use the following command,

```bash
$ sphinx-build -a -j auto -b html docs/source/ docs/build/html
```

This will create a directory `docs/build/html` and you can view the
documentation by opening `docs/build/html/index.html` in your web browser.

## Usage

The main interface for PyPython will be the `pypython` command. This is
documented on [ReadTheDocs](https://pypython.readthedocs.io/en/stable/). Another
valid use-case would be to import pypython (the actual Python package) into
your own scripts.
