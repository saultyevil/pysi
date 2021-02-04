# pypython

pypython is a Python package designed to make using the Monte Carlo radiative 
transfer and ionization code Python a wee bit easier. Historically this package 
was supposed to be simple and focused on creating a tool to plot the output from
Python with a variety of plotting scripts. However, it has now become the final
resting place for a lot of other code abd scripts used throughout my PhD. Some 
of them are quite useful, but some of them are quite specific to certain computing
environments and users.

## Dependencies

pypython requires Python 3.5+, as there is a lot of type hinting throughout.

There are not too many dependencies for pypython and they can all be installed
using pip. The required dependencies should be automatically if you install
using pip.

## Installation

pypython is easiest used and installed by using pip. For example, it can be
installed as follows,

```bash
$ pip install .
```

You may need to use --user if installing on a shared machine where you do not
have administrator privileges, or do not want to install it for all users. If
you are a development user, it's probably best to use,

```bash
$ pip install -e .
```

## Documentation

Documentation of the module can be built by using Sphinx and is located in
the `docs` directory. To build the documentation, run the following command
in the `docs` directory,

```bash
make html
```

HTML documentation will be created in `docs/build/html` and the root 
documentation file will be named `index.html`. At current, there is no 
documentation hosted online and it is also quite incomplete.

## Usage

Provided in the `scripts` directory are scripts which use various parts of
pypython to, for example, plot spectra, process the atomic data or some 
general error checking for models. 

The most useful script is plot `py_plot.py` which creates plots of the various
wind parameters, ions and velocities as well as spectra. `py_run.py` is also
quite nice at helping taking the stress away from running multiple models.
