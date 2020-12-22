# pyPython

A Python package to make using the Monte Carlo radiative transfer and ionization
code Python a wee bit easier. Whilst originally this package was meant to be
simple and focused on creating a tool to analyse the data from Python , it has 
become the final resting place for a lot of other code I have used to do
science over the course of my PhD.

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

## Documentation

Documentation of the module can be created by using Sphinx and is located in
the `docs` directory. To build the documentation, run the following command
in the `docs` directory,

```bash
make html
```

HTML documentation will be created in `docs/build/html` and the root 
documentation file will be named `index.html`.

## Usage

Provided in the `examples` directory are scripts which use various parts of
the package to, for example, plot spectra, process the atomic data or some 
general error checking for models. 


