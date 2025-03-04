#!/usr/bin/env python

"""The entry point for the PyPython CLI."""

import click

from .spectrum import spectrum_entry
from .wind import wind_entry


@click.group(name="plot")
def plot():
    """Commands for plotting synthetic spectra or wind properties"""


plot.add_command(wind_entry)
plot.add_command(spectrum_entry)
