#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The entry point for the PyPython CLI.
"""

import click

from .wind import wind_entry
from .spectrum import spectrum


@click.group(name="plot")
def plot():
    """Commands for plotting synthetic spectra or wind properties"""


plot.add_command(wind_entry)
plot.add_command(spectrum)
