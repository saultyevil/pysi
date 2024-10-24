#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The entry point for the PyPython CLI."""

import click

from .commands.model.model import model
from .commands.plot.plot import plot


@click.group()
@click.pass_context
def cli(ctx: click.Context):
    """Plot, run and evalulate Python models from a simple to use CLI.

    PyPython is a command line interface and Python package designed to make
    running and analysing Python models a bit easier. The main purpose is to
    enable consitent and easy plot creation across all models, for both the
    synthetic spectra and wind properties.
    """


cli.add_command(plot)
cli.add_command(model)
