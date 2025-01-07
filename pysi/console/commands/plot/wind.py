#!/usr/bin/env python

"""The entry point for plotting wind properties."""

import click
import numpy

from pysi.wind import Wind


@click.group(name="wind")
def wind_entry():
    """Plot wind properties."""


@wind_entry.command(name="property")
@click.argument("root")
@click.argument("parameter")
def wind_property(root: str, parameter: str):
    """Plot a single wind property."""
    wind = Wind(root)

    with numpy.errstate(divide="ignore"):
        try:
            wind.plot_parameter(parameter)
        except (ValueError, KeyError):
            click.echo(f"Error: {parameter} not in {root}")
            return

    wind.show_figures()
