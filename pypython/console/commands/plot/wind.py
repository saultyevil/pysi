#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The entry point for plotting wind properties.
"""

import click


@click.group()
def wind():
    """Plot wind properties."""


@wind.command(name="property")
def wind_property():
    """Plot a single wind property."""
