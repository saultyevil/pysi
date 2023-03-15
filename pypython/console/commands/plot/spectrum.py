#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The entry point for the plotting synthetic spectra.
"""

import click


@click.group()
def spectrum():
    """Plot synthetic spectra."""


@spectrum.command()
def observer():
    """Plot the observer spectrum."""
