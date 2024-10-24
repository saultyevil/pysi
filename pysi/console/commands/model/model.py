#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The entry point for the model commands."""

from pathlib import Path

import click
import pysi.sim.model


@click.group()
def model():
    """Commands for evaluating and running models"""


@model.command()
def convergence():
    """Assess the convergence of a model(s)."""
    models = sorted(Path(".").rglob("*.pf"))
    for model in models:
        model_convergence = pysi.sim.model.model_convergence(model.stem, model.parent)
        print(convergence)
