#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The entry point for the model commands.
"""

import click


@click.group()
def model():
    """Commands for evaluating and running models"""
