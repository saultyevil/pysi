#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Console commands for pypython.
"""

import click


@click.group()
def cli():
    """Entry point for pypython console commands."""
    print("In development...")
