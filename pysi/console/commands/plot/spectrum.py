#!/usr/bin/env python

"""The entry point for the plotting synthetic spectra."""

import click

from pysi.spec import Spectrum


@click.group(name="spectrum")
def spectrum_entry():
    """Plot synthetic spectra."""


@spectrum_entry.command()
@click.argument("root")
@click.option("--angle", default=None, help="The angle to plot.")
def observer(root: str, angle: str | None) -> None:
    """Plot the observer spectrum."""
    model_spectra = Spectrum(root)
    extracted_spectra = model_spectra["spec"]
    angles_available = sorted(extracted_spectra["inclinations"], key=int, reverse=True) if not angle else (angle,)

    for _angle in angles_available:
        try:
            model_spectra.plot(_angle)
        except KeyError:
            click.echo(f"Error: {_angle} not in {root} spectra")
            return

    model_spectra.show_figures()
