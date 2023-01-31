"""Test script for PyPython classes.
"""
import pathlib
from IPython import embed

import pypython.wind
import pypython.spectrum

directory = pathlib.Path("~/onedrive/Postgraduate/PySims/tde_optical/grid/3e6/Mdot_acc/0_15").expanduser()

wind = pypython.wind.Wind("tde_opt", directory)
# wind.plot_parameter("t_e")
# wind.show_figures()

spectrum = pypython.spectrum.Spectrum("tde_opt_spec", directory)

embed()
