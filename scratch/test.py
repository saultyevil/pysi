"""Test script for PyPython classes.
"""
import pathlib

import pypython.wind
import pypython.spectrum

directory = pathlib.Path("/home/saultyevil/tde_postprocess/m_2d/full")
wind = pypython.wind.Wind("input", directory)
wind.plot_parameter("converge", log_p=False)
wind.plot_parameter("t_e")
wind.plot_parameter("t_r")
wind.plot_parameter("ntot")
wind.plot_parameter("ne")
wind.show_figures()

# spectrum = pypython.spectrum.Spectrum("tde_opt_spec", directory)
a = 1
