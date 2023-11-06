"""Test script for PyPython classes.
"""
import pathlib

import pypython.wind
import pypython.spectrum

directory = pathlib.Path("~/tde_postprocess/m_2d/full").expanduser()
wind = pypython.wind.Wind("input", directory)

print(wind.parameters.keys())

wind.plot_parameter("converge", log_p=False)
wind.plot_parameter("t_e")
wind.plot_parameter("t_r")
wind.plot_parameter("ntot")
wind.plot_parameter("ne")
wind.plot_parameter("He_i02_frac", vmax=-2, vmin=-8)
wind.show_figures()

# spectrum = pypython.spectrum.Spectrum("input", directory)
# spectrum.plot_extracted_spectrum("Emitted")
# spectrum.show_figures()
