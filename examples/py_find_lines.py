#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script should be used to find the location of emission and absorption
lines in a Python spectrum. The package specutils is required for this
script.
"""

import astropy.units as u
from PyPython import SpectrumUtils
from specutils import Spectrum1D
from specutils.fitting import find_lines_derivative
from matplotlib import pyplot as plt

wd = "./"
root = "tde_uv"

frequency_space = False

file = wd + root + ".spec"
inclination = "10"
SMOOTH = 1

t = SpectrumUtils.read_spec(file)

if frequency_space:
    s = Spectrum1D(t[inclination].values * (u.erg * u.s**-1 * u.cm**-2 * u.AA**-1), t["Freq."].values * u.Hz)
else:
    s = Spectrum1D(SpectrumUtils.smooth(t[inclination].values, SMOOTH) * (u.erg * u.s**-1 * u.cm**-2 * u.AA**-1), t["Lambda"].values * u.AA)

lines = find_lines_derivative(s)

print(lines)

new_lines = lines[lines["line_center"] > 2000 * u.AA]
print(new_lines)

line = float(new_lines[-1]["line_center"] / u.AA)

plt.loglog(t["Lambda"], SpectrumUtils.smooth(t[inclination].values, SMOOTH))
plt.axvline(line)
plt.show()
