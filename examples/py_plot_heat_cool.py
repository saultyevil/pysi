#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The purpose of this script is to provide quick plotting of the wind for a Python
simulation. As such, it is not very flexible with input to modify the output.
The script will create a figure of the "important" wind quantities, such as
the electron temperature and density, as well figures for the ion fractions
for H, He, C, N, O and Si.
"""


import argparse as ap
from sys import exit
from typing import List, Tuple
from matplotlib import pyplot as plt

from pypython import windplot
from pypython import windutil
from pypython import pythonutil
from pypython.error import EXIT_FAIL


plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.labelsize'] = 15


vars = [
    "heat_tot", "heat_comp", "heat_lines", "heat_ff", "heat_photo", "heat_auger", "cool_tot", "cool_comp", "lum_lines",
    "cool_dr", "lum_ff", "cool_rr", "cool_adiab", "heat_shock", "heat_lines_macro", "head_photo_macro"
]




root = "tde_uv"

# def doplot(nrows, ncols):
#     """Do the plotting :----)"""

nrows = 1
ncols = 3

fig, ax = plt.subplots(nrows, ncols, figsize=(12, 5), squeeze=False)


vars = [
    "heat_tot", "cool_tot", "heat/cool"
]

for i in range(len(vars)):

    var = vars[i]

    if vars[i] == "heat/cool":
        var = "heat_tot"

    try:
        x, z, w = windutil.get_wind_variable(
            root, var, "wind", ".", "rectilinear",
        )
    except Exception as e:
        print("\nSomething went wrong :(")
        print(e)
        continue

    if vars[i] == "heat/cool":
        x, z, ww = windutil.get_wind_variable(root, "cool_tot", "wind", ".", "rectilinear")
        w /= ww

    fig, ax = windplot.plot_rectilinear_wind(x, z, w, vars[i], "wind", fig, ax, 0, i)

fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
plt.show()

