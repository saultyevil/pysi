#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup


scripts = [
      "scripts/py_add_parameter",
      "scripts/py_change_parameter",
      "scripts/py_check_convergence",
      "scripts/py_check_errors",
      "scripts/py_convert_to_previous",
      "scripts/py_create_run_script",
      "scripts/py_create_spectrum",
      "scripts/py_create_wind_tables",
      "scripts/py_plot_multiple_spectra",
      "scripts/py_plot_optical_depth",
      "scripts/py_plot_reprocessing",
      "scripts/py_plot_spectrum",
      "scripts/py_plot_wind",
      "scripts/py_rm_data",
      "scripts/py_run",
      "scripts/slurm_add",
      "scripts/slurm_clear",
      "scripts/slurm_create"
]


setup(
      name="pypython",
      python_requires='>=3.7',
      version="3.5.1",
      description="A package to make using Python a wee bit easier.",
      url="https://github.com/saultyevil/pypython",
      author="Edward J. Parkinson",
      author_email="e.j.parkinson@soton.ac.uk",
      license="MIT",
      packages=["pypython", "pypython/math", "pypython/observations", "pypython/physics", "pypython/plot",
                "pypython/simulation", "pypython/spectrum", "pypython/util", "pypython/wind"],
      scripts=scripts,
      zip_safe=False,
      install_requires=[
            "matplotlib", "scipy", "numpy", "pandas", "astropy", "numba",
            "psutil", "sphinx", "karma_sphinx_theme", "sqlalchemy",
            "dust_extinction", "protobuf"
      ]
)
