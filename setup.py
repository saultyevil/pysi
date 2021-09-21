#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup


scripts = [
      "scripts/pyaddp",
      "scripts/pyupdp",
      "scripts/pyconv",
      "scripts/pyerr",
      "scripts/pyprev",
      "scripts/pyscript",
      "scripts/pycspec",
      "scripts/pyctables",
      "scripts/pmspec",
      "scripts/poptd",
      "scripts/prepro",
      "scripts/pspec",
      "scripts/pwind",
      "scripts/pydeld",
      "scripts/pyrun",
      "scripts/slurmadd",
      "scripts/slurmclear",
      "scripts/slurmnew"
]


setup(
      name="pypython",
      python_requires='>=3.7',
      version="3.6.3",
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
