#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
      name="pypython",
      python_requires='>=3.5',
      version="2.0a",
      description="A Python package to make using Python a wee bit easier.",
      url="https://github.com/saultyevil/pyPython",
      author="Edward J. Parkinson",
      author_email="e.j.parkinson@soton.ac.uk",
      license="MIT",
      packages=["pypython", "physics"],
      zip_safe=False,
      install_requires=[
            "matplotlib", "scipy", "numpy", "pandas", "astropy", "numba",
            "psutil", "pathlib", "argparse", "google-api-python-client",
            "google-auth-httplib2", "google-auth-oauthlib", "sphinx",
            "karma_sphinx_theme", "automodinit"
      ]
)
