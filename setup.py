#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
      name="pypython",
      python_requires='>=3.7',
      version="3.2",
      description="A package to make using Python a wee bit easier.",
      url="https://github.com/saultyevil/pypython",
      author="Edward J. Parkinson",
      author_email="e.j.parkinson@soton.ac.uk",
      license="MIT",
      packages=["pypython", "pypython/math", "pypython/physics", "pypython/plot",
                "pypython/simulation", "pypython/spectrum", "pypython/util"],
      zip_safe=False,
      install_requires=[
            "matplotlib", "scipy", "numpy", "pandas", "astropy", "numba",
            "psutil", "google-api-python-client", "google-auth-httplib2",
            "google-auth-oauthlib", "sphinx", "karma_sphinx_theme", "sqlalchemy",
            "dust_extinction", "protobuf"
      ]
)
