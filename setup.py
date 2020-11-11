#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
      name="PyPython",
      python_requires='>=3.5',
      version="1.2",
      description="A Python package to make using Python a wee bit easier.",
      url="https://github.com/saultyevil/PyPython",
      author="Edward J. Parkinson",
      author_email="e.j.parkinson@soton.ac.uk",
      license="MIT",
      packages=["PyPython"],
      zip_safe=False,
      install_requires=[
            "matplotlib", "scipy", "numpy", "pandas", "astropy", "numba", "glob", "subprocess", "psutil",
            "platform", "pathlib", "shutil", "typing", "argparse", "sys"
      ]
      )
