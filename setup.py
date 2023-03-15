#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

# get requirements from file
with open("requirements.txt", "r", encoding="utf-8") as file_in:
    requirements = [line.strip("\n") for line in file_in.readlines()]
# get version from pypython/__init__.py
with open("pypython/__init__.py", "r", encoding="utf-8") as file_in:
    lines = file_in.readlines()
for line in lines:
    line = line.split()
    if len(line) < 1:
        continue
    if line[0] == "__version__":
        __version__ = str(line[2]).strip('"').strip("'")
# setup function
setup(
    name="PyPython",
    python_requires="~=3.10",
    version=__version__,
    description="A package to make using Python a wee bit easier.",
    url="https://github.com/saultyevil/pypython",
    author="Edward J. Parkinson",
    author_email="saultyevil@gmail.com",
    license="MIT",
    install_requires=requirements,
    packages=find_packages(),
    entry_points={"console_scripts": ["pypython = console.cli:cli"]},
)
