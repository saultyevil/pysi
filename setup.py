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
    "scripts/slurmnew",
]


setup(
    name="pypython",
    python_requires=">=3.7",
    version="4.0.0",
    description="A package to make using Python a wee bit easier.",
    url="https://github.com/saultyevil/pypython",
    author="Edward J. Parkinson",
    author_email="saultyevil@gmail.com",
    license="MIT",
    packages=[
        "pypython",
        "pypython/math",
        "pypython/observations",
        "pypython/physics",
        "pypython/plot",
        "pypython/simulation",
        "pypython/spectrum",
        "pypython/util",
        "pypython/wind",
    ],
    scripts=scripts,
    zip_safe=False,
    install_requires=[
        "numba>=0.53.1",
        "numpy>=1.20.2",
        "psutil>=5.8.0",
        "SQLAlchemy>=1.4.15",
        "matplotlib>=3.3.4",
        "pandas>=1.2.4",
        "dust_extinction>=1.0",
        "scipy>=1.6.3",
        "astropy>=4.2.1",
    ],
)
