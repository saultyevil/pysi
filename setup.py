#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

with open("requirements.txt", "r", encoding="utf-8") as file_in:
    requirements = [line.strip("\n") for line in file_in.readlines()]

setup(
    name="pypython",
    python_requires="~=3.10",
    version="4.0.0",
    description="A package to make using Python a wee bit easier.",
    url="https://github.com/saultyevil/pypython",
    author="Edward J. Parkinson",
    author_email="saultyevil@gmail.com",
    license="MIT",
    install_requires=requirements,
    packages=[
        "pypython",
    ],
    entry_points={"console_scripts": ["pypython = console.cli.cli"]},
)
