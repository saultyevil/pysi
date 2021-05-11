#!/usr/bin/env python3

from sys import argv
from subprocess import run
from pathlib import Path


def format_source(fp):
    """Run isort and then yapf to format the python files contained in
    fp. Sends the output to /dev/null.
    Parameters
    ----------
    fp: str
        The file path to search recursively for python files.
    """
    for file in Path(fp).rglob("*.py"):
        print(str(file))
        run(f"isort {file} > /dev/null; yapf -i {file} > /dev/null", shell=True)


def strip_type_hints(fp):
    """Stip type hints from source files.
    Parameters
    ----------
    fp: str
        The file path to search recursively for python files.
    """
    for file in Path(fp).rglob("*.py"):
        print(str(file))
        run(f"strip-hints {file}")


if "--strip_hints" in argv:
    strip_type_hints("pypython")
    strip_type_hints("scripts")

format_source("pypython")
format_source("scripts")
