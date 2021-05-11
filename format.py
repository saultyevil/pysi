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
    style = "'{based_on_style: pep8, column_limit: 120}'"
    for file in Path(fp).rglob("*.py"):
        print("  -", str(file))
        run(f"isort {file} > /dev/null; yapf -i --style={style} {file} > /dev/null", shell=True)


def strip_type_hints(fp):
    """Stip type hints from source files.
    Parameters
    ----------
    fp: str
        The file path to search recursively for python files.
    """
    for file in Path(fp).rglob("*.py"):
        print("  -", str(file))
        run(f"strip-hints {file} > tmp.txt; mv tmp.txt {file}", shell=True)


if "--strip-hints" in argv:
    print("Stripping type hints:")
    strip_type_hints("pypython")
    strip_type_hints("scripts")

print("Reformating source files:")
format_source("pypython")
format_source("scripts")
