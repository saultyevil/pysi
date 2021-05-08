#!/usr/bin/env python

import automodinit
import pathlib
import re
from typing import List

name = "pypython"

# Import all files using automodinit

__all__ = ["I will get rewritten"]
automodinit.automodinit(__name__, __file__, globals())
del automodinit


# Functions


def getf(pattern: str, fp: str = "."):
    """Find files of the given pattern recursively.

    Parameters
    ----------
    pattern: str
        Patterns to search recursively for, i.e. *.pf, *.spec, tde_std.pf
    fp: str [optional]
        The directory to search from, if not specified in the pattern.
    """

    files = pathlib.Path(f"{fp}").rglob(pattern)

    files = [str(thisfile) for thisfile in files]
    if ".pf" in pattern:
        files = [thisfile for thisfile in files if "out.pf" not in thisfile and "py_wind" not in thisfile]
    files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    return files
