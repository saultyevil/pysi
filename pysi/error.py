#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Error codes and custom exceptions for pysi.

These are pretty basic, and don't do much.
"""

EXIT_SUCCESS = 0
EXIT_FAIL = 1


class CoordError(Exception):
    """Exception for when an incorrect coordinate system is used."""


class DimensionError(Exception):
    """Exception for when arrays with incorrect dimensions have been
    supplied."""


class InvalidFileContents(Exception):
    """Exception for when a file has a different contents than expected."""


class InvalidParameter(Exception):
    """Exception for when a parameter is not recognised."""


class PythonError(Exception):
    """Exception for when Python has broken."""


class RunError(Exception):
    """Exception for when windsave2table, or etc., have failed to run."""
