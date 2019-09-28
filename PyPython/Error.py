#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains various custom exceptions which are raised throughout to give
more detailed error messages.
"""


class CoordError(Exception):
    """Exception for when an incorrect coordinate system is used"""
    pass


class DimensionError(Exception):
    """Exception for when arrays with incorrect dimensions have been supplied"""
    pass


class PythonError(Exception):
    """Exception for when Python has broken"""
    pass


class InvalidFileContents(Exception):
    """Exception for when a file has a different contents than expected"""


class InvalidParameter(Exception):
    """Exception for when a parameter is not recognised"""
