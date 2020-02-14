#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Remove all data directories from Python simulations - this was written because
Dropbox is able to follow symbolic links on Linux >:(.

Usage:
        python py_rm_data.py [directory]

        directory: a string for the path to the base directory to search
                   recursively from for data symbolic links
"""

from sys import argv, exit
from PyPython.PythonUtils import remove_data_sym_links


print("--------------------------------------------------------------------------------\n")
if len(argv) > 1:
    if argv[1] == "-h" or argv[1] == "--help":
        print(__doc__)
        exit(0)
    remove_data_sym_links(argv[1], True)
else:
    remove_data_sym_links(verbose=True)
print("\n--------------------------------------------------------------------------------")
