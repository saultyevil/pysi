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

from pypython import cleanup_data

if len(argv) == 1:
    cleanup_data()
elif len(argv) == 2:
    if argv[1] == "-h" or argv[1] == "--help":
        print(__doc__)
        exit(0)
    cleanup_data(argv[1])
else:
    print("Unknown number of arguments.")
    print(__doc__)
