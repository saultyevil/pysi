#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions and classes used to manipulate and plot wind save tables, for
Python simuations."""

import os
from sys import exit
from typing import List, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt

from .math import vector
from .physics.constants import CMS_TO_KMS, PI, C
from .plot import normalize_figure_style
from .util import create_wind_save_tables, get_array_index
