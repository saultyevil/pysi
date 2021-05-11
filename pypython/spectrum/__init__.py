#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contains the spectrum object, as well as utility and plotting functions for
spectra."""

import copy
import os
import textwrap
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import boxcar, convolve

from pypython.error import InvalidParameter

from . import get_root, smooth_array
from .physics.constants import PARSEC
from .plot import (ax_add_line_ids, common_lines, get_y_lims_for_x_lims, normalize_figure_style, photoionization_edges,
                   remove_extra_axes, subplot_dims)
