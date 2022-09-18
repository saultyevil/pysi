#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pypython.wind.properties


class WindPlot(pypython.wind.properties.WindProperties):
    """An extension to the WindGrid base class which adds various plotting
    functionality.
    """

    def __init__(self, root: str, directory: str, **kwargs):
        super().__init__(root, directory, **kwargs)

    def plot(self):
        pass
