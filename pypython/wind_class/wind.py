import pypython.wind_class.plot


class Wind(pypython.wind_class.plot.WindPlot):
    def __init__(self, root, directory, **kwargs):
        super().__init__(root, directory, **kwargs)
