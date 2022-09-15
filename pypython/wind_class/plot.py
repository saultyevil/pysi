import pypython.wind_class.grid


class WindPlot(pypython.wind_class.grid.WindGrid):
    def __init__(self, root, directory, **kwargs):
        super().__init__(root, directory, **kwargs)

        self.plot_this = "do it"

    def plot(self):
        print("You are making a plot", self.plot_this)
