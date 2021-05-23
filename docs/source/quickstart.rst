Quickstart
==========

Requirements
------------

Python 3.6 or above, plus all of the packages in :code:`requirements.txt`.
Having the Monte Carlo radiative transfer and ionization code Python installed
is also a requirement for some of the functions, but isn't required if you
wish to only access the spectrum and most of the utility functions.

Installation
------------

pypython can be installed using :code:`pip` or using :code:`setup.py` -- I don't think
it really matters. pypython can also be used by adding the pypython directory
into your python path.

If you wish to go down the :code:`pip` installation method, then it is recommended
that you install pypython using editable mode. In the root directory, use,

::

    pip install -e .

All of the requirements will be installed during this and if everything has
worked, you should see :code:`Successfully installed pypython` printed to
the screen.

Example usage
-------------

Plotting a spectrum file
^^^^^^^^^^^^^^^^^^^^^^^^

::

    import pypython

    # Create a plot directly using the class
    s = pypython.Spectrum("cv_standard", smooth=3)
    fig1, ax1 = s.plot()       # Plot the the components and observer spectra
    fig2, ax2 = s.plot("62")
    s.show()

    # Or, create a plot using what is contained in plot.spectrum
    fig, ax = pypython.plot.spectrum.spectrum_observer(s, "all", use_flux=True, label_lines=True, display=True)

    # And to plot, i.e., the spec_tot file
    s.set("spec_tot")
    fig, ax = pypython.plot.spectrum.spectrum_components(s, display=True)

