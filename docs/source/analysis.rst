Analysing models
================

pypython has multiple classes which can be used to read simulation data
into memory. The two main classes for this are :code:`pypython.Spectrum` and
:code:`pypython.Wind`, which will read in all of the available spectra and wind
save tables respectively.

pypython also has a large collection of plotting functions to present the data
from a simulation, such as creating figures of the observer spectra or for
creating diagnostic plots to determine the ionization structure of the wind.

The spectrum
------------

The main type of output from Python are the :code:`.spec` files.

Reading the files
^^^^^^^^^^^^^^^^^

::

  import pypython
  spec = pypython.Spectrum("input", distance=1e6, smooth=5)

  spec.available  # the list of spectra which have been read in
  spec.set("spec_tot")  # sets the spectrum

Plotting
^^^^^^^^

aaa

The wind
--------

The cell spectra
----------------

The optical depth
-----------------

Reprocessing processes
----------------------
