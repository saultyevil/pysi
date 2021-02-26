Generating spectra from the raw photons
=======================================

By using reverberation mapping filtering functionality, or by hacking Python
yourself, it is possible to dump out photons as they interact and contribute
to the final spectrum. The weight of these photons can be binned and converted
into a flux by using :code:`pypython.createspectrum`.
