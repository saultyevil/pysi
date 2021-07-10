#!/usr/bin/env python

import pypython

spectrum = pypython.Spectrum("tde_opt_cmf", log_spec=True, default="spec_tot")

print(spectrum.units)
spectrum.plot("Emitted")
spectrum.show()

ltot = pypython.spectrum.integrate_spectrum(spectrum, "Emitted", 3000, 8000)
print(ltot)