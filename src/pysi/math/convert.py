from astropy.constants import c

from pysi.math.constants import ANGSTROM


def angstrom_to_hz(wavelength: float) -> float:
    """Convert a wavelength from Angstroms into a frequency.

    Parameters
    ----------
    wavelength: float
        The wavelength in Angstroms.

    Returns
    -------
    The frequency in Hertz.

    """
    return c.cgs / (wavelength * ANGSTROM)


def hz_to_angstrom(frequency: float) -> float:
    """Convert a frequency in Hz to a wavelength in Angstroms.

    Parameters
    ----------
    frequency: float
        The frequency in Hz.

    Returns
    -------
    The wavelength in Angstroms.

    """
    return c.cgs / frequency / ANGSTROM
