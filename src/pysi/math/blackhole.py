"""Calculate parameters relating to Schwarzchild black holes.

All functions require the mass to be in units of solar masses and
accretion rates in solar masses per year.
"""

from astropy.constants import G, M_sun, c


def gravitational_radius(m_bh: float) -> float:
    """Calculate the gravitational radius of a black hole of given mass Mbh.

    Parameters
    ----------
    m_bh: float
        The mass of the black hole in solar masses.

    Returns
    -------
    The gravitational radius in cm.

    """
    return G.cgs.value * m_bh * M_sun.cgs.value / c.cgs.value**2


def schwarzschild_radius(m_bh: float) -> float:
    """Calculate the Schwarzschild radius of a black hole of given mass Mbh.

    Parameters
    ----------
    m_bh: float
        The mass of the black hole in solar masses.

    Returns
    -------
    The Schwarzschild radius in cm.

    """
    return 2 * gravitational_radius(m_bh)
