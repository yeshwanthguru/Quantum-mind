r"""Quantum limits of a phase measurement, and the acceleration sensitivity of an atom interferometer.

A sensor that reads a phase :math:`\varphi` from :math:`N` independent probes (atoms, photons) repeated :math:`\nu`
times cannot do better than the **standard quantum limit**,

.. math:: \Delta\varphi_{\mathrm{SQL}} = \frac{1}{\sqrt{N\nu}} ,

while entangled probes can in principle reach the **Heisenberg limit** :math:`\Delta\varphi_{\mathrm{H}} = 1/(N\sqrt{\nu})`.
Entangled states are fragile, so real instruments sit near the standard quantum limit.

A light-pulse atom interferometer measures a constant acceleration :math:`a` as a phase
:math:`\varphi = k_{\mathrm{eff}}\, a\, T^2`, with :math:`k_{\mathrm{eff}} = 4\pi/\lambda` the effective two-photon wave
number and :math:`T` the time between pulses (Kasevich and Chu, 1992). Its single-shot acceleration sensitivity is therefore
:math:`\Delta a = \Delta\varphi/(C\, k_{\mathrm{eff}} T^2)` with :math:`C` the fringe contrast.
"""
from __future__ import annotations

import numpy as np

__all__ = ['phase_sensitivity', 'atom_interferometer_sensitivity', 'RB87_WAVELENGTH_M']

#: Wavelength of the rubidium-87 D2 line, a common choice for atom interferometers, in metres.
RB87_WAVELENGTH_M = 780.24e-9


def phase_sensitivity(n_probes, repeats=1, scaling='sql'):
    """Smallest phase resolvable with ``n_probes`` probes repeated ``repeats`` times.

    Parameters
    ----------
    n_probes : float
        Number of probes per shot (atoms or photons), at least 1.
    repeats : float, optional
        Number of independent shots.
    scaling : {'sql', 'heisenberg'}, optional
        Standard quantum limit :math:`1/\\sqrt{N\\nu}` or Heisenberg limit :math:`1/(N\\sqrt{\\nu})`.

    Returns
    -------
    float
        Phase uncertainty in radians.

    Raises
    ------
    ValueError
        For fewer than one probe or one repeat, or an unknown scaling.

    Examples
    --------
    >>> from quantum_mind.sensing import phase_sensitivity
    >>> round(phase_sensitivity(10_000), 4), round(phase_sensitivity(10_000, scaling='heisenberg'), 6)
    (0.01, 0.0001)
    """
    if n_probes < 1 or repeats < 1:
        raise ValueError('need at least one probe and one repeat')
    if scaling == 'sql':
        return float(1.0 / np.sqrt(n_probes * repeats))
    if scaling == 'heisenberg':
        return float(1.0 / (n_probes * np.sqrt(repeats)))
    raise ValueError("scaling must be 'sql' or 'heisenberg'")


def atom_interferometer_sensitivity(n_atoms, interrogation_time, contrast=0.5, wavelength=RB87_WAVELENGTH_M,
                                    scaling='sql'):
    """Single-shot acceleration sensitivity of a light-pulse atom interferometer.

    Parameters
    ----------
    n_atoms : float
        Atoms detected per shot.
    interrogation_time : float
        Time ``T`` between pulses in seconds.
    contrast : float, optional
        Fringe contrast in (0, 1].
    wavelength : float, optional
        Laser wavelength in metres (default: rubidium D2).
    scaling : {'sql', 'heisenberg'}, optional

    Returns
    -------
    float
        Acceleration uncertainty per shot in m/s^2.

    Examples
    --------
    >>> from quantum_mind.sensing import atom_interferometer_sensitivity
    >>> a = atom_interferometer_sensitivity(1e6, 0.1)
    >>> 1e-9 < a < 1e-7
    True
    """
    if interrogation_time <= 0 or not 0 < contrast <= 1:
        raise ValueError('interrogation_time must be positive and contrast in (0, 1]')
    k_eff = 4 * np.pi / wavelength
    return float(phase_sensitivity(n_atoms, 1, scaling) / (contrast * k_eff * interrogation_time ** 2))
