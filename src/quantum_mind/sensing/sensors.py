"""Inertial sensor models: white noise, a drifting bias and a sample rate.

A :class:`Sensor` measures the acceleration along one axis as ``a + bias + noise``. The bias follows a random walk. These
are the standard first-order error terms of an accelerometer (Allan-variance noise terms: velocity random walk and bias
instability); the two presets differ in exactly the way that matters for fusion: a MEMS accelerometer is fast and drifts,
a cold-atom interferometer is slow and almost drift-free.

**The preset values are illustrative assumptions, not measurements of any product.** The atom-interferometer noise is
derived from the quantum limit (:func:`quantum_mind.sensing.atom_interferometer_sensitivity`) for the atom number and
interrogation time you pass; its bias walk is an assumption. Set every number for the sensors you care about.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .limits import atom_interferometer_sensitivity

__all__ = ['Sensor', 'mems_accelerometer', 'atom_interferometer_accelerometer']

G = 9.80665


@dataclass(frozen=True)
class Sensor:
    """One accelerometer.

    Attributes
    ----------
    name : str
    rate_hz : float
        Samples per second.
    noise_std : float
        Standard deviation of the white noise of one sample, in m/s^2.
    bias_walk : float
        Bias random-walk density in (m/s^2)/sqrt(s).
    bias_std0 : float
        Standard deviation of the initial bias in m/s^2.
    """
    name: str
    rate_hz: float
    noise_std: float
    bias_walk: float
    bias_std0: float

    def __post_init__(self):
        if self.rate_hz <= 0 or self.noise_std <= 0 or self.bias_walk < 0 or self.bias_std0 < 0:
            raise ValueError('rate and noise must be positive, bias terms non-negative')


def mems_accelerometer(rate_hz=100.0, noise_density_ug=100.0, bias_walk=2e-4, bias_std0=0.02):
    """A MEMS accelerometer: fast, noisy, with a drifting bias (illustrative values).

    Parameters
    ----------
    rate_hz : float, optional
        Output rate.
    noise_density_ug : float, optional
        Velocity random walk expressed as an acceleration noise density in micro-g per root hertz.
    bias_walk : float, optional
        Bias random-walk density in (m/s^2)/sqrt(s).
    bias_std0 : float, optional
        Initial bias spread in m/s^2.

    Returns
    -------
    Sensor
    """
    sigma = noise_density_ug * 1e-6 * G * np.sqrt(rate_hz)
    return Sensor('MEMS accelerometer', rate_hz, float(sigma), bias_walk, bias_std0)


def atom_interferometer_accelerometer(n_atoms=1e6, interrogation_time=0.1, cycle_time=1.0, contrast=0.5,
                                      bias_walk=1e-8, bias_std0=1e-6):
    """A cold-atom interferometer accelerometer: slow, very sensitive, almost drift-free (illustrative values).

    The per-shot noise comes from the standard quantum limit for ``n_atoms`` and ``interrogation_time``; the rate is
    one sample per ``cycle_time`` (atom preparation and detection dominate it). The bias walk and initial bias are
    assumptions.

    Returns
    -------
    Sensor
    """
    sigma = atom_interferometer_sensitivity(n_atoms, interrogation_time, contrast)
    return Sensor('atom interferometer', 1.0 / cycle_time, float(sigma), bias_walk, bias_std0)
