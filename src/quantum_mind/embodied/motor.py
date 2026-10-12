"""A single DC-motor joint: simulated dynamics, a noisy encoder and a privileged PD expert."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OneMotor:
    """One position-controlled motor joint (simulated).

    ``dtheta = omega``, ``domega = gain*u - damping*omega - coulomb*tanh(omega/0.02)`` with the command ``u`` clipped
    to [-1, 1]. The joint position is normalised to the workspace [-1, 1]. The encoder reads the position with Gaussian
    noise; the velocity estimate is a filtered finite difference of the noisy readings, as on a real drive.

    ``randomise`` draws the gain and damping per episode (domain randomisation), so a policy cannot memorise one plant.
    """
    gain: float = 8.0
    damping: float = 2.0
    coulomb: float = 0.3
    dt: float = 0.05
    encoder_noise: float = 0.005
    velocity_filter: float = 0.4
    randomise: bool = True

    def reset(self, theta0, rng):
        self.rng = rng
        self.g = self.gain * (rng.uniform(0.8, 1.2) if self.randomise else 1.0)
        self.b = self.damping * (rng.uniform(0.8, 1.2) if self.randomise else 1.0)
        self.theta, self.omega = float(theta0), 0.0
        self._last = self.theta + self.encoder_noise * rng.standard_normal()
        self._vel = 0.0
        return self.measure(first=True)

    def measure(self, first=False):
        """Noisy ``(theta, omega)`` as the controller sees them."""
        if first:
            return self._last, 0.0
        z = self.theta + self.encoder_noise * self.rng.standard_normal()
        raw = (z - self._last) / self.dt
        self._vel += self.velocity_filter * (raw - self._vel)
        self._last = z
        return z, self._vel

    def step(self, u):
        u = float(np.clip(u, -1.0, 1.0))
        acc = self.g * u - self.b * self.omega - self.coulomb * np.tanh(self.omega / 0.02)
        self.omega += self.dt * acc
        self.theta = float(np.clip(self.theta + self.dt * self.omega, -1.0, 1.0))
        return self.measure()


def expert_command(target, theta, omega, kp=3.0, kd=1.0):
    """Saturated PD command toward ``target`` (used as the demonstrator; it sees the true target)."""
    return np.clip(kp * (np.asarray(target, float) - theta) - kd * omega, -1.0, 1.0)
