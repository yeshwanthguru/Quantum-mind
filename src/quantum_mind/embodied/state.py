"""State slot of the VLA: a Kalman estimator for the joint that can use the library's accelerometer models.

The joint is a linear stage of half-stroke ``stroke`` metres (the workspace [-1, 1] maps to +/- stroke). An accelerometer on the
carriage is a :class:`quantum_mind.sensing.Sensor` (MEMS or atom interferometer; illustrative values). The estimator state is
position, velocity and the MEMS bias. Modes, from least to most informed: raw encoder; encoder-only Kalman; Kalman with the MEMS
accelerometer as the motion input; the classical rival 'kalman + motor model' predicts with the nominal plant and the command
instead of an accelerometer; the same plus the slow quantum accelerometer, whose near-bias-free samples measure the MEMS bias
(``a_mems - a_quantum = bias + noise``). The encoder can drop out (cable fault, occlusion), which is when the difference shows.
"""
from __future__ import annotations

import numpy as np

from ..sensing import atom_interferometer_accelerometer, mems_accelerometer
from .motor import OneMotor

MODES = ('encoder', 'kalman, encoder only', 'kalman + motor model', 'kalman + MEMS', 'kalman + MEMS + quantum')


class SensedMotor:
    """A :class:`OneMotor` plus an encoder (which can drop out) and the MEMS and quantum accelerometers, all simulated."""

    def __init__(self, motor=None, stroke=0.2, mems=None, quantum=None, enc_std=0.005, outage_at=None, disturbance=0.0, faults=()):
        self.disturbance = disturbance
        self.faults = [dict(f) for f in faults]                 # {'kind': ..., 't': seconds, 'mag': ...}, see :meth:`step`
        self.m = motor or OneMotor()
        self.stroke, self.enc_std, self.outage_at = stroke, enc_std, outage_at
        self.mems, self.quantum = mems or mems_accelerometer(), quantum or atom_interferometer_accelerometer()

    def reset(self, theta0, rng):
        self.rng, self.t, self.k = rng, 0.0, 0
        self.enc_offset = 0.0
        for f in self.faults:
            f['done'] = False
        self.m.reset(theta0, rng)
        self.m.ext = self.disturbance * rng.uniform(-1, 1)                    # constant unmodelled push, workspace units/s^2
        self.bias = self.mems.bias_std0 * rng.standard_normal()                 # m/s^2
        self.per = max(int(round(self.mems.rate_hz * self.m.dt)), 1)            # MEMS samples averaged per control step
        self.q_every = max(int(round(1.0 / (self.quantum.rate_hz * self.m.dt))), 1)         # one quantum sample per interrogation cycle
        return self.theta0_reading(theta0)

    def theta0_reading(self, theta0):
        return float(theta0 + self.enc_std * self.rng.standard_normal())

    def step(self, u):
        """Advance one control step. Faults fire when time passes their ``t``: ``encoder_dropout`` (no reading), ``encoder_offset``
        (reading shifted by ``mag``), ``gain_loss`` (motor gain times ``mag``), ``friction`` (Coulomb friction times ``mag``),
        ``push`` (extra constant external acceleration ``mag``)."""
        for f in self.faults:
            if not f['done'] and self.t >= f['t']:
                f['done'] = True
                k, mag = f['kind'], f['mag']
                if k == 'encoder_dropout':
                    self.outage_at = self.t
                elif k == 'encoder_offset':
                    self.enc_offset = mag
                elif k == 'gain_loss':
                    self.m.g *= mag
                elif k == 'friction':
                    self.m.coulomb *= mag
                elif k == 'push':
                    self.m.ext += mag
        self.m.step(u)
        self.t += self.m.dt
        self.k += 1
        a = self.m.acc * self.stroke                                           # true acceleration, m/s^2
        self.bias += self.mems.bias_walk * np.sqrt(self.m.dt) * self.rng.standard_normal()
        a_mems = a + self.bias + self.mems.noise_std / np.sqrt(self.per) * self.rng.standard_normal()
        a_q = a + self.quantum.noise_std * self.rng.standard_normal() if self.k % self.q_every == 0 else None
        out = self.outage_at is not None and self.t >= self.outage_at
        enc = None if out else float(self.m.theta + self.enc_offset + self.enc_std * self.rng.standard_normal())
        return dict(enc=enc, a_mems=a_mems, a_q=a_q)


class JointEstimator:
    """Kalman filter for (position, velocity, MEMS bias) in workspace units; ``mode`` is one of :data:`MODES`."""

    def __init__(self, mode, dt=0.05, stroke=0.2, mems=None, quantum=None, enc_std=0.005, vel_process=3.0):
        assert mode in MODES
        self.mode, self.dt, self.s, self.enc_std = mode, dt, stroke, enc_std
        self.mems, self.quantum = mems or mems_accelerometer(), quantum or atom_interferometer_accelerometer()
        self.per = max(int(round(self.mems.rate_hz * dt)), 1)
        self.vel_process = vel_process
        self.plant = None                    # optional (gain, damping, Coulomb, push) replacing the nominal plant in 'kalman + motor model'
        self.quantum_bias = False            # also apply the quantum bias update in the motor-model mode

    def uses_accel(self):
        return self.mode in ('kalman + MEMS', 'kalman + MEMS + quantum')

    def reset(self, z0):
        self.x = np.array([z0, 0.0, 0.0])
        self.P = np.diag([self.enc_std ** 2, 1.0, (self.mems.bias_std0 / self.s) ** 2])
        self.last, self.vel = z0, 0.0

    def step(self, meas, u=0.0):
        dt, s = self.dt, self.s
        if self.mode == 'encoder':                                              # raw reading, held when it drops out
            if meas['enc'] is not None:
                self.vel += 0.4 * ((meas['enc'] - self.last) / dt - self.vel)
                self.last = meas['enc']
            return self.last, self.vel
        # the bias enters the motion through the accelerometer input, so position and velocity depend on it (columns 3 of F)
        F = np.array([[1, dt, -0.5 * dt * dt if self.uses_accel() else 0.0], [0, 1, -dt if self.uses_accel() else 0.0], [0, 0, 1.0]])
        if self.mode == 'kalman, encoder only':
            u = 0.0
            Q = np.diag([0.0, (self.vel_process * dt) ** 2, 0.0])
            self.x = F @ self.x
        elif self.mode == 'kalman + motor model':
            nom = OneMotor(randomise=False)                       # nominal plant; the real gain and damping differ by up to 20%
            g, b, c, d = self.plant if self.plant is not None else (nom.gain, nom.damping, nom.coulomb, 0.0)
            acc = g * float(np.clip(u, -1, 1)) - b * self.x[1] - c * np.tanh(self.x[1] / 0.02) + d
            self.x = F @ self.x + np.array([0.5 * dt * dt * acc, dt * acc, 0.0])
            Q = np.diag([0.0, (1.5 * dt) ** 2, 0.0])
        else:
            acc = meas['a_mems'] / s                                            # raw reading, workspace units / s^2; F carries the -bias term
            self.x = F @ self.x + np.array([0.5 * dt * dt * acc, dt * acc, 0.0])
            q_a = (self.mems.noise_std / np.sqrt(self.per) / s * dt) ** 2
            q_b = (self.mems.bias_walk / s) ** 2 * dt
            Q = np.diag([0.25 * dt ** 4 * q_a / dt ** 2, q_a, q_b])
        self.P = F @ self.P @ F.T + Q + 1e-12 * np.eye(3)
        if meas['enc'] is not None:
            self._update(np.array([1.0, 0, 0]), meas['enc'], self.enc_std ** 2)
        if (self.mode == 'kalman + MEMS + quantum' or (self.quantum_bias and self.mode == 'kalman + motor model')) and meas['a_q'] is not None:
            r = (self.mems.noise_std / np.sqrt(self.per)) ** 2 + self.quantum.noise_std ** 2
            self._update(np.array([0, 0, 1.0]), (meas['a_mems'] - meas['a_q']) / s, r / s ** 2)
        return float(self.x[0]), float(self.x[1])

    def _update(self, H, z, r):
        S = H @ self.P @ H + r
        K = self.P @ H / S
        self.x = self.x + K * (z - H @ self.x)
        self.P = (np.eye(3) - np.outer(K, H)) @ self.P
