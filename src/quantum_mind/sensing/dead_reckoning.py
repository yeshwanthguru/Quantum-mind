r"""One-axis dead reckoning with several accelerometers and a multi-rate Kalman filter.

A robot moves along a line with an unknown, smoothly changing acceleration :math:`a(t)` (an Ornstein-Uhlenbeck process). Each
:class:`~quantum_mind.sensing.Sensor` reports :math:`a + b_k + n` at its own rate, where :math:`b_k` is a random-walk bias. A
Kalman filter with state :math:`(p, v, a, b_1, \dots, b_K)` fuses whichever sensors are in use and, optionally, occasional
noisy position fixes (a landmark, a wheel-odometry reset). Integrating a biased acceleration twice makes position error
grow, so what matters is how well the filter can *estimate the bias*: a slow, almost drift-free sensor can calibrate a fast,
drifting one. :func:`compare_sensor_suites` runs the same simulated trajectories through several sensor suites and reports
position error, so the benefit of the quantum sensor (and where it vanishes) can be measured instead of assumed.

The filter is the standard linear Kalman filter with the true noise parameters (a matched filter, an upper bound on what a
real tuned filter gets). The simulation is one axis and illustrative; it is not a navigation system.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ['simulate_dead_reckoning', 'run_filter', 'compare_sensor_suites', 'DeadReckoningData']


@dataclass
class DeadReckoningData:
    """A simulated trajectory with the measurements of every sensor.

    Attributes
    ----------
    dt : float
    pos : numpy.ndarray
        True position at every step.
    accel : numpy.ndarray
        True acceleration at every step.
    readings : dict
        ``{sensor name: array}`` of what the sensor outputs at every step (only the sampled steps are used).
    fixes : numpy.ndarray
        Noisy position at every step (used only at the fix instants).
    sensors : list of Sensor
    accel_std, accel_tau, fix_period, fix_std : float
        Trajectory and fix parameters, needed by the filter.
    """
    dt: float
    pos: np.ndarray
    accel: np.ndarray
    readings: dict
    fixes: np.ndarray
    sensors: list
    accel_std: float
    accel_tau: float
    fix_period: float
    fix_std: float


def simulate_dead_reckoning(sensors, duration=300.0, dt=0.01, accel_std=0.5, accel_tau=2.0, fix_period=None, fix_std=0.1,
                            seed=0):
    """Simulate a trajectory and the readings of every sensor.

    Parameters
    ----------
    sensors : list of Sensor
    duration : float, optional
        Seconds.
    dt : float, optional
        Time step; must be at most the shortest sensor period.
    accel_std, accel_tau : float, optional
        Standard deviation (m/s^2) and correlation time (s) of the true acceleration.
    fix_period : float, optional
        Seconds between position fixes (``None`` for none).
    fix_std : float, optional
        Standard deviation of a position fix in metres.
    seed : int, optional

    Returns
    -------
    DeadReckoningData
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration / dt))
    decay = np.exp(-dt / accel_tau)
    a = np.empty(n)
    a[0] = rng.normal(0, accel_std)
    shocks = rng.normal(0, accel_std * np.sqrt(1 - decay ** 2), n)
    for k in range(1, n):
        a[k] = decay * a[k - 1] + shocks[k]
    v = np.zeros(n)
    p = np.zeros(n)
    for k in range(1, n):
        p[k] = p[k - 1] + v[k - 1] * dt + 0.5 * a[k - 1] * dt ** 2
        v[k] = v[k - 1] + a[k - 1] * dt
    readings = {}
    for s in sensors:
        bias = s.bias_std0 * rng.normal() + np.concatenate([[0.0], np.cumsum(rng.normal(0, s.bias_walk * np.sqrt(dt), n - 1))])
        readings[s.name] = a + bias + rng.normal(0, s.noise_std, n)
    fixes = p + rng.normal(0, fix_std, n)
    return DeadReckoningData(dt, p, a, readings, fixes, list(sensors), accel_std, accel_tau, fix_period, fix_std)


def run_filter(data, use=None, use_fixes=False):
    """Run the multi-rate Kalman filter on a simulated trajectory.

    Parameters
    ----------
    data : DeadReckoningData
    use : list of str, optional
        Names of the sensors the filter may use (default: all).
    use_fixes : bool, optional
        Whether to use the position fixes (requires ``data.fix_period``).

    Returns
    -------
    numpy.ndarray
        Position error (estimate minus truth) at every step.
    """
    sensors = [s for s in data.sensors if use is None or s.name in use]
    if not sensors:
        raise ValueError('no sensors selected')
    K, dt = len(sensors), data.dt
    n = len(data.pos)
    dim = 3 + K
    decay = np.exp(-dt / data.accel_tau)
    F = np.eye(dim)
    F[0, 1], F[0, 2], F[1, 2], F[2, 2] = dt, 0.5 * dt ** 2, dt, decay
    Q = np.zeros((dim, dim))
    Q[2, 2] = data.accel_std ** 2 * (1 - decay ** 2)
    for i, s in enumerate(sensors):
        Q[3 + i, 3 + i] = s.bias_walk ** 2 * dt
    x = np.zeros(dim)
    P = np.diag([1e-6, 1e-6, data.accel_std ** 2] + [s.bias_std0 ** 2 + 1e-12 for s in sensors])
    periods = [max(1, int(round(1.0 / (s.rate_hz * dt)))) for s in sensors]
    fix_every = int(round(data.fix_period / dt)) if (use_fixes and data.fix_period) else None
    err = np.empty(n)
    err[0] = 0.0
    Hs = []
    for i in range(K):
        h = np.zeros(dim)
        h[2], h[3 + i] = 1.0, 1.0
        Hs.append(h)
    hp = np.zeros(dim)
    hp[0] = 1.0

    def update(x, P, h, z, r):
        S = h @ P @ h + r
        gain = P @ h / S
        x = x + gain * (z - h @ x)
        P = P - np.outer(gain, h @ P)
        return x, P

    for k in range(1, n):
        x = F @ x
        P = F @ P @ F.T + Q
        for i, s in enumerate(sensors):
            if k % periods[i] == 0:
                x, P = update(x, P, Hs[i], data.readings[s.name][k], s.noise_std ** 2)
        if fix_every and k % fix_every == 0:
            x, P = update(x, P, hp, data.fixes[k], data.fix_std ** 2)
        err[k] = x[0] - data.pos[k]
    return err


def compare_sensor_suites(sensors, suites, duration=300.0, dt=0.01, accel_std=0.5, accel_tau=2.0, fix_period=None,
                          fix_std=0.1, seeds=range(10)):
    """Run the same simulated trajectories through several sensor suites.

    Parameters
    ----------
    sensors : list of Sensor
        Every sensor that exists in the simulation.
    suites : dict
        ``{label: list of sensor names}`` to give the filter.
    duration, dt, accel_std, accel_tau, fix_period, fix_std : optional
        As in :func:`simulate_dead_reckoning`; the filter uses position fixes whenever ``fix_period`` is set.
    seeds : iterable of int, optional
        One simulated trajectory per seed.

    Returns
    -------
    list of dict
        One row per suite with ``suite``, ``rmse`` and ``final_error`` as ``(mean, std)`` over seeds (metres), and
        ``rmse_per_seed`` (the list), so a paired comparison is possible.
    """
    errs = {label: [] for label in suites}
    for seed in seeds:
        data = simulate_dead_reckoning(sensors, duration, dt, accel_std, accel_tau, fix_period, fix_std, seed)
        for label, names in suites.items():
            errs[label].append(run_filter(data, names, use_fixes=bool(fix_period)))
    rows = []
    for label, runs in errs.items():
        rmse = [float(np.sqrt(np.mean(e ** 2))) for e in runs]
        final = [float(abs(e[-1])) for e in runs]
        rows.append(dict(suite=label, rmse=(float(np.mean(rmse)), float(np.std(rmse))),
                         final_error=(float(np.mean(final)), float(np.std(final))), rmse_per_seed=rmse))
    return rows
