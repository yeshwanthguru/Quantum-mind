r"""Scheduling a slow quantum sensor with a QUBO: thread 1 (quantum computing) applied to thread 2 (quantum sensing).

A cold-atom accelerometer is slow and costs time and power, so on a mission only some of the possible interrogation slots can be used.
Which ones? The answer depends on the Kalman filter that fuses the quantum sensor with a fast, drifting MEMS sensor (see
:mod:`quantum_mind.sensing.dead_reckoning`): a quantum reading pins the MEMS bias, and after it the bias drifts again, so the position
error grows until the next reading. Two things make the choice non-trivial: the mission cares about accuracy at some times more than at
others (``weights``), and a slot is usable only if the interferometer's fringe contrast is high enough there (``contrast``; vibration
and rotation wash it out).

The method (:func:`schedule_interrogations`):

1. **Exact objective.** For any chosen set of slots, the weighted position variance over the mission follows from the Kalman covariance
   recursion, with no simulation noise (:meth:`InterrogationProblem.objective`). It is the quantity every method is judged on.
2. **Local quadratic QUBO.** Around a current schedule :math:`x^0`, the weighted variance is expanded to second order in the flips
   :math:`\delta_i=x_i-x^0_i`, with coefficients taken from exact covariance runs of every single flip and every pair of flips
   (:meth:`InterrogationProblem.local_quadratic`). The model is exact for every schedule within two flips of :math:`x^0`, and
   choosing exactly :math:`k` slots under it is the library's :func:`~quantum_mind.computing.grasp_selection` QUBO, solved by any
   of its solvers (exact,
   annealing, greedy descent, random sampling, Qiskit QAOA). From the empty schedule this is the inclusion-exclusion expansion,
   which fails
   when readings are nearly redundant (the first reading does most of the work), hence the expansion around a good schedule.
3. **Iterate.** The QUBO's solution becomes the new expansion point if the exact objective improves; otherwise stop (a trust-region
   style loop, so the result is never worse than the start). The start is the uniform spacing unless given.
4. **Polish (optional).** A swap local search on the exact objective.

The QUBO is a local approximation, so this does not claim to find the exact optimum; the baselines (uniform spacing, greedy on the exact
objective, random, the exact optimum by enumeration) say how good it is. A quantum solver is a solver of the QUBO and nothing more: it
gives no advantage here, and the benchmark measures that rather than assuming it.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from ..computing import (exhaustive, grasp_selection, greedy_descent, qaoa, random_sampling, simulated_annealing)
from .sensors import atom_interferometer_accelerometer, mems_accelerometer

__all__ = ['InterrogationProblem', 'Schedule', 'schedule_interrogations', 'random_mission']

SOLVERS = {'exact': exhaustive, 'annealing': simulated_annealing, 'greedy descent': greedy_descent, 'random sampling': random_sampling,
           'qaoa': qaoa}


@dataclass
class Schedule:
    """A chosen set of slots.

    Attributes
    ----------
    slots : list of int
        Indices into ``problem.slot_times``.
    objective : float
        Weighted RMS position standard deviation over the mission, in metres (exact, from the covariance recursion).
    method : str
    info : dict
        Method-specific details (for the QUBO methods: ``tau``, ``qubo_energy``, ``polished``).
    """
    slots: list
    objective: float
    method: str
    info: dict = field(default_factory=dict)


class InterrogationProblem:
    """Choose ``k`` interrogation slots of a quantum accelerometer on a mission, to minimise weighted position error.

    Parameters
    ----------
    slot_times : array_like
        Times (s) at which an interrogation can start.
    contrast : array_like
        Fringe contrast in [0, 1] at each slot; a slot below ``min_contrast`` cannot be used, otherwise the reading noise is the sensor's
        noise divided by the contrast.
    weights : array_like or callable
        Importance of position accuracy over time: a function of time (s) or an array on the grid ``arange(0, duration, dt)``.
    k : int
        Number of slots to use.
    duration, dt : float, optional
        Mission length and the time step of the covariance recursion (seconds).
    accel_std, accel_tau : float, optional
        Standard deviation and correlation time of the unknown acceleration (as in :func:`simulate_dead_reckoning`).
    mems, quantum : Sensor, optional
        The two sensors (default: the library's illustrative models).
    min_contrast : float, optional
    """

    def __init__(self, slot_times, contrast, weights, k, duration=120.0, dt=0.1, accel_std=0.5, accel_tau=2.0, mems=None, quantum=None,
                 min_contrast=0.3):
        self.t_slots = np.asarray(slot_times, float)
        self.contrast = np.asarray(contrast, float)
        self.k, self.duration, self.dt = int(k), float(duration), float(dt)
        self.accel_std, self.accel_tau, self.min_contrast = accel_std, accel_tau, min_contrast
        self.mems, self.quantum = mems or mems_accelerometer(), quantum or atom_interferometer_accelerometer()
        self.grid = np.arange(0.0, self.duration, self.dt)
        w = weights(self.grid) if callable(weights) else np.asarray(weights, float)
        if len(w) != len(self.grid):
            raise ValueError('weights must have one value per time step (%d)' % len(self.grid))
        self.w = w / w.sum()
        self.feasible = np.flatnonzero(self.contrast >= min_contrast)
        if self.k > len(self.feasible):
            raise ValueError('only %d usable slots for k=%d' % (len(self.feasible), self.k))
        self._slot_step = np.clip(np.round(self.t_slots / self.dt).astype(int), 1, len(self.grid) - 1)
        self._build_model()

    # ---- the exact objective --------------------------------------------------------------------------------------
    def _build_model(self):
        dt = self.dt
        # The MEMS sensor reports at its own rate; one filter step averages rate*dt readings (noise falls by the square root).
        m = self.mems
        self._r_mems = (m.noise_std ** 2) / max(m.rate_hz * dt, 1.0)
        decay = np.exp(-dt / self.accel_tau)
        F = np.eye(5)
        F[0, 1], F[0, 2], F[1, 2], F[2, 2] = dt, 0.5 * dt ** 2, dt, decay
        Q = np.zeros((5, 5))
        Q[2, 2] = self.accel_std ** 2 * (1 - decay ** 2)
        Q[3, 3] = m.bias_walk ** 2 * dt
        Q[4, 4] = self.quantum.bias_walk ** 2 * dt
        self._F, self._Q = F, Q
        self._P0 = np.diag([1e-6, 1e-6, self.accel_std ** 2, m.bias_std0 ** 2 + 1e-12, self.quantum.bias_std0 ** 2 + 1e-12])
        self._hm = np.array([0, 0, 1.0, 1.0, 0])
        self._hq = np.array([0, 0, 1.0, 0, 1.0])

    def _variance_path(self, slots):
        """Position variance at every step for a set of slots (the Kalman covariance recursion, no simulation noise)."""
        F, Q = self._F, self._Q
        P = self._P0.copy()
        due = {}
        for i in slots:
            due.setdefault(int(self._slot_step[i]), []).append(1.0 / max(self.contrast[i], 1e-3) ** 2)
        var = np.empty(len(self.grid))
        var[0] = P[0, 0]
        hm, hq = self._hm, self._hq
        for k in range(1, len(self.grid)):
            P = F @ P @ F.T + Q
            S = hm @ P @ hm + self._r_mems
            g = P @ hm / S
            P = P - np.outer(g, hm @ P)
            for scale in due.get(k, ()):
                r = (self.quantum.noise_std ** 2) * scale
                S = hq @ P @ hq + r
                g = P @ hq / S
                P = P - np.outer(g, hq @ P)
            var[k] = P[0, 0]
        return var

    def objective(self, slots):
        """Weighted RMS position standard deviation (m) over the mission for a set of slot indices (lower is better)."""
        return float(np.sqrt(self._variance_path(sorted(slots)) @ self.w))

    # ---- Monte Carlo check ----------------------------------------------------------------------------------------
    def monte_carlo(self, slots, seeds=range(20)):
        """Weighted RMS position error of the actual filter on simulated trajectories (an independent check of :meth:`objective`)."""
        dt, n = self.dt, len(self.grid)
        F, Q = self._F, self._Q
        decay = F[2, 2]
        due = {int(self._slot_step[i]): self.quantum.noise_std / max(self.contrast[i], 1e-3) for i in slots}
        out = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            a = np.empty(n)
            a[0] = rng.normal(0, self.accel_std)
            shocks = rng.normal(0, self.accel_std * np.sqrt(1 - decay ** 2), n)
            for k in range(1, n):
                a[k] = decay * a[k - 1] + shocks[k]
            v = np.zeros(n)
            p = np.zeros(n)
            for k in range(1, n):
                p[k] = p[k - 1] + v[k - 1] * dt + 0.5 * a[k - 1] * dt ** 2
                v[k] = v[k - 1] + a[k - 1] * dt
            walk = np.cumsum(rng.normal(0, self.mems.bias_walk * np.sqrt(dt), n - 1))
            bm = self.mems.bias_std0 * rng.normal() + np.concatenate([[0.0], walk])
            bq = self.quantum.bias_std0 * rng.normal()
            x = np.zeros(5)
            P = self._P0.copy()
            err2 = np.empty(n)
            err2[0] = 0.0
            for k in range(1, n):
                x = F @ x
                P = F @ P @ F.T + Q
                z = a[k] + bm[k] + rng.normal(0, np.sqrt(self._r_mems))
                S = self._hm @ P @ self._hm + self._r_mems
                g = P @ self._hm / S
                x = x + g * (z - self._hm @ x)
                P = P - np.outer(g, self._hm @ P)
                if k in due:
                    z = a[k] + bq + rng.normal(0, due[k])
                    S = self._hq @ P @ self._hq + due[k] ** 2
                    g = P @ self._hq / S
                    x = x + g * (z - self._hq @ x)
                    P = P - np.outer(g, self._hq @ P)
                err2[k] = (x[0] - p[k]) ** 2
            out.append(err2 @ self.w)
        return float(np.sqrt(np.mean(out)))

    # ---- baselines ------------------------------------------------------------------------------------------------
    def uniform(self):
        """Evenly spaced usable slots (the strong, simple baseline)."""
        targets = np.linspace(0, self.duration, self.k + 2)[1:-1]
        chosen = []
        for t in targets:
            order = np.argsort(np.abs(self.t_slots[self.feasible] - t))
            chosen.append(next(int(self.feasible[j]) for j in order if int(self.feasible[j]) not in chosen))
        return Schedule(sorted(chosen), self.objective(chosen), 'uniform spacing')

    def greedy(self):
        """Forward selection on the exact objective: add the slot that lowers it most, k times."""
        chosen = []
        for _ in range(self.k):
            best = min((i for i in self.feasible if i not in chosen), key=lambda i: self.objective(chosen + [i]))
            chosen.append(int(best))
        return Schedule(sorted(chosen), self.objective(chosen), 'greedy on the exact objective')

    def random(self, seed=0):
        rng = np.random.default_rng(seed)
        chosen = sorted(int(i) for i in rng.choice(self.feasible, self.k, replace=False))
        return Schedule(chosen, self.objective(chosen), 'random')

    def optimum(self):
        """The exact optimum by enumerating every k-subset of the usable slots (small problems only)."""
        best = min(itertools.combinations(self.feasible, self.k), key=self.objective)
        return Schedule([int(i) for i in best], self.objective(best), 'exact optimum (enumeration)')

    # ---- the QUBO surrogate ---------------------------------------------------------------------------------------
    def variance(self, slots):
        """Weighted position variance (m^2) for a set of slot indices: the quantity the QUBO model expands (``objective`` is its root)."""
        return float(self._variance_path(sorted(set(slots))) @ self.w)

    def local_quadratic(self, base):
        """Quadratic model of the weighted variance around the schedule ``base``, exact within two flips of it.

        Returns ``(linear, quadratic, constant)`` over the usable slots, with
        ``variance(S) = constant + linear @ x + sum_{i<j} quadratic[i, j] x_i x_j`` for the 0/1 vector ``x`` of ``S`` (exact when ``S``
        differs from ``base`` in at most two slots). With an empty ``base`` this is the inclusion-exclusion expansion.
        """
        f = [int(i) for i in self.feasible]
        n = len(f)
        on = set(int(i) for i in base)
        x0 = np.array([1.0 if i in on else 0.0 for i in f])
        s = 1 - 2 * x0                                             # the change in x_i when it is flipped
        j0 = self.variance(on)

        def flipped(*idx):
            cur = set(on)
            for i in idx:
                cur ^= {f[i]}
            return self.variance(cur)
        single = np.array([flipped(i) - j0 for i in range(n)])
        a = single / s
        b = np.zeros((n, n))
        for i, j in itertools.combinations(range(n), 2):
            b[i, j] = b[j, i] = (flipped(i, j) - j0 - single[i] - single[j]) / (s[i] * s[j])
        linear = a - b @ x0
        constant = j0 - a @ x0 + 0.5 * x0 @ b @ x0
        return linear, b, constant

    def qubo(self, base):
        """The slot-selection QUBO around ``base`` (a :class:`~quantum_mind.computing.RobotProblem` over the usable slots)."""
        linear, quad, _ = self.local_quadratic(base)
        scale = np.abs(linear).max() + 1e-12
        return grasp_selection(-linear / scale, quad / scale, self.k)


def schedule_interrogations(problem, method='exact', start='uniform', rounds=3, polish=False, seed=0, **solver_kw):
    """Schedule the quantum sensor: iterate a local quadratic QUBO solved by a thread 1 solver.

    Parameters
    ----------
    problem : InterrogationProblem
    method : {'exact', 'annealing', 'greedy descent', 'random sampling', 'qaoa'}
        The QUBO solver (``'exact'`` enumerates the QUBO: exact for the model, not for the true objective).
    start : {'uniform', 'greedy', 'empty'} or list of int, optional
        Where to expand first. ``'empty'`` with ``rounds=1`` is the plain inclusion-exclusion surrogate.
    rounds : int, optional
        Maximum number of expand-solve rounds; the loop stops as soon as the exact objective does not improve.
    polish : bool, optional
        Finish with a swap local search on the exact objective.
    seed : int, optional
    **solver_kw
        Passed to the solver (for example ``depth`` for QAOA).

    Returns
    -------
    Schedule
        ``info`` holds ``rounds_used``, ``start`` objective and ``polished``.
    """
    if isinstance(start, str):
        base = {'uniform': lambda: problem.uniform().slots, 'greedy': lambda: problem.greedy().slots, 'empty': lambda: []}[start]()
    else:
        base = list(start)
    idx = sorted(int(i) for i in base)
    start_obj = problem.objective(idx) if idx else float('nan')
    cur = problem.variance(idx) if idx else float('inf')
    solver = SOLVERS[method]
    used = 0
    for _ in range(rounds):
        qp = problem.qubo(idx)
        res = solver(qp.qubo, seed=seed, **solver_kw) if method != 'exact' else solver(qp.qubo)
        cand = [int(problem.feasible[j]) for j in np.flatnonzero(res.x)]
        if len(cand) != problem.k:                                 # an infeasible bit string (sampling solvers): ignore this round
            break
        val = problem.variance(cand)
        used += 1
        if val < cur - 1e-12 * max(cur, 1.0):
            idx, cur = sorted(cand), val
        else:
            break
    if len(idx) != problem.k:                                      # nothing feasible was found (empty start): use the uniform schedule
        idx = problem.uniform().slots
    info = dict(rounds_used=used, start_objective=start_obj, polished=False)
    if polish:
        idx, improved = _swap_search(problem, idx)
        info['polished'] = improved
    return Schedule(sorted(idx), problem.objective(idx), 'QUBO (local) + %s%s' % (method, ' + swap polish' if polish else ''), info)


def _swap_search(problem, idx):
    idx = list(idx)
    cur = problem.objective(idx)
    improved = False
    while True:
        best = None
        for a in range(len(idx)):
            for j in problem.feasible:
                j = int(j)
                if j in idx:
                    continue
                cand = idx[:a] + [j] + idx[a + 1:]
                val = problem.objective(cand)
                if val < cur - 1e-12 and (best is None or val < best[0]):
                    best = (val, cand)
        if best is None:
            return sorted(idx), improved
        cur, idx, improved = best[0], best[1], True


def random_mission(seed, n_slots=14, k=4, duration=120.0, dt=0.1, unusable=0.25, **kw):
    """A random mission: slots spread over the window, some with too little contrast, and two phases where accuracy matters most.

    The importance weight is 1 inside two random windows of 15 to 25 s and 0.1 elsewhere. Used by the benchmark and the tests.
    """
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(4.0, duration - 4.0, n_slots))
    contrast = np.where(rng.random(n_slots) < unusable, rng.uniform(0.0, 0.25, n_slots), rng.uniform(0.5, 1.0, n_slots))
    grid = np.arange(0.0, duration, dt)
    w = np.full(len(grid), 0.1)
    for _ in range(2):
        start, width = rng.uniform(0, duration - 25), rng.uniform(15, 25)
        w[(grid >= start) & (grid < start + width)] = 1.0
    return InterrogationProblem(times, contrast, w, k, duration, dt, **kw)
