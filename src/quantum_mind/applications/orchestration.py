r"""Orchestration: choose which module a robot relies on, from calibrated confidence and cost.

A robot often has several ways to act on a situation: a vision model, a reinforcement-learning policy,
an imitation-learned skill, a fault-recovery routine, a human model that predicts what a person will
answer. Each reports a confidence; each costs compute, time or energy. A *gate* picks the module with
the best confidence-minus-cost trade-off,

.. math:: k^* = \arg\max_k\; \hat c_k - \lambda\, \text{cost}_k ,

where :math:`\hat c_k` is the module's confidence *after recalibration*. The gate is only as good as
those confidences: an over-confident module wins too often. This module provides

* :class:`Module`, the description of one module (name, cost);
* :class:`ConfidenceGate`, the cost-penalised gate, optionally with a recalibrator per module and a
  compute budget;
* :class:`MetaCalibratedGate`, which learns each module's confidence offset online from outcomes,
  starting from a prior meta-learned on earlier tasks (empirical Bayes), so that it is well calibrated
  from the first steps of a new task;
* :class:`RoutingTask` and :func:`simulate_routing`, a synthetic task distribution (the one used in
  the supplementary simulation of the companion review on meta-learning orchestration) for testing
  gates before deploying them.

All simulated numbers are synthetic; nothing here is robot data.

Examples
--------
>>> import numpy as np
>>> from quantum_mind.applications.orchestration import Module, ConfidenceGate
>>> gate = ConfidenceGate([Module('vision', 1.0), Module('rl', 3.0)], lam=0.05)
>>> gate.select([0.80, 0.85])          # RL is 0.05 more confident but costs 2 more: vision wins
0
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

__all__ = ['Module', 'ConfidenceGate', 'MetaCalibratedGate', 'RoutingTask', 'simulate_routing', 'meta_fit_offsets']


@dataclass(frozen=True)
class Module:
    """One module the gate can call.

    Parameters
    ----------
    name : str
        Name, for reports.
    cost : float
        Relative cost per call (compute, latency or energy, in any consistent unit).
    """
    name: str
    cost: float = 1.0


class ConfidenceGate:
    """Cost-penalised confidence gate.

    Parameters
    ----------
    modules : sequence of Module
        The modules, in the order of the confidence vectors.
    lam : float, optional
        Price of one unit of cost in units of success probability.
    calibrators : sequence, optional
        One object per module with a ``transform(conf)`` method (for example
        :class:`~quantum_mind.applications.calibration.IsotonicCalibration`), or ``None`` entries to leave a
        module's confidence as reported.
    budget : float, optional
        Largest average cost per call over a sliding window of ``window`` calls. When the window is
        over budget, only modules at most as expensive as the cheapest one plus the remaining headroom
        are considered.
    window : int, optional
        Length of the budget window.

    Attributes
    ----------
    history : list of int
        Indices of the modules chosen so far.
    """

    def __init__(self, modules, lam=0.05, calibrators=None, budget=None, window=50):
        self.modules = list(modules)
        self.cost = np.array([m.cost for m in self.modules], float)
        self.lam, self.budget, self.window = lam, budget, window
        self.calibrators = list(calibrators) if calibrators is not None else [None] * len(self.modules)
        if len(self.calibrators) != len(self.modules):
            raise ValueError('one calibrator (or None) per module is needed')
        self.history = []

    def calibrated(self, conf):
        """Apply the per-module recalibration.

        Parameters
        ----------
        conf : array_like
            Reported confidences, shape ``(K,)`` or ``(n, K)``.

        Returns
        -------
        numpy.ndarray
            Same shape.
        """
        c = np.array(conf, float, copy=True)
        for k, cal in enumerate(self.calibrators):
            if cal is not None:
                c[..., k] = cal.transform(c[..., k])
        return c

    def scores(self, conf):
        """Gate scores :math:`\\hat c_k - \\lambda\\,\\text{cost}_k`.

        Parameters
        ----------
        conf : array_like
            Reported confidences.

        Returns
        -------
        numpy.ndarray
        """
        return self.calibrated(conf) - self.lam * self.cost

    def select(self, conf):
        """Choose a module for one situation and record the choice.

        Parameters
        ----------
        conf : array_like
            Reported confidence of each module, shape ``(K,)``.

        Returns
        -------
        int
            Index of the chosen module.
        """
        s = self.scores(conf)
        if self.budget is not None and self.history:
            recent = self.cost[self.history[-self.window:]]
            headroom = self.budget * (len(recent) + 1) - recent.sum()
            allowed = self.cost <= max(headroom, self.cost.min())
            s = np.where(allowed, s, -np.inf)
        k = int(np.argmax(s))
        self.history.append(k)
        return k

    def select_batch(self, conf):
        """Choose modules for many situations at once (no budget, no history).

        Parameters
        ----------
        conf : array_like
            Shape ``(n, K)``.

        Returns
        -------
        numpy.ndarray
            Chosen index per situation.
        """
        return np.argmax(self.scores(conf), axis=-1)


class MetaCalibratedGate:
    """Gate that learns each module's confidence offset online, from a meta-learned prior.

    Each module's reported confidence is assumed to be off by an unknown, task-specific offset
    :math:`b_k` (for example an imitation-learned skill that is systematically over-confident). After
    calling module k with confidence :math:`c_k` and observing the outcome :math:`y \\in \\{0, 1\\}`, the
    residual :math:`c_k - y` is an unbiased sample of :math:`b_k`. The gate keeps the posterior mean

    .. math:: \\hat b_k = \\frac{n_0\\, \\mu_k + \\sum (c_k - y)}{n_0 + n_k},

    a running mean shrunk toward the prior mean :math:`\\mu_k` with :math:`n_0` pseudo-observations.
    :func:`meta_fit_offsets` estimates :math:`\\mu` and :math:`n_0` from logs of earlier tasks.

    Parameters
    ----------
    modules : sequence of Module
        The modules.
    prior_mean : array_like, optional
        Prior offset per module (default 0).
    prior_n : float or array_like, optional
        Pseudo-counts of the prior (default 1: learn mostly from scratch). A very large value freezes
        the offsets at the prior (no online learning).
    lam : float, optional
        Price of cost.
    explore : float, optional
        Probability of calling a random module, so that every module keeps being checked.
    seed : int, optional
        Seed of the exploration.
    """

    def __init__(self, modules, prior_mean=None, prior_n=1.0, lam=0.05, explore=0.05, seed=0):
        self.modules = list(modules)
        K = len(self.modules)
        self.cost = np.array([m.cost for m in self.modules], float)
        self.prior_mean = np.zeros(K) if prior_mean is None else np.asarray(prior_mean, float)
        self.prior_n = np.broadcast_to(np.asarray(prior_n, float), (K,)).copy()
        self.lam, self.explore = lam, explore
        self.rng = np.random.default_rng(seed)
        self.sum_resid = np.zeros(K)
        self.n = np.zeros(K)

    @property
    def offsets(self):
        """numpy.ndarray: current estimate of each module's confidence offset."""
        return (self.prior_n * self.prior_mean + self.sum_resid) / (self.prior_n + self.n)

    def select(self, conf):
        """Choose a module (with a small exploration probability).

        Parameters
        ----------
        conf : array_like
            Reported confidences, shape ``(K,)``.

        Returns
        -------
        int
        """
        if self.rng.random() < self.explore:
            return int(self.rng.integers(len(self.modules)))
        return int(np.argmax(np.asarray(conf, float) - self.offsets - self.lam * self.cost))

    def update(self, k, conf_k, success):
        """Learn from the outcome of a call.

        Parameters
        ----------
        k : int
            Module that was called.
        conf_k : float
            Confidence it had reported.
        success : bool or int
            Whether it succeeded.
        """
        self.sum_resid[k] += float(conf_k) - float(bool(success))
        self.n[k] += 1


def meta_fit_offsets(logs, residual_var=0.25):
    """Meta-learn the prior of :class:`MetaCalibratedGate` from earlier tasks (empirical Bayes).

    Parameters
    ----------
    logs : sequence
        One entry per earlier task: a list of ``(module index, reported confidence, success)`` tuples,
        collected with an exploratory policy so that every module was tried.
    residual_var : float, optional
        Variance of one residual :math:`c - y` (about :math:`p(1 - p)`, at most 0.25).

    Returns
    -------
    prior_mean : numpy.ndarray
        Mean offset per module across tasks.
    prior_n : numpy.ndarray
        Pseudo-counts :math:`n_0 = \\sigma^2_{\\text{within}} / \\sigma^2_{\\text{between}}`, clipped to
        [1, 200]: many when tasks agree, few when they differ.
    """
    K = 1 + max(k for task in logs for k, _, _ in task)
    est = np.full((len(logs), K), np.nan)
    calls = np.zeros((len(logs), K))
    for i, task in enumerate(logs):
        for k in range(K):
            r = [c - float(bool(y)) for kk, c, y in task if kk == k]
            if r:
                est[i, k] = np.mean(r)
                calls[i, k] = len(r)
    mean = np.nanmean(est, axis=0)
    per_task_noise = residual_var / np.maximum(np.nanmean(calls, axis=0), 1)
    between = np.maximum(np.nanvar(est, axis=0) - per_task_noise, 1e-3)
    return mean, np.clip(residual_var / between, 1, 200)


@dataclass
class RoutingTask:
    """A synthetic orchestration task with four modules and four situation types.

    Each situation type has one module that succeeds 85% of the time; the others succeed 35% of the
    time. Each module reports its success probability plus noise plus a task-specific offset.

    Parameters
    ----------
    offsets : array_like
        Confidence offset of each module in this task.
    mix : array_like
        Probability of each situation type.
    noise : float, optional
        Standard deviation of the confidence noise.
    """
    offsets: np.ndarray
    mix: np.ndarray
    noise: float = 0.08
    p_good: float = 0.85
    p_bad: float = 0.35
    names: tuple = field(default=('Vision', 'RL', 'Imitation', 'Fault-recovery'))

    #: Relative costs of the four default modules.
    COST = (1.0, 3.0, 2.0, 4.0)
    #: Mean offsets across tasks: reinforcement learning and imitation are over-confident.
    MEAN_OFFSETS = (0.00, 0.30, 0.25, 0.05)

    @classmethod
    def sample(cls, rng, spread=0.10):
        """Draw a task: offsets around :attr:`MEAN_OFFSETS` and a random situation mix.

        Parameters
        ----------
        rng : numpy.random.Generator
        spread : float, optional
            Task-to-task standard deviation of the offsets.

        Returns
        -------
        RoutingTask
        """
        return cls(np.asarray(cls.MEAN_OFFSETS) + rng.normal(0, spread, 4), rng.dirichlet(np.ones(4)))

    def modules(self):
        """The four modules with their default costs.

        Returns
        -------
        list of Module
        """
        return [Module(n, c) for n, c in zip(self.names, self.COST)]

    def step(self, rng):
        """One situation.

        Parameters
        ----------
        rng : numpy.random.Generator

        Returns
        -------
        p : numpy.ndarray
            True success probability of each module.
        conf : numpy.ndarray
            Confidence each module reports.
        """
        ctx = rng.choice(4, p=self.mix)
        p = np.full(4, self.p_bad)
        p[ctx] = self.p_good
        conf = np.clip(p + self.offsets + rng.normal(0, self.noise, 4), 0.01, 0.99)
        return p, conf


def simulate_routing(gate, task, steps, rng, learn=True):
    """Run a gate on a synthetic task.

    Parameters
    ----------
    gate : ConfidenceGate or MetaCalibratedGate
        Gate to evaluate.
    task : RoutingTask
        Task.
    steps : int
        Number of situations.
    rng : numpy.random.Generator
        Random number generator for the situations and outcomes.
    learn : bool, optional
        Call ``gate.update`` after each outcome when the gate has one.

    Returns
    -------
    dict
        ``success`` (0/1 per step), ``cost`` (cost per step) and ``choice`` (module per step).
    """
    succ = np.zeros(steps)
    cost = np.zeros(steps)
    choice = np.zeros(steps, int)
    costs = np.array([m.cost for m in gate.modules])
    for t in range(steps):
        p, conf = task.step(rng)
        k = gate.select(conf)
        y = rng.random() < p[k]
        if learn and hasattr(gate, 'update'):
            gate.update(k, conf[k], y)
        succ[t], cost[t], choice[t] = y, costs[k], k
    return {'success': succ, 'cost': cost, 'choice': choice}
