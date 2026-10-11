r"""Robot problems posed as binary optimisation (QUBO), ready for classical and quantum solvers.

Three problems that a robot team actually has, each with its encoding, the true objective (without the
penalty terms) and a feasibility check, so a solver's answer can be judged on the robot's terms and not only on
the QUBO energy:

* :func:`multi_robot_allocation`: assign every task to one robot at the lowest total cost, optionally balancing
  the load.
* :func:`grasp_selection`: pick exactly ``k`` grasp candidates of high quality that do not conflict (for example
  because they need the same finger or the same contact point).
* :func:`slot_scheduling`: give every robot that shares a workspace a time slot so that robots whose motions
  conflict never move together (graph colouring with a fixed number of slots).

The encodings are the standard penalty formulations of Lucas (2014); what this module adds is the robot
vocabulary, the true objectives and the decoding.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..problems import Qubo

__all__ = ['RobotProblem', 'multi_robot_allocation', 'grasp_selection', 'slot_scheduling']


@dataclass
class RobotProblem:
    """A robot problem with its QUBO encoding.

    Attributes
    ----------
    name : str
    qubo : Qubo
        The encoding; variable order is documented by each builder.
    objective : callable
        ``objective(x)``: the true objective of a bit string (lower is better), without penalties.
    feasible : callable
        ``feasible(x)``: whether the bit string satisfies the problem's hard constraints.
    decode : callable
        ``decode(x)``: the bit string in the robot's vocabulary (who does what, which grasps, which slots).
    """
    name: str
    qubo: Qubo
    objective: Callable
    feasible: Callable
    decode: Callable

    @property
    def n_variables(self):
        return self.qubo.n

    def optimum(self):
        """Exhaustive optimum of the true objective over feasible bit strings (n at most 22).

        Returns
        -------
        x : numpy.ndarray
        value : float
        """
        n = self.qubo.n
        k = np.arange(2 ** n)
        X = ((k[:, None] >> np.arange(n)) & 1)
        best, best_x = np.inf, None
        for row in X:
            if self.feasible(row):
                v = self.objective(row)
                if v < best:
                    best, best_x = v, row.copy()
        if best_x is None:
            raise ValueError('the problem has no feasible bit string')
        return best_x, float(best)


def _one_hot_penalty(Q, indices, A):
    """Add ``A * (sum_i x_i - 1)^2`` (without the constant) to ``Q``; returns the constant ``A``."""
    for i in indices:
        Q[i, i] -= A
    for i, j in itertools.combinations(indices, 2):
        Q[i, j] += 2 * A
    return A


def multi_robot_allocation(costs, balance=0.0, penalty=None):
    r"""Assign every task to exactly one robot.

    Parameters
    ----------
    costs : array_like
        ``costs[r, t]``: cost of robot ``r`` doing task ``t`` (travel time, energy, ...).
    balance : float, optional
        Weight of a load-balancing term :math:`\lambda \sum_r (\sum_t x_{r,t})^2` that discourages piling tasks on
        one robot (0 ignores balance).
    penalty : float, optional
        One-robot-per-task penalty (default: large enough to dominate the costs).

    Returns
    -------
    RobotProblem
        Variable :math:`x_{r,t}` has index ``r * n_tasks + t``. ``objective`` is the total cost plus the balance term.

    Examples
    --------
    >>> import numpy as np
    >>> from quantum_mind.computing import multi_robot_allocation
    >>> p = multi_robot_allocation(np.array([[1.0, 5.0], [4.0, 2.0]]))
    >>> x, value = p.optimum()
    >>> p.decode(x), value
    ({0: [0], 1: [1]}, 3.0)
    """
    C = np.asarray(costs, float)
    if C.ndim != 2 or not np.all(np.isfinite(C)):
        raise ValueError('costs must be a finite robots x tasks matrix')
    nr, nt = C.shape
    A = penalty or 2 * (np.abs(C).max() * nt + balance * nt ** 2 + 1)
    idx = lambda r, t: r * nt + t                                        # noqa: E731
    Q = np.zeros((nr * nt, nr * nt))
    offset = 0.0
    for r in range(nr):
        for t in range(nt):
            Q[idx(r, t), idx(r, t)] += C[r, t] + balance
        for t1, t2 in itertools.combinations(range(nt), 2):
            Q[idx(r, t1), idx(r, t2)] += 2 * balance
    for t in range(nt):
        offset += _one_hot_penalty(Q, [idx(r, t) for r in range(nr)], A)
    labels = ['robot%d->task%d' % (r, t) for r in range(nr) for t in range(nt)]

    def grid(x):
        return np.asarray(x).reshape(nr, nt)

    def feasible(x):
        return bool(np.all(grid(x).sum(axis=0) == 1))

    def objective(x):
        X = grid(x)
        return float((C * X).sum() + balance * (X.sum(axis=1) ** 2).sum())

    def decode(x):
        return {r: [int(t) for t in np.flatnonzero(grid(x)[r])] for r in range(nr)}

    return RobotProblem('Multi-robot allocation', Qubo(Q, offset, labels, 'Multi-robot allocation'),
                        objective, feasible, decode)


def grasp_selection(quality, conflict, k, penalty=None):
    r"""Choose exactly ``k`` grasps of high total quality that do not conflict.

    Minimises :math:`-\sum_i q_i x_i + \sum_{i<j} c_{ij} x_i x_j` subject to :math:`\sum_i x_i = k`.

    Parameters
    ----------
    quality : array_like
        ``quality[i]``: quality of grasp candidate ``i`` (for example a force-closure margin), higher is better.
    conflict : array_like
        Symmetric ``(n, n)`` matrix; ``conflict[i, j] > 0`` is the cost of choosing both ``i`` and ``j``.
    k : int
        Number of grasps to choose.
    penalty : float, optional
        Cardinality penalty (default: large enough to dominate).

    Returns
    -------
    RobotProblem
        Variable ``i`` is grasp ``i``. ``objective`` is :math:`-\sum q_i x_i + \sum c_{ij} x_i x_j`.
    """
    q = np.asarray(quality, float)
    c = np.asarray(conflict, float)
    n = len(q)
    if c.shape != (n, n) or not np.allclose(c, c.T) or not 0 < k <= n:
        raise ValueError('conflict must be a symmetric n x n matrix and 0 < k <= n')
    c = c.copy()
    np.fill_diagonal(c, 0.0)                     # a grasp does not conflict with itself
    A = penalty or 2 * (np.abs(q).sum() + np.abs(c).sum() + 1)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] -= q[i]
    for i, j in itertools.combinations(range(n), 2):
        Q[i, j] += c[i, j]
    # A (sum x - k)^2 = A sum x_i + 2A sum_{i<j} x_i x_j - 2Ak sum x_i + A k^2   (x_i^2 = x_i)
    for i in range(n):
        Q[i, i] += A * (1 - 2 * k)
    for i, j in itertools.combinations(range(n), 2):
        Q[i, j] += 2 * A
    labels = ['grasp%d' % i for i in range(n)]

    def feasible(x):
        return int(np.sum(x)) == k

    def objective(x):
        x = np.asarray(x, float)
        return float(-q @ x + 0.5 * x @ c @ x)

    def decode(x):
        return [int(i) for i in np.flatnonzero(x)]

    return RobotProblem('Grasp selection', Qubo(Q, A * k * k, labels, 'Grasp selection'), objective, feasible, decode)


def slot_scheduling(conflicts, n_slots, penalty=None):
    r"""Give every robot a time slot so that conflicting robots never share one.

    Parameters
    ----------
    conflicts : array_like
        Symmetric 0/1 ``(n, n)`` matrix; ``conflicts[i, j] = 1`` if robots ``i`` and ``j`` must not move together.
    n_slots : int
        Number of time slots available.
    penalty : float, optional
        One-slot-per-robot penalty (default 2 times the number of conflicts plus 2).

    Returns
    -------
    RobotProblem
        Variable :math:`x_{i,s}` has index ``i * n_slots + s``. ``objective`` is the number of conflicting pairs that
        share a slot (0 is a valid schedule).
    """
    G = np.asarray(conflicts, float)
    n = len(G)
    if G.shape != (n, n) or not np.allclose(G, G.T) or n_slots < 1:
        raise ValueError('conflicts must be a symmetric n x n matrix and n_slots >= 1')
    pairs = [(i, j) for i, j in itertools.combinations(range(n), 2) if G[i, j]]
    A = penalty or 2 * len(pairs) + 2
    idx = lambda i, s: i * n_slots + s                                   # noqa: E731
    Q = np.zeros((n * n_slots, n * n_slots))
    offset = 0.0
    for i in range(n):
        offset += _one_hot_penalty(Q, [idx(i, s) for s in range(n_slots)], A)
    for i, j in pairs:
        for s in range(n_slots):
            Q[idx(i, s), idx(j, s)] += 1.0
    labels = ['robot%d@slot%d' % (i, s) for i in range(n) for s in range(n_slots)]

    def grid(x):
        return np.asarray(x).reshape(n, n_slots)

    def feasible(x):
        return bool(np.all(grid(x).sum(axis=1) == 1))

    def objective(x):
        X = grid(x)
        return float(sum((X[i] * X[j]).sum() for i, j in pairs))

    def decode(x):
        return {i: [int(s) for s in np.flatnonzero(grid(x)[i])] for i in range(n)}

    return RobotProblem('Slot scheduling', Qubo(Q, offset, labels, 'Slot scheduling'), objective, feasible, decode)
