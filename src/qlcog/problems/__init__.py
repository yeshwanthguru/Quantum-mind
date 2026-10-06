r"""Binary optimisation problems shared by the quantum and quantum-inspired solvers.

Every problem is a QUBO: minimise :math:`x^T Q x + \text{offset}` over :math:`x \in \{0, 1\}^n`
(:math:`Q` upper triangular after :meth:`Qubo.normalised`). The same object can be solved by QAOA
(:mod:`qlcog.quantum`), by simulated quantum annealing or the quantum-inspired evolutionary algorithm
(:mod:`qlcog.inspired`), or exactly by brute force for small n, so the three approaches can be compared
on identical instances.

* :func:`maxcut`: graph partitioning, clustering, network design.
* :func:`knapsack`: resource selection under a budget.
* :func:`task_allocation`: assign tasks to agents (multi-robot, scheduling), one agent per task.
* :func:`portfolio`: select assets, return against risk, with a cardinality budget.
* :func:`from_ising`: any Ising model.

Examples
--------
>>> from qlcog.problems import maxcut
>>> q = maxcut([(0, 1), (1, 2), (2, 0)])             # a triangle: the best cut has 2 edges
>>> x, e = q.brute_force()
>>> e
-2.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import numpy as np

__all__ = ['Qubo', 'maxcut', 'knapsack', 'task_allocation', 'portfolio', 'from_ising']


@dataclass
class Qubo:
    r"""Quadratic unconstrained binary optimisation problem.

    Minimise :math:`x^T Q x + \text{offset}` over :math:`x \in \{0, 1\}^n`.

    Parameters
    ----------
    Q : array_like
        ``(n, n)`` coefficient matrix (any square matrix; see :meth:`normalised`).
    offset : float, optional
        Constant energy offset.
    labels : list of str, optional
        Variable names (default ``x0, x1, ...``).
    name : str, optional
        Problem name.
    """
    Q: np.ndarray
    offset: float = 0.0
    labels: list = field(default_factory=list)
    name: str = 'QUBO'

    def __post_init__(self):
        self.Q = np.asarray(self.Q, float)
        if not self.labels:
            self.labels = ['x%d' % i for i in range(self.n)]

    @property
    def n(self):
        """int: number of binary variables."""
        return self.Q.shape[0]

    def normalised(self):
        """Upper-triangular form with the same energies.

        Returns
        -------
        Qubo
        """
        U = np.triu(self.Q) + np.triu(self.Q.T, 1)
        return Qubo(U, self.offset, list(self.labels), self.name)

    def energy(self, x):
        """Energy of bit strings.

        Parameters
        ----------
        x : array_like
            One bit string (1-D) or one per row (2-D).

        Returns
        -------
        float or numpy.ndarray
        """
        x = np.asarray(x, float)
        if x.ndim == 1:
            return float(x @ self.Q @ x + self.offset)
        return np.einsum('bi,ij,bj->b', x, self.Q, x) + self.offset

    def all_energies(self):
        """Energies of all :math:`2^n` bit strings.

        Index k holds the bits of k with :math:`x_0` the least significant bit (Qiskit ordering).

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        ValueError
            If n > 22.
        """
        if self.n > 22:
            raise ValueError('too many variables for exhaustive enumeration')
        k = np.arange(2 ** self.n)
        X = ((k[:, None] >> np.arange(self.n)) & 1).astype(float)
        return self.energy(X)

    def brute_force(self):
        """Exact minimum by enumeration (small n).

        Returns
        -------
        x : numpy.ndarray
            Optimal bit string.
        energy : float
        """
        e = self.all_energies(); k = int(np.argmin(e))
        return np.array([(k >> i) & 1 for i in range(self.n)]), float(e[k])

    def to_ising(self):
        r"""Equivalent Ising model with spins :math:`s = 1 - 2x \in \{+1, -1\}`.

        Returns
        -------
        h : numpy.ndarray
            Local fields.
        J : numpy.ndarray
            Upper-triangular couplings.
        c : float
            Constant, so that energy :math:`= h \cdot s + \sum_{i<j} J_{ij} s_i s_j + c`.
        """
        U = self.normalised().Q; n = self.n
        h = np.zeros(n); J = np.zeros((n, n)); c = self.offset
        for i in range(n):
            # x_i^2 = x_i = (1 - s_i)/2
            h[i] -= U[i, i] / 2; c += U[i, i] / 2
            for j in range(i + 1, n):
                q = U[i, j] / 4
                J[i, j] += q; h[i] -= q; h[j] -= q; c += q
        return h, J, c


def maxcut(edges, n=None, weights=None):
    """Maximum cut of a graph, written as a minimisation (energy = minus the cut weight).

    Parameters
    ----------
    edges : iterable of tuple
        ``(i, j)`` pairs.
    n : int, optional
        Number of nodes.
    weights : array_like, optional
        Edge weights (default 1).

    Returns
    -------
    Qubo
    """
    edges = list(edges); n = n or 1 + max(max(e) for e in edges)
    w = np.ones(len(edges)) if weights is None else np.asarray(weights, float)
    Q = np.zeros((n, n))
    for (i, j), wij in zip(edges, w):
        # cut(i,j) = x_i + x_j - 2 x_i x_j
        Q[i, i] -= wij; Q[j, j] -= wij; Q[min(i, j), max(i, j)] += 2 * wij
    return Qubo(Q, 0.0, name='MaxCut')


def knapsack(values, weights, capacity, penalty=None):
    """0-1 knapsack: maximise value subject to total weight <= capacity.

    The inequality becomes an equality with binary slack variables, enforced by a quadratic penalty.

    Parameters
    ----------
    values, weights : array_like
        Item values and weights.
    capacity : int
        Weight limit.
    penalty : float, optional
        Penalty coefficient (default large enough to make violations unprofitable).

    Returns
    -------
    Qubo
        Items first, then the slack bits.
    """
    v, w = np.asarray(values, float), np.asarray(weights, float); n = len(v)
    ns = int(np.ceil(np.log2(capacity + 1)))
    coef = np.r_[w, -(2 ** np.arange(ns))]                  # sum w x - slack = 0  (slack <= capacity)
    coef[-1] = -(capacity - (2 ** (ns - 1) - 1))            # slack covers exactly 0..capacity
    A = penalty or 2 * v.sum() / max(w.min(), 1)
    m = n + ns; Q = np.zeros((m, m))
    Q[:n, :n] -= np.diag(v)
    # penalty * (coef . z - 0)^2 with z = (x, slack bits); the constraint is sum w x <= capacity
    Q += A * np.outer(coef, coef)
    labels = ['item%d' % i for i in range(n)] + ['slack%d' % k for k in range(ns)]
    return Qubo(Q, 0.0, labels, 'Knapsack')


def task_allocation(costs, penalty=None):
    """Assignment of tasks to agents; every task goes to exactly one agent.

    Parameters
    ----------
    costs : array_like
        ``costs[a, t]``: cost of agent a doing task t.
    penalty : float, optional
        Penalty for a task with zero or several agents.

    Returns
    -------
    Qubo
        Variable :math:`x_{a,t}` has index ``a * n_tasks + t``.
    """
    C = np.asarray(costs, float); na, nt = C.shape
    A = penalty or 2 * (np.abs(C).max() * nt + 1)
    n = na * nt; Q = np.zeros((n, n)); off = 0.0
    idx = lambda a, t: a * nt + t                            # noqa: E731
    for a in range(na):
        for t in range(nt):
            Q[idx(a, t), idx(a, t)] += C[a, t]
    for t in range(nt):                                     # A (sum_a x_at - 1)^2
        vs = [idx(a, t) for a in range(na)]
        for i in vs:
            Q[i, i] -= A
        for i, j in itertools.combinations(vs, 2):
            Q[i, j] += 2 * A
        off += A
    labels = ['agent%d->task%d' % (a, t) for a in range(na) for t in range(nt)]
    return Qubo(Q, off, labels, 'Task allocation')


def portfolio(mu, cov, budget, risk=0.5, penalty=None):
    r"""Portfolio selection with a cardinality budget.

    Choose exactly ``budget`` assets, minimising :math:`\text{risk}\; x^T \Sigma x - \mu^T x`.

    Parameters
    ----------
    mu : array_like
        Expected returns.
    cov : array_like
        Covariance matrix.
    budget : int
        Number of assets to select.
    risk : float, optional
        Risk aversion.
    penalty : float, optional
        Budget-constraint penalty.

    Returns
    -------
    Qubo
    """
    mu, cov = np.asarray(mu, float), np.asarray(cov, float); n = len(mu)
    A = penalty or 2 * (np.abs(mu).sum() + risk * np.abs(cov).sum())
    Q = risk * cov - np.diag(mu)
    Q = Q + A * (np.ones((n, n)) - 2 * budget * np.eye(n))   # A (sum x - budget)^2, using x_i^2 = x_i
    return Qubo(Q, A * budget ** 2, ['asset%d' % i for i in range(n)], 'Portfolio')


def from_ising(h, J, c=0.0):
    r"""QUBO with the same energies as an Ising model.

    Parameters
    ----------
    h : array_like
        Local fields.
    J : array_like
        Couplings (the strict upper triangle is used).
    c : float, optional
        Constant.

    Returns
    -------
    Qubo
        Energies of :math:`h \cdot s + \sum_{i<j} J_{ij} s_i s_j + c` with :math:`s = 1 - 2x`.
    """
    h = np.asarray(h, float); J = np.triu(np.asarray(J, float), 1); n = len(h)
    Q = np.zeros((n, n)); off = c + h.sum() + J.sum()
    for i in range(n):
        Q[i, i] -= 2 * h[i]
        for j in range(i + 1, n):
            Q[i, j] += 4 * J[i, j]; Q[i, i] -= 2 * J[i, j]; Q[j, j] -= 2 * J[i, j]
    return Qubo(Q, off, name='Ising')
