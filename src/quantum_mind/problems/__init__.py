r"""The binary optimisation container used by trust-aware task allocation.

A :class:`Qubo` is the problem :math:`\min_x\, x^T Q x + \text{offset}` over :math:`x \in \{0, 1\}^n`.
:func:`quantum_mind.applications.allocation.trust_aware_allocation` builds one from each person's trust in
the robot, and small instances are solved exactly with :meth:`Qubo.brute_force`. General-purpose problem
builders (max-cut, knapsack, portfolio) and solvers are left to dedicated optimisation libraries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import numpy as np

__all__ = ['Qubo']


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
