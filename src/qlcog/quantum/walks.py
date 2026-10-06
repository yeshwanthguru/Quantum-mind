r"""Continuous-time quantum walks on graphs: walk probabilities and a quantum-walk centrality, with
PageRank and degree centrality as classical baselines.

A continuous-time quantum walk evolves :math:`|\psi(t)\rangle = e^{-iHt}|\psi(0)\rangle` with
:math:`H` the graph's adjacency matrix (Farhi and Gutmann, PRA 58, 915, 1998). Quantum-walk
centrality ranks nodes by the long-time average of the walk's occupation probabilities, starting from
an equal superposition (Izaac et al., PRA 95, 032318, 2017); the average is computed exactly from the
eigendecomposition of :math:`H`. Everything is simulated classically (the state has one amplitude
per node).

Examples
--------
>>> from qlcog.quantum import adjacency, quantum_walk_centrality, pagerank
>>> A = adjacency([(0, 1), (0, 2), (0, 3)])            # a star: node 0 is the hub
>>> int(quantum_walk_centrality(A).argmax()), int(pagerank(A).argmax())
(0, 0)
"""
from __future__ import annotations

import numpy as np

__all__ = ['adjacency', 'ctqw_probabilities', 'quantum_walk_centrality', 'pagerank', 'degree_centrality']


def adjacency(edges, n=None, directed=False, weights=None):
    """Adjacency matrix from an edge list.

    Parameters
    ----------
    edges : iterable of tuple
        ``(i, j)`` pairs.
    n : int, optional
        Number of nodes (default: largest index + 1).
    directed : bool, optional
        If False, every edge is added in both directions.
    weights : array_like, optional
        Edge weights (default 1).

    Returns
    -------
    numpy.ndarray
        ``(n, n)`` matrix.
    """
    edges = list(edges)
    n = n or 1 + max(max(e) for e in edges)
    A = np.zeros((n, n))
    w = np.ones(len(edges)) if weights is None else np.asarray(weights, float)
    for (i, j), x in zip(edges, w):
        A[i, j] += x
        if not directed:
            A[j, i] += x
    return A


def _start(n, start):
    """Initial state: equal superposition (start=None) or one node."""
    if start is None:
        return np.ones(n) / np.sqrt(n)
    psi = np.zeros(n, complex)
    psi[start] = 1
    return psi


def ctqw_probabilities(A, t, start=None):
    """Occupation probabilities of a continuous-time quantum walk.

    Parameters
    ----------
    A : array_like
        Adjacency matrix; its symmetric part is the Hamiltonian.
    t : float
        Time.
    start : int, optional
        Starting node; ``None`` starts from the equal superposition.

    Returns
    -------
    numpy.ndarray
        Probability of each node at time ``t``.
    """
    H = (np.asarray(A, float) + np.asarray(A, float).T) / 2
    w, V = np.linalg.eigh(H)
    psi = V @ (np.exp(-1j * w * t) * (V.T @ _start(len(H), start)))
    return np.abs(psi) ** 2


def quantum_walk_centrality(A, start=None, tol=1e-9):
    """Quantum-walk centrality: infinite-time average of the occupation probabilities.

    The average is :math:`\\sum_\\lambda |P_\\lambda \\psi(0)|^2` node by node, summed over the
    distinct eigenvalues :math:`\\lambda` of :math:`H`.

    Parameters
    ----------
    A : array_like
        Adjacency matrix.
    start : int, optional
        Starting node (default: equal superposition).
    tol : float, optional
        Eigenvalues closer than ``tol`` are treated as one degenerate eigenspace.

    Returns
    -------
    numpy.ndarray
        Centrality of each node (sums to 1).
    """
    H = (np.asarray(A, float) + np.asarray(A, float).T) / 2
    w, V = np.linalg.eigh(H)
    psi0 = _start(len(H), start)
    out = np.zeros(len(H))
    k = 0
    while k < len(w):
        # group the degenerate eigenvalues w[k..j]
        j = k
        while j + 1 < len(w) and abs(w[j + 1] - w[k]) < tol:
            j += 1
        Vk = V[:, k:j + 1]
        out += np.abs(Vk @ (Vk.T @ psi0)) ** 2           # |P_lambda psi(0)|^2
        k = j + 1
    return out


def pagerank(A, damping=0.85, tol=1e-12, max_iter=1000):
    """Classical PageRank by power iteration.

    Parameters
    ----------
    A : array_like
        Adjacency matrix; ``A[i, j]`` is a link from i to j. Dangling nodes spread their rank evenly.
    damping : float, optional
        Damping factor.
    tol : float, optional
        Convergence tolerance (L1).
    max_iter : int, optional
        Maximum number of iterations.

    Returns
    -------
    numpy.ndarray
        PageRank of each node (sums to 1).
    """
    A = np.asarray(A, float)
    n = len(A)
    out = A.sum(1)
    # row-stochastic transition matrix; rows of dangling nodes are uniform
    T = np.where(out[:, None] > 0, A / np.where(out[:, None] > 0, out[:, None], 1), 1.0 / n)
    r = np.ones(n) / n
    for _ in range(max_iter):
        new = (1 - damping) / n + damping * (r @ T)
        if np.abs(new - r).sum() < tol:
            return new
        r = new
    return r


def degree_centrality(A):
    """Normalised degree centrality.

    Parameters
    ----------
    A : array_like
        Adjacency matrix.

    Returns
    -------
    numpy.ndarray
        Number of neighbours of each node, divided by the total.
    """
    d = (np.asarray(A, float) > 0).sum(1).astype(float)
    return d / d.sum()
