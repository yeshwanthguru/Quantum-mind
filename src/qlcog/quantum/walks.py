"""Continuous-time quantum walks on graphs: walk probabilities and a quantum-walk centrality, with
PageRank and degree centrality as classical baselines.

A continuous-time quantum walk evolves |psi(t)> = exp(-i H t) |psi(0)> with H the graph's adjacency
matrix (Farhi and Gutmann, PRA 58, 915, 1998). Quantum-walk centrality ranks nodes by the long-time
average of the walk's occupation probabilities starting from an equal superposition (Izaac et al.,
PRA 95, 032318, 2017); the average is computed exactly from the eigen-decomposition of H.
Simulated classically here (state size = number of nodes)."""
from __future__ import annotations

import numpy as np

__all__ = ['adjacency', 'ctqw_probabilities', 'quantum_walk_centrality', 'pagerank', 'degree_centrality']


def adjacency(edges, n=None, directed=False, weights=None):
    """Adjacency matrix from an edge list."""
    edges = list(edges); n = n or 1 + max(max(e) for e in edges)
    A = np.zeros((n, n)); w = np.ones(len(edges)) if weights is None else np.asarray(weights, float)
    for (i, j), x in zip(edges, w):
        A[i, j] += x
        if not directed:
            A[j, i] += x
    return A


def _start(n, start):
    if start is None:
        return np.ones(n) / np.sqrt(n)
    psi = np.zeros(n, complex); psi[start] = 1
    return psi


def ctqw_probabilities(A, t, start=None):
    """Occupation probabilities of each node at time t (start: a node index, or None for the equal
    superposition). Uses the symmetric part of A as the Hamiltonian."""
    H = (np.asarray(A, float) + np.asarray(A, float).T) / 2
    w, V = np.linalg.eigh(H)
    psi = V @ (np.exp(-1j * w * t) * (V.T @ _start(len(H), start)))
    return np.abs(psi) ** 2


def quantum_walk_centrality(A, start=None, tol=1e-9):
    """Infinite-time average of the occupation probabilities: sum over distinct eigenvalues of
    |P_lambda psi(0)|^2 node by node (degenerate eigenspaces grouped within tol)."""
    H = (np.asarray(A, float) + np.asarray(A, float).T) / 2
    w, V = np.linalg.eigh(H); psi0 = _start(len(H), start)
    out = np.zeros(len(H)); k = 0
    while k < len(w):
        j = k
        while j + 1 < len(w) and abs(w[j + 1] - w[k]) < tol:
            j += 1
        Vk = V[:, k:j + 1]; out += np.abs(Vk @ (Vk.T @ psi0)) ** 2
        k = j + 1
    return out


def pagerank(A, damping=0.85, tol=1e-12, max_iter=1000):
    """Classical PageRank (power iteration; links i -> j from A[i, j]; dangling nodes spread evenly)."""
    A = np.asarray(A, float); n = len(A); out = A.sum(1)
    T = np.where(out[:, None] > 0, A / np.where(out[:, None] > 0, out[:, None], 1), 1.0 / n)
    r = np.ones(n) / n
    for _ in range(max_iter):
        new = (1 - damping) / n + damping * (r @ T)
        if np.abs(new - r).sum() < tol:
            return new
        r = new
    return r


def degree_centrality(A):
    """Normalised (undirected) degree."""
    d = (np.asarray(A, float) > 0).sum(1).astype(float)
    return d / d.sum()
