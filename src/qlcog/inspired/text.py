"""Quantum language model (QLM) for document ranking, with a classical query-likelihood baseline.

Sordoni, Nie and Bengio, "Modeling term dependencies with quantum language models for IR", SIGIR 2013.
A document is a density matrix over the vocabulary space, estimated by maximum likelihood from
projectors: one per term occurrence (|e_w><e_w|) and one per pair of terms that co-occur within a
window (|v><v|, v = (e_a + e_b)/sqrt 2), which lets the model represent term dependencies that a
bag of words cannot. The estimate is found by the R rho R iteration (Lvovsky, J. Opt. B 6, S556, 2004)
and smoothed with the collection model. Documents are ranked by the negative von Neumann divergence
from the query's density matrix. Classical (quantum-inspired) linear algebra; vocabulary up to a few
hundred terms."""
from __future__ import annotations

import re
import numpy as np

__all__ = ['tokenize', 'QuantumLanguageModel', 'QueryLikelihoodModel']


def tokenize(text):
    """Lower-case word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _projectors(tokens, index, window):
    ids = [index[t] for t in tokens if t in index]
    out = [('t', i) for i in ids]
    for a in range(len(ids)):
        for b in range(a + 1, min(len(ids), a + window)):
            if ids[a] != ids[b]:
                out.append(('d', ids[a], ids[b]))
    return out


def _vec(p, V):
    v = np.zeros(V)
    if p[0] == 't':
        v[p[1]] = 1
    else:
        v[p[1]] = v[p[2]] = 1 / np.sqrt(2)
    return v


def _rhor(vectors, V, iters=60):
    """Maximum-likelihood density matrix for rank-1 projectors |v><v| (R rho R iteration)."""
    if not len(vectors):
        return np.eye(V) / V
    M = np.array(vectors); rho = np.eye(V) / V
    for _ in range(iters):
        p = np.einsum('iv,vw,iw->i', M, rho, M)
        R = (M.T / np.maximum(p, 1e-12)) @ M
        rho = R @ rho @ R; rho = (rho + rho.T) / 2; rho /= np.trace(rho)
    return rho


def _logm_psd(rho, eps=1e-10):
    w, U = np.linalg.eigh(rho)
    return (U * np.log(np.maximum(w, eps))) @ U.T


class QuantumLanguageModel:
    """fit(documents) then rank(query). Parameters: window (dependency window in tokens), smoothing
    (weight of the collection density matrix), uniform (weight of the maximally mixed state I/V, which
    keeps every document matrix full rank so that its logarithm is finite)."""

    def __init__(self, window=3, smoothing=0.2, uniform=0.01, iters=60):
        self.window, self.smoothing, self.uniform, self.iters = window, smoothing, uniform, iters

    def fit(self, documents):
        """documents: list of strings; returns self."""
        toks = [tokenize(d) for d in documents]
        self.vocab = sorted({t for d in toks for t in d}); self.index = {t: i for i, t in enumerate(self.vocab)}
        V = len(self.vocab)
        doc_vecs = [[_vec(p, V) for p in _projectors(d, self.index, self.window)] for d in toks]
        coll = _rhor([v for d in doc_vecs for v in d], V, self.iters)
        a, u = self.smoothing, self.uniform
        self.rhos = [(1 - a - u) * _rhor(d, V, self.iters) + a * coll + u * np.eye(V) / V for d in doc_vecs]
        self.logs = [_logm_psd(r) for r in self.rhos]
        return self

    def query_density(self, query):
        """Density matrix of a query (terms outside the vocabulary are ignored)."""
        V = len(self.vocab)
        return _rhor([_vec(p, V) for p in _projectors(tokenize(query), self.index, self.window)], V, self.iters)

    def scores(self, query):
        """Negative von Neumann divergence -S(rho_q || rho_d) for each document (higher = better)."""
        q = self.query_density(query); lq = _logm_psd(q)
        return np.array([-float(np.trace(q @ (lq - ld))) for ld in self.logs])

    def rank(self, query):
        """Document indices, best first."""
        return list(np.argsort(-self.scores(query), kind='stable'))


class QueryLikelihoodModel:
    """Classical baseline: unigram query likelihood with Dirichlet smoothing (mu)."""

    def __init__(self, mu=50.0):
        self.mu = mu

    def fit(self, documents):
        """documents: list of strings; returns self."""
        self.docs = [tokenize(d) for d in documents]
        allt = [t for d in self.docs for t in d]
        self.coll = {t: allt.count(t) / len(allt) for t in set(allt)}
        return self

    def scores(self, query):
        """Log-likelihood of the query under each smoothed document model."""
        q = [t for t in tokenize(query) if t in self.coll]
        out = []
        for d in self.docs:
            n = len(d)
            out.append(sum(np.log((d.count(t) + self.mu * self.coll[t]) / (n + self.mu)) for t in q))
        return np.array(out)

    def rank(self, query):
        """Document indices, best first."""
        return list(np.argsort(-self.scores(query), kind='stable'))
