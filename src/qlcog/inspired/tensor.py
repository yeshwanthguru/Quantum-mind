"""Tensor-network (matrix product state) classifier: quantum-inspired supervised learning.

Each feature x_j in [0, 1] is mapped to a local state of dimension d: for d = 2 the 'qubit' state
(cos(pi x_j / 2), sin(pi x_j / 2)); for d > 2 the spin-coherent state with components
sqrt(C(d-1, s)) cos^(d-1-s) sin^s, a polynomial feature map of degree d - 1. The input becomes a
product state in a d^n-dimensional space. The classifier is a weight tensor in
that space stored as a matrix product state with bond dimension D (Stoudenmire and Schwab, NeurIPS
2016), so its cost is linear in the number of features. It is trained here by gradient descent on
the softmax cross-entropy, with exact gradients from left and right environments."""
from __future__ import annotations

import numpy as np

__all__ = ['MPSClassifier']


class MPSClassifier:
    """Parameters: bond (bond dimension D), local_dim (d), epochs, lr (Adam step), batch, seed.
    The class (label) index sits on the last site."""

    def __init__(self, bond=6, local_dim=2, epochs=60, lr=0.02, batch=32, seed=0):
        self.D, self.d, self.epochs, self.lr, self.batch, self.seed = bond, local_dim, epochs, lr, batch, seed

    def _phi(self, Xs):
        from scipy.special import comb
        c, s = np.cos(np.pi * Xs / 2), np.sin(np.pi * Xs / 2); d = self.d
        return np.stack([np.sqrt(comb(d - 1, k)) * c ** (d - 1 - k) * s ** k for k in range(d)], -1)   # (B, n, d)

    def _scale(self, X):
        span = np.where(self.mx > self.mn, self.mx - self.mn, 1.0)
        return np.clip((np.asarray(X, float) - self.mn) / span, 0, 1)

    def _contract(self, Phi):
        """Left environments L_j (B, D_j) and the scores (B, C)."""
        B, n, _ = Phi.shape; L = [np.ones((B, 1))]
        for j in range(n - 1):
            M = np.einsum('bs,lsr->blr', Phi[:, j], self.cores[j])
            L.append(np.einsum('bl,blr->br', L[-1], M))
        scores = np.einsum('bl,bs,lsc->bc', L[-1], Phi[:, -1], self.cores[-1])
        return L, scores

    def _grads(self, Phi, Y):
        B, n, _ = Phi.shape
        L, S = self._contract(Phi)
        S = S - S.max(1, keepdims=True); Pr = np.exp(S); Pr /= Pr.sum(1, keepdims=True)
        loss = -np.mean(np.log(np.clip((Pr * Y).sum(1), 1e-12, None)))
        dS = (Pr - Y) / B                                                   # (B, C)
        grads = [None] * n
        grads[-1] = np.einsum('bl,bs,bc->lsc', L[-1], Phi[:, -1], dS)
        R = np.einsum('bs,lsc,bc->bl', Phi[:, -1], self.cores[-1], dS)       # right env including dS
        for j in range(n - 2, -1, -1):
            grads[j] = np.einsum('bl,bs,br->lsr', L[j], Phi[:, j], R)
            R = np.einsum('bs,lsr,br->bl', Phi[:, j], self.cores[j], R)
        return loss, grads, Pr

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        self.classes_ = np.unique(y); C = len(self.classes_)
        self.mn, self.mx = X.min(0), X.max(0)
        Phi = self._phi(self._scale(X)); n = X.shape[1]
        Y = (y[:, None] == self.classes_[None, :]).astype(float)
        rng = np.random.default_rng(self.seed); D = self.D
        dims = [1] + [D] * (n - 1) + [1]
        self.cores = []
        for j in range(n):
            last = j == n - 1
            core = np.zeros((dims[j], self.d, C if last else dims[j + 1]))
            if last:
                core += rng.normal(0, 0.1, core.shape)
            else:                                         # near-identity cores keep scores well scaled
                for s in range(self.d):
                    core[:, s, :] = np.eye(dims[j], dims[j + 1]) + rng.normal(0, 0.05, (dims[j], dims[j + 1]))
            self.cores.append(core)
        m = [np.zeros_like(c) for c in self.cores]; v = [np.zeros_like(c) for c in self.cores]
        t = 0; self.history_ = []
        for _ in range(self.epochs):
            idx = rng.permutation(len(X)); ep = []
            for s in range(0, len(X), self.batch):
                b = idx[s:s + self.batch]
                loss, g, _ = self._grads(Phi[b], Y[b]); t += 1; ep.append(loss)
                for j in range(n):                          # Adam
                    m[j] = 0.9 * m[j] + 0.1 * g[j]; v[j] = 0.999 * v[j] + 0.001 * g[j] ** 2
                    self.cores[j] -= self.lr * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
            self.history_.append(float(np.mean(ep)))
        return self

    def predict_proba(self, X):
        _, S = self._contract(self._phi(self._scale(X)))
        S = S - S.max(1, keepdims=True); P = np.exp(S); return P / P.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    @property
    def n_parameters(self):
        return int(sum(c.size for c in self.cores))
