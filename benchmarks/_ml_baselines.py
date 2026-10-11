"""Flattened-sequence NumPy baselines (logistic regression, MLP) for the COBRA benchmark.

Both expose ``fit(x, y)`` and ``predict_proba(x)`` on sequences of shape (N, T, m) so they are drop-in
comparable to :class:`quantum_mind.belief.COBRA`.
"""
from __future__ import annotations

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class _Adam:
    def __init__(self, shapes, lr):
        self.lr, self.t = lr, 0
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]

    def step(self, params, grads):
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= 0.9; m += 0.1 * g
            v *= 0.999; v += 0.001 * g * g
            p -= self.lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)


class MLP:
    """Flattened-sequence MLP (tanh hidden layer). ``hidden=0`` is logistic regression."""

    def __init__(self, n_hyp: int = 3, hidden: int = 32, epochs: int = 300,
                 lr: float = 0.02, l2: float = 1e-3, seed: int = 0) -> None:
        self.d, self.h, self.epochs, self.lr, self.l2, self.seed = n_hyp, hidden, epochs, lr, l2, seed

    def _init(self, n_in):
        rng = np.random.default_rng(self.seed)
        if self.h:
            self.W1 = rng.standard_normal((n_in, self.h)) / np.sqrt(n_in)
            self.b1 = np.zeros(self.h)
            self.W2 = rng.standard_normal((self.h, self.d)) / np.sqrt(self.h)
        else:
            self.W1 = self.b1 = None
            self.W2 = 0.01 * rng.standard_normal((n_in, self.d))
        self.b2 = np.zeros(self.d)
        self.n_params = int(sum(a.size for a in (self.W1, self.b1, self.W2, self.b2) if a is not None))

    def _fwd(self, f):
        if self.h:
            a = np.tanh(f @ self.W1 + self.b1)
            return a, _softmax(a @ self.W2 + self.b2)
        return f, _softmax(f @ self.W2 + self.b2)

    def fit(self, x, y):
        f = np.asarray(x, float).reshape(len(x), -1)
        self.mu, self.sd = f.mean(0), f.std(0) + 1e-6
        f = (f - self.mu) / self.sd
        y = np.asarray(y, int)
        self._init(f.shape[1])
        rng = np.random.default_rng(self.seed + 1)
        params = [p for p in (self.W1, self.b1, self.W2, self.b2) if p is not None]
        opt = _Adam([p.shape for p in params], self.lr)
        n_val = max(len(f) // 6, 1) if len(f) >= 30 else 0
        order = rng.permutation(len(f))
        val, tr = order[:n_val], order[n_val:]
        best, best_val = [p.copy() for p in params], np.inf
        bs = min(128, len(tr))
        for ep in range(self.epochs):
            idx = rng.choice(tr, bs, replace=False)
            a, p = self._fwd(f[idx])
            g = p.copy()
            g[np.arange(bs), y[idx]] -= 1
            g /= bs
            if self.h:
                gW2 = a.T @ g + self.l2 * self.W2
                gb2 = g.sum(0)
                da = (g @ self.W2.T) * (1 - a ** 2)
                grads = [f[idx].T @ da + self.l2 * self.W1, da.sum(0), gW2, gb2]
            else:
                grads = [a.T @ g + self.l2 * self.W2, g.sum(0)]
            opt.step(params, grads)
            if n_val and (ep % 10 == 0 or ep == self.epochs - 1):
                pv = self._fwd(f[val])[1]
                v = -np.log(pv[np.arange(len(val)), y[val]]).mean()
                if v < best_val:
                    best_val, best = v, [q.copy() for q in params]
        if n_val:
            for q, b in zip(params, best):
                q[...] = b
        return self

    def predict_proba(self, x):
        f = (np.asarray(x, float).reshape(len(x), -1) - self.mu) / self.sd
        return self._fwd(f)[1]


def LogisticRegression(n_hyp: int = 3, **kw) -> MLP:
    return MLP(n_hyp, hidden=0, **kw)
