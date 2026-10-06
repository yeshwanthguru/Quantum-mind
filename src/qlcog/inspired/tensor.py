"""Tensor-network (matrix product state) classifier: quantum-inspired supervised learning.

Each feature x_j in [0, 1] is mapped to a local state of dimension d: for d = 2 the 'qubit' state
(cos(pi x_j / 2), sin(pi x_j / 2)); for d > 2 the spin-coherent state with components
sqrt(C(d-1, s)) cos^(d-1-s) sin^s, a polynomial feature map of degree d - 1. The input becomes a
product state in a d^n-dimensional space. The classifier is a weight tensor in that space stored as a
matrix product state with bond dimension at most D (Stoudenmire and Schwab, NeurIPS 2016), so its
cost is linear in the number of features. One core carries the class (label) index.

Two training methods:
  method='sweep'  DMRG-style sweeps as in Stoudenmire and Schwab: the two cores on a bond are merged,
                  optimised together, and split again by an SVD truncated to D, which adapts the bond
                  dimensions and moves the label index along the chain. Loss: softmax cross-entropy
                  (the original paper used a squared loss).
  method='adam'   all cores updated together by Adam with exact gradients from left and right
                  environments; fixed bond dimension, label on the last site (fast, the default)."""
from __future__ import annotations

import numpy as np

__all__ = ['MPSClassifier']


def _softmax(S):
    S = S - S.max(1, keepdims=True); P = np.exp(S); return P / P.sum(1, keepdims=True)


class MPSClassifier:
    """Parameters: bond (maximum bond dimension D), local_dim (d), method ('adam' or 'sweep'),
    epochs, lr, batch (Adam training); sweeps (back-and-forth sweeps), steps (optimisation steps per
    bond), sweep_lr, cutoff (relative singular-value cutoff) (sweep training); seed.

    On the package's test problems the two methods reach similar accuracy (two-moons: both 1.00 with
    d = 4; an 8-feature, 3-class Gaussian task: Adam 0.85, sweeps 0.81); Adam is faster, sweeps adapt
    the bond dimensions."""

    def __init__(self, bond=6, local_dim=2, epochs=60, lr=0.02, batch=32, seed=0, method='adam', sweeps=6,
                 steps=40, sweep_lr=0.05, cutoff=1e-10):
        if method not in ('adam', 'sweep'):
            raise ValueError("method must be 'adam' or 'sweep'")
        self.D, self.d, self.epochs, self.lr, self.batch, self.seed = bond, local_dim, epochs, lr, batch, seed
        self.method, self.sweeps, self.steps, self.sweep_lr, self.cutoff = method, sweeps, steps, sweep_lr, cutoff

    # ------------------------------------------------------------------------------- feature map
    def _phi(self, Xs):
        from scipy.special import comb
        c, s = np.cos(np.pi * Xs / 2), np.sin(np.pi * Xs / 2); d = self.d
        return np.stack([np.sqrt(comb(d - 1, k)) * c ** (d - 1 - k) * s ** k for k in range(d)], -1)   # (B, n, d)

    def _scale(self, X):
        span = np.where(self.mx > self.mn, self.mx - self.mn, 1.0)
        return np.clip((np.asarray(X, float) - self.mn) / span, 0, 1)

    # ------------------------------------------------------------------------------- contraction
    # cores[j] has shape (left, d, right), except cores[self.label_], which has shape (left, d, C, right)
    def _left(self, Phi, upto):
        """Environment of sites 0..upto-1, shape (B, left bond of site upto)."""
        L = np.ones((Phi.shape[0], 1))
        for j in range(upto):
            L = np.einsum('bl,bs,lsr->br', L, Phi[:, j], self.cores[j])
        return L

    def _right(self, Phi, frm):
        """Environment of sites frm..n-1, shape (B, right bond of site frm-1)."""
        R = np.ones((Phi.shape[0], 1))
        for j in range(Phi.shape[1] - 1, frm - 1, -1):
            R = np.einsum('bs,lsr,br->bl', Phi[:, j], self.cores[j], R)
        return R

    def _scores(self, Phi):
        p = self.label_
        L, R = self._left(Phi, p), self._right(Phi, p + 1)
        return np.einsum('bl,bs,lscr,br->bc', L, Phi[:, p], self.cores[p], R)

    # ------------------------------------------------------------------------------- Adam training
    def _grads(self, Phi, Y):
        """Loss and gradients of all cores (label on the last site)."""
        B, n, _ = Phi.shape
        Ls = [np.ones((B, 1))]
        for j in range(n - 1):
            Ls.append(np.einsum('bl,bs,lsr->br', Ls[-1], Phi[:, j], self.cores[j]))
        last = self.cores[-1][..., 0]                                       # (l, d, C)
        Pr = _softmax(np.einsum('bl,bs,lsc->bc', Ls[-1], Phi[:, -1], last))
        loss = -np.mean(np.log(np.clip((Pr * Y).sum(1), 1e-12, None)))
        dS = (Pr - Y) / B
        grads = [None] * n
        grads[-1] = np.einsum('bl,bs,bc->lsc', Ls[-1], Phi[:, -1], dS)[..., None]
        R = np.einsum('bs,lsc,bc->bl', Phi[:, -1], last, dS)
        for j in range(n - 2, -1, -1):
            grads[j] = np.einsum('bl,bs,br->lsr', Ls[j], Phi[:, j], R)
            R = np.einsum('bs,lsr,br->bl', Phi[:, j], self.cores[j], R)
        return loss, grads, Pr

    def _fit_adam(self, Phi, Y, rng):
        n = Phi.shape[1]
        m = [np.zeros_like(c) for c in self.cores]; v = [np.zeros_like(c) for c in self.cores]; t = 0
        for _ in range(self.epochs):
            idx = rng.permutation(len(Phi)); ep = []
            for s in range(0, len(Phi), self.batch):
                b = idx[s:s + self.batch]
                loss, g, _ = self._grads(Phi[b], Y[b]); t += 1; ep.append(loss)
                for j in range(n):
                    m[j] = 0.9 * m[j] + 0.1 * g[j]; v[j] = 0.999 * v[j] + 0.001 * g[j] ** 2
                    self.cores[j] -= self.lr * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
            self.history_.append(float(np.mean(ep)))

    # ------------------------------------------------------------------------------- sweep training
    def _bond_update(self, Phi, Y, j, move):
        """Merge cores j and j+1 (one of them carries the label), optimise the merged tensor, split by a
        truncated SVD and move the label to j+1 (move='right') or j (move='left')."""
        a, b = self.cores[j], self.cores[j + 1]
        T = np.einsum('lscm,mtr->lstcr', a, b) if self.label_ == j else np.einsum('lsm,mtcr->lstcr', a, b)
        L, R = self._left(Phi, j), self._right(Phi, j + 2)
        p1, p2 = Phi[:, j], Phi[:, j + 1]; N = len(Phi)
        m = np.zeros_like(T); v = np.zeros_like(T); loss = None
        for t in range(1, self.steps + 1):                     # Adam on the merged two-site tensor
            P = _softmax(np.einsum('bl,bs,bt,lstcr,br->bc', L, p1, p2, T, R, optimize=True))
            loss = -np.mean(np.log(np.clip((P * Y).sum(1), 1e-12, None)))
            g = np.einsum('bl,bs,bt,bc,br->lstcr', L, p1, p2, (P - Y) / N, R, optimize=True)
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2
            T = T - self.sweep_lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        l, s, tt, c, r = T.shape
        if move == 'right':                                    # label goes to j + 1
            U, S, Vt = np.linalg.svd(T.reshape(l * s, tt * c * r), full_matrices=False)
        else:                                                  # label goes to j
            U, S, Vt = np.linalg.svd(T.transpose(0, 1, 3, 2, 4).reshape(l * s * c, tt * r), full_matrices=False)
        k = int(max(1, min(self.D, np.sum(S > self.cutoff * S[0]))))
        U, S, Vt = U[:, :k], S[:k], Vt[:k]
        if move == 'right':
            self.cores[j] = U.reshape(l, s, k)
            self.cores[j + 1] = (S[:, None] * Vt).reshape(k, tt, c, r); self.label_ = j + 1
        else:
            self.cores[j] = (U * S[None, :]).reshape(l, s, c, k)
            self.cores[j + 1] = Vt.reshape(k, tt, r); self.label_ = j
        return loss

    def _fit_sweep(self, Phi, Y):
        n = Phi.shape[1]
        if n < 2:
            raise ValueError("method='sweep' needs at least two features")
        for _ in range(self.sweeps):
            losses = [self._bond_update(Phi, Y, j, 'left') for j in range(n - 2, -1, -1)]     # label: end -> start
            losses += [self._bond_update(Phi, Y, j, 'right') for j in range(n - 1)]          # label: start -> end
            self.history_.append(float(losses[-1]))

    # ------------------------------------------------------------------------------- public API
    def fit(self, X, y):
        """Train on X (samples x features) and labels y; returns self."""
        X = np.asarray(X, float); y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError('X must be samples x features with one label per sample')
        self.classes_ = np.unique(y); C = len(self.classes_)
        if C < 2:
            raise ValueError('need at least two classes in y')
        self.mn, self.mx = X.min(0), X.max(0)
        Phi = self._phi(self._scale(X)); n = X.shape[1]
        Y = (y[:, None] == self.classes_[None, :]).astype(float)
        rng = np.random.default_rng(self.seed); D = self.D
        dims = [1] + [D] * (n - 1) + [1]
        self.cores = []
        for j in range(n):
            if j == n - 1:
                core = rng.normal(0, 0.1, (dims[j], self.d, C, 1))
            else:                                         # near-identity cores keep scores well scaled
                core = np.zeros((dims[j], self.d, dims[j + 1]))
                for s in range(self.d):
                    core[:, s, :] = np.eye(dims[j], dims[j + 1]) + rng.normal(0, 0.05, (dims[j], dims[j + 1]))
            self.cores.append(core)
        self.label_ = n - 1; self.history_ = []
        if self.method == 'adam':
            self._fit_adam(Phi, Y, rng)
        else:
            self._fit_sweep(Phi, Y)
        return self

    def predict_proba(self, X):
        """Class probabilities (softmax of the MPS scores)."""
        return _softmax(self._scores(self._phi(self._scale(X))))

    def predict(self, X):
        """Most probable class for each sample."""
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

    def score(self, X, y):
        """Accuracy on (X, y)."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    @property
    def bond_dimensions(self):
        """Bond dimensions between neighbouring sites (adapted by SVD truncation in sweep training)."""
        return [c.shape[-1] for c in self.cores[:-1]]

    @property
    def n_parameters(self):
        """Number of trainable numbers in the MPS cores."""
        return int(sum(c.size for c in self.cores))
