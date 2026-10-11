"""COBRA: Coherent Order-sensitive Belief and Risk-aware Action.

COBRA is a trainable robot belief filter. The robot keeps a density matrix
``rho`` over a small set of latent hypotheses (for example the human's intent).
Each sensor frame ``x_t`` updates it in two steps:

1. **Context-rotated evidence** - a generalized measurement
   ``rho -> M rho M / tr`` with ``M = U diag(exp(0.5 (W x_t + b))) U^dagger``.
   The diagonal part is ordinary Bayes-style evidence; the unitary
   ``U = exp(-i H(tanh(beta x_{t-1})))`` is a basis rotation driven by the
   *previous* frame. So what a reading means depends on what came before it
   (an order effect), which a commutative filter cannot represent.
2. **Dephasing** - ``rho -> (1 - g) rho + g diag(rho)`` with learned ``g``,
   controlling how long phase information survives between frames.

Hypothesis probabilities are the diagonal of ``rho`` (Born rule). Two built-in
reference points keep the quantum-style part honest: with ``H = 0`` COBRA is a
classical Bayes filter, and with ``g = 1`` it is a classical *incoherent*
context-dependent filter (no phase interference between frames). The
benchmark ablates both.

The off-diagonal mass of ``rho`` (l1 coherence) is exposed as a diagnostic.
Decisions are made outside the model by :func:`min_cost_action`, which turns
calibrated probabilities and an explicit cost matrix into an action, with an
"ask" action available for hedging.

This is a classical numerical simulation of a quantum-probability
representation; it needs only NumPy and makes no quantum-advantage claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

_EXPM_SCALE = 3      # scaling-and-squaring levels
_EXPM_ORDER = 9      # Taylor order


def _hermitian(vec: np.ndarray, d: int) -> np.ndarray:
    """Build Hermitian matrices from real parameters (..., d*d)."""
    out = np.zeros(vec.shape[:-1] + (d, d), dtype=complex)
    idx = np.arange(d)
    out[..., idx, idx] = vec[..., :d]
    iu, ju = np.triu_indices(d, 1)
    n_off = len(iu)
    re = vec[..., d:d + n_off]
    im = vec[..., d + n_off:d + 2 * n_off]
    out[..., iu, ju] = re + 1j * im
    out[..., ju, iu] = re - 1j * im
    return out


def _expm_skew(h: np.ndarray) -> np.ndarray:
    """Return exp(-i h) for batched Hermitian h by scaling and squaring."""
    scale = 2 ** _EXPM_SCALE
    a = (-1j / scale) * h
    eye = np.eye(h.shape[-1], dtype=complex)
    term = np.broadcast_to(eye, h.shape).copy()
    total = term.copy()
    for k in range(1, _EXPM_ORDER + 1):
        term = term @ a / k
        total += term
    for _ in range(_EXPM_SCALE):
        total = total @ total
    return total


@dataclass(frozen=True)
class Trace:
    """Per-frame outputs of a COBRA forward pass."""
    probs: np.ndarray       # (N, T, d) hypothesis probabilities after each frame
    coherence: np.ndarray   # (N, T) l1 coherence of rho after each frame


class COBRA:
    """Learned density-matrix belief filter over ``n_hyp`` hypotheses.

    Args:
        n_hyp: number of latent hypotheses (Hilbert-space dimension).
        n_features: sensor features per frame.
        beta: gain of the tanh squashing applied before the rotation generator.
        steps, batch, lr: Adam training settings (gradients by central finite
            differences, which is exact enough for ~50 parameters).
        l2: weight decay on all parameters.
        restarts: symmetry-breaking initialisations tried before full training.
        seed: deterministic initialisation and minibatch order.
    """

    def __init__(self, n_hyp: int = 3, n_features: int = 3, *, beta: float = 2.0,
                 steps: int = 100, batch: int = 160, lr: float = 0.1,
                 l2: float = 1e-4, restarts: int = 4, seed: int = 0) -> None:
        if n_hyp < 2 or n_features < 1:
            raise ValueError("need n_hyp >= 2 and n_features >= 1")
        self.d, self.m = int(n_hyp), int(n_features)
        self.beta, self.steps, self.batch = float(beta), int(steps), int(batch)
        self.lr, self.l2, self.seed = float(lr), float(l2), int(seed)
        self.restarts = int(restarts)
        self.trial_div = 4
        rng = np.random.default_rng(seed)
        self._n_gen = self.d * self.d
        self._shapes = [
            ("gens", (self.m + 1, self._n_gen)),
            ("W", (self.d, self.m)),
            ("b", (self.d,)),
            ("prior", (self.d,)),
            ("gamma", (1,)),
        ]
        self.theta = np.concatenate([
            0.15 * rng.standard_normal((self.m + 1) * self._n_gen),
            0.10 * rng.standard_normal(self.d * self.m),
            np.zeros(self.d), np.zeros(self.d), np.zeros(1),
        ])

    # ---- parameters -----------------------------------------------------
    @property
    def n_params(self) -> int:
        return int(self.theta.size)

    def _unpack(self, theta: np.ndarray) -> dict[str, np.ndarray]:
        out, pos = {}, 0
        for name, shape in self._shapes:
            size = int(np.prod(shape))
            out[name] = theta[pos:pos + size].reshape(shape)
            pos += size
        return out

    # ---- filtering ------------------------------------------------------
    def _forward(self, theta: np.ndarray, x: np.ndarray, *, trace: bool = False):
        p = self._unpack(theta)
        n, t_len, _ = x.shape
        herm = _hermitian(p["gens"], self.d)                  # (m+1, d, d)
        gamma = 1.0 / (1.0 + np.exp(-p["gamma"][0]))
        prior = np.exp(p["prior"] - p["prior"].max())
        prior /= prior.sum()
        rho = np.broadcast_to(np.diag(prior).astype(complex), (n, self.d, self.d)).copy()
        probs = np.empty((n, t_len, self.d))
        coh = np.empty((n, t_len))
        for t in range(t_len):
            xt = x[:, t]
            prev = x[:, t - 1] if t else np.zeros_like(xt)
            logit = xt @ p["W"].T + p["b"]
            w = np.exp(0.5 * (logit - logit.max(axis=1, keepdims=True)))
            gen = herm[0] + np.einsum("nk,kij->nij", np.tanh(self.beta * prev), herm[1:])
            u = _expm_skew(gen)
            # context-rotated evidence operator M = U diag(w) U^dagger
            meas = (u * w[:, None, :]) @ u.conj().transpose(0, 2, 1)
            rho = meas @ rho @ meas.conj().transpose(0, 2, 1)
            rho /= np.einsum("nii->n", rho).real[:, None, None]
            diag = np.einsum("nii->ni", rho)
            rho = (1.0 - gamma) * rho + gamma * (diag[:, :, None] * np.eye(self.d))
            probs[:, t] = np.clip(diag.real, 1e-9, None)
            probs[:, t] /= probs[:, t].sum(axis=1, keepdims=True)
            coh[:, t] = np.abs(rho).sum(axis=(1, 2)) - np.abs(diag).sum(axis=1)
        return (probs, coh) if trace else probs

    def _loss(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        probs = self._forward(theta, x)
        t_len = x.shape[1]
        wt = np.arange(1, t_len + 1, dtype=float)
        wt /= wt.sum()
        nll = -np.log(probs[np.arange(len(y)), :, y])          # (N, T)
        return float((nll * wt).sum(axis=1).mean() + self.l2 * float(theta @ theta))

    # ---- training -------------------------------------------------------
    def _warm_start(self, x: np.ndarray, y: np.ndarray) -> None:
        """Start the evidence step at the classical Gaussian log-likelihood.

        With zero rotation this is a Bayes filter; training then only has to
        discover which context rotations help beyond it.
        """
        flat = x.reshape(-1, self.m)
        lab = np.repeat(y, x.shape[1])
        var = flat.var(axis=0) + 1e-3
        mu = np.stack([flat[lab == k].mean(0) if np.any(lab == k) else flat.mean(0)
                       for k in range(self.d)])
        counts = np.maximum(np.bincount(y, minlength=self.d), 1)
        p = self._unpack(self.theta)
        p["W"][...] = mu / var
        p["b"][...] = -0.5 * (mu ** 2 / var).sum(axis=1)
        p["prior"][...] = np.log(counts / counts.sum())

    def _train(self, theta, x, y, tr, val, steps, rng):
        """Adam on finite-difference gradients; returns (best theta, val loss)."""
        theta, m1, m2 = theta.copy(), np.zeros_like(theta), np.zeros_like(theta)
        best_theta, best_val = theta.copy(), np.inf
        eps = 1e-4
        for step in range(1, steps + 1):
            idx = rng.choice(tr, size=min(self.batch, len(tr)), replace=False)
            xb, yb = x[idx], y[idx]
            grad = np.empty_like(theta)
            for k in range(theta.size):
                old = theta[k]
                theta[k] = old + eps
                hi = self._loss(theta, xb, yb)
                theta[k] = old - eps
                lo = self._loss(theta, xb, yb)
                theta[k] = old
                grad[k] = (hi - lo) / (2 * eps)
            m1 = 0.9 * m1 + 0.1 * grad
            m2 = 0.999 * m2 + 0.001 * grad ** 2
            lr = self.lr * (0.5 * (1 + np.cos(np.pi * step / steps)) * 0.9 + 0.1)
            theta -= lr * (m1 / (1 - 0.9 ** step)) / (np.sqrt(m2 / (1 - 0.999 ** step)) + 1e-8)
            if len(val) and (step % 10 == 0 or step == steps):
                v = self._loss(theta, x[val], y[val])
                if v < best_val:
                    best_val, best_theta = v, theta.copy()
        if not len(val):
            best_theta, best_val = theta, self._loss(theta, x[tr], y[tr])
        return best_theta, best_val

    def fit(self, x: np.ndarray, y: np.ndarray) -> "COBRA":
        """Train on sequences ``x`` (N, T, m) and integer labels ``y`` (N,).

        The context rotation and the evidence weights must grow together (a
        product of two near-zero terms is a saddle), so training uses
        ``restarts`` symmetry-breaking initialisations, keeps the one with the
        best validation loss after a short run, and then trains it fully. Every
        start begins from the classical Gaussian log-likelihood evidence.
        """
        x, y = np.asarray(x, float), np.asarray(y, int)
        if x.ndim != 3 or x.shape[2] != self.m or len(x) != len(y):
            raise ValueError("x must be (N, T, n_features) with len(y) == N")
        rng = np.random.default_rng(self.seed + 1)
        n_val = max(len(x) // 6, 1) if len(x) >= 30 else 0
        order = rng.permutation(len(x))
        val, tr = order[:n_val], order[n_val:]
        n_gen = (self.m + 1) * self._n_gen
        starts = []
        for r in range(max(self.restarts, 1)):
            r_rng = np.random.default_rng(self.seed * 101 + r)
            self._warm_start(x, y)
            th = self.theta.copy()
            if self.restarts > 1 or r > 0:
                th[:n_gen] = 0.7 * r_rng.standard_normal(n_gen)
                p = self._unpack(th)
                p["W"][...] += 0.8 * r_rng.standard_normal(p["W"].shape)
            starts.append(th)
        if len(starts) == 1:
            self.theta, _ = self._train(starts[0], x, y, tr, val, self.steps, rng)
            return self
        trial = max(self.steps // self.trial_div, 10)
        scored = [self._train(th, x, y, tr, val, trial, rng) for th in starts]
        best = min(scored, key=lambda s: s[1])[0]
        self.theta, _ = self._train(best, x, y, tr, val, self.steps, rng)
        return self

    # ---- inference ------------------------------------------------------
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Hypothesis probabilities after the last frame, shape (N, d)."""
        return self._forward(self.theta, np.asarray(x, float))[:, -1]

    def trace(self, x: np.ndarray) -> Trace:
        probs, coh = self._forward(self.theta, np.asarray(x, float), trace=True)
        return Trace(probs=probs, coherence=coh)

    def coherence(self, x: np.ndarray) -> np.ndarray:
        """l1 coherence of the final belief (0 for a classical belief)."""
        return self.trace(x).coherence[:, -1]


def min_cost_action(probs: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Pick the action with minimum expected cost.

    ``cost[a, z]`` is the cost of action ``a`` when hypothesis ``z`` is true.
    Include an "ask"/"wait" row to let the robot hedge under ambiguity.
    """
    return np.argmin(np.asarray(probs) @ np.asarray(cost).T, axis=1)


__all__ = ["COBRA", "Trace", "min_cost_action"]
