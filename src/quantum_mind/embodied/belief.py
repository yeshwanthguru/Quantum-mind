"""Belief slot of the VLA: fuse an instruction with the person's clarifying answers.

Three interchangeable slots share one interface (``fit`` on held-out sequences, ``posterior`` over the objects on the table):

* :class:`BayesSlot`: the product of the instruction belief and each answer's likelihood (answers independent and equally reliable).
* :class:`TemperedSlot` (``shared=True``: one exponent, calibrated Bayes): the classical control for context effects: the same product, but each answer is weighted by a learned
  exponent that depends on the *context* of the answer (first answer, same kind as the previous one, different kind).
* :class:`QuantumSlot`: a quantum-probability belief. The instruction belief is a state ``psi = sqrt(g0)`` over the objects, each
  answer is a measurement ``M = U_c diag(sqrt(L)) U_c^dagger`` whose rotation ``U_c = exp(iH_c)`` depends on the context ``c``,
  and the posterior is the Born rule. Measurements in different contexts need not commute, so the order of the answers can matter.
  With every ``H_c = 0`` it is exactly the Bayes slot; it is the same construction as the COBRA belief filter, on a table of objects.

The slots work on at most three objects, ordered left to right, padded with zeros; ``mask`` marks the real ones.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from .scene import COLORS, clarification, encode

K = 3
N_CTX = 3          # 0: first answer, 1: same kind as the previous answer, 2: different kind
EPS = 0.02         # likelihood floor shared by all slots, so one wrong answer is never an infinite penalty


def _floor(L, mask):
    return (1 - EPS) * L + EPS * mask[:, None, :] / mask.sum(-1)[:, None, None]


def _ctx_matrices(h, n=N_CTX):
    out = []
    for c in range(n):
        a = h[c * 9:(c + 1) * 9].reshape(3, 3)
        out.append(expm(1j * ((a + a.T) / 2 + 1j * (a - a.T) / 2)))
    return np.stack(out)


class BayesSlot:
    n_params = 0

    def posterior(self, g0, L, ctx, mask):
        p = g0 * np.prod(_floor(L, mask), axis=1)
        return p / np.maximum(p.sum(-1, keepdims=True), 1e-12)

    def fit(self, batch):
        return self


class _Fitted:
    def fit(self, batch, seed=0):
        rng = np.random.default_rng(seed)
        x0 = self.init(rng)

        def loss(x):
            p = self._post(x, batch['g0'], batch['L'], batch['ctx'], batch['mask'])
            return -np.mean(np.log(p[np.arange(len(batch['y'])), batch['y']] + 1e-9)) + 1e-3 * np.sum(x ** 2) / len(x)
        self.x = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 200}).x
        return self

    def posterior(self, g0, L, ctx, mask):
        return self._post(self.x, g0, L, ctx, mask)


class TemperedSlot(_Fitted):
    """Product rule with learned exponents. ``shared=True`` learns a single exponent (calibrated Bayes, no context)."""

    def __init__(self, shared=False):
        self.shared = shared
        self.n_params = 1 if shared else N_CTX

    def init(self, rng):
        return np.ones(self.n_params)

    def _post(self, x, g0, L, ctx, mask):
        ctx = np.zeros_like(ctx) if self.shared else ctx
        logp = np.log(g0 + 1e-12) + np.sum(x[ctx][:, :, None] * np.log(_floor(L, mask) + 1e-12), axis=1)
        logp = np.where(mask > 0, logp, -np.inf)
        p = np.exp(logp - logp.max(-1, keepdims=True))
        return p / p.sum(-1, keepdims=True)


class QuantumSlot(_Fitted):
    n_params = 9 * N_CTX

    def init(self, rng):
        return 0.05 * rng.standard_normal(9 * N_CTX)

    def _post(self, x, g0, L, ctx, mask):
        U = _ctx_matrices(x)
        Lf = _floor(L, mask)
        psi = np.sqrt(g0).astype(complex)
        for j in range(L.shape[1]):
            Uj = U[ctx[:, j]]                                             # (N, 3, 3)
            M = np.einsum('nij,nj,nkj->nik', Uj, np.sqrt(Lf[:, j]), Uj.conj())
            psi = np.einsum('nik,nk->ni', M, psi)
            psi = psi / np.maximum(np.linalg.norm(psi, axis=1, keepdims=True), 1e-12)
        p = np.abs(psi) ** 2 * mask
        return p / np.maximum(p.sum(-1, keepdims=True), 1e-12)


# ------------------------------------------------------------------------------------------------ simulated person
def _words(scene, k):
    """Descriptor words true of the k-th object (in left-to-right order): ``{kind: word}``."""
    order = np.argsort(scene.cells)
    cells = [scene.cells[i] for i in order]
    cols = [scene.colors[i] for i in order]
    w = {'color': COLORS[cols[k]]}
    if k == 0:
        w['position'] = 'leftmost'
    elif k == len(cells) - 1:
        w['position'] = 'rightmost'
    return w


def person_answers(scene, rng, e_first=0.05, e_same=0.10, e_switch=0.35, n_answers=2):
    """Simulated person who answers ``n_answers`` clarifying questions about the intended object.

    Each answer describes the intended object by its position or its colour. An answer is unreliable: with a probability that
    depends on the *context* (first answer; same kind as the previous answer; different kind) it describes another object instead.
    This is a classical, context-dependent noise model, chosen to make the answers context-dependent without being quantum.
    """
    cells = sorted(scene.cells)
    k = cells.index(scene.target)
    kinds = list(_words(scene, k))
    out, prev = [], None
    for j in range(n_answers):
        kind = kinds[int(rng.integers(len(kinds)))]
        ctx = 0 if prev is None else (1 if kind == prev else 2)
        e = (e_first, e_same, e_switch)[ctx]
        obj = k
        if rng.random() < e and len(cells) > 1:
            obj = int(rng.choice([i for i in range(len(cells)) if i != k]))
        w = _words(scene, obj)
        word = w.get(kind) or next(iter(w.values()))
        out.append((encode(('go', 'to', 'the', word)), ctx))
        prev = kind
    return out


def sequence_batch(model, scenes, rng, person=person_answers, **kw):
    """Turn scenes into the arrays the slots use: instruction belief, answer likelihoods, contexts, mask, target index."""
    n = len(scenes)
    g0, L = np.zeros((n, K)), np.zeros((n, 2, K))
    ctx, mask, y = np.zeros((n, 2), int), np.zeros((n, K)), np.zeros(n, int)
    for i, s in enumerate(scenes):
        cells = sorted(s.cells)
        m = len(cells)
        mask[i, :m] = 1
        g0[i, :m] = _restrict(model.ground(s.image, s.text), cells)
        for j, (text, c) in enumerate(person(s, rng, **kw)):
            L[i, j, :m] = _restrict(model.ground(s.image, text), cells)
            ctx[i, j] = c
        y[i] = cells.index(s.target)
    return dict(g0=g0, L=L, ctx=ctx, mask=mask, y=y)


def _restrict(g, cells):
    v = np.asarray(g)[cells]
    return v / v.sum() if v.sum() > 1e-9 else np.full(len(cells), 1.0 / len(cells))
