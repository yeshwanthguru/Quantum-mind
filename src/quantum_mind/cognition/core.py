"""The quantum-cognition core: a belief over hypotheses updated by context-dependent generalised measurements, trained by gradients.

One model that replaces the separate COBRA filter and the VLA belief slot. The belief is a density matrix ``rho`` over ``n_hyp`` hypotheses
(what the person means, what the object is, ...). Each evidence step ``t`` brings a feature vector ``x_t`` and a context vector ``c_t`` and
applies the measurement

.. math::
    M_t = U(c_t)\\,\\mathrm{diag}(w(x_t))\\,U(c_t)^\\dagger,\\qquad w = e^{(W x_t + b)/2},\\qquad U(c) = \\exp\\Big(i\\sum_k c_k H_k\\Big),

then ``rho <- M rho M^dagger / tr`` and partial dephasing ``rho <- (1-g) rho + g diag(rho)``. The prediction is the Born rule,
``diag(rho)``.

* With every ``H_k = 0`` it is exactly a Bayes filter whose evidence map is ``W x + b`` (:class:`BayesCore`); the tests check this.
* The rotations ``H_k`` are what makes the order and context of the evidence matter in a way a product of likelihoods cannot, and they are
  penalised toward zero, so coherence is switched on only when the data ask for it.
* Everything is differentiable (JAX), so the model trains by Adam on real data sizes, unlike the finite-difference COBRA.

The classical competitors share the interface and the evidence map: :class:`BayesCore` (no rotations), :class:`ContextWeightedCore`
(each context scales how much its evidence counts), :class:`MarkovContextCore` (a context-dependent stochastic transition between
hypotheses, the classical analogue of a rotation) and :class:`GRUCore` (a small recurrent network).

Needs the optional ``cognition`` extra: ``pip install "quantum-mind[cognition]"`` (JAX and Optax).
"""
from __future__ import annotations

import numpy as np

from .._optional import require

jax = require('jax', extra='cognition')
optax = require('optax', extra='cognition')
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
from jax.scipy.linalg import expm  # noqa: E402

CLIP = 8.0


def hermitian(A):
    """Hermitian matrix from a real square matrix: symmetric part plus ``i`` times the antisymmetric part."""
    return (A + jnp.swapaxes(A, -1, -2)) / 2 + 0.5j * (A - jnp.swapaxes(A, -1, -2))


def measurement(w, U):
    """The generalised measurement operator ``U diag(w) U^dagger``."""
    return U @ (w[:, None] * jnp.conj(U.T))


def update_state(rho, w, U):
    """Unnormalised post-measurement state ``M rho M^dagger`` (its trace is the probability of the outcome)."""
    M = measurement(w, U)
    return M @ rho @ jnp.conj(M.T)


def _prep(X, C, mask):
    X, C = jnp.asarray(X, jnp.float32), jnp.asarray(C, jnp.float32)
    mask = jnp.ones(X.shape[:2], bool) if mask is None else jnp.asarray(mask, bool)
    return X, C, mask


class _Model:
    """Shared fitting loop. Subclasses define ``init``, ``probs`` (batch of sequences -> probabilities) and ``penalty``."""

    def __init__(self, n_hyp=3, n_features=3, n_context=2, *, steps=500, lr=0.03, batch=256, restarts=3, seed=0, val_fraction=0.15):
        self.d, self.f, self.k = n_hyp, n_features, n_context
        self.steps, self.lr, self.batch, self.restarts, self.seed, self.val_fraction = steps, lr, batch, restarts, seed, val_fraction
        self.params = None

    @property
    def n_params(self):
        return int(sum(np.size(v) for v in jax.tree_util.tree_leaves(self.init(jax.random.PRNGKey(0)))))

    _jit_cache = {}

    def _key(self):
        return (type(self).__name__, self.d, self.f, self.k, getattr(self, 'coherent', None), getattr(self, 'dephasing', None),
                getattr(self, 'rotation_penalty', None), getattr(self, 'hidden', None))

    def _compiled(self):
        """JIT-compiled loss gradient and validation loss, shared by every model with the same configuration."""
        key = self._key()
        if key not in _Model._jit_cache:
            loss = lambda p, X, C, mask, y: self._nll(p, X, C, mask, y) + self.penalty(p)           # noqa: E731
            _Model._jit_cache[key] = (jax.jit(jax.value_and_grad(loss)), jax.jit(self._nll))
        return _Model._jit_cache[key]

    def _nll(self, params, X, C, mask, y):
        p = self.probs(params, X, C, mask)
        return -jnp.mean(jnp.log(p[jnp.arange(len(y)), y] + 1e-7))

    def fit(self, X, C, y, mask=None):
        """Fit by minibatch Adam with restarts; the restart with the best held-out log-loss is kept.

        Parameters
        ----------
        X : array (N, T, F)
            Evidence features per step.
        C : array (N, T, K)
            Context vector per step (a one-hot context id, or any real features).
        y : array (N,)
            Index of the hypothesis that turned out to be true.
        mask : array (N, T) of bool, optional
            ``False`` for padded steps of shorter sequences.
        """
        X, C, mask = _prep(X, C, mask)
        y = jnp.asarray(y, jnp.int32)
        n = len(y)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(n)
        n_val = max(int(n * self.val_fraction), 1)
        va, tr = order[:n_val], order[n_val:]
        grad, val_fn = self._compiled()
        val = lambda p: val_fn(p, X[va], C[va], mask[va], y[va])                                  # noqa: E731
        best, best_val = None, np.inf
        for r in range(self.restarts):
            params = self.init(jax.random.PRNGKey(self.seed * 1000 + r))
            opt = optax.adam(self.lr)
            state = opt.init(params)
            run_best, run_val = params, np.inf
            for s in range(self.steps):
                idx = tr[rng.integers(0, len(tr), min(self.batch, len(tr)))]
                _, g = grad(params, X[idx], C[idx], mask[idx], y[idx])
                upd, state = opt.update(g, state)
                params = optax.apply_updates(params, upd)
                if s % 20 == 0 or s == self.steps - 1:
                    v = float(val(params))
                    if v < run_val:
                        run_best, run_val = params, v
            if run_val < best_val:
                best, best_val = run_best, run_val
        self.params, self.val_nll = best, best_val
        return self

    def predict_proba(self, X, C, mask=None):
        """Probabilities over the hypotheses, shape ``(N, n_hyp)``."""
        X, C, mask = _prep(X, C, mask)
        return np.asarray(self.probs(self.params, X, C, mask))

    def log_loss(self, X, C, y, mask=None):
        p = self.predict_proba(X, C, mask)
        return float(-np.mean(np.log(p[np.arange(len(y)), np.asarray(y)] + 1e-7)))

    def penalty(self, params):
        return 1e-4 * sum(jnp.sum(v ** 2) for v in jax.tree_util.tree_leaves(params))


class QuantumCognitionCore(_Model):
    """The quantum-cognition core (see the module docstring).

    Parameters
    ----------
    n_hyp, n_features, n_context : int
        Number of hypotheses, evidence features per step and context features per step.
    coherent : bool, optional
        ``False`` removes the rotations (a Bayes filter); default ``True``.
    dephasing : bool, optional
        Learn a partial dephasing after each update; default ``True``.
    rotation_penalty : float, optional
        L2 penalty pulling the rotation generators toward zero (toward Bayes); default 1e-2.
    steps, lr, batch, restarts, seed : optional
        Optimisation settings (see :meth:`fit`).
    """

    def __init__(self, n_hyp=3, n_features=3, n_context=2, *, coherent=True, dephasing=True, rotation_penalty=1e-2, **kw):
        super().__init__(n_hyp, n_features, n_context, **kw)
        self.coherent, self.dephasing, self.rotation_penalty = coherent, dephasing, rotation_penalty

    def init(self, key):
        k = jax.random.split(key, 4)
        d = self.d
        return {'a': 0.1 * jax.random.normal(k[0], (d,)), 'phi': jnp.zeros(d), 'W': 0.3 * jax.random.normal(k[1], (self.f, d)),
                'b': jnp.zeros(d), 'H': 0.05 * jax.random.normal(k[2], (self.k, d, d)), 'g': jnp.asarray(-2.0)}

    def penalty(self, params):
        return 1e-4 * jnp.sum(params['W'] ** 2) + self.rotation_penalty * jnp.mean(params['H'] ** 2)

    def _prior(self, params):
        amp = jnp.sqrt(jax.nn.softmax(params['a']))
        psi = amp * jnp.exp(1j * params['phi']) if self.coherent else amp.astype(jnp.complex64)
        return jnp.outer(psi, jnp.conj(psi))

    def _step(self, params, rho, x, c):
        w = jnp.exp(0.5 * jnp.clip(x @ params['W'] + params['b'], -CLIP, CLIP))
        if self.coherent:
            U = expm(1j * jnp.einsum('k,kij->ij', c, hermitian(params['H'])))
        else:
            U = jnp.eye(self.d, dtype=jnp.complex64)
        rho = update_state(rho, w, U)
        rho = rho / jnp.maximum(jnp.real(jnp.trace(rho)), 1e-12)
        if self.dephasing:
            g = jax.nn.sigmoid(params['g'])
            rho = (1 - g) * rho + g * jnp.diag(jnp.diag(rho))
        return rho

    def state(self, params, X, C, mask):
        """Final density matrix of one sequence (``X`` of shape ``(T, F)``)."""
        def body(rho, inp):
            x, c, m = inp
            return jnp.where(m, self._step(params, rho, x, c), rho), None
        rho, _ = lax.scan(body, self._prior(params), (X, C, mask))
        return rho

    def probs(self, params, X, C, mask):
        rho = jax.vmap(lambda x, c, m: self.state(params, x, c, m))(X, C, mask)
        p = jnp.real(jnp.diagonal(rho, axis1=1, axis2=2))
        return p / jnp.sum(p, axis=1, keepdims=True)

    def coherence(self, X, C, mask=None):
        """l1 norm of the off-diagonal part of the final state: how much interference the model uses (0 for a Bayes filter)."""
        X, C, mask = _prep(X, C, mask)
        rho = jax.vmap(lambda x, c, m: self.state(self.params, x, c, m))(X, C, mask)
        return np.asarray(jnp.sum(jnp.abs(rho), axis=(1, 2)) - jnp.sum(jnp.abs(jnp.diagonal(rho, axis1=1, axis2=2)), axis=1))

    @property
    def rotation_size(self):
        """Root-mean-square of the learned rotation generators (0 means Bayes)."""
        return float(jnp.sqrt(jnp.mean(self.params['H'] ** 2)))


class BayesCore(QuantumCognitionCore):
    """The same evidence map without rotations or dephasing: a Bayes filter with a learned likelihood."""

    def __init__(self, *a, **kw):
        super().__init__(*a, coherent=False, dephasing=False, **kw)


class ContextWeightedCore(_Model):
    """Classical context effect: each context scales how much its evidence counts, ``log p = a + sum_t lambda(c_t) (W x_t + b)``."""

    def init(self, key):
        k = jax.random.split(key, 2)
        return {'a': 0.1 * jax.random.normal(k[0], (self.d,)), 'W': 0.3 * jax.random.normal(k[1], (self.f, self.d)),
                'b': jnp.zeros(self.d), 'lam': jnp.ones(self.k)}

    def probs(self, params, X, C, mask):
        ev = jnp.clip(X @ params['W'] + params['b'], -CLIP, CLIP)
        lam = C @ params['lam']
        logit = params['a'] + jnp.sum(jnp.where(mask[..., None], lam[..., None] * ev, 0.0), axis=1)
        return jax.nn.softmax(logit, axis=1)


class MarkovContextCore(_Model):
    """Classical analogue of a rotation: a context-dependent stochastic transition between hypotheses, then the evidence."""

    def init(self, key):
        k = jax.random.split(key, 2)
        return {'a': 0.1 * jax.random.normal(k[0], (self.d,)), 'W': 0.3 * jax.random.normal(k[1], (self.f, self.d)),
                'b': jnp.zeros(self.d), 'T': 3.0 * jnp.eye(self.d)[None].repeat(self.k, 0)}

    def probs(self, params, X, C, mask):
        def seq(x, c, m):
            def body(p, inp):
                xt, ct, mt = inp
                T = jax.nn.softmax(jnp.einsum('k,kij->ij', ct, params['T']), axis=-1)
                q = (T.T @ p) * jnp.exp(0.5 * jnp.clip(xt @ params['W'] + params['b'], -CLIP, CLIP) * 2)
                q = q / jnp.sum(q)
                return jnp.where(mt, q, p), None
            p, _ = lax.scan(body, jax.nn.softmax(params['a']), (x, c, m))
            return p
        return jax.vmap(seq)(X, C, mask)


class GRUCore(_Model):
    """A small GRU over the concatenated evidence and context, the generic recurrent baseline."""

    def __init__(self, *a, hidden=6, **kw):
        super().__init__(*a, **kw)
        self.hidden = hidden

    def init(self, key):
        k = jax.random.split(key, 4)
        n_in, h = self.f + self.k, self.hidden
        s = 1 / np.sqrt(n_in + h)
        return {'Wz': s * jax.random.normal(k[0], (n_in + h, h)), 'Wr': s * jax.random.normal(k[1], (n_in + h, h)),
                'Wh': s * jax.random.normal(k[2], (n_in + h, h)), 'bz': jnp.zeros(h), 'br': jnp.zeros(h), 'bh': jnp.zeros(h),
                'Wo': s * jax.random.normal(k[3], (h, self.d)), 'bo': jnp.zeros(self.d)}

    def probs(self, params, X, C, mask):
        def seq(x, c, m):
            def body(h, inp):
                xt, ct, mt = inp
                v = jnp.concatenate([xt, ct, h])
                z = jax.nn.sigmoid(v @ params['Wz'] + params['bz'])
                r = jax.nn.sigmoid(v @ params['Wr'] + params['br'])
                cand = jnp.tanh(jnp.concatenate([xt, ct, r * h]) @ params['Wh'] + params['bh'])
                return jnp.where(mt, (1 - z) * h + z * cand, h), None
            h, _ = lax.scan(body, jnp.zeros(self.hidden), (x, c, m))
            return jax.nn.softmax(h @ params['Wo'] + params['bo'])
        return jax.vmap(seq)(X, C, mask)
