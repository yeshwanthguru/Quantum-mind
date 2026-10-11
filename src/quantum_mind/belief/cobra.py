r"""COBRA: a context-rotated density-matrix belief filter for robots that read people.

A robot watching a person receives a stream of sensor frames (gaze, hand motion, force). What a reading
*means* often depends on what came just before it: the same hand movement signals a hand-over after a
look at the robot and a retrieval after a look away. A classical Bayes filter treats the frames as
conditionally independent and cannot represent that. COBRA keeps the robot's belief as a density matrix
:math:`\rho` over :math:`d` hypotheses and updates it with a *generalised measurement* whose basis is
rotated by the previous frame:

.. math::

    M_t = U(x_{t-1})\,\mathrm{diag}\bigl(e^{(Wx_t + b)/2}\bigr)\,U(x_{t-1})^\dagger , \qquad
    \rho_t \propto M_t \rho_{t-1} M_t ,

    U(x) = \exp\bigl(-i\,[\,H_0 + \textstyle\sum_k \tanh(\beta x_k)\,H_k\,]\bigr),

followed by partial dephasing :math:`\rho \leftarrow (1-g)\rho + g\,\mathrm{diag}(\rho)`. The probability of
hypothesis :math:`z` is :math:`\rho_{zz}` (Born rule). The :math:`H_k` are learned Hermitian matrices.

Two special cases make the model honest about what is quantum-like in it. With :math:`H = 0` it is a
classical Bayes filter with log-linear likelihoods. With :math:`g = 1` it is a classical *incoherent*
context-dependent filter: the previous frame still changes the meaning of the next one, but no phase
survives between frames. Comparing the three isolates the effect of the context rotation and of the
coherence.

This is a classical simulation of the quantum-probability formalism; it needs only NumPy and does not claim
a quantum computational advantage. Training uses finite-difference Adam with symmetry-breaking restarts,
because the rotation and the evidence weights must grow together (a product of two near-zero terms is a
saddle); the benchmark in ``benchmarks/context_intent_benchmark.py`` shows that the context mechanism is
found on some restarts and seeds and not on others.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ['COBRA', 'BeliefTrace', 'OnlineBelief', 'GaussianBayesFilter', 'min_cost_action']

_EXPM_SCALE = 3      # scaling-and-squaring levels
_EXPM_ORDER = 9      # Taylor order


def _hermitian(vec, d):
    """Hermitian matrices from real parameters of shape (..., d*d)."""
    out = np.zeros(vec.shape[:-1] + (d, d), dtype=complex)
    idx = np.arange(d)
    out[..., idx, idx] = vec[..., :d]
    iu, ju = np.triu_indices(d, 1)
    n_off = len(iu)
    re, im = vec[..., d:d + n_off], vec[..., d + n_off:d + 2 * n_off]
    out[..., iu, ju] = re + 1j * im
    out[..., ju, iu] = re - 1j * im
    return out


def _expm_skew(h):
    """exp(-i h) for batched Hermitian h, by scaling and squaring."""
    scale = 2 ** _EXPM_SCALE
    a = (-1j / scale) * h
    term = np.broadcast_to(np.eye(h.shape[-1], dtype=complex), h.shape).copy()
    total = term.copy()
    for k in range(1, _EXPM_ORDER + 1):
        term = term @ a / k
        total += term
    for _ in range(_EXPM_SCALE):
        total = total @ total
    return total


def min_cost_action(probs, cost):
    """Action with minimum expected cost under a belief.

    Parameters
    ----------
    probs : array_like, shape (N, d) or (d,)
        Belief over the hypotheses.
    cost : array_like, shape (n_actions, d)
        ``cost[a, z]`` is the cost of action ``a`` when hypothesis ``z`` is true. Include an "ask" or
        "wait" row so the robot can hedge under ambiguity.

    Returns
    -------
    numpy.ndarray of int
        Index of the best action for each belief.

    Examples
    --------
    >>> from quantum_mind.belief import min_cost_action
    >>> cost = [[0, 6, 10], [1.5, 0, 0], [0.8, 0.8, 0.8]]      # hand over, wait, ask
    >>> min_cost_action([[0.97, 0.02, 0.01], [0.7, 0.2, 0.1]], cost).tolist()
    [0, 2]
    """
    probs, cost = np.atleast_2d(np.asarray(probs, float)), np.asarray(cost, float)
    if probs.shape[1] != cost.shape[1]:
        raise ValueError('cost must have one column per hypothesis')
    return np.argmin(probs @ cost.T, axis=1)


@dataclass(frozen=True)
class BeliefTrace:
    """Per-frame outputs of a COBRA forward pass.

    Attributes
    ----------
    probs : numpy.ndarray, shape (N, T, d)
        Hypothesis probabilities after each frame.
    coherence : numpy.ndarray, shape (N, T)
        l1 coherence (sum of off-diagonal moduli) of :math:`\\rho` after each frame.
    rho : numpy.ndarray, shape (N, T, d, d)
        Density matrix after each frame.
    """
    probs: np.ndarray
    coherence: np.ndarray
    rho: np.ndarray


class OnlineBelief:
    """A COBRA belief updated one frame at a time, as a robot would run it.

    Create it with :meth:`COBRA.start`. Call :meth:`update` with each new sensor frame; read
    :attr:`probs` and :attr:`coherence` after every update.

    Examples
    --------
    >>> import numpy as np
    >>> from quantum_mind.belief import COBRA
    >>> belief = COBRA(3, 3, restarts=1, steps=1).start()
    >>> p = belief.update([0.5, -0.2, 0.1])
    >>> bool(abs(p.sum() - 1) < 1e-9)
    True
    """

    def __init__(self, model, rho, prev):
        self._model, self.rho, self._prev = model, rho, prev
        self.frames = 0

    @property
    def probs(self):
        p = np.clip(np.real(np.diag(self.rho)), 1e-9, None)
        return p / p.sum()

    @property
    def coherence(self):
        return float(np.abs(self.rho).sum() - np.abs(np.diag(self.rho)).sum())

    def update(self, frame):
        """Fold one sensor frame into the belief and return the new probabilities."""
        x = np.asarray(frame, float).reshape(1, -1)
        if x.shape[1] != self._model.m or not np.all(np.isfinite(x)):
            raise ValueError('frame must be finite with %d features' % self._model.m)
        self.rho = self._model._step(self._model.theta, self.rho[None], x, self._prev)[0]
        self._prev = x
        self.frames += 1
        return self.probs


class COBRA:
    """Context-rotated density-matrix belief filter (see the module docstring for the equations).

    Parameters
    ----------
    n_hyp : int, optional
        Number of latent hypotheses (the Hilbert-space dimension), for example intents.
    n_features : int, optional
        Sensor features per frame.
    beta : float, optional
        Gain of the tanh squashing applied before the rotation generator.
    steps, batch, lr : optional
        Adam settings. Gradients are central finite differences, which is exact enough for the few dozen
        parameters.
    l2 : float, optional
        Weight decay on all parameters.
    restarts : int, optional
        Symmetry-breaking initialisations tried before full training; the one with the best validation
        loss after a short run is trained fully.
    seed : int, optional
        Deterministic initialisation and minibatch order.

    Attributes
    ----------
    theta : numpy.ndarray
        All learned parameters (``n_params`` values).

    Examples
    --------
    >>> import numpy as np
    >>> from quantum_mind.belief import COBRA
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=(60, 3, 2)); y = (x[:, :, 0].sum(1) > 0).astype(int)
    >>> model = COBRA(2, 2, steps=20, restarts=1).fit(x, y)
    >>> model.predict_proba(x).shape
    (60, 2)
    """

    def __init__(self, n_hyp=3, n_features=3, *, beta=2.0, steps=100, batch=160, lr=0.1, l2=1e-4, restarts=4, seed=0):
        if n_hyp < 2 or n_features < 1:
            raise ValueError('need n_hyp >= 2 and n_features >= 1')
        self.d, self.m = int(n_hyp), int(n_features)
        self.beta, self.steps, self.batch = float(beta), int(steps), int(batch)
        self.lr, self.l2, self.seed = float(lr), float(l2), int(seed)
        self.restarts, self.trial_div = int(restarts), 4
        rng = np.random.default_rng(seed)
        self._n_gen = self.d * self.d
        self._shapes = [('gens', (self.m + 1, self._n_gen)), ('W', (self.d, self.m)), ('b', (self.d,)),
                        ('prior', (self.d,)), ('gamma', (1,))]
        self.theta = np.concatenate([0.15 * rng.standard_normal((self.m + 1) * self._n_gen),
                                     0.10 * rng.standard_normal(self.d * self.m),
                                     np.zeros(self.d), np.zeros(self.d), np.zeros(1)])

    # ------------------------------------------------------------------ parameters
    @property
    def n_params(self):
        """Number of learned parameters."""
        return int(self.theta.size)

    def _unpack(self, theta):
        out, pos = {}, 0
        for name, shape in self._shapes:
            size = int(np.prod(shape))
            out[name] = theta[pos:pos + size].reshape(shape)
            pos += size
        return out

    @property
    def dephasing(self):
        """The learned dephasing strength :math:`g` in (0, 1)."""
        return float(1.0 / (1.0 + np.exp(-self._unpack(self.theta)['gamma'][0])))

    # ------------------------------------------------------------------ filtering
    def _prior_rho(self, theta, n):
        p = self._unpack(theta)
        prior = np.exp(p['prior'] - p['prior'].max())
        prior /= prior.sum()
        return np.broadcast_to(np.diag(prior).astype(complex), (n, self.d, self.d)).copy()

    def _step(self, theta, rho, xt, prev):
        """One frame: context-rotated generalised measurement, then dephasing."""
        p = self._unpack(theta)
        gamma = 1.0 / (1.0 + np.exp(-p['gamma'][0]))
        herm = _hermitian(p['gens'], self.d)
        logit = xt @ p['W'].T + p['b']
        w = np.exp(0.5 * (logit - logit.max(axis=1, keepdims=True)))
        gen = herm[0] + np.einsum('nk,kij->nij', np.tanh(self.beta * prev), herm[1:])
        u = _expm_skew(gen)
        meas = (u * w[:, None, :]) @ u.conj().transpose(0, 2, 1)
        rho = meas @ rho @ meas.conj().transpose(0, 2, 1)
        rho /= np.einsum('nii->n', rho).real[:, None, None]
        diag = np.einsum('nii->ni', rho)
        return (1.0 - gamma) * rho + gamma * (diag[:, :, None] * np.eye(self.d))

    def _forward(self, theta, x, *, trace=False):
        n, t_len, _ = x.shape
        rho = self._prior_rho(theta, n)
        probs = np.empty((n, t_len, self.d))
        coh = np.empty((n, t_len))
        rhos = np.empty((n, t_len, self.d, self.d), dtype=complex) if trace else None
        for t in range(t_len):
            prev = x[:, t - 1] if t else np.zeros_like(x[:, t])
            rho = self._step(theta, rho, x[:, t], prev)
            diag = np.einsum('nii->ni', rho)
            if trace:
                rhos[:, t] = rho
            probs[:, t] = np.clip(diag.real, 1e-9, None)
            probs[:, t] /= probs[:, t].sum(axis=1, keepdims=True)
            coh[:, t] = np.abs(rho).sum(axis=(1, 2)) - np.abs(diag).sum(axis=1)
        return (probs, coh, rhos) if trace else probs

    def _loss(self, theta, x, y):
        probs = self._forward(theta, x)
        wt = np.arange(1, x.shape[1] + 1, dtype=float)
        wt /= wt.sum()
        nll = -np.log(probs[np.arange(len(y)), :, y])
        return float((nll * wt).sum(axis=1).mean() + self.l2 * float(theta @ theta))

    # ------------------------------------------------------------------ training
    def _warm_start(self, x, y):
        """Start the evidence step at the classical Gaussian log-likelihood (a Bayes filter at ``H = 0``)."""
        flat, lab = x.reshape(-1, self.m), np.repeat(y, x.shape[1])
        var = flat.var(axis=0) + 1e-3
        mu = np.stack([flat[lab == k].mean(0) if np.any(lab == k) else flat.mean(0) for k in range(self.d)])
        counts = np.maximum(np.bincount(y, minlength=self.d), 1)
        p = self._unpack(self.theta)
        p['W'][...] = mu / var
        p['b'][...] = -0.5 * (mu ** 2 / var).sum(axis=1)
        p['prior'][...] = np.log(counts / counts.sum())

    def _train(self, theta, x, y, tr, val, steps, rng):
        theta, m1, m2 = theta.copy(), np.zeros_like(theta), np.zeros_like(theta)
        best_theta, best_val, eps = theta.copy(), np.inf, 1e-4
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

    def fit(self, x, y):
        """Train on sequences ``x`` of shape (N, T, n_features) and integer labels ``y`` of shape (N,).

        Returns
        -------
        COBRA
            ``self``, so calls can be chained.

        Raises
        ------
        ValueError
            If the shapes do not match the model or a label is outside ``range(n_hyp)``.
        """
        x, y = np.asarray(x, float), np.asarray(y, int)
        if x.ndim != 3 or x.shape[2] != self.m or len(x) != len(y):
            raise ValueError('x must be (N, T, n_features) with len(y) == N')
        if y.min() < 0 or y.max() >= self.d:
            raise ValueError('labels must be integers in range(n_hyp)')
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
                self._unpack(th)['W'][...] += 0.8 * r_rng.standard_normal((self.d, self.m))
            starts.append(th)
        if len(starts) == 1:
            self.theta, _ = self._train(starts[0], x, y, tr, val, self.steps, rng)
            return self
        trial = max(self.steps // self.trial_div, 10)
        best = min((self._train(th, x, y, tr, val, trial, rng) for th in starts), key=lambda s: s[1])[0]
        self.theta, _ = self._train(best, x, y, tr, val, self.steps, rng)
        return self

    # ------------------------------------------------------------------ inference
    def predict_proba(self, x):
        """Hypothesis probabilities after the last frame, shape (N, d)."""
        return self._forward(self.theta, np.asarray(x, float))[:, -1]

    def trace(self, x):
        """Probabilities, coherence and density matrix after every frame (a :class:`BeliefTrace`)."""
        probs, coh, rhos = self._forward(self.theta, np.asarray(x, float), trace=True)
        return BeliefTrace(probs=probs, coherence=coh, rho=rhos)

    def coherence(self, x):
        """l1 coherence of the final belief (0 for a classical belief), shape (N,)."""
        return self.trace(x).coherence[:, -1]

    def start(self):
        """Begin an :class:`OnlineBelief` for streaming use."""
        return OnlineBelief(self, self._prior_rho(self.theta, 1)[0], np.zeros((1, self.m)))


class GaussianBayesFilter:
    """Commutative naive-Bayes filter with independent Gaussian features per hypothesis.

    The classical reference for :class:`COBRA`: frames are conditionally independent given the
    hypothesis, so reordering them cannot change the belief.

    Examples
    --------
    >>> import numpy as np
    >>> from quantum_mind.belief import GaussianBayesFilter
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 2, 80); x = rng.normal(size=(80, 3, 2)) + y[:, None, None]
    >>> GaussianBayesFilter(2).fit(x, y).predict_proba(x).shape
    (80, 2)
    """

    def __init__(self, n_hyp=3):
        self.d = int(n_hyp)
        self.n_params = 0

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, int)
        flat, lab = x.reshape(-1, x.shape[2]), np.repeat(y, x.shape[1])
        self.mu = np.stack([flat[lab == k].mean(0) if np.any(lab == k) else flat.mean(0) for k in range(self.d)])
        self.var = np.stack([flat[lab == k].var(0) + 1e-3 if np.sum(lab == k) > 1 else flat.var(0) + 1e-3
                             for k in range(self.d)])
        self.logp = np.log(np.maximum(np.bincount(y, minlength=self.d), 1) / len(y))
        self.n_params = int(self.mu.size + self.var.size + self.d)
        return self

    def predict_proba(self, x):
        x = np.asarray(x, float)
        ll = -0.5 * (((x[:, :, None, :] - self.mu) ** 2) / self.var + np.log(self.var))
        z = self.logp + ll.sum(axis=(1, 3))
        z = np.exp(z - z.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)
