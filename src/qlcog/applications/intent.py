"""Intent resolution from ambiguous commands and context cues (robots, assistants, interfaces).

A user's intent is one of K options ("fetch the red cup", "fetch the blue cup", ...). Each context cue
(a word, a gesture, a gaze direction, the time of day) carries a likelihood over intents. Classical
Bayesian updating multiplies the likelihoods, so the order in which cues arrive does not matter.

QuantumIntentResolver represents belief as amplitudes psi = sqrt(prior) and applies each cue c as a
Kraus operator K_c = R(theta_c) diag(sqrt(L_c)) R(theta_c)^T, where R(theta_c) rotates the intent
basis by the cue's incompatibility angle theta_c (in the plane of the cue's two most likely intents).
With every theta_c = 0 the posterior is exactly Bayes' rule; theta_c != 0 makes cues incompatible, so
their order matters, as in question-order effects. The angles can be fitted to observed choices.

    r = QuantumIntentResolver(['red cup', 'blue cup', 'bowl'], prior=[1/3, 1/3, 1/3])
    r.add_cue('word "cup"', [0.45, 0.45, 0.10], theta=0.0)
    r.add_cue('points left', [0.70, 0.20, 0.10], theta=0.4)
    r.posterior(['word "cup"', 'points left'])        # intent probabilities
    r.order_effect('word "cup"', 'points left')        # total variation between the two orders"""
from __future__ import annotations

import numpy as np

__all__ = ['QuantumIntentResolver', 'BayesIntentResolver']


class BayesIntentResolver:
    """Classical baseline: posterior proportional to prior times the product of cue likelihoods."""

    def __init__(self, intents, prior=None):
        self.intents = list(intents); K = len(self.intents)
        self.prior = np.full(K, 1 / K) if prior is None else _prob(prior, K)
        self.cues = {}

    def add_cue(self, name, likelihood, theta=0.0):
        """Register a cue with its likelihood over intents (theta is ignored: classical cues commute)."""
        self.cues[name] = _prob(likelihood, len(self.intents), normalise=False)
        return self

    def posterior(self, cues):
        """Intent probabilities after the cues (order-free)."""
        p = self.prior.copy()
        for c in cues:
            p = p * self.cues[c]; p = p / p.sum()
        return p

    def order_effect(self, a, b):
        """Total variation between the posteriors for cue orders (a, b) and (b, a); always 0 here."""
        return 0.5 * float(np.abs(self.posterior([a, b]) - self.posterior([b, a])).sum())


class QuantumIntentResolver(BayesIntentResolver):
    """Quantum-like resolver; nests BayesIntentResolver (all theta = 0)."""

    def add_cue(self, name, likelihood, theta=0.0):
        """Register a cue: likelihood over intents and incompatibility angle theta (radians)."""
        L = _prob(likelihood, len(self.intents), normalise=False)
        i, j = np.argsort(L)[::-1][:2]                      # rotate in the plane of the two leading intents
        R = np.eye(len(L)); c, s = np.cos(theta), np.sin(theta)
        R[[i, i, j, j], [i, j, i, j]] = [c, -s, s, c]
        self.cues[name] = R @ np.diag(np.sqrt(L)) @ R.T
        return self

    def posterior(self, cues):
        """Intent probabilities |psi|^2 after applying the cues in the given order."""
        psi = np.sqrt(self.prior)
        for c in cues:
            psi = self.cues[c] @ psi; psi = psi / np.linalg.norm(psi)
        return psi ** 2

    def resolve(self, cues):
        """(most likely intent, its probability, entropy of the posterior in bits)."""
        p = self.posterior(cues); k = int(np.argmax(p))
        H = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
        return self.intents[k], float(p[k]), H


def _prob(v, K, normalise=True):
    v = np.asarray(v, float)
    if v.shape != (K,) or np.any(v < 0) or v.sum() <= 0:
        raise ValueError('expected %d non-negative weights' % K)
    return v / v.sum() if normalise else v
