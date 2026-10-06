r"""Intent resolution from ambiguous commands and context cues (robots, assistants, interfaces).

A user's intent is one of K options ("fetch the red cup", "fetch the blue cup", ...). Each context cue
(a word, a gesture, a gaze direction, the time of day) carries a likelihood over intents. Classical
Bayesian updating multiplies the likelihoods, so the order in which cues arrive does not matter.

:class:`QuantumIntentResolver` represents belief as amplitudes :math:`\psi = \sqrt{\text{prior}}` and
applies each cue c as a Kraus operator

.. math:: K_c = R(\theta_c)\,\mathrm{diag}\bigl(\sqrt{L_c}\bigr)\,R(\theta_c)^T,

where :math:`R(\theta_c)` rotates the intent basis by the cue's incompatibility angle
:math:`\theta_c`, in the plane of the cue's two most likely intents. With every
:math:`\theta_c = 0` the posterior is exactly Bayes' rule; :math:`\theta_c \neq 0` makes cues
incompatible, so their order matters, as in question-order effects. The angles can be fitted to
observed choices.

Examples
--------
>>> from quantum_mind.applications.intent import QuantumIntentResolver
>>> r = QuantumIntentResolver(['red cup', 'blue cup', 'bowl'])
>>> _ = r.add_cue('points left', [0.70, 0.20, 0.10], theta=0.5)
>>> _ = r.add_cue('looks right', [0.25, 0.60, 0.15], theta=0.5)
>>> r.posterior(['points left', 'looks right']).round(3)
array([0.482, 0.498, 0.02 ])
>>> round(r.order_effect('points left', 'looks right'), 3)
0.079
"""
from __future__ import annotations

import numpy as np

__all__ = ['QuantumIntentResolver', 'BayesIntentResolver']


class BayesIntentResolver:
    """Classical baseline: posterior proportional to the prior times the cue likelihoods.

    Parameters
    ----------
    intents : sequence of str
        Names of the K possible intents.
    prior : array_like, optional
        Prior over intents (normalised); uniform by default.

    Attributes
    ----------
    cues : dict
        Registered cues, by name.
    """

    def __init__(self, intents, prior=None):
        self.intents = list(intents)
        K = len(self.intents)
        self.prior = np.full(K, 1 / K) if prior is None else _prob(prior, K)
        self.cues = {}

    def add_cue(self, name, likelihood, theta=0.0):
        """Register a cue.

        Parameters
        ----------
        name : str
            Cue name.
        likelihood : array_like
            Non-negative likelihood of the cue under each intent.
        theta : float, optional
            Ignored here (classical cues commute); used by :class:`QuantumIntentResolver`.

        Returns
        -------
        BayesIntentResolver
            ``self``, so calls can be chained.
        """
        self.cues[name] = _prob(likelihood, len(self.intents), normalise=False)
        return self

    def posterior(self, cues):
        """Intent probabilities after a sequence of cues.

        Parameters
        ----------
        cues : sequence of str
            Names of registered cues.

        Returns
        -------
        numpy.ndarray
            Posterior over intents (independent of the order of ``cues``).
        """
        p = self.prior.copy()
        for c in cues:
            p = p * self.cues[c]
            p = p / p.sum()
        return p

    def order_effect(self, a, b):
        """Size of the order effect between two cues.

        Parameters
        ----------
        a, b : str
            Cue names.

        Returns
        -------
        float
            Total variation distance between the posteriors for orders (a, b) and (b, a); always 0 for
            the Bayesian resolver.
        """
        return 0.5 * float(np.abs(self.posterior([a, b]) - self.posterior([b, a])).sum())


class QuantumIntentResolver(BayesIntentResolver):
    """Quantum-like intent resolver; it reduces to :class:`BayesIntentResolver` when every angle is 0.

    Parameters
    ----------
    intents : sequence of str
        Names of the K possible intents.
    prior : array_like, optional
        Prior over intents; the initial amplitudes are its square roots.
    """

    def add_cue(self, name, likelihood, theta=0.0):
        """Register a cue as a Kraus operator.

        Parameters
        ----------
        name : str
            Cue name.
        likelihood : array_like
            Non-negative likelihood of the cue under each intent.
        theta : float, optional
            Incompatibility angle in radians (0: the cue commutes with every other cue).

        Returns
        -------
        QuantumIntentResolver
            ``self``.
        """
        L = _prob(likelihood, len(self.intents), normalise=False)
        i, j = np.argsort(L)[::-1][:2]                      # rotate in the plane of the two leading intents
        R = np.eye(len(L))
        c, s = np.cos(theta), np.sin(theta)
        R[[i, i, j, j], [i, j, i, j]] = [c, -s, s, c]
        self.cues[name] = R @ np.diag(np.sqrt(L)) @ R.T
        return self

    def posterior(self, cues):
        """Intent probabilities :math:`|\\psi|^2` after applying the cues in order.

        Parameters
        ----------
        cues : sequence of str
            Names of registered cues, in the order they arrived.

        Returns
        -------
        numpy.ndarray
        """
        psi = np.sqrt(self.prior)
        for c in cues:
            psi = self.cues[c] @ psi
            psi = psi / np.linalg.norm(psi)
        return psi ** 2

    def resolve(self, cues):
        """Most likely intent after the cues.

        Parameters
        ----------
        cues : sequence of str
            Names of registered cues, in order.

        Returns
        -------
        intent : str
            Most likely intent.
        probability : float
            Its posterior probability.
        entropy : float
            Entropy of the posterior in bits (high values suggest asking for clarification).
        """
        p = self.posterior(cues)
        k = int(np.argmax(p))
        H = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
        return self.intents[k], float(p[k]), H


def _prob(v, K, normalise=True):
    """Validate K non-negative weights and optionally normalise them."""
    v = np.asarray(v, float)
    if v.shape != (K,) or np.any(v < 0) or v.sum() <= 0:
        raise ValueError('expected %d non-negative weights' % K)
    return v / v.sum() if normalise else v
