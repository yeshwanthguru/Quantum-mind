"""Choosing the conditions (questions, orders, displays) that best tell competing models apart.

A quantum-like model and its classical baselines often predict almost the same answers in some
conditions and very different answers in others. Asking where they agree wastes a person's time.
:func:`information_gain` is the expected reduction in uncertainty about *which model is right* from
one observation in a condition: the mutual information between the model identity and the outcome,

.. math:: I(c) = H\\Big(\\sum_m w_m\\, p_m(c)\\Big) - \\sum_m w_m\\, H\\big(p_m(c)\\big),

the weighted Jensen-Shannon divergence of the models' predictions (in bits). Ranking conditions by
it is the optimal-design approach to model discrimination of Myung and Pitt (2009). After the answer
arrives, :func:`model_posterior` updates the model weights, so a robot can alternate the two and
ask the most informative question next.

The functions accept anything with a ``predict(design)`` method, including fitted models and
:class:`~quantum_mind.core.online.OnlinePersonModel`.

Examples
--------
>>> from quantum_mind.core.design import information_gain
>>> from quantum_mind.families.order_effects import QuantumOrderModel4D, BayesOrderModel
>>> q, b = QuantumOrderModel4D(phi=0.6), BayesOrderModel()
>>> information_gain([q, b], 'AB') >= 0
True
"""
from __future__ import annotations

import numpy as np

__all__ = ['information_gain', 'rank_conditions', 'model_posterior']


def _entropy_bits(p):
    p = np.clip(np.asarray(p, float), 0, None)
    p = p / p.sum()
    nz = p > 0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def _weights(models, weights):
    w = np.ones(len(models)) if weights is None else np.asarray(weights, float)
    if len(w) != len(models) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('weights must be non-negative, one per model, with a positive sum')
    return w / w.sum()


def information_gain(models, condition, design=None, weights=None):
    """Expected information about the model identity from one observation in a condition.

    Parameters
    ----------
    models : list
        Objects with ``predict(design)`` returning probability vectors.
    condition : hashable
        Condition to evaluate.
    design : iterable, optional
        Design passed to ``predict``.
    weights : array_like, optional
        Current probabilities of the models (default equal).

    Returns
    -------
    float
        Mutual information in bits, between 0 and :math:`\\log_2` of the number of models.
    """
    w = _weights(models, weights)
    preds = [np.asarray(m.predict(design)[condition], float) for m in models]
    mix = sum(wi * p for wi, p in zip(w, preds))
    return max(0.0, _entropy_bits(mix) - float(sum(wi * _entropy_bits(p) for wi, p in zip(w, preds))))


def rank_conditions(models, conditions, design=None, weights=None):
    """Rank candidate conditions by :func:`information_gain`.

    Parameters
    ----------
    models : list
        Competing models.
    conditions : iterable
        Candidate conditions (they must all appear in ``predict(design)``).
    design : iterable, optional
        Design passed to ``predict``.
    weights : array_like, optional
        Current model probabilities.

    Returns
    -------
    list of tuple
        ``[(condition, bits)]`` from most to least informative.
    """
    gains = [(c, information_gain(models, c, design, weights)) for c in conditions]
    return sorted(gains, key=lambda t: -t[1])


def model_posterior(models, data, design=None, prior=None):
    """Posterior probabilities of the models after observing outcome counts.

    The models' parameters are held fixed (for example at population estimates, or the posterior
    predictive of an :class:`~quantum_mind.core.online.OnlinePersonModel`).

    Parameters
    ----------
    models : list
        Competing models.
    data : dict
        ``{condition: counts}``.
    design : iterable, optional
        Design passed to ``predict``.
    prior : array_like, optional
        Prior model probabilities (default equal).

    Returns
    -------
    numpy.ndarray
        Posterior probability of each model.
    """
    w = _weights(models, prior)
    ll = np.zeros(len(models))
    for i, m in enumerate(models):
        pred = m.predict(design)
        for c, counts in data.items():
            ll[i] += float(np.dot(np.asarray(counts, float), np.log(np.clip(np.asarray(pred[c], float), 1e-12, 1))))
    lp = np.log(w) + ll
    lp -= lp.max()
    post = np.exp(lp)
    return post / post.sum()
