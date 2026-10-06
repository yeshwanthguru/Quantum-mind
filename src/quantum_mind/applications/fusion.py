r"""Multimodal fusion for robots: from detector scores, speech and gaze to one belief over intents.

A request such as "bring me that cup" is resolved from several sources: an object detector's scores,
the speech recogniser's n-best list, where the person looks and where they point. This module turns
each source into a *likelihood over the same hypotheses* and offers three ways to combine them:

* :func:`bayes_fusion`: independent cues multiply (order-free);
* :func:`dempster_shafer_fusion`: Dempster's rule with discounted sources, which keeps an explicit
  "don't know" mass when sources conflict;
* :class:`~quantum_mind.applications.intent.QuantumIntentResolver`: quantum-like fusion in which incompatible
  cues are applied as rotated Kraus operators, so cue order matters (it reduces to Bayes when every
  incompatibility angle is zero); :func:`fit_incompatibility` estimates the angles from logged
  choices, and :func:`compare_fusion` compares the three rules on held-out data.

Adapters: :func:`detector_likelihood`, :func:`asr_likelihood`, :func:`direction_likelihood` (gaze or
pointing).

Examples
--------
>>> import numpy as np
>>> from quantum_mind.applications.fusion import detector_likelihood, direction_likelihood, bayes_fusion
>>> objects = ['red cup', 'blue cup', 'bowl']
>>> vision = detector_likelihood({'red cup': 0.6, 'blue cup': 0.5, 'bowl': 0.2}, objects)
>>> gaze = direction_likelihood([1, 0, 0], {'red cup': [0.9, 0.1, 0], 'blue cup': [0, 1, 0],
...                                         'bowl': [-1, 0, 0]}, objects, kappa=4)
>>> post = bayes_fusion([vision, gaze])
>>> objects[int(np.argmax(post))]
'red cup'
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

__all__ = ['detector_likelihood', 'asr_likelihood', 'direction_likelihood', 'bayes_fusion',
           'dempster_shafer_fusion', 'fit_incompatibility', 'compare_fusion']


def _norm(v, floor):
    """Floor and normalise a non-negative vector."""
    v = np.maximum(np.asarray(v, float), 0) + floor
    return v / v.sum()


def detector_likelihood(scores, hypotheses, temperature=1.0, floor=1e-3):
    """Likelihood over hypotheses from an object detector's scores.

    Parameters
    ----------
    scores : dict or array_like
        ``{hypothesis: score}`` (missing hypotheses score 0) or scores in the order of ``hypotheses``.
        Scores are confidences in [0, 1] or logits.
    hypotheses : sequence
        Hypotheses (object or intent names).
    temperature : float, optional
        Softens (> 1) or sharpens (< 1) the scores before normalising.
    floor : float, optional
        Added to every entry so that no hypothesis is ruled out by one cue.

    Returns
    -------
    numpy.ndarray
        Normalised likelihood, one entry per hypothesis.
    """
    s = np.array([scores.get(h, 0.0) for h in hypotheses], float) if isinstance(scores, dict) else np.asarray(scores, float)
    if np.all((s >= 0) & (s <= 1)):              # confidences: power-sharpen or soften
        return _norm(s ** (1.0 / temperature), floor)
    z = s / temperature                           # logits: softmax
    return _norm(np.exp(z - z.max()), floor)


def asr_likelihood(nbest, keywords, hypotheses, floor=0.02):
    """Likelihood over hypotheses from a speech recogniser's n-best list.

    Parameters
    ----------
    nbest : sequence of tuple
        ``[(transcript, confidence), ...]``.
    keywords : dict
        ``{hypothesis: [words that refer to it]}``.
    hypotheses : sequence
    floor : float, optional

    Returns
    -------
    numpy.ndarray
        Each transcript's confidence is shared among the hypotheses it mentions (or spread over all
        hypotheses when it mentions none).
    """
    L = np.zeros(len(hypotheses))
    for text, conf in nbest:
        words = set(text.lower().replace(',', ' ').split())
        hit = np.array([any(k.lower() in words or k.lower() in text.lower() for k in keywords.get(h, [])) for h in hypotheses])
        L += conf * (hit / hit.sum() if hit.any() else np.full(len(hypotheses), 1 / len(hypotheses)))
    return _norm(L, floor)


def direction_likelihood(direction, targets, hypotheses, kappa=5.0, floor=1e-3):
    """Likelihood over hypotheses from a gaze or pointing direction (von Mises-Fisher).

    Parameters
    ----------
    direction : array_like
        Direction vector (2D or 3D) from the person's head or hand.
    targets : dict
        ``{hypothesis: direction to that object}`` from the same origin.
    hypotheses : sequence
    kappa : float, optional
        Concentration: larger values trust the direction more.
    floor : float, optional

    Returns
    -------
    numpy.ndarray
        :math:`\\propto \\exp(\\kappa \\cos \\angle(d, t_h))`.
    """
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    cos = []
    for h in hypotheses:
        t = np.asarray(targets[h], float)
        cos.append(float(d @ t / np.linalg.norm(t)))
    return _norm(np.exp(kappa * (np.array(cos) - 1)), floor)


def bayes_fusion(likelihoods, prior=None):
    """Posterior of independent cues: prior times the product of the likelihoods.

    Parameters
    ----------
    likelihoods : sequence of array_like
        One likelihood vector per cue.
    prior : array_like, optional
        Prior over hypotheses (uniform by default).

    Returns
    -------
    numpy.ndarray
    """
    L = np.asarray(likelihoods, float)
    post = (np.full(L.shape[1], 1 / L.shape[1]) if prior is None else np.asarray(prior, float)) * np.prod(L, axis=0)
    return post / post.sum()


def dempster_shafer_fusion(likelihoods, reliabilities=None):
    """Dempster's rule for singleton masses with discounting.

    Each cue becomes a mass function on the single hypotheses, discounted by its reliability r: a
    fraction 1 - r of its mass goes to "any hypothesis" (ignorance). Conflict between cues is
    renormalised away by Dempster's rule.

    Parameters
    ----------
    likelihoods : sequence of array_like
        One normalised likelihood per cue.
    reliabilities : sequence of float, optional
        Reliability of each cue in [0, 1] (default 0.9).

    Returns
    -------
    belief : numpy.ndarray
        Pignistic probability over hypotheses (ignorance shared equally).
    ignorance : float
        Remaining mass on "any hypothesis".
    """
    L = np.asarray(likelihoods, float)
    K = L.shape[1]
    r = np.full(len(L), 0.9) if reliabilities is None else np.asarray(reliabilities, float)
    m = np.zeros(K)
    omega = 1.0                                    # mass on the whole frame (start: total ignorance)
    for li, ri in zip(L, r):
        mi = ri * li / li.sum()
        wi = 1 - ri
        # combine (m, omega) with (mi, wi) on singletons plus the frame
        new = m * mi + m * wi + omega * mi
        new_omega = omega * wi
        z = new.sum() + new_omega                  # 1 - conflict
        m, omega = new / z, new_omega / z
    return m + omega / K, float(omega)


def fit_incompatibility(resolver_cls, intents, cue_likelihoods, trials, prior=None, bounds=(0.0, np.pi / 2)):
    """Fit the incompatibility angle of each cue of a quantum-like resolver from logged choices.

    Parameters
    ----------
    resolver_cls : type
        :class:`~quantum_mind.applications.intent.QuantumIntentResolver`.
    intents : sequence of str
        Intent names.
    cue_likelihoods : dict
        ``{cue name: likelihood over intents}``.
    trials : sequence of tuple
        ``[(cue order, chosen intent), ...]``: the order in which cues arrived and what the person
        actually wanted (or chose).
    prior : array_like, optional
        Prior over intents.
    bounds : tuple, optional
        Range of the angles.

    Returns
    -------
    resolver : QuantumIntentResolver
        Resolver with the fitted angles.
    angles : dict
        ``{cue name: fitted angle}``.
    """
    names = list(cue_likelihoods)
    idx = {h: i for i, h in enumerate(intents)}

    def build(theta):
        r = resolver_cls(intents, prior)
        for n, t in zip(names, theta):
            r.add_cue(n, cue_likelihoods[n], t)
        return r

    def nll(theta):
        r = build(theta)
        return -sum(np.log(max(r.posterior(order)[idx[c]], 1e-12)) for order, c in trials)

    best = min((minimize(nll, np.full(len(names), x0), method='L-BFGS-B', bounds=[bounds] * len(names))
                for x0 in (0.05, 0.4, 0.9)), key=lambda r: r.fun)
    return build(best.x), dict(zip(names, (float(v) for v in best.x)))


def compare_fusion(intents, cue_likelihoods, train, test, prior=None):
    """Compare Bayesian, Dempster-Shafer and fitted quantum-like fusion on held-out trials.

    Parameters
    ----------
    intents : sequence of str
    cue_likelihoods : dict
        ``{cue name: likelihood over intents}``.
    train, test : sequence of tuple
        ``[(cue order, chosen intent), ...]``; the quantum-like angles are fitted on ``train``.
    prior : array_like, optional

    Returns
    -------
    dict
        ``{rule: {'log_loss': ..., 'accuracy': ...}}`` on ``test`` (lower log loss is better), plus
        ``'angles'``.
    """
    from .intent import QuantumIntentResolver
    idx = {h: i for i, h in enumerate(intents)}
    q, angles = fit_incompatibility(QuantumIntentResolver, intents, cue_likelihoods, train, prior)
    rules = {
        'bayes': lambda order: bayes_fusion([cue_likelihoods[c] for c in order], prior),
        'dempster_shafer': lambda order: dempster_shafer_fusion([cue_likelihoods[c] for c in order])[0],
        'quantum_like': q.posterior,
    }
    out = {}
    for name, rule in rules.items():
        P = [rule(order) for order, _ in test]
        y = [idx[c] for _, c in test]
        out[name] = {'log_loss': float(-np.mean([np.log(max(p[i], 1e-12)) for p, i in zip(P, y)])),
                     'accuracy': float(np.mean([int(np.argmax(p)) == i for p, i in zip(P, y)]))}
    out['angles'] = angles
    return out
