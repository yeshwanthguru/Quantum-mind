"""Belief over what the speaker means across the turns of a short dialogue: language model evidence, quantum-cognition fusion.

Each turn of a dialogue ("go to the block", then "no, the left one") is read by the small language model into a belief over the
cells. The beliefs of the objects on the table become the evidence of one step of the quantum-cognition core
(:class:`QuantumCognitionCore`), the context of the step is the kind of turn (an instruction or a follow-up), and the core fuses
the turns. The same interface fits the classical competitors (:class:`BayesCore`, :class:`ContextWeightedCore`), so the fusion
rule is chosen by held-out log-loss, not by assumption."""
from __future__ import annotations

import numpy as np

from . import corpus as C
from .core import BayesCore, ContextWeightedCore, QuantumCognitionCore

__all__ = ['DialogueBelief', 'make_dialogues']

SLOTS = 3
FUSIONS = {'bayes': BayesCore, 'context weights': ContextWeightedCore, 'quantum core': QuantumCognitionCore}


def make_dialogues(n, seed, e_after_clear=0.25, e_after_unclear=0.05):
    """Two-turn dialogues about one scene: an instruction, then a follow-up that sometimes misspeaks.

    The follow-up describes a wrong object with probability ``e_after_clear`` when the instruction already pinned the target down and
    ``e_after_unclear`` when it was ambiguous. This is a classical, context-dependent noise model chosen by the author; it is not data.
    """
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        u1 = C.sample_utterance(rng)
        if u1.kind not in ('simple', 'polite', 'ambiguous'):
            continue
        wrong = rng.random() < (e_after_unclear if u1.ambiguous else e_after_clear)
        out.append([u1, C.make_followup(u1, rng, wrong)])
    return out


class DialogueBelief:
    """Fuse the turns of a dialogue into a belief over the objects on the table.

    Parameters
    ----------
    lm : RobotLanguageModel
        Reads each turn into a belief over cells.
    fusion : {'bayes', 'context weights', 'quantum core'}
        The fusion rule. ``'quantum core'`` is :class:`QuantumCognitionCore`; the others are its classical competitors.
    **kw
        Optimisation settings passed to the fusion model (``steps``, ``restarts``, ``seed``, ...).
    """

    def __init__(self, lm, fusion='quantum core', **kw):
        self.lm, self.fusion = lm, fusion
        self.model = FUSIONS[fusion](SLOTS, SLOTS, 2, **kw)

    def features(self, dialogues):
        """Evidence ``X`` (log belief of each object slot, per turn) and context ``Cx`` (instruction or follow-up) for the dialogues."""
        T = len(dialogues[0])
        X = np.full((len(dialogues), T, SLOTS), -8.0, np.float32)
        Cx = np.zeros((len(dialogues), T, 2), np.float32)
        for t in range(T):
            beliefs = self.lm.ground([d[t] for d in dialogues])
            for i, d in enumerate(dialogues):
                p = beliefs[i][d[t].cells]
                p = p / max(p.sum(), 1e-9)
                X[i, t, :len(p)] = np.log(p + 1e-3)
                Cx[i, t, 0 if d[t].kind != 'followup' else 1] = 1.0
        return X, Cx

    @staticmethod
    def labels(dialogues):
        return np.array([d[0].cells.index(d[0].target) for d in dialogues])

    def fit(self, dialogues):
        X, Cx = self.features(dialogues)
        self.model.fit(X, Cx, self.labels(dialogues))
        return self

    def predict_proba(self, dialogues):
        """Probabilities over the object slots (left to right), padded slots get probability near 0."""
        X, Cx = self.features(dialogues)
        p = np.array(self.model.predict_proba(X, Cx))
        for i, d in enumerate(dialogues):
            p[i, len(d[0].cells):] = 0
        return p / p.sum(1, keepdims=True)

    def log_loss(self, dialogues):
        p = self.predict_proba(dialogues)
        y = self.labels(dialogues)
        return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-7)))

    def accuracy(self, dialogues):
        return float(np.mean(self.predict_proba(dialogues).argmax(1) == self.labels(dialogues)))
