r"""Probability judgements of single events, conjunctions and disjunctions (conjunction and
disjunction fallacies).

**Design.** A list of judgement types among ``'A'``, ``'B'``, ``'A&B'`` and ``'A|B'``. Each
condition's prediction is a one-element vector, the judged probability. Data are observed mean
judgements, fitted by least squares.

**Models.**

* :class:`QuantumConjunctionModel`: three-dimensional real state; events A and B are projectors
  (rank 1 or 2). The conjunction is judged sequentially, the more likely event first:
  :math:`P(A \wedge B) = \lVert P_B P_A \psi \rVert^2` if :math:`P(A) \ge P(B)` (Busemeyer et al., 2011).
  The disjunction is 1 - P(not-A then not-B) in the same order. Incompatible events can make
  :math:`P(A \wedge B) > P(B)`, the conjunction fallacy.
* :class:`ClassicalJointModel`: a joint distribution, so :math:`P(A \wedge B) \le \min(P(A), P(B))`.
* :class:`AveragingModel`: the conjunction is a weighted average of the two judgements.
* :class:`PTNModel`: probability theory plus noise (Costello and Watts, 2014).
"""
from __future__ import annotations

import itertools
import numpy as np
from ...core import Model, Param, projector, luders

#: Judgement types of the default design.
TYPES = ('A', 'B', 'A&B', 'A|B')
#: Every combination of event-subspace ranks (pass to ``fit(structures=...)``).
RANK_STRUCTURES = [{'ranks': r} for r in itertools.product((1, 2), repeat=2)]


class QuantumConjunctionModel(Model):
    """Quantum-like conjunction and disjunction judgements.

    Two events are projectors in a real 3D space; the conjunction is judged by asking the more likely
    event first (Lüders rule), which can produce conjunction and disjunction fallacies.

    Parameters
    ----------
    a, b, g : float
        Angles of the event vectors :math:`u_A = (\\cos a, \\sin a, 0)` and
        :math:`u_B = (\\cos b, \\sin b \\cos g, \\sin b \\sin g)`; the state is :math:`e_1`.
    ranks : tuple of int, optional
        Rank (1 or 2) of each event's subspace; default ``(1, 1)``.
    """
    PARAMS = [Param('a', 'angle', 0.8), Param('b', 'angle', 1.2), Param('g', 'angle', 0.3)]
    LOSS = 'sse'
    name = 'Quantum-like'

    def projectors(self):
        """Projectors of the two events.

        Returns
        -------
        dict
            ``{'A': P_A, 'B': P_B}``.
        """
        a, b, g = self.a, self.b, self.g
        uA = np.array([np.cos(a), np.sin(a), 0.0])
        uB = np.array([np.cos(b), np.sin(b) * np.cos(g), np.sin(b) * np.sin(g)])
        r = dict(zip('AB', self.options.get('ranks', (1, 1))))
        return {q: (projector(u) if r[q] == 1 else np.eye(3) - projector(u)) for q, u in (('A', uA), ('B', uB))}

    def judgements(self):
        """Predicted probability judgements.

        Returns
        -------
        dict
            ``{'A': ..., 'B': ..., 'A&B': ..., 'A|B': ...}``.
        """
        psi = np.array([1.0, 0, 0])
        P = self.projectors()
        pA, pB = luders(psi, P['A'])[0], luders(psi, P['B'])[0]
        # the more likely event is evaluated first
        first, second = ('A', 'B') if pA >= pB else ('B', 'A')
        p1, v = luders(psi, P[first])
        p2, _ = luders(v, P[second])
        # disjunction: 1 - P(not first, then not second)
        n1, w = luders(psi, np.eye(3) - P[first])
        n2, _ = luders(w, np.eye(3) - P[second])
        return {'A': pA, 'B': pB, 'A&B': p1 * p2, 'A|B': 1 - n1 * n2}

    def predict(self, design=None):
        """Judged probability (one-element vector) for each judgement type in the design."""
        j = self.judgements()
        return {t: np.array([j[t]]) for t in (design or TYPES)}


class ClassicalJointModel(Model):
    """Classical baseline: one joint distribution, so no fallacies are possible.

    Parameters
    ----------
    pA, pB : float
        Event probabilities.
    rho : float
        Correlation in (-1, 1), scaled between the Fréchet bounds.
    """
    PARAMS = [Param('pA', 'prob', 0.5), Param('pB', 'prob', 0.5), Param('rho', 'bounded', 0.0, -1, 1)]
    LOSS = 'sse'
    name = 'Classical joint'

    def predict(self, design=None):
        """Judged probability for each judgement type."""
        pA, pB = self.pA, self.pB
        lo, hi = max(0.0, pA + pB - 1), min(pA, pB)
        ab = pA * pB + self.rho * ((hi - pA * pB) if self.rho > 0 else (pA * pB - lo))
        j = {'A': pA, 'B': pB, 'A&B': ab, 'A|B': pA + pB - ab}
        return {t: np.array([j[t]]) for t in (design or TYPES)}


class AveragingModel(Model):
    """Baseline: conjunctions and disjunctions are weighted averages of the event judgements.

    :math:`P(A \\wedge B) = w\\,\\min + (1 - w)\\max` and :math:`P(A \\vee B) = w\\,\\max + (1 - w)\\min`.

    Parameters
    ----------
    pA, pB : float
        Event probabilities.
    w : float
        Averaging weight in (0, 1).
    """
    PARAMS = [Param('pA', 'prob', 0.5), Param('pB', 'prob', 0.5), Param('w', 'prob', 0.5)]
    LOSS = 'sse'
    name = 'Averaging'

    def predict(self, design=None):
        """Judged probability for each judgement type."""
        lo, hi = sorted([self.pA, self.pB])
        j = {'A': self.pA, 'B': self.pB, 'A&B': self.w * lo + (1 - self.w) * hi, 'A|B': self.w * hi + (1 - self.w) * lo}
        return {t: np.array([j[t]]) for t in (design or TYPES)}


class PTNModel(Model):
    """Probability theory plus noise (Costello and Watts, 2014).

    Classical probabilities are read with random noise, which regresses judgements toward 0.5:
    every estimate is :math:`(1 - 2d)\\,p + d`, with a larger noise ``d + dd`` for conjunctions and
    disjunctions.

    Parameters
    ----------
    pA, pB, rho : float
        Joint distribution, as in :class:`ClassicalJointModel`.
    d : float
        Noise for single events, in (0, 0.5).
    dd : float
        Extra noise for conjunctions and disjunctions.
    """
    PARAMS = [Param('pA', 'prob', 0.5), Param('pB', 'prob', 0.5), Param('rho', 'bounded', 0.0, -1, 1),
              Param('d', 'bounded', 0.1, 0, 0.5), Param('dd', 'bounded', 0.05, 0, 0.5)]
    LOSS = 'sse'
    name = 'Probability theory plus noise'

    def predict(self, design=None):
        """Judged probability for each judgement type."""
        base = ClassicalJointModel(pA=self.pA, pB=self.pB, rho=self.rho).predict(TYPES)
        d2 = min(self.d + self.dd, 0.5)
        out = {}
        for t in (design or TYPES):
            d = self.d if t in ('A', 'B') else d2
            out[t] = (1 - 2 * d) * base[t] + d
        return out


def fallacy_rate(judgements):
    """Check a set of judgements for the conjunction and disjunction fallacies.

    Parameters
    ----------
    judgements : dict
        ``{'A', 'B', 'A&B', 'A|B'}`` to a number or one-element array.

    Returns
    -------
    dict
        ``conjunction_fallacy``: :math:`P(A \\wedge B) > \\min(P(A), P(B))`;
        ``disjunction_fallacy``: :math:`P(A \\vee B) < \\max(P(A), P(B))`.
    """
    j = {k: float(np.ravel(v)[0]) for k, v in judgements.items()}
    return {'conjunction_fallacy': j['A&B'] > min(j['A'], j['B']), 'disjunction_fallacy': j['A|B'] < max(j['A'], j['B'])}
