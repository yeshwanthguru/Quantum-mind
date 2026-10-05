"""Probability judgements of single events, conjunctions and disjunctions (conjunction and
disjunction fallacies).

Design: a list of judgement types among 'A', 'B', 'A&B', 'A|B'. Each condition's prediction is a
one-element vector: the judged probability. Data: observed mean judgements (fitted by least squares)
or, for 'fallacy rate' data, see `fallacy_rate`.

Models
  QuantumConjunctionModel  three-dimensional real state; events A and B are projectors (rank 1 or 2).
                           The conjunction is judged sequentially, the more likely event first:
                           P(A&B) = ||P_B P_A psi||^2 if P(A) >= P(B) (Busemeyer et al., 2011);
                           the disjunction is 1 - P(not-A then not-B) in the same order. Incompatible
                           events can make P(A&B) > P(B) (conjunction fallacy).
  ClassicalJointModel      a joint distribution; P(A&B) <= min(P(A), P(B)) always.
  AveragingModel           P(A&B) = w P(A) + (1-w) P(B) with the less likely event weighted by w;
                           P(A|B) = w P(A) + (1-w) P(B) with the more likely event weighted by w.
  PTNModel                 probability theory plus noise (Costello and Watts, 2014): every estimate is
                           (1 - 2d) p + d with noise d, larger (d + dd) for conjunctions and disjunctions."""
from __future__ import annotations

import itertools
import numpy as np
from ...core import Model, Param, projector, luders

TYPES = ('A', 'B', 'A&B', 'A|B')
RANK_STRUCTURES = [{'ranks': r} for r in itertools.product((1, 2), repeat=2)]


class QuantumConjunctionModel(Model):
    """Quantum-like conjunction and disjunction judgements: two events as projectors in a real 3D space,
    with the conjunction judged by asking the more likely event first (Lueders rule); can produce
    conjunction and disjunction fallacies."""
    PARAMS = [Param('a', 'angle', 0.8), Param('b', 'angle', 1.2), Param('g', 'angle', 0.3)]
    LOSS = 'sse'
    name = 'Quantum-like'

    def projectors(self):
        """Projectors {'A': P_A, 'B': P_B} of the two events."""
        a, b, g = self.a, self.b, self.g
        uA = np.array([np.cos(a), np.sin(a), 0.0]); uB = np.array([np.cos(b), np.sin(b) * np.cos(g), np.sin(b) * np.sin(g)])
        r = dict(zip('AB', self.options.get('ranks', (1, 1))))
        return {q: (projector(u) if r[q] == 1 else np.eye(3) - projector(u)) for q, u in (('A', uA), ('B', uB))}

    def judgements(self):
        """Predicted judgements {A, B, A&B, A|B}."""
        psi = np.array([1.0, 0, 0]); P = self.projectors()
        pA, pB = luders(psi, P['A'])[0], luders(psi, P['B'])[0]
        first, second = ('A', 'B') if pA >= pB else ('B', 'A')
        p1, v = luders(psi, P[first]); p2, _ = luders(v, P[second])
        n1, w = luders(psi, np.eye(3) - P[first]); n2, _ = luders(w, np.eye(3) - P[second])
        return {'A': pA, 'B': pB, 'A&B': p1 * p2, 'A|B': 1 - n1 * n2}

    def predict(self, design=None):
        j = self.judgements()
        return {t: np.array([j[t]]) for t in (design or TYPES)}


class ClassicalJointModel(Model):
    """Classical baseline: one joint distribution, so P(A and B) <= min(P(A), P(B)) (no fallacies)."""
    PARAMS = [Param('pA', 'prob', 0.5), Param('pB', 'prob', 0.5), Param('rho', 'bounded', 0.0, -1, 1)]
    LOSS = 'sse'
    name = 'Classical joint'

    def predict(self, design=None):
        pA, pB = self.pA, self.pB
        lo, hi = max(0.0, pA + pB - 1), min(pA, pB)
        ab = pA * pB + self.rho * ((hi - pA * pB) if self.rho > 0 else (pA * pB - lo))
        j = {'A': pA, 'B': pB, 'A&B': ab, 'A|B': pA + pB - ab}
        return {t: np.array([j[t]]) for t in (design or TYPES)}


class AveragingModel(Model):
    """Baseline: the conjunction is judged as a weighted average of the two event judgements."""
    PARAMS = [Param('pA', 'prob', 0.5), Param('pB', 'prob', 0.5), Param('w', 'prob', 0.5)]
    LOSS = 'sse'
    name = 'Averaging'

    def predict(self, design=None):
        lo, hi = sorted([self.pA, self.pB])
        j = {'A': self.pA, 'B': self.pB, 'A&B': self.w * lo + (1 - self.w) * hi, 'A|B': self.w * hi + (1 - self.w) * lo}
        return {t: np.array([j[t]]) for t in (design or TYPES)}


class PTNModel(Model):
    """Probability theory plus noise baseline (Costello and Watts): classical probabilities read with random
    noise, which regresses judgements towards 0.5."""
    PARAMS = [Param('pA', 'prob', 0.5), Param('pB', 'prob', 0.5), Param('rho', 'bounded', 0.0, -1, 1),
              Param('d', 'bounded', 0.1, 0, 0.5), Param('dd', 'bounded', 0.05, 0, 0.5)]
    LOSS = 'sse'
    name = 'Probability theory plus noise'

    def predict(self, design=None):
        base = ClassicalJointModel(pA=self.pA, pB=self.pB, rho=self.rho).predict(TYPES)
        d2 = min(self.d + self.dd, 0.5)
        out = {}
        for t in (design or TYPES):
            d = self.d if t in ('A', 'B') else d2
            out[t] = (1 - 2 * d) * base[t] + d
        return out


def fallacy_rate(judgements):
    """Whether a set of judgements commits the conjunction fallacy (P(A&B) > min(P(A), P(B))) and the
    disjunction fallacy (P(A|B) < max(P(A), P(B)))."""
    j = {k: float(np.ravel(v)[0]) for k, v in judgements.items()}
    return {'conjunction_fallacy': j['A&B'] > min(j['A'], j['B']), 'disjunction_fallacy': j['A|B'] < max(j['A'], j['B'])}
