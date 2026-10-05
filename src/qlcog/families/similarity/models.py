"""Similarity judgements and their asymmetry.

Design: list of ordered pairs (A, B): 'how similar is A to B'. Each condition's prediction is a
one-element vector: the similarity on a 0-1 scale. Data: mean ratings rescaled to 0-1 (least squares).

Models
  QuantumSimilarityModel  (Pothos, Busemeyer and Trueblood, 2013) concepts are subspaces of R^3; the
                          neutral state psi = e1; Sim(A, B) = ||P_B P_A psi||^2 (think of A, then of
                          B). A concept with a larger subspace (option ranks: 1 or 2 per concept,
                          standing for how much is known about it) makes the judgement asymmetric.
                          Each concept's direction has two angles.
  BiasedGeometricModel    concepts are points in a plane; Sim(A, B) = b_B exp(-c d(A, B)), with a
                          bias b per concept (Nosofsky, 1991): a classical account of asymmetry.
  GeometricModel          the same with no bias (symmetric)."""
from __future__ import annotations

import numpy as np
from ...core import Model, Param, projector

MAX_CONCEPTS = 6


def _concept_params():
    ps = []
    for i in range(MAX_CONCEPTS):
        ps += [Param('t%d' % i, 'angle', 0.5 + 0.3 * i), Param('p%d' % i, 'angle', 0.2 * i)]
    return ps


class QuantumSimilarityModel(Model):
    """Options: concepts (list of names, at most 6), ranks {name: 1 or 2} (default 1).
    Use for_concepts(QuantumSimilarityModel, names) to get a class with exactly 2 angles per concept."""
    PARAMS = _concept_params()
    LOSS = 'sse'
    name = 'Quantum similarity'

    def projectors(self):
        out = {}
        for i, c in enumerate(self.options['concepts']):
            t, p = getattr(self, 't%d' % i), getattr(self, 'p%d' % i)
            u = np.array([np.cos(t), np.sin(t) * np.cos(p), np.sin(t) * np.sin(p)])
            P = projector(u)
            out[c] = P if self.options.get('ranks', {}).get(c, 1) == 1 else np.eye(3) - P
        return out

    def predict(self, design=None):
        P = self.projectors(); psi = np.array([1.0, 0, 0]); out = {}
        for a, b in design:
            v = P[b] @ (P[a] @ psi); out[(a, b)] = np.array([float(v @ v)])
        return out


def _point_params():
    ps = []
    for i in range(MAX_CONCEPTS):
        ps += [Param('x%d' % i, 'real', np.cos(i)), Param('y%d' % i, 'real', np.sin(i))]
    return ps


class GeometricModel(Model):
    PARAMS = _point_params() + [Param('c', 'positive', 1.0)]
    LOSS = 'sse'
    name = 'Geometric (symmetric)'

    def bias(self, name):
        return 1.0

    def predict(self, design=None):
        idx = {c: i for i, c in enumerate(self.options['concepts'])}; out = {}
        for a, b in design:
            i, j = idx[a], idx[b]
            d = np.hypot(getattr(self, 'x%d' % i) - getattr(self, 'x%d' % j), getattr(self, 'y%d' % i) - getattr(self, 'y%d' % j))
            out[(a, b)] = np.array([self.bias(b) * np.exp(-self.c * d)])
        return out


class BiasedGeometricModel(GeometricModel):
    PARAMS = GeometricModel.PARAMS + [Param('b%d' % i, 'prob', 0.9) for i in range(MAX_CONCEPTS)]
    name = 'Biased geometric (Nosofsky)'

    def bias(self, name):
        return getattr(self, 'b%d' % self.options['concepts'].index(name))


def asymmetry(pred, a, b):
    """Sim(A, B) - Sim(B, A)."""
    return float(np.ravel(pred[(a, b)])[0] - np.ravel(pred[(b, a)])[0])


def for_concepts(cls, concepts):
    """Subclass of a similarity model whose parameters cover exactly the given concepts (so that
    information criteria count only the parameters in use). Pass the same concepts as an option."""
    n = len(concepts)
    if n > MAX_CONCEPTS:
        raise ValueError('at most %d concepts' % MAX_CONCEPTS)
    keep = [p for p in cls.PARAMS if not (p.name[0] in 'tpxyb' and p.name[1:].isdigit() and int(p.name[1:]) >= n)]
    return type(cls.__name__, (cls,), {'PARAMS': keep})
