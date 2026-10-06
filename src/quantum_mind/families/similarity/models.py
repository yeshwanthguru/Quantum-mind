r"""Similarity judgements and their asymmetry.

**Design.** A list of ordered pairs ``(A, B)``: "how similar is A to B". Each condition's prediction
is a one-element vector, the similarity on a 0-1 scale. Data are mean ratings rescaled to 0-1
(least squares).

**Models.**

* :class:`QuantumSimilarityModel` (Pothos, Busemeyer and Trueblood, 2013): concepts are subspaces of
  :math:`\mathbb{R}^3`; the neutral state is :math:`e_1` and
  :math:`\mathrm{Sim}(A, B) = \lVert P_B P_A \psi \rVert^2` (think of A, then of B). A concept with a
  larger subspace (more knowledge about it) makes the judgement asymmetric.
* :class:`BiasedGeometricModel`: concepts are points in a plane and
  :math:`\mathrm{Sim}(A, B) = b_B\,e^{-c\,d(A, B)}`, with a bias per concept (Nosofsky, 1991), a
  classical account of asymmetry.
* :class:`GeometricModel`: the same without bias (symmetric).
"""
from __future__ import annotations

import numpy as np
from ...core import Model, Param, projector

#: Largest number of concepts a similarity model can hold.
MAX_CONCEPTS = 6


def _concept_params():
    """Two angles (t_i, p_i) per concept for the quantum model."""
    ps = []
    for i in range(MAX_CONCEPTS):
        ps += [Param('t%d' % i, 'angle', 0.5 + 0.3 * i), Param('p%d' % i, 'angle', 0.2 * i)]
    return ps


class QuantumSimilarityModel(Model):
    """Quantum similarity model (Pothos, Busemeyer and Trueblood, 2013).

    Use ``for_concepts(QuantumSimilarityModel, names)`` to get a class with exactly two angles per
    concept, so that information criteria count only the parameters in use.

    Parameters
    ----------
    t0, p0, ..., t5, p5 : float
        Polar and azimuthal angle of each concept's direction.
    concepts : list of str
        Concept names, at most 6 (option).
    ranks : dict, optional
        ``{name: 1 or 2}``: dimension of each concept's subspace (default 1).
    """
    PARAMS = _concept_params()
    LOSS = 'sse'
    name = 'Quantum similarity'

    def projectors(self):
        """Projector of each concept subspace.

        Returns
        -------
        dict
            ``{concept: 3 x 3 projector}``.
        """
        out = {}
        for i, c in enumerate(self.options['concepts']):
            t, p = getattr(self, 't%d' % i), getattr(self, 'p%d' % i)
            u = np.array([np.cos(t), np.sin(t) * np.cos(p), np.sin(t) * np.sin(p)])
            P = projector(u)
            out[c] = P if self.options.get('ranks', {}).get(c, 1) == 1 else np.eye(3) - P
        return out

    def predict(self, design=None):
        """Similarity of each ordered pair ``(A, B)``."""
        P = self.projectors()
        psi = np.array([1.0, 0, 0])
        out = {}
        for a, b in design:
            v = P[b] @ (P[a] @ psi)              # think of A, then of B
            out[(a, b)] = np.array([float(v @ v)])
        return out


def _point_params():
    """Two coordinates (x_i, y_i) per concept for the geometric models."""
    ps = []
    for i in range(MAX_CONCEPTS):
        ps += [Param('x%d' % i, 'real', np.cos(i)), Param('y%d' % i, 'real', np.sin(i))]
    return ps


class GeometricModel(Model):
    """Classical baseline: similarity decreases with distance in a psychological space (symmetric).

    Parameters
    ----------
    x0, y0, ..., x5, y5 : float
        Coordinates of each concept.
    c : float
        Generalisation gradient.
    concepts : list of str
        Concept names (option).
    """
    PARAMS = _point_params() + [Param('c', 'positive', 1.0)]
    LOSS = 'sse'
    name = 'Geometric (symmetric)'

    def bias(self, name):
        """Response bias toward a concept (always 1 in the symmetric model).

        Parameters
        ----------
        name : str
            Concept name.

        Returns
        -------
        float
        """
        return 1.0

    def predict(self, design=None):
        """Similarity of each ordered pair ``(A, B)``."""
        idx = {c: i for i, c in enumerate(self.options['concepts'])}
        out = {}
        for a, b in design:
            i, j = idx[a], idx[b]
            d = np.hypot(getattr(self, 'x%d' % i) - getattr(self, 'x%d' % j), getattr(self, 'y%d' % i) - getattr(self, 'y%d' % j))
            out[(a, b)] = np.array([self.bias(b) * np.exp(-self.c * d)])
        return out


class BiasedGeometricModel(GeometricModel):
    """Classical baseline with asymmetry (Nosofsky, 1991): geometric similarity plus a bias per concept.

    Parameters
    ----------
    x0, y0, ..., x5, y5, c : float
        As in :class:`GeometricModel`.
    b0, ..., b5 : float
        Bias toward each concept, in (0, 1).
    """
    PARAMS = GeometricModel.PARAMS + [Param('b%d' % i, 'prob', 0.9) for i in range(MAX_CONCEPTS)]
    name = 'Biased geometric (Nosofsky)'

    def bias(self, name):
        """Bias parameter of a concept.

        Parameters
        ----------
        name : str
            Concept name.

        Returns
        -------
        float
        """
        return getattr(self, 'b%d' % self.options['concepts'].index(name))


def asymmetry(pred, a, b):
    """Asymmetry of a similarity judgement.

    Parameters
    ----------
    pred : dict
        Predictions or data ``{(A, B): value}``.
    a, b : str
        Concepts.

    Returns
    -------
    float
        :math:`\\mathrm{Sim}(A, B) - \\mathrm{Sim}(B, A)`.
    """
    return float(np.ravel(pred[(a, b)])[0] - np.ravel(pred[(b, a)])[0])


def for_concepts(cls, concepts):
    """Similarity-model subclass with parameters for exactly the given concepts.

    Parameters
    ----------
    cls : type
        :class:`QuantumSimilarityModel`, :class:`GeometricModel` or :class:`BiasedGeometricModel`.
    concepts : list of str
        Concepts; pass the same list as the ``concepts`` option.

    Returns
    -------
    type
        Subclass with the unused parameters removed.

    Raises
    ------
    ValueError
        If more than :data:`MAX_CONCEPTS` concepts are given.
    """
    n = len(concepts)
    if n > MAX_CONCEPTS:
        raise ValueError('at most %d concepts' % MAX_CONCEPTS)
    keep = [p for p in cls.PARAMS if not (p.name[0] in 'tpxyb' and p.name[1:].isdigit() and int(p.name[1:]) >= n)]
    return type(cls.__name__, (cls,), {'PARAMS': keep})
