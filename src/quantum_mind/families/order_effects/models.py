r"""Question-order effects: two yes/no questions A and B asked in either order.

**Design.** Conditions ``'AB'`` (A asked first) and ``'BA'`` (B asked first). The outcomes of each
condition are the four answer pairs ``[yy, yn, ny, nn]``, where the first letter is the answer to the
question asked first.

**Models.**

* :class:`QuantumOrderModel`: three-dimensional real quantum-like model (Lüders rule), with the rank
  (1 or 2) of each question's "yes" subspace as a structural option; three parameters; satisfies the
  QQ equality.
* :class:`QuantumOrderModel4D`: four-dimensional model with an incompatibility angle; ``phi = 0`` is
  exactly the Bayesian model, so it nests it (four parameters). Recommended default.
* :class:`BayesOrderModel`: one order-free joint distribution (no order effect).
* :class:`AnchoringOrderModel`: the second answer is pulled toward (``w > 0``) or away from
  (``w < 0``) the first, with a weight per question; violates the QQ equality when the weights differ.
* :class:`SaturatedOrderModel`: a free distribution per order (6 parameters; reference only).
* :class:`ProjectiveQuestionModel`: any state and any projectors in any dimension (not fitted).

**QQ equality.** Every projective model satisfies

.. math:: p_{AB}(yn) + p_{AB}(ny) = p_{BA}(yn) + p_{BA}(ny),

a parameter-free prediction tested by :func:`qq_test` (Wang and Busemeyer, 2013).
"""
from __future__ import annotations

import itertools
import numpy as np
from ...core import Model, Param, projector, sequence_probabilities, normalize

#: The two question orders of the design.
ORDERS = ('AB', 'BA')
#: Every combination of "yes"-subspace ranks for :class:`QuantumOrderModel` (pass to ``fit(structures=...)``).
RANK_STRUCTURES = [{'ranks': r} for r in itertools.product((1, 2), repeat=2)]
DESIGN = ORDERS


def _cells(probs):
    """Answer-sequence probabilities as the vector [yy, yn, ny, nn]."""
    return np.array([probs[(1, 1)], probs[(1, 0)], probs[(0, 1)], probs[(0, 0)]])


class ProjectiveQuestionModel:
    """General projective model of two questions.

    Parameters
    ----------
    psi : array_like
        Belief state (real or complex); normalised on construction.
    projectors : dict
        ``{'A': P_A, 'B': P_B}``: projectors of the "yes" answers.
    """
    def __init__(self, psi, projectors):
        self.psi = normalize(psi)
        self.projectors = projectors

    def answer_probs(self, order):
        """Probabilities of all answer sequences.

        Parameters
        ----------
        order : sequence of str
            Questions in the order they are asked, for example ``('A', 'B')``.

        Returns
        -------
        dict
            ``{(a1, a2): probability}`` from the Lüders rule.
        """
        return sequence_probabilities(self.psi, self.projectors, order)

    def predict(self, design=None):
        """Answer distribution for each question order.

        Parameters
        ----------
        design : sequence of str, optional
            Orders, default ``('AB', 'BA')``.

        Returns
        -------
        dict
            ``{order: [yy, yn, ny, nn]}``.
        """
        return {o: _cells(self.answer_probs(tuple(o))) for o in (design or ORDERS)}


class QuantumOrderModel(Model):
    """Three-dimensional real quantum-like model of two questions.

    The belief state is :math:`\\psi = e_1`; the questions are defined by the unit vectors
    :math:`u_A = (\\cos a, \\sin a, 0)` and :math:`u_B = (\\cos b, \\sin b \\cos g, \\sin b \\sin g)`.

    Parameters
    ----------
    a, b, g : float
        Angles (fitted).
    ranks : tuple of int, optional
        ``(rA, rB)``. Rank 1: "yes" is the ray of :math:`u_q`; rank 2: the plane orthogonal to
        :math:`u_q`. Default ``(1, 1)``; fit all four with ``structures=RANK_STRUCTURES``.
    """
    PARAMS = [Param('a', 'angle', 0.8), Param('b', 'angle', 1.2), Param('g', 'angle', 0.3)]
    name = 'Quantum-like'

    def vectors(self):
        """Unit vectors that define the two questions.

        Returns
        -------
        dict
            ``{'A': u_A, 'B': u_B}``.
        """
        a, b, g = self.a, self.b, self.g
        return {'A': np.array([np.cos(a), np.sin(a), 0.0]),
                'B': np.array([np.cos(b), np.sin(b) * np.cos(g), np.sin(b) * np.sin(g)])}

    def projectors(self):
        """Projectors of the "yes" answers.

        Returns
        -------
        dict
            ``{'A': P_A, 'B': P_B}``, 3 x 3 matrices of rank 1 or 2.
        """
        ranks = dict(zip('AB', self.options.get('ranks', (1, 1))))
        out = {}
        for q, u in self.vectors().items():
            P = projector(u)
            out[q] = P if ranks[q] == 1 else np.eye(3) - P
        return out

    def as_projective(self):
        """The same model as a :class:`ProjectiveQuestionModel`.

        Returns
        -------
        ProjectiveQuestionModel
        """
        return ProjectiveQuestionModel(np.array([1.0, 0.0, 0.0]), self.projectors())

    def predict(self, design=None):
        """Answer distribution ``[yy, yn, ny, nn]`` for each order (default ``'AB'``, ``'BA'``)."""
        return self.as_projective().predict(design)


class QuantumOrderModel4D(Model):
    """Four-dimensional real model in which the questions can be compatible or incompatible.

    The basis :math:`e_1, \\dots, e_4` is (yes, yes), (yes, no), (no, yes), (no, no) of a classical
    2 x 2 table, and :math:`P_A = \\mathrm{span}(e_1, e_2)`. :math:`P_B = U\\,\\mathrm{span}(e_1, e_3)\\,U^T`,
    where :math:`U` rotates by the incompatibility angle :math:`\\varphi` in the planes
    :math:`(e_1, e_4)` and :math:`(e_2, e_3)`.

    With :math:`\\varphi = 0` the projectors commute and the model is exactly the order-free Bayesian
    model (:math:`\\psi_k^2` is the joint table), so it nests :class:`BayesOrderModel`;
    :math:`\\varphi \\neq 0` produces order effects, and the QQ equality still holds.

    Parameters
    ----------
    t1, t2, t3 : float
        Hyperspherical angles of the state :math:`\\psi`.
    phi : float
        Incompatibility angle.
    """
    PARAMS = [Param('t1', 'angle', 0.8), Param('t2', 'angle', 0.8), Param('t3', 'angle', 0.8), Param('phi', 'angle', 0.2)]
    name = 'Quantum-like (4D, nests Bayes)'

    def projectors(self):
        """Projectors of the "yes" answers.

        Returns
        -------
        dict
            ``{'A': P_A, 'B': P_B}``, 4 x 4 matrices of rank 2.
        """
        from ...core import angles_to_unit  # noqa: F401
        c, s_ = np.cos(self.phi), np.sin(self.phi)
        # U rotates by phi in the (e1, e4) and (e2, e3) planes
        U = np.eye(4)
        U[0, 0] = U[3, 3] = c
        U[0, 3] = -s_
        U[3, 0] = s_
        U[1, 1] = U[2, 2] = c
        U[1, 2] = -s_
        U[2, 1] = s_
        E = np.eye(4)
        PA = projector([E[0], E[1]])
        PB = U @ projector([E[0], E[2]]) @ U.T
        return {'A': PA, 'B': PB}

    def as_projective(self):
        """The same model as a :class:`ProjectiveQuestionModel`.

        Returns
        -------
        ProjectiveQuestionModel
        """
        from ...core import angles_to_unit
        return ProjectiveQuestionModel(angles_to_unit([self.t1, self.t2, self.t3]), self.projectors())

    def predict(self, design=None):
        """Answer distribution ``[yy, yn, ny, nn]`` for each order (default ``'AB'``, ``'BA'``)."""
        return self.as_projective().predict(design)


class BayesOrderModel(Model):
    """Classical baseline: one order-free joint distribution.

    Parameters
    ----------
    pA, pB : float
        Marginal "yes" probabilities.
    rho : float
        Correlation in (-1, 1), scaled between the bounds that the marginals allow.
    """
    PARAMS = [Param('pA', 'prob', 0.5), Param('pB', 'prob', 0.5), Param('rho', 'bounded', 0.0, -1, 1)]
    name = 'Bayesian (order-free)'

    def joint(self):
        """Joint distribution with A first.

        Returns
        -------
        numpy.ndarray
            ``[yy, yn, ny, nn]``.
        """
        pA, pB = self.pA, self.pB
        # Fréchet bounds on P(A and B), relative to independence
        lo, hi = max(0.0, pA + pB - 1) - pA * pB, min(pA, pB) - pA * pB
        yy = pA * pB + self.rho * (hi if self.rho > 0 else -lo)
        return np.array([yy, pA - yy, pB - yy, 1 - pA - pB + yy])        # A first: yy, yn, ny, nn

    def predict(self, design=None):
        """Answer distribution for each order; the BA order swaps the ``yn`` and ``ny`` cells."""
        j = self.joint()
        out = {}
        for o in (design or ORDERS):
            out[o] = j if o == 'AB' else j[[0, 2, 1, 3]]
        return out


class AnchoringOrderModel(Model):
    """Classical baseline: the second answer is anchored on the first.

    The first answer is given at its first-position rate. For ``w >= 0``,
    P(second yes | first yes) = p2 + w (1 - p2) and P(second yes | first no) = p2 (1 - w); the mirror
    image applies for ``w < 0``.

    Parameters
    ----------
    pA, pB : float
        "Yes" rates of each question when asked first.
    wA, wB : float
        Anchoring weight when A (respectively B) is the second question, in (-0.99, 0.99).
    tied : bool, optional
        If True, ``wB`` is ignored and ``wA`` is used for both questions.
    """
    PARAMS = [Param('pA', 'prob', 0.5), Param('pB', 'prob', 0.5),
              Param('wA', 'bounded', 0.2, -0.99, 0.99), Param('wB', 'bounded', 0.2, -0.99, 0.99)]
    name = 'Anchoring'

    def predict(self, design=None):
        """Answer distribution ``[yy, yn, ny, nn]`` for each order."""
        p = {'A': self.pA, 'B': self.pB}
        w = {'A': self.wA, 'B': self.wB if not self.options.get('tied') else self.wA}
        out = {}
        for o in (design or ORDERS):
            p1, p2, wq = p[o[0]], p[o[1]], w[o[1]]
            if wq >= 0:                     # pulled toward the first answer
                yy, ny = p2 + wq * (1 - p2), p2 * (1 - wq)
            else:                           # pushed away from it
                yy, ny = p2 * (1 + wq), p2 - wq * (1 - p2)
            out[o] = np.array([p1 * yy, p1 * (1 - yy), (1 - p1) * ny, (1 - p1) * (1 - ny)])
        return out


class SaturatedOrderModel(Model):
    """Reference model with a free answer distribution per order.

    Six parameters (softmax logits, three per order); no model can fit the data better.

    Parameters
    ----------
    x0, ..., x5 : float
        Logits of ``yy``, ``yn``, ``ny`` (relative to ``nn``) for AB, then for BA.
    """
    PARAMS = [Param('x%d' % i, 'real', 0.0) for i in range(6)]
    name = 'Saturated'

    def predict(self, design=None):
        """Answer distribution for each order (softmax of the logits)."""
        def sm(v):
            e = np.exp(np.r_[v, 0.0] - max(np.max(v), 0))
            return e / e.sum()
        v = [getattr(self, 'x%d' % i) for i in range(6)]
        return {'AB': sm(v[:3]), 'BA': sm(v[3:])}


def rates(pred):
    """First- and second-position "yes" rates.

    Parameters
    ----------
    pred : dict
        ``{'AB': cells, 'BA': cells}`` from ``predict`` or from data proportions.

    Returns
    -------
    numpy.ndarray
        ``[A first, B first, A second, B second]``.
    """
    ab, ba = pred['AB'], pred['BA']
    return np.array([ab[0] + ab[1], ba[0] + ba[1], ba[0] + ba[2], ab[0] + ab[2]])


def qq_statistic(cells_ab, cells_ba):
    """QQ statistic.

    Parameters
    ----------
    cells_ab, cells_ba : array_like
        ``[yy, yn, ny, nn]`` probabilities for each order.

    Returns
    -------
    float
        :math:`[p_{AB}(yn) + p_{AB}(ny)] - [p_{BA}(yn) + p_{BA}(ny)]`; zero for every projective model.
    """
    return float((cells_ab[1] + cells_ab[2]) - (cells_ba[1] + cells_ba[2]))


def qq_test(counts):
    """Two-sided z test of the QQ equality.

    Parameters
    ----------
    counts : dict
        ``{'AB': [yy, yn, ny, nn], 'BA': [...]}`` answer counts.

    Returns
    -------
    q : float
        Observed QQ statistic.
    z : float
        z score.
    p_value : float
        Two-sided p-value; a small value rejects every projective (quantum-like) model.

    Examples
    --------
    >>> from quantum_mind.families.order_effects import qq_test
    >>> q, z, p = qq_test({'AB': [40, 10, 10, 40], 'BA': [40, 10, 10, 40]})
    >>> q, p
    (0.0, 1.0)
    """
    from scipy.stats import norm
    ab, ba = np.asarray(counts['AB'], float), np.asarray(counts['BA'], float)
    p1, p2 = (ab[1] + ab[2]) / ab.sum(), (ba[1] + ba[2]) / ba.sum()
    se = np.sqrt(p1 * (1 - p1) / ab.sum() + p2 * (1 - p2) / ba.sum())
    z = (p1 - p2) / se if se > 0 else 0.0
    return float(p1 - p2), float(z), float(2 * norm.sf(abs(z)))
