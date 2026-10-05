"""Question-order effects: two yes/no questions A and B asked in either order.

Design: conditions 'AB' (A asked first) and 'BA' (B asked first). Outcomes of each condition are the
four answer pairs [yy, yn, ny, nn], where the first letter is the answer to the first question asked.

Models
  QuantumOrderModel   three-dimensional real quantum-like model (Lueders rule), with the rank (1 or 2)
                      of each question's 'yes' subspace as a structural option; satisfies the QQ equality.
                      Three parameters; cannot represent order-free data with intermediate correlation.
  QuantumOrderModel4D four-dimensional model with an incompatibility angle; phi = 0 is exactly the
                      Bayesian model, so it nests it (four parameters). Recommended default.
  BayesOrderModel     one order-free joint distribution (no order effect).
  AnchoringOrderModel the second answer is pulled toward (w > 0) or away from (w < 0) the first, with a
                      weight per question; violates the QQ equality when the weights differ.
  SaturatedOrderModel a free distribution per order (6 parameters; reference only).
  ProjectiveQuestionModel  any state and any projectors in any dimension (not fitted directly)."""
from __future__ import annotations

import itertools
import numpy as np
from ...core import Model, Param, projector, sequence_probabilities, normalize

ORDERS = ('AB', 'BA')
RANK_STRUCTURES = [{'ranks': r} for r in itertools.product((1, 2), repeat=2)]
DESIGN = ORDERS


def _cells(probs):
    return np.array([probs[(1, 1)], probs[(1, 0)], probs[(0, 1)], probs[(0, 0)]])


class ProjectiveQuestionModel:
    """General projective model: psi (unit vector, real or complex) and a dict of 'yes' projectors."""
    def __init__(self, psi, projectors):
        self.psi = normalize(psi); self.projectors = projectors

    def answer_probs(self, order):
        """Probabilities of all answer sequences for questions asked in `order` (Lueders rule)."""
        return sequence_probabilities(self.psi, self.projectors, order)

    def predict(self, design=None):
        """Answer distribution [yy, yn, ny, nn] for each question order in the design (default AB, BA)."""
        return {o: _cells(self.answer_probs(tuple(o))) for o in (design or ORDERS)}


class QuantumOrderModel(Model):
    """psi = e1; u_A = (cos a, sin a, 0); u_B = (cos b, sin b cos g, sin b sin g).
    Option ranks=(rA, rB): rank 1 -> 'yes' is the ray of u_q; rank 2 -> the plane orthogonal to u_q."""
    PARAMS = [Param('a', 'angle', 0.8), Param('b', 'angle', 1.2), Param('g', 'angle', 0.3)]
    name = 'Quantum-like'

    def vectors(self):
        """Unit vectors u_A and u_B that define the two questions."""
        a, b, g = self.a, self.b, self.g
        return {'A': np.array([np.cos(a), np.sin(a), 0.0]),
                'B': np.array([np.cos(b), np.sin(b) * np.cos(g), np.sin(b) * np.sin(g)])}

    def projectors(self):
        """'Yes' projectors of the two questions {'A': P_A, 'B': P_B}."""
        ranks = dict(zip('AB', self.options.get('ranks', (1, 1))))
        out = {}
        for q, u in self.vectors().items():
            P = projector(u)
            out[q] = P if ranks[q] == 1 else np.eye(3) - P
        return out

    def as_projective(self):
        """The same model as a ProjectiveQuestionModel (state and projectors)."""
        return ProjectiveQuestionModel(np.array([1.0, 0.0, 0.0]), self.projectors())

    def predict(self, design=None):
        return self.as_projective().predict(design)


class QuantumOrderModel4D(Model):
    """Four-dimensional real model in which the questions can be compatible or incompatible.
    Basis e1..e4 = (yes,yes), (yes,no), (no,yes), (no,no) of a classical 2 x 2 table; P_A = span(e1, e2).
    P_B = U span(e1, e3) U^T, where U rotates by the incompatibility angle phi in the planes (e1, e4)
    and (e2, e3). The state psi has three hyperspherical angles. With phi = 0 the projectors commute
    and the model is exactly the order-free Bayesian model (psi_k^2 is the joint table), so the model
    nests BayesOrderModel; phi != 0 produces order effects (and the QQ equality still holds)."""
    PARAMS = [Param('t1', 'angle', 0.8), Param('t2', 'angle', 0.8), Param('t3', 'angle', 0.8), Param('phi', 'angle', 0.2)]
    name = 'Quantum-like (4D, nests Bayes)'

    def projectors(self):
        """'Yes' projectors of the two questions {'A': P_A, 'B': P_B}."""
        from ...core import angles_to_unit  # noqa: F401
        c, s_ = np.cos(self.phi), np.sin(self.phi)
        U = np.eye(4)
        U[0, 0] = U[3, 3] = c; U[0, 3] = -s_; U[3, 0] = s_
        U[1, 1] = U[2, 2] = c; U[1, 2] = -s_; U[2, 1] = s_
        E = np.eye(4)
        PA = projector([E[0], E[1]])
        PB = U @ projector([E[0], E[2]]) @ U.T
        return {'A': PA, 'B': PB}

    def as_projective(self):
        """The same model as a ProjectiveQuestionModel (state and projectors)."""
        from ...core import angles_to_unit
        return ProjectiveQuestionModel(angles_to_unit([self.t1, self.t2, self.t3]), self.projectors())

    def predict(self, design=None):
        return self.as_projective().predict(design)


class BayesOrderModel(Model):
    """Order-free joint distribution with marginals pA, pB and correlation rho in (-1, 1) (scaled
    between the bounds allowed by the marginals)."""
    PARAMS = [Param('pA', 'prob', 0.5), Param('pB', 'prob', 0.5), Param('rho', 'bounded', 0.0, -1, 1)]
    name = 'Bayesian (order-free)'

    def joint(self):
        """Order-free joint distribution [yy, yn, ny, nn] with A first."""
        pA, pB = self.pA, self.pB
        lo, hi = max(0.0, pA + pB - 1) - pA * pB, min(pA, pB) - pA * pB
        yy = pA * pB + self.rho * (hi if self.rho > 0 else -lo)
        return np.array([yy, pA - yy, pB - yy, 1 - pA - pB + yy])        # A first: yy, yn, ny, nn

    def predict(self, design=None):
        j = self.joint(); out = {}
        for o in (design or ORDERS):
            out[o] = j if o == 'AB' else j[[0, 2, 1, 3]]
        return out


class AnchoringOrderModel(Model):
    """First answer at its first-position rate; P(second yes | first yes) = p2 + w (1 - p2) and
    P(second yes | first no) = p2 (1 - w) for w >= 0 (mirror image for w < 0). w depends on the
    second question (wA, wB). Option tied=True forces wA = wB (one weight)."""
    PARAMS = [Param('pA', 'prob', 0.5), Param('pB', 'prob', 0.5),
              Param('wA', 'bounded', 0.2, -0.99, 0.99), Param('wB', 'bounded', 0.2, -0.99, 0.99)]
    name = 'Anchoring'

    def predict(self, design=None):
        p = {'A': self.pA, 'B': self.pB}
        w = {'A': self.wA, 'B': self.wB if not self.options.get('tied') else self.wA}
        out = {}
        for o in (design or ORDERS):
            p1, p2, wq = p[o[0]], p[o[1]], w[o[1]]
            if wq >= 0:
                yy, ny = p2 + wq * (1 - p2), p2 * (1 - wq)
            else:
                yy, ny = p2 * (1 + wq), p2 - wq * (1 - p2)
            out[o] = np.array([p1 * yy, p1 * (1 - yy), (1 - p1) * ny, (1 - p1) * (1 - ny)])
        return out


class SaturatedOrderModel(Model):
    """Reference model with a free answer distribution per order (6 parameters); the best any model can fit."""
    PARAMS = [Param('x%d' % i, 'real', 0.0) for i in range(6)]
    name = 'Saturated'

    def predict(self, design=None):
        def sm(v):
            e = np.exp(np.r_[v, 0.0] - max(np.max(v), 0)); return e / e.sum()
        v = [getattr(self, 'x%d' % i) for i in range(6)]
        return {'AB': sm(v[:3]), 'BA': sm(v[3:])}


def rates(pred):
    """First- and second-position 'yes' rates [A first, B first, A second, B second]."""
    ab, ba = pred['AB'], pred['BA']
    return np.array([ab[0] + ab[1], ba[0] + ba[1], ba[0] + ba[2], ab[0] + ab[2]])


def qq_statistic(cells_ab, cells_ba):
    """[p_AB(yn) + p_AB(ny)] - [p_BA(yn) + p_BA(ny)]; zero for every projective model."""
    return float((cells_ab[1] + cells_ab[2]) - (cells_ba[1] + cells_ba[2]))


def qq_test(counts):
    """Two-sided z test of the QQ equality from answer counts {'AB': [...], 'BA': [...]}.
    Returns (q, z, p_value)."""
    from scipy.stats import norm
    ab, ba = np.asarray(counts['AB'], float), np.asarray(counts['BA'], float)
    p1, p2 = (ab[1] + ab[2]) / ab.sum(), (ba[1] + ba[2]) / ba.sum()
    se = np.sqrt(p1 * (1 - p1) / ab.sum() + p2 * (1 - p2) / ba.sum())
    z = (p1 - p2) / se if se > 0 else 0.0
    return float(p1 - p2), float(z), float(2 * norm.sf(abs(z)))
