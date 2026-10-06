r"""Choices between risky prospects (lotteries).

**Design.** ``{problem: (lottery_1, lottery_2)}``, a lottery being a list of
``(outcome, probability)`` pairs. The outcomes of each problem are ``[P(choose 1), P(choose 2)]``.

**Models.**

* :class:`QDTModel`: quantum decision theory (Yukalov and Sornette). The choice probability is
  :math:`p(\pi_n) = f(\pi_n) + q(\pi_n)`. The utility factor :math:`f` is a logit of expected utility
  with power utility :math:`u(x) = \mathrm{sign}(x)|x|^\alpha`. The attraction factor :math:`q` has
  magnitude ``q0`` (the "quarter law" suggests 0.25 as a non-informative value) and a negative sign
  for the more uncertain prospect (larger outcome variance). :math:`q` is clipped so that every
  probability stays in [0, 1], and :math:`q_1 + q_2 = 0`. This default sign rule encodes uncertainty
  aversion; in the loss domain it conflicts with the usual risk seeking (reflection effect), so signs
  can be set per problem with the option ``signs={problem: +1 or -1}`` (+1 favours prospect 1).
* :class:`ExpectedUtilityModel`: logit choice on expected power utility (the :math:`q = 0` case).
* :class:`ProspectTheoryModel`: logit choice on prospect value with value function
  :math:`x^\alpha` (gains), :math:`-\lambda(-x)^\alpha` (losses) and probability weighting
  :math:`w(p) = p^g / (p^g + (1-p)^g)^{1/g}` (Tversky and Kahneman, 1992 functional forms, applied
  outcome by outcome rather than rank-dependently).
"""
from __future__ import annotations

import numpy as np
from ...core import Model, Param


def _u(x, alpha):
    """Sign-preserving power utility."""
    x = np.asarray(x, float)
    return np.sign(x) * np.abs(x) ** alpha


def expected_utility(lottery, alpha):
    """Expected power utility of a lottery.

    Parameters
    ----------
    lottery : list of tuple
        ``[(outcome, probability), ...]``.
    alpha : float
        Utility exponent; :math:`u(x) = \\mathrm{sign}(x)|x|^\\alpha`.

    Returns
    -------
    float
    """
    return float(sum(p * _u(x, alpha) for x, p in lottery))


def outcome_variance(lottery):
    """Variance of the outcomes of a lottery.

    Parameters
    ----------
    lottery : list of tuple
        ``[(outcome, probability), ...]``.

    Returns
    -------
    float
    """
    m = sum(p * x for x, p in lottery)
    return float(sum(p * (x - m) ** 2 for x, p in lottery))


def _softmax2(v1, v2, beta):
    """Logit choice probabilities between two values with inverse temperature beta."""
    z = np.clip(beta * (v1 - v2), -500, 500)
    p1 = 1 / (1 + np.exp(-z))
    return np.array([p1, 1 - p1])


class ExpectedUtilityModel(Model):
    """Classical baseline: logit choice by expected utility.

    Parameters
    ----------
    alpha : float
        Utility exponent in (0.05, 2).
    beta : float
        Inverse temperature (choice consistency).
    """
    PARAMS = [Param('alpha', 'bounded', 0.8, 0.05, 2.0), Param('beta', 'positive', 1.0)]
    name = 'Expected utility (logit)'

    def predict(self, design=None):
        """``[P(choose 1), P(choose 2)]`` for each problem."""
        return {k: _softmax2(expected_utility(l1, self.alpha), expected_utility(l2, self.alpha), self.beta)
                for k, (l1, l2) in design.items()}


class QDTModel(Model):
    """Quantum decision theory (Yukalov and Sornette).

    Choice probability :math:`p = f + q`: a utility factor plus an attraction factor that favours the
    less uncertain prospect.

    Parameters
    ----------
    alpha : float
        Utility exponent.
    beta : float
        Inverse temperature.
    q0 : float
        Magnitude of the attraction factor, in (0, 0.5).
    signs : dict, optional
        ``{problem: +1 or -1}`` overriding the default sign rule.
    """
    PARAMS = [Param('alpha', 'bounded', 0.8, 0.05, 2.0), Param('beta', 'positive', 1.0), Param('q0', 'bounded', 0.25, 0.0, 0.5)]
    name = 'Quantum decision theory'

    def predict(self, design=None):
        """``[P(choose 1), P(choose 2)]`` for each problem."""
        out = {}
        for k, (l1, l2) in design.items():
            f = _softmax2(expected_utility(l1, self.alpha), expected_utility(l2, self.alpha), self.beta)
            s = self.options.get('signs', {}).get(k)
            if s is None:
                # default: the prospect with the larger outcome variance is less attractive
                v1, v2 = outcome_variance(l1), outcome_variance(l2)
                s = 0 if np.isclose(v1, v2) else (-1 if v1 > v2 else 1)
            q1 = float(np.clip(s * self.q0, -f[0], 1 - f[0]))     # keep both probabilities in [0, 1]
            out[k] = np.array([f[0] + q1, f[1] - q1])
        return out


class ProspectTheoryModel(Model):
    """Classical baseline: prospect-theory value with logit choice.

    Parameters
    ----------
    alpha : float
        Value-function curvature.
    lam : float
        Loss aversion :math:`\\lambda`.
    g : float
        Probability-weighting curvature.
    beta : float
        Inverse temperature.
    """
    PARAMS = [Param('alpha', 'bounded', 0.88, 0.05, 2.0), Param('lam', 'bounded', 2.25, 0.2, 5.0),
              Param('g', 'bounded', 0.65, 0.2, 1.5), Param('beta', 'positive', 1.0)]
    name = 'Prospect theory (logit)'

    def _w(self, p):
        """Probability weighting function."""
        g = self.g
        return p ** g / (p ** g + (1 - p) ** g) ** (1 / g)

    def value(self, lottery):
        """Prospect-theory value of a lottery.

        Parameters
        ----------
        lottery : list of tuple
            ``[(outcome, probability), ...]``.

        Returns
        -------
        float
        """
        v = 0.0
        for x, p in lottery:
            vx = abs(x) ** self.alpha if x >= 0 else -self.lam * abs(x) ** self.alpha
            v += self._w(p) * vx
        return v

    def predict(self, design=None):
        """``[P(choose 1), P(choose 2)]`` for each problem."""
        return {k: _softmax2(self.value(l1), self.value(l2), self.beta) for k, (l1, l2) in design.items()}
