r"""Violations of the law of total probability (disjunction effect, sure-thing principle).

A decision ("act" or "not act") is made when an earlier event is known to be X, known to be Y, or
unknown. Classical probability requires P(act | unknown) to lie between P(act | X) and P(act | Y);
people often violate this (Prisoner's Dilemma, two-stage gamble; Tversky and Shafir, 1992).

**Design.** Conditions ``'known_1'``, ``'known_2'`` and ``'unknown'``; outcomes ``[act, not act]``.

**Quantum-like law of total probability.** The unknown condition superposes the two paths with
amplitudes :math:`\sqrt{c}` and :math:`\sqrt{1-c}` and relative phase :math:`\theta`:

.. math::

   A &= c\,p_1 + (1-c)\,p_2 + 2\cos\theta\,\sqrt{c\,p_1\,(1-c)\,p_2} \quad\text{(act)}\\
   B &= c\,q_1 + (1-c)\,q_2 + 2\cos\theta\,\sqrt{c\,q_1\,(1-c)\,q_2}, \quad q = 1 - p \quad\text{(not act)}

and :math:`P(\text{act}\mid\text{unknown}) = A / (A + B)`, the normalised form used in quantum-like
Bayesian networks (Moreira and Wichert, 2014). ``normalized=False`` uses :math:`A` alone, clipped to
[0, 1]. :math:`\theta = \pi/2` recovers the classical law. The interference circuit in
:mod:`quantum_mind.circuits` implements the normalised form exactly (a which-path qubit erased by a Hadamard
gate and post-selected).
"""
from __future__ import annotations

import numpy as np
from ...core import Model, Param

#: Conditions of the default design.
CONDITIONS = ('known_1', 'known_2', 'unknown')


def _pair(p):
    """[p, 1 - p] with p clipped to [0, 1]."""
    p = float(np.clip(p, 0, 1))
    return np.array([p, 1 - p])


class InterferenceModel(Model):
    """Quantum-like law of total probability with an interference term between the two paths.

    Parameters
    ----------
    p1, p2 : float
        P(act) when the event is known to be X or Y.
    c : float
        Weight of path X when the event is unknown.
    theta : float
        Relative phase of the two paths; ``pi/2`` gives the classical law.
    normalized : bool, optional
        Use the normalised form ``A / (A + B)`` (default True).
    """
    PARAMS = [Param('p1', 'prob', 0.9), Param('p2', 'prob', 0.8), Param('c', 'prob', 0.5), Param('theta', 'angle', 2.0)]
    name = 'Quantum-like interference'

    def p_unknown(self):
        """Probability of acting when the event is unknown.

        Returns
        -------
        float
        """
        c, p1, p2, ct = self.c, self.p1, self.p2, np.cos(self.theta)
        A = c * p1 + (1 - c) * p2 + 2 * ct * np.sqrt(c * p1 * (1 - c) * p2)
        if not self.options.get('normalized', True):
            return float(np.clip(A, 0, 1))
        q1, q2 = 1 - p1, 1 - p2
        B = c * q1 + (1 - c) * q2 + 2 * ct * np.sqrt(c * q1 * (1 - c) * q2)
        return float(A / (A + B)) if A + B > 1e-12 else 0.5

    def predict(self, design=None):
        """``[P(act), P(not act)]`` for each condition."""
        v = {'known_1': self.p1, 'known_2': self.p2, 'unknown': self.p_unknown()}
        return {k: _pair(v[k]) for k in (design or CONDITIONS)}


class ClassicalMixtureModel(Model):
    """Classical baseline: the unknown condition mixes the known ones (law of total probability).

    Parameters
    ----------
    p1, p2 : float
        P(act) when the event is known to be X or Y.
    c : float
        Subjective probability of X.
    """
    PARAMS = [Param('p1', 'prob', 0.9), Param('p2', 'prob', 0.8), Param('c', 'prob', 0.5)]
    name = 'Classical (total probability)'

    def predict(self, design=None):
        """``[P(act), P(not act)]`` for each condition."""
        v = {'known_1': self.p1, 'known_2': self.p2, 'unknown': self.c * self.p1 + (1 - self.c) * self.p2}
        return {k: _pair(v[k]) for k in (design or CONDITIONS)}


def total_probability_bounds(p1, p2):
    """Classical range of P(act | unknown).

    Parameters
    ----------
    p1, p2 : float
        P(act) in the two known conditions.

    Returns
    -------
    tuple of float
        ``(min(p1, p2), max(p1, p2))``; values outside violate the law of total probability.
    """
    return min(p1, p2), max(p1, p2)


def interference_phase(p1, p2, p_unknown, c=0.5):
    """Phase that reproduces an observed P(act | unknown) with the unnormalised law.

    Parameters
    ----------
    p1, p2 : float
        P(act) in the two known conditions.
    p_unknown : float
        Observed P(act | unknown).
    c : float, optional
        Path weight.

    Returns
    -------
    float
        :math:`\\theta` in [0, pi], or ``nan`` if no phase can reproduce ``p_unknown``.
    """
    cosv = (p_unknown - c * p1 - (1 - c) * p2) / (2 * np.sqrt(c * p1 * (1 - c) * p2))
    return float(np.arccos(cosv)) if -1 <= cosv <= 1 else float('nan')
