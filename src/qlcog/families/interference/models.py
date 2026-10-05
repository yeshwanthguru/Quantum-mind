"""Violations of the law of total probability (disjunction effect, sure-thing principle).

A decision ('act' or 'not act') is made when an earlier event is known to be X, known to be Y, or
unknown. Classical probability requires P(act | unknown) to lie between P(act | X) and P(act | Y);
people often violate this (Prisoner's Dilemma, two-stage gamble).

Design: conditions 'known_1', 'known_2', 'unknown'; outcomes [act, not act].

Models
  InterferenceModel     quantum-like law of total probability. The unknown condition superposes the
                        two paths with amplitudes sqrt(c) and sqrt(1-c) and relative phase theta:
                          A = c p1 + (1-c) p2 + 2 cos(theta) sqrt(c p1 (1-c) p2)          (act)
                          B = c q1 + (1-c) q2 + 2 cos(theta) sqrt(c q1 (1-c) q2), q = 1 - p (not act)
                        P(act | unknown) = A / (A + B) (normalised form, as in quantum-like Bayesian
                        networks; Moreira and Wichert, 2014). Option normalized=False uses A alone,
                        clipped to [0, 1] (the unnormalised form common in the literature).
                        theta = pi/2 recovers the classical law.
  ClassicalMixtureModel P(act | unknown) = c p1 + (1-c) p2.
The interference circuit in qlcog.circuits implements the normalised form exactly (a which-path
qubit erased by a Hadamard gate and post-selected)."""
from __future__ import annotations

import numpy as np
from ...core import Model, Param

CONDITIONS = ('known_1', 'known_2', 'unknown')


def _pair(p):
    p = float(np.clip(p, 0, 1)); return np.array([p, 1 - p])


class InterferenceModel(Model):
    PARAMS = [Param('p1', 'prob', 0.9), Param('p2', 'prob', 0.8), Param('c', 'prob', 0.5), Param('theta', 'angle', 2.0)]
    name = 'Quantum-like interference'

    def p_unknown(self):
        c, p1, p2, ct = self.c, self.p1, self.p2, np.cos(self.theta)
        A = c * p1 + (1 - c) * p2 + 2 * ct * np.sqrt(c * p1 * (1 - c) * p2)
        if not self.options.get('normalized', True):
            return float(np.clip(A, 0, 1))
        q1, q2 = 1 - p1, 1 - p2
        B = c * q1 + (1 - c) * q2 + 2 * ct * np.sqrt(c * q1 * (1 - c) * q2)
        return float(A / (A + B)) if A + B > 1e-12 else 0.5

    def predict(self, design=None):
        v = {'known_1': self.p1, 'known_2': self.p2, 'unknown': self.p_unknown()}
        return {k: _pair(v[k]) for k in (design or CONDITIONS)}


class ClassicalMixtureModel(Model):
    PARAMS = [Param('p1', 'prob', 0.9), Param('p2', 'prob', 0.8), Param('c', 'prob', 0.5)]
    name = 'Classical (total probability)'

    def predict(self, design=None):
        v = {'known_1': self.p1, 'known_2': self.p2, 'unknown': self.c * self.p1 + (1 - self.c) * self.p2}
        return {k: _pair(v[k]) for k in (design or CONDITIONS)}


def total_probability_bounds(p1, p2):
    """Classical range of P(act | unknown)."""
    return min(p1, p2), max(p1, p2)


def interference_phase(p1, p2, p_unknown, c=0.5):
    """Phase theta for which the unnormalised quantum-like law reproduces p_unknown (nan if impossible)."""
    cosv = (p_unknown - c * p1 - (1 - c) * p2) / (2 * np.sqrt(c * p1 * (1 - c) * p2))
    return float(np.arccos(cosv)) if -1 <= cosv <= 1 else float('nan')
