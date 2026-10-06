"""Quantum games: the Eisert-Wilkens-Lewenstein (EWL) protocol for two-player 2 x 2 games, with the
classical game (and its Nash equilibria) as the baseline it must reproduce.

Each player owns one qubit (|0> = cooperate C, |1> = defect D). A referee entangles the pair with
J = exp(i gamma D (x) D / 2), D = i sigma_y; the players apply local unitaries U_A, U_B; the referee
applies J^dagger and measures. Strategies are U(theta, phi) = [[e^{i phi} cos(theta/2), sin(theta/2)],
[-sin(theta/2), e^{-i phi} cos(theta/2)]]: C = U(0, 0), D = U(pi, 0), Q = U(0, pi/2).
With gamma = 0 the game is exactly the classical game for every pair of classical strategies; with
gamma = pi/2 in the Prisoner's Dilemma, (Q, Q) is a Nash equilibrium of the two-parameter strategy
set with the Pareto-optimal payoff, and Q exploits a defector.

Eisert, Wilkens and Lewenstein, Physical Review Letters 83, 3077 (1999).
Note: with unrestricted SU(2) strategies (three parameters) the (Q, Q) equilibrium disappears
(Benjamin and Hayden, PRL 87, 069801, 2001); `best_response` lets this be checked numerically."""
from __future__ import annotations

import itertools
import numpy as np

__all__ = ['PRISONERS_DILEMMA', 'CHICKEN', 'STAG_HUNT', 'strategy', 'C', 'D', 'Q', 'EWLGame',
           'classical_nash_equilibria']

# payoffs[a, b] = (payoff of A, payoff of B) for actions a, b in {0: C, 1: D}
PRISONERS_DILEMMA = np.array([[[3, 3], [0, 5]], [[5, 0], [1, 1]]], float)
CHICKEN = np.array([[[3, 3], [1, 4]], [[4, 1], [0, 0]]], float)
STAG_HUNT = np.array([[[4, 4], [0, 3]], [[3, 0], [2, 2]]], float)

_SY = np.array([[0, -1j], [1j, 0]])
_D_OP = 1j * _SY                                              # = [[0, 1], [-1, 0]]


def strategy(theta, phi=0.0):
    """EWL two-parameter strategy U(theta, phi); theta in [0, pi], phi in [0, pi/2]."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[np.exp(1j * phi) * c, s], [-s, np.exp(-1j * phi) * c]])


C, D, Q = strategy(0.0), strategy(np.pi), strategy(0.0, np.pi / 2)


class EWLGame:
    """Two-player quantum game. payoffs: array (2, 2, 2) as PRISONERS_DILEMMA; gamma: entanglement in
    [0, pi/2] (0 = classical game)."""

    def __init__(self, payoffs=PRISONERS_DILEMMA, gamma=np.pi / 2):
        self.payoffs = np.asarray(payoffs, float)
        if self.payoffs.shape != (2, 2, 2):
            raise ValueError('payoffs must have shape (2, 2, 2)')
        self.gamma = float(gamma)
        DD = np.kron(_D_OP, _D_OP)                            # (D (x) D)^2 = I, so J = cos I + i sin DD
        self.J = np.cos(self.gamma / 2) * np.eye(4) + 1j * np.sin(self.gamma / 2) * DD

    def outcome_probabilities(self, UA, UB):
        """P(CC), P(CD), P(DC), P(DD) after the protocol (first letter: player A)."""
        psi = self.J.conj().T @ np.kron(UA, UB) @ self.J @ np.array([1, 0, 0, 0], complex)
        return np.abs(psi) ** 2

    def payoff(self, UA, UB):
        """Expected payoffs (A, B)."""
        p = self.outcome_probabilities(UA, UB).reshape(2, 2)
        return float(np.sum(p * self.payoffs[..., 0])), float(np.sum(p * self.payoffs[..., 1]))

    def best_response(self, U_other, player='A', grid=61, phases=True, three_parameter=False):
        """Best payoff and strategy against `U_other` over a grid of strategies: (theta, phi) of the EWL
        set, or the full SU(2) set U(theta, phi, alpha) with three_parameter=True."""
        best = (-np.inf, None)
        thetas = np.linspace(0, np.pi, grid)
        phis = np.linspace(0, np.pi / 2, grid // 2 + 1) if phases else [0.0]
        alphas = np.linspace(0, np.pi, grid // 2 + 1) if three_parameter else [None]
        for th, ph, al in itertools.product(thetas, phis, alphas):
            U = strategy(th, ph) if al is None else _su2(th, ph, al)
            pay = self.payoff(U, U_other)[0] if player == 'A' else self.payoff(U_other, U)[1]
            if pay > best[0] + 1e-12:
                best = (pay, (th, ph) if al is None else (th, ph, al))
        return best

    def is_nash(self, UA, UB, **kw):
        """True if neither player gains (beyond 1e-9) by deviating within the grid of strategies."""
        pa, pb = self.payoff(UA, UB)
        return self.best_response(UB, 'A', **kw)[0] <= pa + 1e-9 and self.best_response(UA, 'B', **kw)[0] <= pb + 1e-9


def _su2(theta, phi, alpha):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[np.exp(1j * phi) * c, np.exp(1j * alpha) * s], [-np.exp(-1j * alpha) * s, np.exp(-1j * phi) * c]])


def classical_nash_equilibria(payoffs=PRISONERS_DILEMMA, tol=1e-12):
    """All Nash equilibria of the classical 2 x 2 game: pure ones and the mixed one if it exists.
    Returns a list of (p_A(C), p_B(C), payoff_A, payoff_B)."""
    P = np.asarray(payoffs, float); out = []
    for a, b in itertools.product((0, 1), repeat=2):
        if P[a, b, 0] >= P[1 - a, b, 0] - tol and P[a, b, 1] >= P[a, 1 - b, 1] - tol:
            out.append((1.0 - a, 1.0 - b, P[a, b, 0], P[a, b, 1]))
    # mixed: q = P(B plays C) makes A indifferent; p = P(A plays C) makes B indifferent
    A, B = P[..., 0], P[..., 1]
    den_q = A[0, 0] - A[0, 1] - A[1, 0] + A[1, 1]; den_p = B[0, 0] - B[1, 0] - B[0, 1] + B[1, 1]
    if abs(den_q) > tol and abs(den_p) > tol:
        q = (A[1, 1] - A[0, 1]) / den_q; p = (B[1, 1] - B[1, 0]) / den_p
        if 0 < p < 1 and 0 < q < 1:
            probs = np.outer([p, 1 - p], [q, 1 - q])
            out.append((p, q, float(np.sum(probs * A)), float(np.sum(probs * B))))
    return out
