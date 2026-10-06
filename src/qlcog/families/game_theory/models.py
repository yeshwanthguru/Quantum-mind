r"""Quantum games: the Eisert-Wilkens-Lewenstein (EWL) protocol for two-player 2 x 2 games, with the
classical game and its Nash equilibria as the baseline it must reproduce.

Each player owns one qubit (:math:`|0\rangle` = cooperate C, :math:`|1\rangle` = defect D). A referee
entangles the pair with :math:`J = \exp(i\gamma\, D \otimes D / 2)`, :math:`D = i\sigma_y`; the players
apply local unitaries :math:`U_A, U_B`; the referee applies :math:`J^\dagger` and measures. The
strategies are

.. math:: U(\theta, \phi) = \begin{pmatrix} e^{i\phi}\cos\frac\theta2 & \sin\frac\theta2 \\
          -\sin\frac\theta2 & e^{-i\phi}\cos\frac\theta2 \end{pmatrix},

with :math:`C = U(0, 0)`, :math:`D = U(\pi, 0)` and :math:`Q = U(0, \pi/2)`.

With :math:`\gamma = 0` the game is exactly the classical game for every pair of classical strategies.
With :math:`\gamma = \pi/2` in the Prisoner's Dilemma, (Q, Q) is a Nash equilibrium of the
two-parameter strategy set with the Pareto-optimal payoff, and Q exploits a defector (Eisert, Wilkens
and Lewenstein, Physical Review Letters 83, 3077, 1999).

With unrestricted SU(2) strategies (three parameters) the (Q, Q) equilibrium disappears (Benjamin and
Hayden, PRL 87, 069801, 2001); :meth:`EWLGame.best_response` lets this be checked numerically.

Examples
--------
>>> from qlcog.families.game_theory import EWLGame, Q, D
>>> game = EWLGame(gamma=0.0)            # no entanglement: the classical game
>>> game.payoff(D, D)
(1.0, 1.0)
"""
from __future__ import annotations

import itertools
import numpy as np

__all__ = ['PRISONERS_DILEMMA', 'CHICKEN', 'STAG_HUNT', 'strategy', 'C', 'D', 'Q', 'EWLGame',
           'classical_nash_equilibria']

# payoffs[a, b] = (payoff of A, payoff of B) for actions a, b in {0: C, 1: D}
#: Prisoner's Dilemma payoffs, shape (2, 2, 2): ``[a, b] -> (payoff A, payoff B)``, action 0 = C, 1 = D.
PRISONERS_DILEMMA = np.array([[[3, 3], [0, 5]], [[5, 0], [1, 1]]], float)
#: Chicken (hawk-dove) payoffs.
CHICKEN = np.array([[[3, 3], [1, 4]], [[4, 1], [0, 0]]], float)
#: Stag hunt payoffs.
STAG_HUNT = np.array([[[4, 4], [0, 3]], [[3, 0], [2, 2]]], float)

_SY = np.array([[0, -1j], [1j, 0]])
_D_OP = 1j * _SY                                              # = [[0, 1], [-1, 0]]


def strategy(theta, phi=0.0):
    """EWL two-parameter strategy.

    Parameters
    ----------
    theta : float
        Angle in [0, pi]; 0 is "cooperate", pi is "defect".
    phi : float, optional
        Phase in [0, pi/2]; only non-zero phases use the quantum part of the strategy space.

    Returns
    -------
    numpy.ndarray
        2 x 2 unitary :math:`U(\\theta, \\phi)`.
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[np.exp(1j * phi) * c, s], [-s, np.exp(-1j * phi) * c]])


#: Cooperate, defect and the quantum strategy Q.
C, D, Q = strategy(0.0), strategy(np.pi), strategy(0.0, np.pi / 2)


class EWLGame:
    """Two-player quantum game (EWL protocol).

    Parameters
    ----------
    payoffs : array_like, optional
        Shape (2, 2, 2), as :data:`PRISONERS_DILEMMA` (the default).
    gamma : float, optional
        Entanglement in [0, pi/2]; 0 is the classical game, pi/2 (default) is maximal.

    Raises
    ------
    ValueError
        If ``payoffs`` does not have shape (2, 2, 2).
    """

    def __init__(self, payoffs=PRISONERS_DILEMMA, gamma=np.pi / 2):
        self.payoffs = np.asarray(payoffs, float)
        if self.payoffs.shape != (2, 2, 2):
            raise ValueError('payoffs must have shape (2, 2, 2)')
        self.gamma = float(gamma)
        DD = np.kron(_D_OP, _D_OP)                            # (D (x) D)^2 = I, so J = cos I + i sin DD
        self.J = np.cos(self.gamma / 2) * np.eye(4) + 1j * np.sin(self.gamma / 2) * DD

    def outcome_probabilities(self, UA, UB):
        """Probabilities of the four outcomes.

        Parameters
        ----------
        UA, UB : numpy.ndarray
            2 x 2 strategies of players A and B.

        Returns
        -------
        numpy.ndarray
            ``[P(CC), P(CD), P(DC), P(DD)]`` (first letter: player A).
        """
        # |psi> = J^dagger (UA x UB) J |00>
        psi = self.J.conj().T @ np.kron(UA, UB) @ self.J @ np.array([1, 0, 0, 0], complex)
        return np.abs(psi) ** 2

    def payoff(self, UA, UB):
        """Expected payoffs.

        Parameters
        ----------
        UA, UB : numpy.ndarray
            Strategies of players A and B.

        Returns
        -------
        tuple of float
            ``(payoff of A, payoff of B)``.
        """
        p = self.outcome_probabilities(UA, UB).reshape(2, 2)
        return float(np.sum(p * self.payoffs[..., 0])), float(np.sum(p * self.payoffs[..., 1]))

    def best_response(self, U_other, player='A', grid=61, phases=True, three_parameter=False):
        """Best response to a fixed strategy, by grid search.

        Parameters
        ----------
        U_other : numpy.ndarray
            Strategy of the other player.
        player : {'A', 'B'}, optional
            Player who responds.
        grid : int, optional
            Number of grid points for theta (about half as many for the phases).
        phases : bool, optional
            Include the phase phi (False restricts to classical mixed strategies).
        three_parameter : bool, optional
            Search the full SU(2) set :math:`U(\\theta, \\phi, \\alpha)` instead of the EWL set.

        Returns
        -------
        payoff : float
            Best payoff found.
        params : tuple
            ``(theta, phi)`` or ``(theta, phi, alpha)`` of the best strategy.
        """
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
        """Check whether a strategy pair is a Nash equilibrium on the grid.

        Parameters
        ----------
        UA, UB : numpy.ndarray
            Strategies.
        **kw
            Passed to :meth:`best_response` (for example ``three_parameter=True``).

        Returns
        -------
        bool
            True if neither player gains more than 1e-9 by deviating.
        """
        pa, pb = self.payoff(UA, UB)
        return self.best_response(UB, 'A', **kw)[0] <= pa + 1e-9 and self.best_response(UA, 'B', **kw)[0] <= pb + 1e-9


def _su2(theta, phi, alpha):
    """General SU(2) element with three parameters."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[np.exp(1j * phi) * c, np.exp(1j * alpha) * s], [-np.exp(-1j * alpha) * s, np.exp(-1j * phi) * c]])


def classical_nash_equilibria(payoffs=PRISONERS_DILEMMA, tol=1e-12):
    """Nash equilibria of the classical 2 x 2 game.

    Parameters
    ----------
    payoffs : array_like, optional
        Shape (2, 2, 2).
    tol : float, optional
        Tolerance for ties.

    Returns
    -------
    list of tuple
        ``(p_A(C), p_B(C), payoff_A, payoff_B)`` for every pure equilibrium and for the fully mixed
        one if it exists.

    Examples
    --------
    >>> from qlcog.families.game_theory import classical_nash_equilibria
    >>> [tuple(float(x) for x in e) for e in classical_nash_equilibria()]
    [(0.0, 0.0, 1.0, 1.0)]
    """
    P = np.asarray(payoffs, float)
    out = []
    for a, b in itertools.product((0, 1), repeat=2):
        if P[a, b, 0] >= P[1 - a, b, 0] - tol and P[a, b, 1] >= P[a, 1 - b, 1] - tol:
            out.append((1.0 - a, 1.0 - b, P[a, b, 0], P[a, b, 1]))
    # mixed: q = P(B plays C) makes A indifferent; p = P(A plays C) makes B indifferent
    A, B = P[..., 0], P[..., 1]
    den_q = A[0, 0] - A[0, 1] - A[1, 0] + A[1, 1]
    den_p = B[0, 0] - B[1, 0] - B[0, 1] + B[1, 1]
    if abs(den_q) > tol and abs(den_p) > tol:
        q = (A[1, 1] - A[0, 1]) / den_q
        p = (B[1, 1] - B[1, 0]) / den_p
        if 0 < p < 1 and 0 < q < 1:
            probs = np.outer([p, 1 - p], [q, 1 - q])
            out.append((p, q, float(np.sum(probs * A)), float(np.sum(probs * B))))
    return out
