"""Quantum games: the Prisoner's Dilemma
=====================================

Game theory: the quantum Prisoner's Dilemma (Eisert-Wilkens-Lewenstein protocol).

With no entanglement the quantum game is the classical game: mutual defection is the only Nash
equilibrium. With maximal entanglement and the two-parameter strategy set, the quantum strategy Q
gives a Nash equilibrium with the cooperative payoff, and Q exploits a defector. Allowing every SU(2)
strategy (three parameters) removes that equilibrium again (Benjamin and Hayden, 2001).
"""
# sphinx_gallery_start_ignore
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parents[1] / "src"))  # run without installing
# sphinx_gallery_end_ignore
# sphinx_gallery_thumbnail_path = '_static/thumbs/ql.png'
import numpy as np
from quantum_mind.families.game_theory import EWLGame, PRISONERS_DILEMMA, CHICKEN, C, D, Q, classical_nash_equilibria

# %%
# The classical games
# -------------------
# Nash equilibria of the classical Prisoner's Dilemma and Chicken.
print('classical Nash equilibria (p_A(C), p_B(C), payoff A, payoff B):')
for name, g in (("Prisoner's Dilemma", PRISONERS_DILEMMA), ('Chicken', CHICKEN)):
    print('  %-18s %s' % (name, [tuple(round(float(v), 2) for v in e) for e in classical_nash_equilibria(g)]))

# %%
# Increasing the entanglement
# ---------------------------
# Payoffs of four strategy pairs and two Nash checks at three values of the entanglement gamma.
for gamma in (0.0, np.pi / 4, np.pi / 2):
    G = EWLGame(PRISONERS_DILEMMA, gamma)
    print('gamma = %.2f  (C,C) %s  (D,D) %s  (Q,Q) %s  (D,Q) %s  | (Q,Q) Nash: %s, (D,D) Nash: %s'
          % (gamma, G.payoff(C, C), G.payoff(D, D), tuple(round(x, 2) for x in G.payoff(Q, Q)),
             tuple(round(x, 2) for x in G.payoff(D, Q)), G.is_nash(Q, Q, grid=31), G.is_nash(D, D, grid=31)))

# %%
# Unrestricted strategies
# -----------------------
# With all SU(2) strategies a player can always beat Q, so (Q, Q) is no longer an equilibrium.
G = EWLGame(PRISONERS_DILEMMA, np.pi / 2)
print('with all SU(2) strategies (three parameters): (Q,Q) Nash: %s' % G.is_nash(Q, Q, grid=17, three_parameter=True))
