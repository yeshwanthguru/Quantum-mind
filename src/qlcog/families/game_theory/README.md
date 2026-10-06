# Quantum games (`qlcog.families.game_theory`)

Two-player 2 × 2 games played with the **Eisert–Wilkens–Lewenstein (EWL)** protocol, with the
classical game as the baseline it must reproduce.

| Name | What it is |
|---|---|
| `EWLGame(payoffs, gamma)` | entangling gate J = exp(iγ D⊗D/2) with D = iσ<sub>y</sub>; `payoff(U_A, U_B)`, `outcome_probabilities`, `best_response`, `is_nash` |
| `strategy(theta, phi)`, `C`, `D`, `Q` | EWL two-parameter strategies: cooperate, defect, and the quantum strategy Q = U(0, π/2) |
| `classical_nash_equilibria(payoffs)` | classical baseline: all pure and mixed Nash equilibria of the 2 × 2 game |
| `PRISONERS_DILEMMA`, `CHICKEN`, `STAG_HUNT` | payoff tables `(2, 2, 2)` |

Checked results (tests): with γ = 0 every pair of classical strategies gives the classical payoffs and
(D, D) is the only equilibrium; with γ = π/2 classical strategies still give the classical payoffs,
(Q, Q) gives (3, 3) and is a Nash equilibrium of the two-parameter strategy set, and Q exploits a
defector (D vs Q gives (0, 5)). Allowing every SU(2) strategy (three parameters) removes the (Q, Q)
equilibrium, as Benjamin and Hayden pointed out.

**Scope.** These are normative models of strategic interaction with quantum resources, not models of
how people play; for human cooperation data (the disjunction effect in the Prisoner's Dilemma) use the
`interference` and `qlbn` families.

References: Eisert, Wilkens and Lewenstein, *PRL* 83, 3077 (1999); Benjamin and Hayden, *PRL* 87,
069801 (2001); Du et al., *PRL* 88, 137902 (2002).
