r"""Trust-aware hand-over: a robot policy built on the trust qubit.

The robot hands objects to a person. Whether the person takes the object smoothly depends on how much
they trust the robot at that moment, and trust changes with every success and failure
(:class:`~quantum_mind.families.dynamics.OpenSystemBelief`). The robot can also *ask* about trust, which tells it
where trust stands but, in the quantum-like model, also changes it (the question collapses the belief).

At each step the policy compares the expected cost of four actions:

=================  ==========================================================================
``handover``       normal speed; fails with probability :math:`1 - p_{\text{trust}}`
``slow_handover``  slower and more legible; costs ``slow_cost`` but halves the failure risk
``ask``            costs ``ask_cost``; the answer collapses the belief, then the better of the
                   two hand-overs is chosen for the next step (one-step look-ahead)
``wait``           costs ``wait_cost``; the belief keeps dephasing toward its mixed state
=================  ==========================================================================

The same policy can run on the classical :class:`~quantum_mind.families.dynamics.MarkovBelief`, where asking
only reads trust; comparing the two shows when the order-aware model changes the robot's behaviour.

Examples
--------
>>> from quantum_mind.families.dynamics import OpenSystemBelief
>>> from quantum_mind.applications.handover import TrustAwareHandover
>>> policy = TrustAwareHandover(OpenSystemBelief(phi0=0.6, a_pos=0.8, a_neg=1.2, gamma=0.3))
>>> policy.decide()['action']                      # high trust: hand over normally
'handover'
>>> policy.observe(0); policy.observe(0)           # two failures
>>> policy.decide()['action'] != 'handover'
True
"""
from __future__ import annotations

import numpy as np

__all__ = ['TrustAwareHandover', 'simulate_handover_session']


class TrustAwareHandover:
    """Hand-over policy on a trust belief.

    Parameters
    ----------
    model : OpenSystemBelief or MarkovBelief
        Trust model, with its parameters (fitted to the person or the population).
    fail_cost : float, optional
        Cost of a failed hand-over (dropped object, startled person).
    slow_cost : float, optional
        Extra cost of a slow hand-over.
    ask_cost : float, optional
        Cost of asking "do you trust me to hand this over?".
    wait_cost : float, optional
        Cost of waiting one step.
    slow_factor : float, optional
        Failure probability of a slow hand-over relative to a normal one.

    Attributes
    ----------
    rho : numpy.ndarray
        Current trust belief as a 2 x 2 density matrix (open-system model), or ``None``.
    p : float
        Current P(trust) (Markov model), or ``None``.
    """

    def __init__(self, model, fail_cost=10.0, slow_cost=1.0, ask_cost=0.5, wait_cost=0.3, slow_factor=0.5):
        self.model = model
        self.fail_cost, self.slow_cost, self.ask_cost = fail_cost, slow_cost, ask_cost
        self.wait_cost, self.slow_factor = wait_cost, slow_factor
        self.quantum = hasattr(model, 'phi0') and hasattr(model, 'gamma')
        if self.quantum:
            v = np.array([np.cos(model.phi0 / 2), np.sin(model.phi0 / 2)])
            self.rho = np.outer(v, v)
            self.p = None
        else:
            self.rho = None
            self.p = float(model.p0)

    # ------------------------------------------------------------------ belief
    @property
    def p_trust(self):
        """float: current probability that the person trusts the robot (says "yes")."""
        return float(np.clip(self.rho[0, 0], 0, 1)) if self.quantum else self.p

    def _rotate(self, rho, event):
        """Open-system update after one event: rotation, then dephasing."""
        a = -self.model.a_pos if event else self.model.a_neg
        c, s = np.cos(a / 2), np.sin(a / 2)
        U = np.array([[c, -s], [s, c]])
        r = U @ rho @ U.T
        f = 1 - self.model.gamma
        return np.array([[r[0, 0], f * r[0, 1]], [f * r[1, 0], r[1, 1]]])

    def observe(self, event):
        """Update the belief after a hand-over outcome.

        Parameters
        ----------
        event : int
            1 for a success, 0 for a failure.
        """
        if self.quantum:
            self.rho = self._rotate(self.rho, event)
        else:
            self.p = self.model.step(self.p, event)

    def answer(self, yes):
        """Update the belief after the person answers the trust question.

        In the open-system model the answer collapses the belief onto "trust" or "no trust"; in the
        Markov model it sets P(trust) to 1 or 0.

        Parameters
        ----------
        yes : bool or int
        """
        if self.quantum:
            self.rho = np.diag([1.0, 0.0]) if yes else np.diag([0.0, 1.0])
        else:
            self.p = 1.0 if yes else 0.0

    def wait(self):
        """Let one step pass without a hand-over (dephasing only, open-system model)."""
        if self.quantum:
            f = 1 - self.model.gamma
            self.rho = np.array([[self.rho[0, 0], f * self.rho[0, 1]], [f * self.rho[1, 0], self.rho[1, 1]]])

    # ------------------------------------------------------------------ decision
    def _handover_costs(self, p):
        """Expected cost of a normal and a slow hand-over at trust probability p."""
        return self.fail_cost * (1 - p), self.slow_cost + self.fail_cost * (1 - p) * self.slow_factor

    def decide(self):
        """Expected cost of each action and the best one.

        Returns
        -------
        dict
            ``action`` (the cheapest), ``costs`` (``{action: expected cost}``) and ``p_trust``.
        """
        p = self.p_trust
        normal, slow = self._handover_costs(p)
        # asking: pay ask_cost, then hand over optimally given the collapsed answer
        after_yes, after_no = min(self._handover_costs(1.0)), min(self._handover_costs(0.0))
        ask = self.ask_cost + p * after_yes + (1 - p) * after_no
        costs = {'handover': normal, 'slow_handover': slow, 'ask': ask, 'wait': self.wait_cost + min(normal, slow)}
        return {'action': min(costs, key=costs.get), 'costs': costs, 'p_trust': p}


def simulate_handover_session(policy, person, events_rng, n_steps=20):
    """Run a policy against a simulated person whose trust follows a model.

    The person's real trust (a separate copy of the model) determines whether each hand-over succeeds
    and how the trust question is answered; the policy only sees outcomes and answers.

    Parameters
    ----------
    policy : TrustAwareHandover
        The robot's policy (its own belief).
    person : TrustAwareHandover
        The simulated person's true trust state (only its belief is used).
    events_rng : numpy.random.Generator
    n_steps : int, optional

    Returns
    -------
    dict
        ``cost`` (total), ``failures``, ``asks`` and ``actions`` (list).
    """
    total = 0.0
    failures = asks = 0
    actions = []
    for _ in range(n_steps):
        d = policy.decide()
        a = d['action']
        actions.append(a)
        if a == 'ask':
            total += policy.ask_cost
            yes = events_rng.random() < person.p_trust
            person.answer(yes)                      # asking changes the person's trust too
            policy.answer(yes)
            asks += 1
            continue
        if a == 'wait':
            total += policy.wait_cost
            person.wait()
            policy.wait()
            continue
        p_fail = 1 - person.p_trust
        if a == 'slow_handover':
            total += policy.slow_cost
            p_fail *= policy.slow_factor
        ok = events_rng.random() >= p_fail
        total += 0 if ok else policy.fail_cost
        failures += int(not ok)
        person.observe(int(ok))
        policy.observe(int(ok))
    return {'cost': total, 'failures': failures, 'asks': asks, 'actions': actions}
