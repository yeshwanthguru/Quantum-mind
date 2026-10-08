"""Hand-over policies and simulated people.

The robot hands an object to a person. A hand-over fails with probability ``1 - P(trust)`` (half that
when the robot hands over slowly), and the person's trust moves after every success and failure. The
robot chooses, before each hand-over, between handing over, handing over slowly, asking "Are you
ready?" and waiting, by expected cost (:class:`quantum_mind.applications.handover.TrustAwareHandover`).

Two robot policies differ only in their model of trust, both set to population averages rather than
to the person in front of the robot:

* ``quantum-like``: trust as a qubit (:class:`~quantum_mind.families.dynamics.OpenSystemBelief`);
  asking changes the person's trust as well as revealing it;
* ``markov``: trust as a probability that moves up after a success and down after a failure
  (:class:`~quantum_mind.families.dynamics.MarkovBelief`); asking only reads it.

Two fixed baselines (always hand over, always hand over slowly) need no model.
"""
from __future__ import annotations

from quantum_mind.applications.handover import TrustAwareHandover
from quantum_mind.families.dynamics import OpenSystemBelief, MarkovBelief

__all__ = ['POLICIES', 'make_policy', 'quantum_like_people', 'markov_people', 'person_state']

#: Population-average trust models the robot starts from (illustrative values, not fitted to people).
QUANTUM_LIKE_POPULATION = dict(phi0=1.45, a_pos=0.8, a_neg=1.2, gamma=0.3)
MARKOV_POPULATION = dict(p0=0.55, up=0.3, down=0.4)


class AlwaysHandOver(TrustAwareHandover):
    """Baseline: hand over at normal speed every time."""

    def decide(self):
        return {'action': 'handover', 'costs': {}, 'p_trust': self.p_trust}


class AlwaysSlow(TrustAwareHandover):
    """Baseline: hand over slowly every time."""

    def decide(self):
        return {'action': 'slow_handover', 'costs': {}, 'p_trust': self.p_trust}


#: Policy names, in report order.
POLICIES = ('quantum-like', 'markov', 'always hand over', 'always slow')


def make_policy(name, **costs):
    """A fresh robot policy.

    Parameters
    ----------
    name : str
        One of :data:`POLICIES`.
    **costs
        ``fail_cost``, ``slow_cost``, ``ask_cost``, ``wait_cost`` and ``slow_factor`` of
        :class:`~quantum_mind.applications.handover.TrustAwareHandover`.

    Returns
    -------
    TrustAwareHandover
    """
    if name == 'quantum-like':
        return TrustAwareHandover(OpenSystemBelief(**QUANTUM_LIKE_POPULATION), **costs)
    if name == 'markov':
        return TrustAwareHandover(MarkovBelief(**MARKOV_POPULATION), **costs)
    if name == 'always hand over':
        return AlwaysHandOver(OpenSystemBelief(**QUANTUM_LIKE_POPULATION), **costs)
    if name == 'always slow':
        return AlwaysSlow(OpenSystemBelief(**QUANTUM_LIKE_POPULATION), **costs)
    raise ValueError('unknown policy %r; choose from %s' % (name, ', '.join(POLICIES)))


def quantum_like_people(n, rng):
    """Simulated people whose trust follows the quantum-like model, with spread-out parameters.

    Parameters
    ----------
    n : int
    rng : numpy.random.Generator

    Returns
    -------
    list of OpenSystemBelief
    """
    return [OpenSystemBelief(phi0=rng.uniform(0.3, 2.6), a_pos=rng.uniform(0.4, 1.2),
                             a_neg=rng.uniform(0.8, 1.6), gamma=rng.uniform(0.1, 0.5)) for _ in range(n)]


def markov_people(n, rng):
    """Simulated people whose trust follows the classical Markov model, with spread-out parameters.

    Parameters
    ----------
    n : int
    rng : numpy.random.Generator

    Returns
    -------
    list of MarkovBelief
    """
    return [MarkovBelief(p0=rng.uniform(0.2, 0.9), up=rng.uniform(0.1, 0.5), down=rng.uniform(0.2, 0.6))
            for _ in range(n)]


def person_state(model):
    """The simulated person's true trust state (a policy object whose belief is the person's trust)."""
    return TrustAwareHandover(model)
