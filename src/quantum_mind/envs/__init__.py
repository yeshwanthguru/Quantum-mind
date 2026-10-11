"""Reinforcement-learning environments with simulated people (Gymnasium API).

* :class:`ClarificationEnv`: a robot resolves an ambiguous request by asking yes/no questions whose
  answers show order effects, then acts.
* :class:`TrustHandoverEnv`: a robot hands objects to a person whose trust follows the trust qubit.

The environments subclass ``gymnasium.Env`` when Gymnasium is installed (``pip install "quantum-mind[rl]"``)
and otherwise provide the same ``reset`` / ``step`` interface. :func:`register_envs` registers them as
``quantum_mind/Clarification-v0`` and ``quantum_mind/TrustHandover-v0``.
:func:`sample_people` draws a population of simulated people from a fitted model's uncertainty, so an
agent can be trained against many plausible people rather than one.
"""
from .hri import ClarificationEnv, TrustHandoverEnv, register_envs, sample_people

__all__ = ['ClarificationEnv', 'TrustHandoverEnv', 'register_envs', 'sample_people']
