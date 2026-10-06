"""Reinforcement-learning environments with simulated people (Gymnasium API).

* :class:`ClarificationEnv`: a robot resolves an ambiguous request by asking yes/no questions whose
  answers show order effects, then acts.
* :class:`TrustHandoverEnv`: a robot hands objects to a person whose trust follows the trust qubit.
* :class:`GridWorldEnv`: the grid world of :mod:`quantum_mind.inspired.rl` with the Gymnasium interface.

The environments subclass ``gymnasium.Env`` when Gymnasium is installed (``pip install "quantum-mind[rl]"``)
and otherwise provide the same ``reset`` / ``step`` interface. :func:`register_envs` registers them as
``quantum_mind/Clarification-v0``, ``quantum_mind/TrustHandover-v0`` and ``quantum_mind/GridWorld-v0``.
"""
from .hri import ClarificationEnv, TrustHandoverEnv, GridWorldEnv, register_envs

__all__ = ['ClarificationEnv', 'TrustHandoverEnv', 'GridWorldEnv', 'register_envs']
