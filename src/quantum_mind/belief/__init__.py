"""Belief filters that keep a robot's belief about a person as a density matrix.

* :class:`COBRA`: a context-rotated density-matrix belief filter. The previous sensor frame rotates the
  measurement basis of the next one, so the meaning of a reading depends on what came before it.
* :class:`OnlineBelief`: the same filter run one frame at a time, as a robot would.
* :class:`GaussianBayesFilter`: the commutative classical reference.
* :func:`min_cost_action`: turn a belief and a cost matrix into an action, with an "ask" row to hedge.

See README.md in this folder and ``benchmarks/context_intent_benchmark.py`` for an honest comparison.
"""
from .cobra import COBRA, BeliefTrace, GaussianBayesFilter, OnlineBelief, min_cost_action

__all__ = ['COBRA', 'BeliefTrace', 'GaussianBayesFilter', 'OnlineBelief', 'min_cost_action']
