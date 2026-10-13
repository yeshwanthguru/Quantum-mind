"""Quantum cognition as a trainable model: a density-matrix belief with context-dependent measurements, and its classical competitors.

Needs the optional ``cognition`` extra (JAX and Optax). See :mod:`quantum_mind.cognition.core`.
"""
from .core import (BayesCore, ContextWeightedCore, GRUCore, MarkovContextCore, QuantumCognitionCore, hermitian, measurement,
                   update_state)

from . import corpus  # noqa: E402
from .bridge import BridgeResult, LLMBridge, belief_from_parse, llm_parse, normalise  # noqa: E402
from .dialogue import DialogueBelief, make_dialogues  # noqa: E402
from .language import RobotLanguageModel, language_plan  # noqa: E402

__all__ = ['QuantumCognitionCore', 'BayesCore', 'ContextWeightedCore', 'MarkovContextCore', 'GRUCore', 'hermitian', 'measurement',
           'update_state', 'RobotLanguageModel', 'language_plan', 'corpus', 'LLMBridge', 'BridgeResult', 'normalise', 'llm_parse',
           'belief_from_parse', 'DialogueBelief', 'make_dialogues']
