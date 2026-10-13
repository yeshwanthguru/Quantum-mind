"""Quantum cognition as a trainable model: a density-matrix belief with context-dependent measurements, and its classical competitors.

Needs the optional ``cognition`` extra (JAX and Optax). See :mod:`quantum_mind.cognition.core`.
"""
from .core import (BayesCore, ContextWeightedCore, GRUCore, MarkovContextCore, QuantumCognitionCore, hermitian, measurement,
                   update_state)

__all__ = ['QuantumCognitionCore', 'BayesCore', 'ContextWeightedCore', 'MarkovContextCore', 'GRUCore', 'hermitian', 'measurement',
           'update_state']
