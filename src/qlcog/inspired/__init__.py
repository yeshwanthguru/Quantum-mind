"""Quantum-inspired models: classical algorithms that borrow quantum ideas. No quantum computer is used.

  optimisers   QIEA (binary evolutionary), QPSO (continuous swarm), SQA (simulated quantum annealing),
               simulated_annealing (classical baseline)
  tensor       MPSClassifier (matrix product state / tensor-network classifier)

See README.md in this folder."""
from .optimisers import OptimResult, QIEA, QPSO, SQA, simulated_annealing
from .tensor import MPSClassifier

__all__ = ['OptimResult', 'QIEA', 'QPSO', 'SQA', 'simulated_annealing', 'MPSClassifier']
