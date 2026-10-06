"""Quantum-inspired models: classical algorithms that borrow quantum ideas. No quantum computer is used.

  optimisers   QIEA (binary evolutionary), QPSO (continuous swarm), SQA (simulated quantum annealing),
               simulated_annealing (classical baseline)
  tensor       MPSClassifier (matrix product state / tensor-network classifier)
  rl           QuantumInspiredQLearning (Dong et al. 2008) on a GridWorld; QLearning baseline
  text         QuantumLanguageModel (density-matrix document ranking); QueryLikelihoodModel baseline

See README.md in this folder."""
from .optimisers import OptimResult, QIEA, QPSO, SQA, simulated_annealing
from .tensor import MPSClassifier
from .rl import GridWorld, QuantumInspiredQLearning, QLearning, train
from .text import tokenize, QuantumLanguageModel, QueryLikelihoodModel

__all__ = ['OptimResult', 'QIEA', 'QPSO', 'SQA', 'simulated_annealing', 'MPSClassifier', 'GridWorld',
           'QuantumInspiredQLearning', 'QLearning', 'train', 'tokenize', 'QuantumLanguageModel', 'QueryLikelihoodModel']
