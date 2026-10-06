"""Quantum-inspired models: classical algorithms that borrow quantum ideas. No quantum computer is used.

* :mod:`~quantum_mind.inspired.optimisers`: QIEA (binary evolutionary), QPSO (continuous swarm), SQA
  (simulated quantum annealing) and the classical ``simulated_annealing`` baseline.
* :mod:`~quantum_mind.inspired.tensor`: :class:`MPSClassifier`, a matrix product state (tensor-network)
  classifier.
* :mod:`~quantum_mind.inspired.rl`: :class:`QuantumInspiredQLearning` (Dong et al., 2008) on a
  :class:`GridWorld`, with a :class:`QLearning` baseline.
* :mod:`~quantum_mind.inspired.text`: :class:`QuantumLanguageModel` (density-matrix document ranking), with
  a :class:`QueryLikelihoodModel` baseline.
* :mod:`~quantum_mind.inspired.exploration`: amplitude (Born-rule) exploration for tabular agents, with
  epsilon-greedy, Boltzmann and UCB baselines.
* :mod:`~quantum_mind.inspired.tensor_layers`: tensor-train compression of network layers.

See README.md in this folder.
"""
from .optimisers import OptimResult, QIEA, QPSO, SQA, simulated_annealing
from .tensor import MPSClassifier
from .rl import GridWorld, QuantumInspiredQLearning, QLearning, train
from .text import tokenize, QuantumLanguageModel, QueryLikelihoodModel
from .exploration import EpsilonGreedy, Boltzmann, UCB, AmplitudeExploration, TabularAgent, StateIndexer, run_episodes
from .tensor_layers import TTMatrix, factorise, compress_layers, apply_mlp

__all__ = ['OptimResult', 'QIEA', 'QPSO', 'SQA', 'simulated_annealing', 'MPSClassifier', 'GridWorld',
           'QuantumInspiredQLearning', 'QLearning', 'train', 'tokenize', 'QuantumLanguageModel', 'QueryLikelihoodModel',
           'EpsilonGreedy', 'Boltzmann', 'UCB', 'AmplitudeExploration', 'TabularAgent', 'StateIndexer', 'run_episodes',
           'TTMatrix', 'factorise', 'compress_layers', 'apply_mlp']
