"""qlcog: quantum-like, quantum and quantum-inspired models, with classical baselines, Qiskit circuits
and a 3D Bloch-sphere viewer.

Subpackages
  qlcog.core          linear algebra, Model base class, fitting, comparison, model recovery
  qlcog.families      quantum-like models of judgement and decision: eleven families (order_effects,
                      conjunction, interference, qlbn, dynamics, decision, contextuality, similarity,
                      game_theory, memory, concepts)
  qlcog.applications  robotics: question domains, questioning designs, trust, intent resolution,
                      human-model ensemble
  qlcog.quantum       quantum machine learning and algorithms (VQC, regressor, quantum kernel classifier,
                      anomaly detection and clustering, QAOA, VQE, Grover, QFT, amplitude estimation,
                      quantum walks)
  qlcog.inspired      quantum-inspired algorithms (QIEA, QPSO, simulated quantum annealing, MPS classifier,
                      quantum reinforcement learning, quantum language model)
  qlcog.problems      QUBO builders shared by the quantum and quantum-inspired solvers
  qlcog.circuits      Qiskit circuits for the families and run() for simulators and cloud hardware
  qlcog.viz           Bloch-sphere viewer: trajectories, animations, interactive HTML, live updates
  qlcog.data          published aggregate data sets
"""
__version__ = '1.4.0'
from .core import fit, compare, recovery, Model, Param   # noqa: F401
