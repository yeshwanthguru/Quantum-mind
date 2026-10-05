"""qlcog: quantum-like, quantum and quantum-inspired models, with classical baselines, Qiskit circuits
and a 3D Bloch-sphere viewer.

Subpackages
  qlcog.core          linear algebra, Model base class, fitting, comparison, model recovery
  qlcog.families      quantum-like models of judgement and decision: eight families (order_effects,
                      conjunction, interference, qlbn, dynamics, decision, contextuality, similarity)
  qlcog.applications  robotics: question domains, questioning designs, trust, human-model ensemble
  qlcog.quantum       quantum machine learning and algorithms (VQC, quantum kernel, QAOA, VQE, Grover)
  qlcog.inspired      quantum-inspired algorithms (QIEA, QPSO, simulated quantum annealing, MPS classifier)
  qlcog.problems      QUBO builders shared by the quantum and quantum-inspired solvers
  qlcog.circuits      Qiskit circuits for the families and run() for simulators and cloud hardware
  qlcog.viz           Bloch-sphere viewer: trajectories, animations, interactive HTML, live updates
  qlcog.data          published aggregate data sets
"""
__version__ = '1.1.0'
from .core import fit, compare, recovery, Model, Param   # noqa: F401
