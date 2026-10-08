"""quantum_mind: quantum-like, quantum and quantum-inspired models, with classical baselines, Qiskit circuits
and a 3D Bloch-sphere viewer.

The package also includes model-agnostic evaluation and reproducibility tools in
quantum_mind.evaluation.
"""
__version__ = '2.0.1'
from .core import fit, compare, recovery, Model, Param, bootstrap, OnlinePersonModel  # noqa: F401
