"""qlcog: quantum-like models of cognition and decision, with classical baselines and Qiskit circuits.

Subpackages
  qlcog.core         linear algebra, Model base class, fitting, comparison, model recovery
  qlcog.families     eight model families (order_effects, conjunction, interference, qlbn,
                     dynamics, decision, contextuality, similarity)
  qlcog.circuits     Qiskit circuits for the families and run() for simulators and cloud hardware
  qlcog.data         published aggregate data sets
"""
__version__ = '1.0.0'
from .core import fit, compare, recovery, Model, Param   # noqa: F401
