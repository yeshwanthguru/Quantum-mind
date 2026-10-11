"""Quantum state-vector simulator used by the Bloch-sphere viewer and the circuit exports.

* :mod:`~quantum_mind.quantum.statevector`: batched NumPy simulator for parameterised circuits, adjoint
  gradients and ``to_qiskit()`` export, so a model's circuit can run on Aer, IBM Quantum or Amazon Braket.

This is infrastructure, not a model. The general-purpose quantum algorithms (VQE, QAOA, Grover,
quantum kernels and the like) are intentionally left to Qiskit and PennyLane.
"""
from .statevector import Circuit, W, X, XX, apply_matrix, expectation_z

__all__ = ['Circuit', 'W', 'X', 'XX', 'apply_matrix', 'expectation_z']
