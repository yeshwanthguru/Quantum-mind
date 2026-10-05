"""Quantum models: quantum machine learning and quantum algorithms (gate-model circuits).

  statevector   batched NumPy simulator for parameterised circuits; to_qiskit() export
  ansatz        angle encoding, ZZ feature map, hardware-efficient ansatz, parameter-shift gradient
  classifiers   VariationalClassifier (data re-uploading), QuantumKernel, QuantumKernelClassifier
  algorithms    QAOA (for qlcog.problems.Qubo), VQE (Pauli Hamiltonians), Grover search

See README.md in this folder."""
from .statevector import Circuit, W, X, XX, apply_matrix, expectation_z
from .ansatz import angle_encoding, zz_feature_map, hardware_efficient, reuploading_classifier_circuit, parameter_shift
from .classifiers import VariationalClassifier, QuantumKernel, QuantumKernelClassifier
from .algorithms import QAOA, QAOAResult, Hamiltonian, VQE, pauli_matrix, grover, GroverResult

__all__ = ['Circuit', 'W', 'X', 'XX', 'apply_matrix', 'expectation_z', 'angle_encoding', 'zz_feature_map',
           'hardware_efficient', 'reuploading_classifier_circuit', 'parameter_shift', 'VariationalClassifier',
           'QuantumKernel', 'QuantumKernelClassifier', 'QAOA', 'QAOAResult', 'Hamiltonian', 'VQE', 'pauli_matrix',
           'grover', 'GroverResult']
