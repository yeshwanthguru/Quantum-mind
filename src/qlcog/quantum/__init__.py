"""Quantum models: quantum machine learning and quantum algorithms (gate-model circuits).

* :mod:`~qlcog.quantum.statevector`: batched NumPy simulator for parameterised circuits, adjoint
  gradients, ``to_qiskit()`` export.
* :mod:`~qlcog.quantum.ansatz`: angle encoding, ZZ feature map, hardware-efficient ansatz,
  parameter-shift gradient.
* :mod:`~qlcog.quantum.classifiers`: variational classifier and regressor, quantum kernel, kernel
  classifier, anomaly detection and clustering (with RBF baselines).
* :mod:`~qlcog.quantum.algorithms`: QAOA (for :class:`qlcog.problems.Qubo`), VQE (Pauli
  Hamiltonians), Grover search.
* :mod:`~qlcog.quantum.fourier`: quantum Fourier transform and period finding.
* :mod:`~qlcog.quantum.estimation`: amplitude estimation (maximum likelihood) for expected values,
  with a Monte Carlo baseline.
* :mod:`~qlcog.quantum.walks`: continuous-time quantum walks and quantum-walk centrality, with
  PageRank and degree baselines.

Everything runs on the built-in simulator; circuits export to Qiskit for Aer, IBM Quantum or Amazon
Braket. See README.md in this folder.
"""
from .statevector import Circuit, W, X, XX, apply_matrix, expectation_z
from .ansatz import angle_encoding, zz_feature_map, hardware_efficient, reuploading_classifier_circuit, parameter_shift
from .classifiers import (VariationalClassifier, VariationalRegressor, QuantumKernel, QuantumKernelClassifier,
                          QuantumKernelAnomalyDetector, QuantumKernelClustering, rbf_kernel)
from .fourier import qft_circuit, qft_matrix, periodic_state, find_period
from .estimation import AmplitudeEstimation, AEResult, monte_carlo_estimate
from .walks import adjacency, ctqw_probabilities, quantum_walk_centrality, pagerank, degree_centrality
from .algorithms import QAOA, QAOAResult, Hamiltonian, VQE, pauli_matrix, grover, GroverResult

__all__ = ['Circuit', 'W', 'X', 'XX', 'apply_matrix', 'expectation_z', 'angle_encoding', 'zz_feature_map',
           'hardware_efficient', 'reuploading_classifier_circuit', 'parameter_shift', 'VariationalClassifier',
           'QuantumKernel', 'QuantumKernelClassifier', 'QAOA', 'QAOAResult', 'Hamiltonian', 'VQE', 'pauli_matrix',
           'grover', 'GroverResult', 'VariationalRegressor', 'QuantumKernelAnomalyDetector', 'QuantumKernelClustering',
           'rbf_kernel', 'qft_circuit', 'qft_matrix', 'periodic_state', 'find_period', 'AmplitudeEstimation', 'AEResult',
           'monte_carlo_estimate', 'adjacency', 'ctqw_probabilities', 'quantum_walk_centrality', 'pagerank',
           'degree_centrality']
