"""Quantum models: quantum machine learning and quantum algorithms (gate-model circuits).

  statevector   batched NumPy simulator for parameterised circuits; to_qiskit() export
  ansatz        angle encoding, ZZ feature map, hardware-efficient ansatz, parameter-shift gradient
  classifiers   VariationalClassifier (data re-uploading), QuantumKernel, QuantumKernelClassifier
  algorithms    QAOA (for qlcog.problems.Qubo), VQE (Pauli Hamiltonians), Grover search
  fourier       quantum Fourier transform, period finding
  estimation    amplitude estimation (maximum likelihood) for expected values / risk; Monte Carlo baseline
  walks         continuous-time quantum walks, quantum-walk centrality; PageRank and degree baselines
  classifiers   also VariationalRegressor, QuantumKernelAnomalyDetector, QuantumKernelClustering

See README.md in this folder."""
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
