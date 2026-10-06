# Quantum models (`quantum_mind.quantum`)

Gate-model quantum machine learning and quantum algorithms. Every model is trained or optimised on a
fast built-in state-vector simulator (pure NumPy, batched over samples and over parameter shifts), and
every circuit exports to Qiskit with `to_qiskit()`, so it runs unchanged on Aer, on IBM Quantum and on
Amazon Braket through `quantum_mind.circuits.run`.

| Model | What it does | Typical use | Reference |
|---|---|---|---|
| `VariationalClassifier` | Data re-uploading classifier, any number of classes, exact adjoint gradients | Classification from a few features (triage, user intent, acceptance of a robot's offer) | Pérez-Salinas et al., *Quantum* 4, 226 (2020) |
| `QuantumKernel`, `QuantumKernelClassifier` | Fidelity kernel of the ZZ feature map; kernel ridge classifier; kernel matrix for any kernel method | Small-data classification; kernel for scikit-learn `SVC(kernel='precomputed')` | Havlíček et al., *Nature* 567, 209 (2019) |
| `QAOA` | Quantum Approximate Optimisation Algorithm for any `quantum_mind.problems.Qubo`; schedule search by depth-1 grid, layer-wise interpolation (INTERP) and random restarts | Task allocation, scheduling, MaxCut, portfolios | Farhi, Goldstone and Gutmann, arXiv:1411.4028 (2014) |
| `VQE`, `Hamiltonian` | Variational Quantum Eigensolver for weighted Pauli strings (or a QUBO as an Ising Hamiltonian) | Ground states of spin models and small molecules | Peruzzo et al., *Nat. Commun.* 5, 4213 (2014) |
| `VariationalRegressor` | Re-uploading regressor (prediction = scaled ⟨Z₀⟩), exact adjoint gradients | Small forecasting tasks from a window of past values | Pérez-Salinas et al. (2020); Mitarai et al., *PRA* 98, 032309 (2018) |
| `QuantumKernelAnomalyDetector`, `QuantumKernelClustering` | Distance to the mean feature vector; spectral clustering; `kernel='rbf'` gives the classical baseline with the same rule | Fraud and fault detection, grouping | Schuld and Killoran, *PRL* 122, 040504 (2019) |
| `qft_circuit`, `find_period` | Quantum Fourier transform (H, controlled phase, swap); period finding | Signal processing, the core of Shor's algorithm | Nielsen and Chuang, ch. 5 |
| `AmplitudeEstimation`, `monte_carlo_estimate` | Maximum-likelihood amplitude estimation of E[f] (no phase-estimation register), Monte Carlo baseline with equal calls | Risk and pricing | Suzuki et al., *QIP* 19, 75 (2020); Woerner and Egger, *npj QI* 5, 15 (2019) |
| `ctqw_probabilities`, `quantum_walk_centrality`, `pagerank` | Continuous-time quantum walk on a graph; centrality from its long-time average; PageRank and degree baselines | Network analysis | Farhi and Gutmann (1998); Izaac et al., *PRA* 95, 032318 (2017) |
| `grover` | Grover search with a phase oracle from a list of solutions or a predicate; exports a gate-level circuit (X and multi-controlled Z) or a diagonal-gate circuit | Constraint satisfaction, unstructured search | Grover, *STOC* (1996) |

Building blocks: `Circuit` (gates `h x y z s sdg rx ry rz p cx cz swap rzz`, `unitary`, `diagonal`;
angles are numbers, trainable weights `W(k)`, features `X(j)` or products `XX(i, j)`),
`angle_encoding`, `zz_feature_map`, `hardware_efficient` (CZ or CX entanglers, ring or linear),
`parameter_shift`, `shifted_weights`, and `Circuit.value_and_grad` (adjoint-method gradients: one
forward and one backward pass whatever the number of weights, samples processed in chunks).

**Sizes.** Up to 22 qubits are accepted. Training cost grows as 2ⁿ: 8 features × 400 samples train in
about 80 s on a laptop CPU. Grover's gate-level export needs about 360 CNOTs per iteration at 8 qubits
and 660 at 10 (diagonal export: 510 and 2,040, growing exponentially); use `style='diagonal'` only
for small cases.

```python
from quantum_mind.quantum import VariationalClassifier, QAOA, VQE, Hamiltonian, grover
from quantum_mind.problems import task_allocation
from quantum_mind.circuits import run

clf = VariationalClassifier(layers=3).fit(X_train, y_train)
print(clf.score(X_test, y_test))
counts = run(clf.to_qiskit(X_test[0]), 'aer:FakeTorino', shots=4000)    # same circuit, IBM noise model

res = QAOA(task_allocation(costs), p=3).run()                          # res.x, res.energy, res.p_optimal
vqe = VQE(Hamiltonian([(-1, 'ZZ'), (-0.5, 'XI'), (-0.5, 'IX')]), rotations=('ry',)).run()
g = grover(5, lambda x: sum(x) == 2)                                   # g.success, g.to_qiskit()
```

**Verification.** The simulator is tested gate by gate against Qiskit's `Statevector`; the adjoint and
parameter-shift gradients against finite differences; the exported QAOA and Grover circuits against
the simulated distributions; and a trained classifier circuit against an Aer run
(`tests/test_quantum.py`, `tests/test_robustness.py`).

**Measured in the examples (simulated data).** Amplitude estimation: RMSE 0.0014 against 0.0049 for
Monte Carlo with the same 6,800 calls (50 runs). Forecasting R²: 0.962 against 0.984 for linear
autoregression. Anomaly AUC: 0.905 with the quantum kernel, 1.000 with the RBF kernel; with angle
encoding, inputs outside the training range wrap around (the encoding is periodic), so out-of-range
anomalies can look normal. Clustering accuracy 0.931 against 1.000.

**What to expect.** These are small, exactly simulable models (up to about 12–16 qubits on a
laptop). There is no proven quantum advantage for them on classical data, and at low depth QAOA puts
only a few percent of its probability on the optimum of a 9-variable problem
(`examples/14_qaoa_multi_robot_allocation.py`). They belong in the package so that
quantum, quantum-like and quantum-inspired approaches can be compared on the same tasks, and so the
circuits can be studied on real hardware.
