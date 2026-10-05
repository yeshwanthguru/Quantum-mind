# Changelog

## 1.1.0 — 5 October 2026

Three pillars in one package: quantum-like, quantum and quantum-inspired, plus a 3D Bloch-sphere
viewer. Backward compatible with 1.0.0.

- **Quantum (`qlcog.quantum`)**: batched NumPy state-vector simulator for parameterised circuits
  (checked gate by gate against Qiskit), `to_qiskit()` export; angle encoding, ZZ feature map,
  hardware-efficient ansatz (CZ or CX entanglers); exact parameter-shift gradients evaluated in one
  batch; `VariationalClassifier` (data re-uploading, multi-class), `QuantumKernel` and
  `QuantumKernelClassifier`, `QAOA` for any QUBO (schedule search: depth-1 grid, INTERP layer interpolation
  of Zhou et al. 2020, random restarts), `VQE` with `Hamiltonian` (Pauli strings or a QUBO),
  `grover`.
- **Quantum-inspired (`qlcog.inspired`)**: `QIEA`, `QPSO`, `SQA` (path-integral Monte Carlo),
  `simulated_annealing` baseline, `MPSClassifier` (matrix product state with polynomial local feature
  maps and exact gradients).
- **Problems (`qlcog.problems`)**: `Qubo` with energies, brute force and Ising conversion; builders for
  MaxCut, knapsack, multi-robot task allocation, portfolio selection and Ising models.
- **Viewer (`qlcog.viz`)**: Bloch vectors of every qubit (reduced states), `circuit_trajectory` for any
  Qiskit circuit, `belief_trajectory` for the trust model, `bloch_tomography` on any backend;
  Matplotlib spheres and GIF/MP4 animation; Plotly interactive figures and HTML animations;
  `LiveBloch` for real-time updates in Jupyter or a window; dark and light themes.
- Eight new examples (13–20) in medicine, robotics, finance, physics, scheduling, control and
  visualisation; 16 new tests (35 in total); concepts page; README with banner, animations and
  catalogue; `docs/make_assets.py` regenerates every figure from the package.
- Extras: `[viz]`; `[dev]` adds ruff. CI installs the viewer, runs the tests, lints and smoke-tests
  the examples.

## 1.0.0 — 5 October 2026

First release.

- Core: projectors, Lüders rule, answer sequences, density matrices and Lindblad dynamics; `Model`
  base class with typed parameters; multi-start fitting over continuous parameters and discrete
  structures; BIC/AIC comparison; model-recovery studies.
- Families: order effects (3D and 4D quantum-like models, Bayes, anchoring, saturated, QQ test);
  conjunction and disjunction judgements (quantum-like, classical, averaging, probability theory plus
  noise); interference in decisions (normalised and unnormalised quantum-like law, classical mixture);
  quantum-like Bayesian networks; belief dynamics (Markov, quantum and open-system walks; Markov and
  open-system belief updating); quantum decision theory (with expected utility and prospect theory);
  contextuality (CHSH, Contextuality-by-Default for cyclic systems); similarity (quantum, biased and
  symmetric geometric).
- Circuits for every family except decision, in dynamic and deferred-measurement forms; `run()` for
  Aer, IBM fake-backend noise models, Braket local simulators, IBM Quantum and Amazon Braket hardware.
- Robotics application: human-robot question domains, questioning designs, trust protocol and an
  evidence-weighted human-model ensemble with uncertainty output.
- Published aggregate data sets, twelve domain examples and 19 tests; continuous integration on
  Python 3.10-3.12.
- Licence: Apache-2.0.
