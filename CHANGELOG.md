# Changelog

## Unreleased

- Every people model gains three tools that the source papers do not provide:
  `quantum_mind.core.bootstrap` (parameter and predictive intervals),
  `quantum_mind.core.OnlinePersonModel` (a per-person posterior updated after every answer, by
  sequential Monte Carlo from the population fit) and `quantum_mind.core.rank_conditions` /
  `model_posterior` (the question or order that best tells competing models apart, after Myung and
  Pitt, 2009). `fit` accepts a starting point `x0`.
- Model atlas: every entry is labelled by origin (unique to Quantum Mind, published model extended
  here, published method, software tool), with counts on the atlas index and a new page, "What
  Quantum Mind adds to published models".
- Citations added: Roeder et al. (2023) for the trust model and Rosenthal, Dey and Veloso (2009) as
  related work for the question planner.

## 2.0.1 (7 October 2026)

Documentation:

- **Foundations**: ten pages that introduce robotics, AI, machine learning, deep learning,
  reinforcement learning, perception and computer vision, embodied AI, quantum computing,
  quantum-like cognition and quantum-inspired algorithms from the basics, each with a diagram, a
  runnable example and references. The documentation home has a tab per topic.
- **Tutorials**: eighteen step-by-step tutorials (`tutorials/*.md`, converted to notebooks and executed
  when the documentation is built), from an introduction to every model group, each with figures and
  a classical baseline. They report where the classical method wins (quantum policy against a linear
  policy, quanvolution against a random filter, TT compression without retraining).
- **Model atlas**: a block diagram for every model (64 diagrams), with its baselines, API, tutorial and
  a reference.
- **Mathematics**: the equations of every model as the code computes them (core, quantum-like
  families, decision layer, perception and fusion, learning, quantum, quantum-inspired), with the
  implementing class or function named next to each equation; linked from every atlas page.
- `quantum_mind.viz.use_mpl_style()` applies the documentation's colour theme to Matplotlib figures.
- CI executes the tutorials.
- **Help**: a glossary, a "which model do I need?" decision tree, an FAQ and a validation page that
  states what has and has not been validated with real people, with a pilot-study protocol.
- New tutorials: an end-to-end robot (calibration, fusion, question planning and trust-aware
  hand-over against a naive robot), the judgement families, and the ROS 2 service.
- Tutorials check their stated results with `assert` statements every time the documentation is
  built; numbers that varied between runs were replaced by qualitative statements.
- Tests keep the documentation in step with the code: the mathematics against the implementations,
  atlas coverage of every public model, tutorial listing and claim checks, the family and example
  counts on the home page, the Foundations examples, and the website's results table.
- Accessibility: every diagram has a title and description for screen readers, and tutorial figures
  have alt text. Diagrams and tables scroll on narrow screens.
- Per-example gallery thumbnails; page descriptions and social-card tags for link previews.
- `docs/_ext/check_refs.py` and a manually triggered workflow check the Foundations references
  against Crossref and report title, year and DOI for review.

Fixes:

- Website and documentation links use the repository's exact name, `Quantum-mind`: GitHub Pages
  addresses are case-sensitive, so the lowercase links in 2.0.0 (including those on the PyPI page)
  returned 404.
- `SplitConformalClassifier` used one order statistic above the conformal quantile
  ceil((n+1)(1-alpha)), giving slightly larger sets than needed; it now uses the exact rank.
- The package docstring's model table was malformed; it now lists all twelve families.
- README and documentation counts (twelve families, 36 examples) are consistent.

Examples:

- 32 calibration and conformal sets, 33 multimodal fusion, 34 bistable perception with the Zeno
  model, 35 reinforcement learning with simulated people (CartPole), 36 trust-aware hand-over.

Website:

- Robot-first landing page with a "New in 2.0" section, results for the 2.0 modules (including the
  cases where a classical method wins), badges and audience cards; the Plotly bundle is served once
  instead of twice.
- PyPI download counts (per month and in total) are shown on the website, the documentation home and
  the README.

## 2.0.0 (6 October 2026)

First release on PyPI.

**New name: Quantum Mind.** The package is now installed with `pip install quantum-mind` and imported
as `import quantum_mind` (previously `qlcog`); the repository is `quantum-mind`. Replace `qlcog` by
`quantum_mind` in imports. The ROS 2 package is now `quantum_mind_ros`.

- **Robot decision layer** (`quantum_mind.applications`): calibration toolkit (`calibration`),
  orchestration gate with meta-learned calibration (`orchestration`), value-of-information question
  planner with order-dependent answer models (`questioning`), personalised human models with partial
  pooling (`personalisation`), trust-aware hand-over policy (`handover`), hazard-aware ask-or-act
  (`robotics.risk_aware_ask_or_act`).
- **Perception**: bistable-perception family (`families.perception`: quantum Zeno, Markov, gamma
  renewal) and multimodal fusion (`applications.fusion`: detector, speech and gaze adapters; Bayesian,
  Dempster-Shafer and quantum-like fusion).
- **Learning**: Gymnasium environments with simulated people (`envs`), exploration strategies for
  tabular reinforcement learning (`inspired.exploration`), a variational quantum policy trained with
  REINFORCE (`quantum.policy`), tensor-train compression of network layers (`inspired.tensor_layers`)
  and a quanvolutional image filter with a classical baseline (`quantum.quanvolution`; on 8 x 8 digits
  it is less accurate than both the random classical filter and the raw pixels).
- New optional extras: `rl` (Gymnasium) and `robotics` (Gymnasium and py_trees).
- New logo and website/documentation branding.

## 1.5.0 (6 October 2026)

Prepared for PyPI as `qlcog`; never uploaded. Superseded by 2.0.0 under the new name.

- **Documentation site (Sphinx).** User guide, an example gallery with the output of every example,
  the executed notebooks and an API reference generated from the docstrings, with the
  pydata-sphinx-theme in the project colours. Built into the GitHub Pages site under `/docs/` and
  ready for Read the Docs (`.readthedocs.yaml`); CI builds it with warnings as errors.
- **Docstrings.** Every public class, function and method now has a NumPy-style docstring with
  parameters, returns and errors; doctests in the docstrings run in CI.
- **Examples** are split into explained steps (`# %%` blocks) and locate the package through
  `sys.argv[0]`, so they run both as scripts and in the gallery.
- README and website: emoji removed, wording made plainer, stale counts updated.
- **Packaging.** SPDX licence metadata (PEP 639), Python-version classifiers, project URLs for the
  website and documentation; README links are absolute so they work on PyPI; install instructions use
  `pip install qlcog`. The release workflow checks that the tag matches the package version.

## 1.4.0 (6 October 2026)

New model families and algorithms (examples 23–31, `tests/test_v14.py`).

- **Quantum-like families.** `families.game_theory`: Eisert–Wilkens–Lewenstein quantum games
  (entangler with D = iσ_y), best responses and Nash checks, including the Benjamin–Hayden result that
  (Q, Q) stops being an equilibrium once all SU(2) strategies are allowed. `families.memory`: quantum
  episodic overdistribution model (Brainerd) against an additive baseline. `families.concepts`: Aerts'
  Fock-space model of concept combination against product, minimum and weighted-average baselines.
- **Applications.** `applications.intent`: Bayesian and quantum-like intent resolvers for robot
  commands; the quantum-like one has order effects controlled by an incompatibility angle and
  reduces to Bayes when the angle is zero.
- **Quantum.** Quantum Fourier transform and period finding; maximum-likelihood amplitude estimation
  with a Qiskit export; continuous-time quantum walks and walk centrality; `VariationalRegressor`;
  quantum-kernel anomaly detection and spectral clustering (with RBF baselines); controlled-phase gate
  `cp` in the simulator, with adjoint gradients.
- **Quantum-inspired.** Quantum reinforcement learning (Dong et al. 2008, with a bounded TD-error
  rotation) against Q-learning on a grid world; quantum language model (Sordoni et al. 2013) against
  query likelihood with Dirichlet smoothing.
- Results are reported as measured, including where the classical baseline wins (forecasting, anomaly
  detection, clustering).
- Project website (`site/`, deployed to GitHub Pages by `.github/workflows/pages.yml`): landing page, an
  in-browser two-qubit playground (exact state-vector simulation; checked against the package's
  simulator in `tests/test_site.py`), interactive animations, documentation pages, executed notebooks,
  social preview card, sitemap. plotly.js is self-hosted.

## 1.3.0 (6 October 2026)

Closes the gaps listed as "not yet" in 1.2.0.

- **IBM Quantum.** Jobs use the client-side Qiskit Runtime Sampler (`executor_sampler.Sampler`), with a
  fallback to `SamplerV2` on releases before 0.50; the deprecation warning is gone.
- **QIEA.** Han and Kim's rotation lookup table (Table I, adapted to minimisation) is now the default;
  the simplified fixed-step rule remains as `rotation='simple'` (exact optimum on 27/30 vs 26/30 random
  8–14-variable problems).
- **MPS classifier.** `method='sweep'`: DMRG-style two-site sweeps with SVD truncation that adapt the
  bond dimensions and move the label index (Stoudenmire and Schwab); `bond_dimensions` property. Adam
  training stays the default.
- **Viewer.** `concurrence`, `reduced_density_pair`, `entanglement_summary`, `Trajectory.concurrence`,
  `plot_entanglement` (Bloch lengths and concurrence along a circuit) and `plot_qsphere`; concurrence
  is checked against Qiskit. `seaborn` added to `[viz]` (needed by Qiskit's Q-sphere).
- **Individual differences.** `fit_individuals`, `compare_individuals` and `IndividualFits`; example 22
  shows a pooled fit selecting the quantum-like model for a half-anchoring population, and per-person
  fits recovering every person's model at about 3,000 answers each.
- **Robotics and ROS 2.** `HumanModelService` (answers in, prediction + uncertainty + ask/act out, JSON
  messages) and the `qlcog_ros` ROS 2 package in `integrations/ros2` (node, launch file); the node's
  callbacks are tested with stand-in `rclpy` / `std_msgs` modules, not yet in a ROS 2 installation.
- 7 new tests (51 in total), about 90% line coverage.

## 1.2.0 (5 October 2026)

Fixes from a critical audit of 1.1.0. Backward compatible, except that QAOA results now carry
`p_optimal` and `weights` as regular fields.

- **Scaling.** Exact adjoint-method gradients (`Circuit.value_and_grad`): one forward and one backward
  pass whatever the number of weights, samples simulated in chunks. The variational classifier on
  8 features × 400 samples now trains in about 80 s (151 steps, 85 MB); 1.1.0 did not finish three
  steps in five minutes. VQE and QAOA use the same gradients. Size limit of 22 qubits with a clear
  error.
- **QAOA.** L-BFGS with exact gradients plus a COBYLA polish (optimum found on 16/16 random
  8-variable instances, 14/16 without the polish).
- **Grover.** `to_qiskit()` exports a gate-level oracle and diffuser (X and multi-controlled Z);
  measured CNOT counts per iteration documented against the diagonal export.
- **Hardware paths.** IBM and Braket submission code split into `run_on_ibm_backend` and
  `run_on_braket_device` and tested against a fake IBM device and the Braket local simulator. The
  "ready" badges were replaced by a status table; no hardware results are claimed.
- **Fairer comparisons.** Example 13 runs 10 seeds (mean ± sd) and adds a quadratic-feature logistic
  regression, which is the most accurate model on that task. QIEA and the MPS classifier are
  documented as simplified variants of the cited algorithms.
- **Robotics.** `ask_or_act` decision rule and example 21 (when to ask for help); README states what
  the robotics part does and does not include (no ROS 2 package, no human-subject data).
- **Notebooks.** `notebooks/01_bloch_sphere_live.ipynb` and `02_robot_questioning_and_trust.ipynb`,
  executed in CI. Running them found that Plotly 6 widgets need `anywidget`, now in `[viz]`.
- **Robustness.** Input validation with clear errors (shapes, NaN, one class, sizes); missing optional
  dependencies name the pip extra to install; 9 new tests (44 in total), about 90% line coverage,
  80% enforced in CI.
- **Documentation.** `docs/API.md` generated from docstrings (every public item documented; CI checks
  it is current); status and limits section in the README.
- **Project.** Author and maintainer email yeshwanth445@gmail.com; SECURITY.md; .zenodo.json for a DOI;
  release workflow for PyPI trusted publishing; CI builds and checks the wheel and installs it in a
  clean environment; README media reduced from 6.5 MB to 2.9 MB (interactive HTML is generated by the
  examples instead of committed); static test-count badge removed.
- Known issue: Qiskit Runtime `SamplerV2` is deprecated as of qiskit-ibm-runtime 0.50.

## 1.1.0 (5 October 2026)

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

## 1.0.0 (5 October 2026)

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
