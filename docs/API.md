# API reference

Generated from the docstrings by `docs/make_api.py`; each entry shows the signature and the first
paragraph of the docstring. Full docstrings: `help(qlcog.<module>.<name>)`.

## `qlcog.core`

Core layer: linear algebra of quantum-like models, the Model base class, fitting and comparison.

- `angles_to_unit(angles)`: Unit vector from hyperspherical angles.
- `compare(candidates, data, design=None, restarts=8, rng=None, criterion='bic')`: Fit several model classes to the same data and rank them.
- `compare_individuals(candidates, data_by_person, design=None, restarts=4, rng=None, criterion='bic')`: Fit every candidate to every person and compare them.
- `complement(P)`: Projector onto the orthogonal complement, :math:`I - P`.
- `density(psi)`: Density matrix of a pure state.
- `dephase(rho, strength, basis_projectors=None)`: Partial dephasing channel.
- `evolve_density(rho, superop, t)`: Evolve a density matrix under a Lindblad generator.
- `fit(model_cls, data, design=None, restarts=8, rng=None, structures=None, method='Nelder-Mead', **options)`: Fit a model class to data by multi-start optimisation.
- `fit_individuals(model_cls, data_by_person, design=None, restarts=4, rng=None, **fit_kwargs)`: Fit a model class separately to each person's data.
- **class `FitResult(model: 'object', loss: 'float', loglik: 'float', k: 'int', n: 'int', options: 'dict' = <factory>) -> None`**: Result of :func:`fit`.
  - `summary(self)`: Summary of the fit.
- `givens_frame(angles, d)`: Orthonormal frame built from Givens rotations.
- **class `IndividualFits(model_name: 'str', results: 'dict') -> None`**: Per-person fits of one model class, returned by :func:`fit_individuals`.
  - `parameters(self)`: Fitted parameter values across people.
- `is_projector(P, tol=1e-09)`: Test whether a matrix is an orthogonal projector.
- `kl(p, q)`: Kullback-Leibler divergence.
- `lindblad_superoperator(H, jumps, rates)`: Lindblad generator as a matrix acting on the row-stacked density matrix.
- `luders(psi, P)`: Apply the Lüders rule for one projective measurement outcome.
- **class `Model(**params)`**: Base class of every quantum-like and classical model.
  - `loglik(self, data, design)`: Multinomial log-likelihood of outcome counts.
  - `predict(self, design)`: Predictions for every condition of a design.
  - `sample(self, design, n, rng=None)`: Simulate outcome counts from the model.
  - `sse(self, data, design)`: Sum of squared errors between predicted and observed values.
  - `to_vector(self)`: Unconstrained vector of the fitted parameters.
- `normalize(v)`: Scale a vector to unit length.
- **class `Param(name: 'str', kind: 'str' = 'real', default: 'float' = 0.0, lo: 'float' = 0.0, hi: 'float' = 1.0) -> None`**: Declaration of one model parameter.
  - `from_free(self, x)`: Inverse of :meth:`to_free`.
  - `random_free(self, rng)`: Random unconstrained starting value for multi-start fitting.
  - `to_free(self, v)`: Map a parameter value to the unconstrained real line used by the optimiser.
- `projector(basis)`: Orthogonal projector onto the span of a set of vectors.
- `recovery(generators, candidates, design, n, reps=20, rng=None, criterion='bic', restarts=8)`: Model-recovery study.
- `sequence_probabilities(psi, projectors, order)`: Probabilities of every yes/no answer sequence to a sequence of questions.
- `tvd(p, q)`: Total variation distance.
- `unitary(H, t)`: Time-evolution operator :math:`e^{-iHt}`.

## `qlcog.families.order_effects`

Question-order effects (see README.md in this folder).

- **class `AnchoringOrderModel(**params)`**: Classical baseline: the second answer is anchored on the first.
  - `predict(self, design=None)`: Answer distribution ``[yy, yn, ny, nn]`` for each order.
- **class `BayesOrderModel(**params)`**: Classical baseline: one order-free joint distribution.
  - `joint(self)`: Joint distribution with A first.
  - `predict(self, design=None)`: Answer distribution for each order; the BA order swaps the ``yn`` and ``ny`` cells.
- `ORDERS` (constant)
- **class `ProjectiveQuestionModel(psi, projectors)`**: General projective model of two questions.
  - `answer_probs(self, order)`: Probabilities of all answer sequences.
  - `predict(self, design=None)`: Answer distribution for each question order.
- `qq_statistic(cells_ab, cells_ba)`: QQ statistic.
- `qq_test(counts)`: Two-sided z test of the QQ equality.
- **class `QuantumOrderModel(**params)`**: Three-dimensional real quantum-like model of two questions.
  - `as_projective(self)`: The same model as a :class:`ProjectiveQuestionModel`.
  - `predict(self, design=None)`: Answer distribution ``[yy, yn, ny, nn]`` for each order (default ``'AB'``, ``'BA'``).
  - `projectors(self)`: Projectors of the "yes" answers.
  - `vectors(self)`: Unit vectors that define the two questions.
- **class `QuantumOrderModel4D(**params)`**: Four-dimensional real model in which the questions can be compatible or incompatible.
  - `as_projective(self)`: The same model as a :class:`ProjectiveQuestionModel`.
  - `predict(self, design=None)`: Answer distribution ``[yy, yn, ny, nn]`` for each order (default ``'AB'``, ``'BA'``).
  - `projectors(self)`: Projectors of the "yes" answers.
- `RANK_STRUCTURES` (constant)
- `rates(pred)`: First- and second-position "yes" rates.
- **class `SaturatedOrderModel(**params)`**: Reference model with a free answer distribution per order.
  - `predict(self, design=None)`: Answer distribution for each order (softmax of the logits).

## `qlcog.families.conjunction`

Conjunction and disjunction judgements (see README.md in this folder).

- **class `AveragingModel(**params)`**: Baseline: conjunctions and disjunctions are weighted averages of the event judgements.
  - `predict(self, design=None)`: Judged probability for each judgement type.
- **class `ClassicalJointModel(**params)`**: Classical baseline: one joint distribution, so no fallacies are possible.
  - `predict(self, design=None)`: Judged probability for each judgement type.
- `fallacy_rate(judgements)`: Check a set of judgements for the conjunction and disjunction fallacies.
- **class `PTNModel(**params)`**: Probability theory plus noise (Costello and Watts, 2014).
  - `predict(self, design=None)`: Judged probability for each judgement type.
- **class `QuantumConjunctionModel(**params)`**: Quantum-like conjunction and disjunction judgements.
  - `judgements(self)`: Predicted probability judgements.
  - `predict(self, design=None)`: Judged probability (one-element vector) for each judgement type in the design.
  - `projectors(self)`: Projectors of the two events.
- `RANK_STRUCTURES` (constant)
- `TYPES` (constant)

## `qlcog.families.interference`

Interference in decisions under uncertainty: disjunction effect (see README.md in this folder).

- **class `ClassicalMixtureModel(**params)`**: Classical baseline: the unknown condition mixes the known ones (law of total probability).
  - `predict(self, design=None)`: ``[P(act), P(not act)]`` for each condition.
- `CONDITIONS` (constant)
- `interference_phase(p1, p2, p_unknown, c=0.5)`: Phase that reproduces an observed P(act | unknown) with the unnormalised law.
- **class `InterferenceModel(**params)`**: Quantum-like law of total probability with an interference term between the two paths.
  - `p_unknown(self)`: Probability of acting when the event is unknown.
  - `predict(self, design=None)`: ``[P(act), P(not act)]`` for each condition.
- `total_probability_bounds(p1, p2)`: Classical range of P(act | unknown).

## `qlcog.families.qlbn`

Quantum-like Bayesian networks (see README.md in this folder).

- **class `BayesNet(variables, parents, cpt)`**: Discrete Bayesian network.
  - `configurations(self, names)`: Iterate over all assignments of some variables.
  - `joint(self, assignment)`: Joint probability of a full assignment.
  - `prob(self, var, value, assignment)`: Conditional probability of one variable.
- `classical_marginal(net, query, evidence=None)`: Classical posterior of one variable by enumeration.
- **class `ClassicalBNModel(**params)`**: Classical baseline for :class:`QLBNModel`: the same network without interference.
  - `predict(self, design=None)`: Distribution of the query variable in each condition.
- `for_network(net, query, conditions)`: :class:`QLBNModel` subclass with exactly the phases a network needs.
- `n_hidden_configurations(net, query, evidence=None)`: Number of hidden configurations, i.e. of interference phases.
- **class `QLBNModel(**params)`**: Fittable quantum-like Bayesian network.
  - `predict(self, design=None)`: Distribution of the query variable in each condition.
- `quantum_like_marginal(net, query, evidence=None, phases=None)`: Quantum-like posterior of one variable (amplitudes summed over hidden configurations).

## `qlcog.families.dynamics`

Belief dynamics: Markov, quantum and open-system models (see README.md in this folder).

- `final_yes(model, events, queries)`: P(yes) at the last query, marginalised over the earlier answers.
- **class `MarkovBelief(**params)`**: Classical baseline: a yes-probability updated by each event.
  - `answer_probs(self, events, queries)`: Probabilities of the answer sequences.
  - `predict(self, design=None)`: Distribution over answer sequences for each condition.
  - `step(self, p, e)`: Update the yes-probability after one event.
- **class `MarkovWalk(**kw)`**: Classical baseline: Markov random walk over N belief states.
  - `joint(self, t1, t2)`: Joint distribution of judgements at two times.
  - `state_probs(self, t)`: Distribution over belief states.
- **class `OpenSystemBelief(**params)`**: Belief (for example trust) as a qubit.
  - `answer_probs(self, events, queries)`: Probabilities of the answer sequences (rotation, dephasing, collapse).
- **class `OpenSystemWalk(**kw)`**: Quantum walk with Lindblad dephasing.
  - `joint(self, t1, t2)`: Joint distribution of judgements at ``t1`` and ``t2``, with collapse at ``t1``.
  - `state_probs(self, t)`: Distribution over belief states at time ``t`` under Lindblad dynamics.
- **class `QuantumWalk(**kw)`**: Quantum walk over N belief states (Busemeyer, Kvam and Pleskac).
  - `U(self, t)`: Unitary :math:`e^{-iHt}`.
  - `joint(self, t1, t2)`: Joint distribution of judgements at ``t1`` and ``t2``, with collapse at ``t1`` (Lüders rule).
  - `state_probs(self, t)`: Distribution over belief states at time ``t`` (Born rule).
- `question_effect(model, events, last, intermediate)`: Effect of an intermediate question on a later answer.

## `qlcog.families.decision`

Risky choice: quantum decision theory and classical baselines (see README.md in this folder).

- `expected_utility(lottery, alpha)`: Expected power utility of a lottery.
- **class `ExpectedUtilityModel(**params)`**: Classical baseline: logit choice by expected utility.
  - `predict(self, design=None)`: ``[P(choose 1), P(choose 2)]`` for each problem.
- `outcome_variance(lottery)`: Variance of the outcomes of a lottery.
- **class `ProspectTheoryModel(**params)`**: Classical baseline: prospect-theory value with logit choice.
  - `predict(self, design=None)`: ``[P(choose 1), P(choose 2)]`` for each problem.
  - `value(self, lottery)`: Prospect-theory value of a lottery.
- **class `QDTModel(**params)`**: Quantum decision theory (Yukalov and Sornette).
  - `predict(self, design=None)`: ``[P(choose 1), P(choose 2)]`` for each problem.

## `qlcog.families.contextuality`

Contextuality analysis: CHSH and Contextuality-by-Default for cyclic systems (see README.md).

- `chsh(E11, E12, E21, E22)`: CHSH value.
- `correlations_from_data(pairs)`: Expectations from raw +-1 observations.
- `cyclic_contextuality(product_expectations, means_pairs)`: Contextuality-by-Default criterion for a cyclic system.
- `qubit_chsh_correlations(a0=0.0, a1=1.5707963267948966, b0=0.7853981633974483, b1=-0.7853981633974483)`: Quantum correlations of a Bell pair.
- `s_odd(values)`: Largest signed sum with an odd number of minus signs.

## `qlcog.families.similarity`

Similarity judgements and asymmetry (see README.md in this folder).

- `asymmetry(pred, a, b)`: Asymmetry of a similarity judgement.
- **class `BiasedGeometricModel(**params)`**: Classical baseline with asymmetry (Nosofsky, 1991): geometric similarity plus a bias per concept.
  - `bias(self, name)`: Bias parameter of a concept.
- `for_concepts(cls, concepts)`: Similarity-model subclass with parameters for exactly the given concepts.
- **class `GeometricModel(**params)`**: Classical baseline: similarity decreases with distance in a psychological space (symmetric).
  - `bias(self, name)`: Response bias toward a concept (always 1 in the symmetric model).
  - `predict(self, design=None)`: Similarity of each ordered pair ``(A, B)``.
- **class `QuantumSimilarityModel(**params)`**: Quantum similarity model (Pothos, Busemeyer and Trueblood, 2013).
  - `predict(self, design=None)`: Similarity of each ordered pair ``(A, B)``.
  - `projectors(self)`: Projector of each concept subspace.

## `qlcog.families.game_theory`

Quantum games (EWL protocol) with the classical game as baseline (see README.md in this folder).

- `C` (constant)
- `CHICKEN` (constant)
- `classical_nash_equilibria(payoffs=array([[[3., 3.],
        [0., 5.]],

       [[5., 0.],
        [1., 1.]]]), tol=1e-12)`: Nash equilibria of the classical 2 x 2 game.
- `D` (constant)
- **class `EWLGame(payoffs=array([[[3., 3.],
        [0., 5.]],

       [[5., 0.],
        [1., 1.]]]), gamma=1.5707963267948966)`**: Two-player quantum game (EWL protocol).
  - `best_response(self, U_other, player='A', grid=61, phases=True, three_parameter=False)`: Best response to a fixed strategy, by grid search.
  - `is_nash(self, UA, UB, **kw)`: Check whether a strategy pair is a Nash equilibrium on the grid.
  - `outcome_probabilities(self, UA, UB)`: Probabilities of the four outcomes.
  - `payoff(self, UA, UB)`: Expected payoffs.
- `PRISONERS_DILEMMA` (constant)
- `Q` (constant)
- `STAG_HUNT` (constant)
- `strategy(theta, phi=0.0)`: EWL two-parameter strategy.

## `qlcog.families.memory`

Episodic memory: overdistribution, quantum versus additive classical model (see README.md).

- **class `AdditiveMemoryModel(**params)`**: Classical baseline: V and G are exclusive, so P(V or G) = P(V) + P(G).
  - `predict(self, design=None)`: Predicted proportion for each ``(probe, question)``.
  - `probe_probabilities(self, probe)`: Answer probabilities for one probe type (softmax over verbatim, gist, neither).
- `overdistribution(values, probe)`: Overdistribution for one probe type.
- **class `QuantumEpisodicModel(**params)`**: Quantum episodic memory model.
  - `predict(self, design=None)`: Predicted proportion for each ``(probe, question)``.
  - `probe_probabilities(self, probe)`: Answer probabilities for one probe type, from the projections of the memory state.

## `qlcog.families.concepts`

Concept combination (guppy effect): Aerts' Fock-space model versus classical rules (see README.md).

- **class `FockSpaceConceptModel(**params)`**: Aerts' Fock-space model of concept combination.
- `interference_bound(mA, mB)`: Largest possible size of the interference term.
- **class `MinConceptModel(**params)`**: Fuzzy-set baseline: :math:`\min(\mu_A, \mu_B)` (no parameter).
- `overextension(design, values)`: Overextension of each item.
- **class `ProductConceptModel(**params)`**: Classical baseline: independent product :math:`\mu_A \mu_B` (no parameter).
- **class `WeightedAverageModel(**params)`**: Baseline: :math:`w\,\mu_A + (1 - w)\,\mu_B`.

## `qlcog.applications.robotics`

Human-robot interaction: order-aware human models for robots that ask questions and track trust.

- **class `AnchoringOrderModel(**params)`**: Classical baseline: the second answer is anchored on the first.
  - `predict(self, design=None)`: Answer distribution ``[yy, yn, ny, nn]`` for each order.
- `ask_or_act(p_success, uncertainty=None, ask_cost=1.0, error_cost=5.0, max_disagreement_bits=0.05)`: Decision rule for a robot that may ask a person before acting.
- **class `BayesOrderModel(**params)`**: Classical baseline: one order-free joint distribution.
  - `joint(self)`: Joint distribution with A first.
  - `predict(self, design=None)`: Answer distribution for each order; the BA order swaps the ``yn`` and ``ny`` cells.
- `compare(candidates, data, design=None, restarts=8, rng=None, criterion='bic')`: Fit several model classes to the same data and rank them.
- `domain_models(domain)`: Simulated populations for one domain.
- `estimate_unprimed_rates(design, generator, n, rng=None, model=None, probe=0.1)`: Estimate the unprimed "yes" rates from people who answer both questions.
- `fit(model_cls, data, design=None, restarts=8, rng=None, structures=None, method='Nelder-Mead', **options)`: Fit a model class to data by multi-start optimisation.
- `HRI_DOMAINS` (constant)
- **class `HumanModelEnsemble(candidates=None, design=None)`**: Competing human models weighted by evidence.
  - `predict(self, condition)`: Mixture prediction and its uncertainty.
  - `summary(self)`: Ensemble weights.
  - `update(self, data, restarts=8, rng=None)`: Fit every candidate model and compute BIC weights.
- **class `HumanModelService(candidates=None, refit_every=20, min_answers=20, restarts=4, seed=0)`**: Middleware-independent human-model service a robot can run next to its planner.
  - `add_answer(self, msg)`: Record one person's answers.
  - `query(self, msg=None)`: Predict answers and decide whether to ask.
- **class `MarkovBelief(**params)`**: Classical baseline: a yes-probability updated by each event.
  - `answer_probs(self, events, queries)`: Probabilities of the answer sequences.
  - `predict(self, design=None)`: Distribution over answer sequences for each condition.
  - `step(self, p, e)`: Update the yes-probability after one event.
- **class `OpenSystemBelief(**params)`**: Belief (for example trust) as a qubit.
  - `answer_probs(self, events, queries)`: Probabilities of the answer sequences (rotation, dephasing, collapse).
- **class `QuantumOrderModel(**params)`**: Three-dimensional real quantum-like model of two questions.
  - `as_projective(self)`: The same model as a :class:`ProjectiveQuestionModel`.
  - `predict(self, design=None)`: Answer distribution ``[yy, yn, ny, nn]`` for each order (default ``'AB'``, ``'BA'``).
  - `projectors(self)`: Projectors of the "yes" answers.
  - `vectors(self)`: Unit vectors that define the two questions.
- **class `QuantumOrderModel4D(**params)`**: Four-dimensional real model in which the questions can be compatible or incompatible.
  - `as_projective(self)`: The same model as a :class:`ProjectiveQuestionModel`.
  - `predict(self, design=None)`: Answer distribution ``[yy, yn, ny, nn]`` for each order (default ``'AB'``, ``'BA'``).
  - `projectors(self)`: Projectors of the "yes" answers.
- `RANK_STRUCTURES` (constant)
- `TRUST_EVENTS` (constant)
- `TRUST_MODELS` (constant)
- `TRUST_PROTOCOL` (constant)

## `qlcog.applications.intent`

Intent resolution from ambiguous commands and context cues (robots, assistants, interfaces).

- **class `BayesIntentResolver(intents, prior=None)`**: Classical baseline: posterior proportional to the prior times the cue likelihoods.
  - `add_cue(self, name, likelihood, theta=0.0)`: Register a cue.
  - `order_effect(self, a, b)`: Size of the order effect between two cues.
  - `posterior(self, cues)`: Intent probabilities after a sequence of cues.
- **class `QuantumIntentResolver(intents, prior=None)`**: Quantum-like intent resolver; it reduces to :class:`BayesIntentResolver` when every angle is 0.
  - `add_cue(self, name, likelihood, theta=0.0)`: Register a cue as a Kraus operator.
  - `posterior(self, cues)`: Intent probabilities :math:`|\psi|^2` after applying the cues in order.
  - `resolve(self, cues)`: Most likely intent after the cues.

## `qlcog.quantum`

Quantum models: quantum machine learning and quantum algorithms (gate-model circuits).

- `adjacency(edges, n=None, directed=False, weights=None)`: Adjacency matrix from an edge list.
- **class `AEResult(estimate: 'float', exact: 'float', oracle_calls: 'int', powers: 'list', hits: 'list', shots: 'int') -> None`**: Result of :meth:`AmplitudeEstimation.run`.
- **class `AmplitudeEstimation(probabilities, payoff)`**: Maximum-likelihood amplitude estimation of an expected payoff.
  - `good_probability(self, m)`: Probability of measuring the ancilla in :math:`|1\rangle` after :math:`Q^m A`.
  - `run(self, powers=(0, 1, 2, 4, 8, 16), shots=100, rng=None)`: Sample measurements at several powers and return the maximum-likelihood estimate.
  - `to_qiskit(self, m=0, measure=True)`: Qiskit circuit for :math:`Q^m A`.
- `angle_encoding(n_features, n_qubits=None, scale=1.0, circ=None)`: Angle encoding: RY(scale * x_j) on qubit j.
- `apply_matrix(state, M, qubits, n)`: Apply a k-qubit matrix to a batch of states.
- **class `Circuit(n)`**: Parameterised quantum circuit with a batched simulator.
  - `compose(self, other)`: Append the gates of another circuit.
  - `cp(self, p, c, t)`: Controlled phase diag(1, 1, 1, e^{ip}) on qubits ``c`` and ``t`` (symmetric).
  - `cx(self, c, t)`: CNOT with control ``c`` and target ``t``.
  - `cz(self, a, b)`: Controlled Z on qubits ``a`` and ``b``.
  - `diagonal(self, phases, label='D')`: Diagonal unitary diag(exp(i phases)) on all qubits (for example a cost or oracle layer).
  - `h(self, q)`: Hadamard on qubit ``q``.
  - `p(self, p, q)`: Phase gate diag(1, e^{ip}) on qubit ``q``.
  - `probabilities(self, w=None, X_=None)`: Born-rule probabilities of the final states.
  - `rx(self, p, q)`: RX rotation on qubit ``q``; ``p`` is a number, :class:`W`, :class:`X` or :class:`XX`.
  - `ry(self, p, q)`: RY rotation on qubit ``q``; ``p`` is a number or an angle expression.
  - `rz(self, p, q)`: RZ rotation on qubit ``q``; ``p`` is a number or an angle expression.
  - `rzz(self, p, a, b)`: RZZ(p) = exp(-i p Z_a Z_b / 2) on qubits ``a`` and ``b``.
  - `s(self, q)`: S gate (phase pi/2) on qubit ``q``.
  - `sdg(self, q)`: S-dagger gate on qubit ``q``.
  - `state(self, w=None, X_=None, init=None)`: Simulate the circuit.
  - `swap(self, a, b)`: Swap qubits ``a`` and ``b``.
  - `to_qiskit(self, w=None, x=None, measure=True)`: Export to Qiskit with every angle bound.
  - `unitary(self, M, qubits, label='U')`: Fixed unitary on some qubits.
  - `value_and_grad(self, w, X_, loss, chunk=None)`: Loss and its exact gradient with respect to the weights (adjoint method).
  - `x(self, q)`: Pauli X on qubit ``q``.
  - `y(self, q)`: Pauli Y on qubit ``q``.
  - `z(self, q)`: Pauli Z on qubit ``q``.
- `ctqw_probabilities(A, t, start=None)`: Occupation probabilities of a continuous-time quantum walk.
- `degree_centrality(A)`: Normalised degree centrality.
- `expectation_z(psi, q, n)`: Expectation of Pauli Z on one qubit.
- `find_period(n, r, offset=0, shots=None, rng=None)`: Recover a period with the QFT.
- `grover(n, marked, iterations=None)`: Grover search.
- **class `GroverResult(probabilities: 'np.ndarray', iterations: 'int', marked: 'list', success: 'float', circuit: 'Circuit' = None, n: 'int' = 0) -> None`**: Result of :func:`grover`.
  - `to_qiskit(self, measure=True, style='gates')`: Qiskit circuit of the search.
- **class `Hamiltonian(terms)`**: Sum of weighted Pauli strings.
  - `expectation(self, psi)`: Energy expectation of a batch of states.
  - `ground_energy(self)`: Exact ground-state energy (dense diagonalisation).
- `hardware_efficient(n_qubits, layers=2, start=0, circ=None, entangle='ring', rotations=('ry', 'rz'), entangler='cz')`: Hardware-efficient ansatz.
- `monte_carlo_estimate(probabilities, payoff, samples, rng=None)`: Classical Monte Carlo baseline.
- `pagerank(A, damping=0.85, tol=1e-12, max_iter=1000)`: Classical PageRank by power iteration.
- `parameter_shift(f, w, shift=1.5707963267948966)`: Exact gradient by the parameter-shift rule.
- `pauli_matrix(label)`: Matrix of a Pauli string.
- `periodic_state(n, r, offset=0)`: Uniform superposition over a periodic set of basis states.
- **class `QAOA(qubo, p=2, restarts=6, maxiter=300, seed=0)`**: Quantum Approximate Optimisation Algorithm (Farhi, Goldstone and Gutmann, 2014).
  - `expectation(self, w)`: Expected QUBO energy of the QAOA state.
  - `run(self)`: Optimise the angles and sample the result.
  - `to_qiskit(self, result, measure=True)`: Qiskit circuit with optimised angles bound.
- **class `QAOAResult(x: 'np.ndarray', energy: 'float', expectation: 'float', probabilities: 'np.ndarray', gammas: 'np.ndarray', betas: 'np.ndarray', circuit: 'Circuit' = None, optimum: 'float' = None, p_optimal: 'float' = 0.0, weights: 'np.ndarray' = None) -> None`**: Result of :meth:`QAOA.run`.
- `qft_circuit(n, inverse=False, swaps=True)`: Circuit for the QFT or its inverse.
- `qft_matrix(n)`: The QFT as a matrix.
- `quantum_walk_centrality(A, start=None, tol=1e-09)`: Quantum-walk centrality: infinite-time average of the occupation probabilities.
- **class `QuantumKernel(reps=2, scale=1.0)`**: Fidelity quantum kernel with the ZZ feature map (Havlíček et al., 2019).
  - `fit(self, X)`: Fit the feature scaling and build the feature map.
  - `states(self, X)`: Feature-map states.
  - `to_qiskit(self, x1, x2)`: Compute-uncompute circuit whose all-zeros probability is ``k(x1, x2)``.
- **class `QuantumKernelAnomalyDetector(kernel='quantum', reps=1, scale=0.5, gamma=0.5, quantile=0.95)`**: Kernel anomaly detector (distance to the mean in feature space).
  - `fit(self, X)`: Learn the normal data.
  - `predict(self, X)`: Anomaly labels.
  - `score_samples(self, X)`: Anomaly score of each sample (higher is more anomalous).
- **class `QuantumKernelClassifier(reps=2, scale=1.0, ridge=0.01)`**: Kernel ridge classifier on the quantum kernel (one-versus-rest, targets +-1).
  - `decision_function(self, X)`: Scores per class (one-versus-rest).
  - `fit(self, X, y)`: Solve the kernel ridge system.
  - `predict(self, X)`: Class with the highest score for each sample.
  - `score(self, X, y)`: Accuracy on ``(X, y)``.
- **class `QuantumKernelClustering(n_clusters=2, kernel='quantum', reps=1, scale=0.5, gamma=0.5, restarts=10, seed=0)`**: Spectral clustering on a kernel.
  - `fit_predict(self, X)`: Cluster the samples.
- `rbf_kernel(X1, X2=None, gamma=1.0)`: Classical Gaussian (RBF) kernel, the baseline for the quantum kernel.
- `reuploading_classifier_circuit(n_features, n_qubits, layers, reupload=True)`: Data re-uploading classifier circuit (Pérez-Salinas et al., Quantum 4, 226, 2020).
- **class `VariationalClassifier(layers=3, n_qubits=None, reupload=True, l2=0.001, maxiter=200, seed=0)`**: Variational quantum classifier with data re-uploading.
  - `fit(self, X, y)`: Train the classifier.
  - `predict(self, X)`: Most probable class for each sample.
  - `predict_proba(self, X)`: Class probabilities.
  - `score(self, X, y)`: Accuracy.
  - `to_qiskit(self, x, measure=True)`: Circuit for one input sample, with the trained weights bound.
- **class `VariationalRegressor(layers=3, n_qubits=None, reupload=True, l2=0.001, maxiter=200, seed=0)`**: Variational (data re-uploading) regressor.
  - `fit(self, X, y)`: Train the regressor.
  - `predict(self, X)`: Predicted values.
  - `score(self, X, y)`: Coefficient of determination :math:`R^2`.
  - `to_qiskit(self, x, measure=True)`: Circuit for one input with the trained weights bound; ``<Z_0>`` gives the prediction.
- **class `VQE(hamiltonian, layers=2, restarts=4, maxiter=400, seed=0, rotations=('ry', 'rz'), entangle='linear', entangler='cx')`**: Variational Quantum Eigensolver (Peruzzo et al., Nature Communications 5, 4213, 2014).
  - `energy(self, w)`: Energy of the ansatz state.
  - `run(self)`: Optimise from several random starts.
  - `to_qiskit(self, measure=True)`: Qiskit circuit of the optimised ansatz.
- **class `W(k: 'int', scale: 'float' = 1.0) -> None`**: Trainable weight as a gate angle.
- **class `X(j: 'int', scale: 'float' = 1.0, shift: 'float' = 0.0) -> None`**: Input feature as a gate angle.
- **class `XX(i: 'int', j: 'int', a: 'float' = 3.141592653589793, scale: 'float' = 1.0) -> None`**: Product feature used by ZZ feature maps.
- `zz_feature_map(n_features, reps=2, circ=None)`: Second-order Pauli-Z feature map (Havlíček et al., Nature 567, 209-212, 2019).

## `qlcog.inspired`

Quantum-inspired models: classical algorithms that borrow quantum ideas. No quantum computer is used.

- **class `GridWorld(size=6, walls=frozenset({(2, 3), (1, 1), (3, 1), (4, 4)}), step_reward=0.0, max_steps=200)`**: Deterministic grid world.
  - `reset(self)`: Return to the start cell.
  - `shortest_path(self)`: Length of the shortest path from start to goal (breadth-first search).
  - `step(self, a)`: Take one action.
- **class `MPSClassifier(bond=6, local_dim=2, epochs=60, lr=0.02, batch=32, seed=0, method='adam', sweeps=6, steps=40, sweep_lr=0.05, cutoff=1e-10)`**: Matrix product state classifier.
  - `fit(self, X, y)`: Train the classifier.
  - `predict(self, X)`: Most probable class for each sample.
  - `predict_proba(self, X)`: Class probabilities (softmax of the MPS scores).
  - `score(self, X, y)`: Accuracy.
- **class `OptimResult(x: 'np.ndarray', value: 'float', history: 'list' = <factory>, evaluations: 'int' = 0) -> None`**: Result of an optimiser.
- **class `QIEA(problem, n=None, pop=20, generations=300, rotation='lookup', delta=0.031415926535897934, migrate_every=20, groups=4, seed=0)`**: Quantum-inspired evolutionary algorithm for binary minimisation.
  - `run(self)`: Run the evolutionary search.
- **class `QLearning(n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.2, decay=0.995, seed=0)`**: Classical baseline: tabular Q-learning with decaying epsilon-greedy exploration.
  - `act(self, s)`: Epsilon-greedy action (ties broken at random).
  - `greedy(self, s)`: Greedy action.
  - `update(self, s, a, r, s2, done)`: Q-learning update; epsilon decays at the end of each episode.
- **class `QPSO(f, lower, upper, particles=30, iterations=300, beta0=1.0, beta1=0.5, seed=0)`**: Quantum-behaved particle swarm optimisation for continuous minimisation on a box.
  - `run(self)`: Run the swarm.
- **class `QuantumInspiredQLearning(n_states, n_actions, alpha=0.2, gamma=0.95, k=2.0, max_step=0.3, floor=0.02, seed=0)`**: Quantum-inspired reinforcement learning agent (Dong et al., 2008, with a bounded rotation).
  - `act(self, s)`: Choose an action by "measuring" the state's action amplitudes (Born rule).
  - `greedy(self, s)`: Most probable action in a state.
  - `update(self, s, a, r, s2, done)`: Learn from one transition.
- **class `QuantumLanguageModel(window=3, smoothing=0.2, uniform=0.01, iters=60)`**: Quantum language model for ranking documents against a query.
  - `fit(self, documents)`: Estimate a density matrix for every document.
  - `query_density(self, query)`: Density matrix of a query (terms outside the vocabulary are ignored).
  - `rank(self, query)`: Rank the documents.
  - `scores(self, query)`: Score every document.
- **class `QueryLikelihoodModel(mu=50.0)`**: Classical baseline: unigram query likelihood with Dirichlet smoothing.
  - `fit(self, documents)`: Count terms in every document and in the collection.
  - `rank(self, query)`: Rank the documents.
  - `scores(self, query)`: Log-likelihood of the query under each smoothed document model.
- `simulated_annealing(qubo, sweeps=400, T0=2.0, T1=0.01, seed=0)`: Classical simulated annealing baseline (single-spin Metropolis on the same Ising model).
- **class `SQA(qubo, replicas=16, sweeps=400, T=0.05, gamma0=3.0, gamma1=0.001, seed=0)`**: Simulated quantum annealing (path-integral Monte Carlo) for a QUBO.
  - `run(self)`: Run the annealing schedule.
- `tokenize(text)`: Split text into lower-case word tokens.
- `train(agent, env, episodes=300)`: Run training episodes.

## `qlcog.problems`

Binary optimisation problems shared by the quantum and quantum-inspired solvers.

- `from_ising(h, J, c=0.0)`: QUBO with the same energies as an Ising model.
- `knapsack(values, weights, capacity, penalty=None)`: 0-1 knapsack: maximise value subject to total weight <= capacity.
- `maxcut(edges, n=None, weights=None)`: Maximum cut of a graph, written as a minimisation (energy = minus the cut weight).
- `portfolio(mu, cov, budget, risk=0.5, penalty=None)`: Portfolio selection with a cardinality budget.
- **class `Qubo(Q: 'np.ndarray', offset: 'float' = 0.0, labels: 'list' = <factory>, name: 'str' = 'QUBO') -> None`**: Quadratic unconstrained binary optimisation problem.
  - `all_energies(self)`: Energies of all :math:`2^n` bit strings.
  - `brute_force(self)`: Exact minimum by enumeration (small n).
  - `energy(self, x)`: Energy of bit strings.
  - `normalised(self)`: Upper-triangular form with the same energies.
  - `to_ising(self)`: Equivalent Ising model with spins :math:`s = 1 - 2x \in \{+1, -1\}`.
- `task_allocation(costs, penalty=None)`: Assignment of tasks to agents; every task goes to exactly one agent.

## `qlcog.circuits`

Qiskit circuits for the model families, and a single :func:`run` for simulators and cloud hardware (see README.md in this folder). Needs the ``qiskit`` extra.

- `belief_circuit(model, events, queries, form='dynamic')`: Circuit for an :class:`~qlcog.families.dynamics.OpenSystemBelief` over a sequence of events.
- `chsh_circuit(a, b)`: Bell pair measured in the x-z plane.
- `conjunction_circuit(model, form='dynamic')`: Circuit for a :class:`~qlcog.families.conjunction.QuantumConjunctionModel`.
- `interference_circuit(p1, p2, c, theta, condition='unknown')`: Two-path interference circuit (normalised form of the interference model).
- `n_qubits(d)`: Number of qubits needed for a dimension.
- `order_effects_circuit(model, order='AB', form='dynamic')`: Circuit for a question-order model.
- `projective_sequence_circuit(psi, projectors, order, form='dynamic')`: Circuit for a sequence of yes/no questions.
- `qlbn_circuit(net, query, evidence=None, phases=None)`: Quantum-like Bayesian network inference as a circuit.
- `run(qc, backend='aer', shots=10000, seed=7)`: Run a circuit and return counts.
- `similarity_circuit(model, a, b, form='dynamic')`: Circuit for Sim(a, b) of a :class:`~qlcog.families.similarity.QuantumSimilarityModel`.
- `subspace_unitary(P, n)`: Unitary that maps the range of a projector onto the first basis states.
- `to_qasm3(qc)`: Export a circuit as OpenQASM 3 in Braket's gate names.
- `walk_circuit(model, t)`: Quantum-walk circuit at time t.

## `qlcog.viz`

Bloch-sphere visualisation: watch qubits evolve in 3D.

- `animate_bloch(traj, theme='dark', fps=30, trail=True, title=None, height=500, max_frames=400)`: Animated interactive figure with play and pause buttons and a slider labelled with the frame labels.
- `animate_trajectory(traj, theme='dark', interval=40, trail=True, save=None, fps=25, size=3.6, rotate=0.0, title=None, dpi=90)`: Animate a trajectory, one sphere per qubit.
- `belief_trajectory(model, events, steps=12)`: Trajectory of the belief qubit of the trust model over a sequence of events.
- `bloch_figure(vectors, names=None, theme='dark', title=None, height=460)`: Interactive figure with one sphere per qubit.
- `bloch_tomography(qc, backend='aer', shots=4000, seed=7)`: Bloch vectors estimated from measurements on a simulator or quantum hardware.
- `bloch_vector(state)`: Bloch vector of one qubit.
- `bloch_vectors(state, n=None)`: Bloch vectors of every qubit.
- **class `BlochSphere(ax=None, title='', theme='dark', figsize=(5, 5), elev=20, azim=35)`**: One Bloch sphere on a Matplotlib 3D axis.
  - `add_points(self, points, color=None, size=12, alpha=0.9)`: Scatter points on or inside the sphere.
  - `add_state(self, state, **kw)`: Arrow for a qubit state.
  - `add_trajectory(self, points, color=None, lw=1.8, alpha=0.8)`: Draw a path of Bloch vectors.
  - `add_vector(self, r, color=None, label=None, lw=3)`: Arrow from the origin to a Bloch vector.
  - `save(self, path, dpi=150)`: Save the figure.
  - `show(self)`: Show the figure.
- `circuit_trajectory(qc, steps=12, initial=None)`: Smooth Bloch-sphere trajectory of every qubit of a Qiskit circuit, gate by gate.
- `concurrence(rho)`: Wootters concurrence of a two-qubit state.
- `entanglement_summary(state, n=None)`: Entanglement overview of a pure multi-qubit state.
- **class `LiveBloch(n_qubits=1, names=None, theme='dark', backend='auto', trail=True, title=None)`**: Real-time Bloch spheres.
  - `play(self, traj, fps=30)`: Replay a trajectory in real time.
  - `reset(self)`: Clear the trails.
  - `show(self)`: Display the spheres (a widget in Jupyter, a window elsewhere).
  - `update(self, state, label='')`: Move the vectors to a new state.
- `plot_bloch(vectors, names=None, theme='dark', title=None, size=3.6)`: Static figure with one sphere per qubit.
- `plot_entanglement(traj, pairs=None, theme='dark', size=(8, 3.2))`: Entanglement timeline of a circuit trajectory.
- `plot_qsphere(state, **kw)`: Q-sphere of a multi-qubit state, drawn by Qiskit's ``plot_state_qsphere``.
- `reduced_density(state, q, n)`: Reduced density matrix of one qubit.
- `reduced_density_pair(state, a, b, n)`: Reduced density matrix of two qubits.
- `rotation_trajectory(axis, angle, start=(0, 0, 1), steps=60)`: Rotation of a Bloch vector about an axis (for demonstrations).
- `save_html(fig, path, auto_open=False)`: Write a standalone HTML file (Plotly JavaScript loaded from its CDN).
- `state_from_bloch(r)`: Density matrix of a Bloch vector.
- `THEMES` (constant)
- `tomography_circuits(qc)`: Circuits for single-qubit state tomography.
- **class `Trajectory(vectors: 'np.ndarray', labels: 'list' = <factory>, names: 'list' = <factory>, title: 'str' = '', states: 'list' = <factory>) -> None`**: Bloch vectors over time.
  - `concurrence(self, a=0, b=1)`: Concurrence of two qubits in every frame.
  - `purity(self)`: Bloch-vector length per frame and qubit.

## `qlcog.data`

Published aggregate data used in the examples and tests.

- `CLINTON_GORE` (constant)
- `LINDA` (constant)
- `PRISONERS_DILEMMA` (constant)
- `proportions_to_counts(props, n)`: Turn proportions into binary counts.
- `TWO_STAGE_GAMBLE` (constant)
