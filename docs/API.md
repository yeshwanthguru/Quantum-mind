# API reference

Generated from the docstrings by `docs/make_api.py`; each entry shows the signature and the first
paragraph of the docstring. Full docstrings: `help(qlcog.<module>.<name>)`.

## `qlcog.core`

Core layer: linear algebra of quantum-like models, the Model base class, fitting and comparison.

- `angles_to_unit(angles)` — Unit vector in R^(len(angles)+1) from hyperspherical angles.
- `compare(candidates, data, design=None, restarts=8, rng=None, criterion='bic')` — Fit several models and rank them. candidates: list of model classes or (class, fit kwargs). Returns a list of FitResult sorted by the criterion (lower is better).
- `compare_individuals(candidates, data_by_person, design=None, restarts=4, rng=None, criterion='bic')` — Fit every candidate to every person. Returns {'group': [(model, group criterion)] sorted (lower is better), 'best_counts': {model: number of people it fits best}, 'fits': {model: IndividualFits}}.
- `complement(P)` — I - P.
- `density(psi)` — Density matrix |psi><psi| of a (normalised) state vector.
- `dephase(rho, strength, basis_projectors=None)` — Partial dephasing: rho -> (1 - s) rho + s * sum_k P_k rho P_k (in the computational basis if no projectors are given).
- `evolve_density(rho, superop, t)` — Evolve a density matrix for time t under a Lindblad superoperator (matrix exponential).
- `fit(model_cls, data, design=None, restarts=8, rng=None, structures=None, method='Nelder-Mead', **options)` — Fit a model class to data by multi-start optimisation.
- `fit_individuals(model_cls, data_by_person, design=None, restarts=4, rng=None, **fit_kwargs)` — Fit `model_cls` separately to each person's data {person: data}. Aggregate fitting assumes that everyone follows the same parameters; per-person fits test that assumption.
- **class `FitResult(model: 'object', loss: 'float', loglik: 'float', k: 'int', n: 'int', options: 'dict' = <factory>) -> None`** — Result of `fit`: the fitted model, loss, log-likelihood, number of parameters k, observations n and options; properties aic and bic.
  - `summary(self)` — Dictionary with model name, parameters, options, log-likelihood, k, n, AIC and BIC.
- `givens_frame(angles, d)` — Orthonormal frame (d x d orthogonal matrix) from d(d-1)/2 Givens rotation angles.
- **class `IndividualFits(model_name: 'str', results: 'dict') -> None`** — Per-person fits of one model class: results {person: FitResult}; summed log-likelihood, total parameters and group AIC/BIC (each person has their own parameters).
  - `parameters(self)` — {parameter: array over people} of the fitted parameter values.
- `is_projector(P, tol=1e-09)` — True if P is Hermitian and idempotent within tol.
- `kl(p, q)` — Kullback-Leibler divergence KL(p || q) in nats (probabilities clipped at 1e-12).
- `lindblad_superoperator(H, jumps, rates)` — Superoperator L (acting on row-stacked vec(rho)) of d rho/dt = -i[H, rho] + sum_k g_k (L_k rho L_k^+ - 1/2 {L_k^+ L_k, rho}).
- `luders(psi, P)` — Lueders update: returns (probability, post-measurement state) for projector P.
- **class `Model(**params)`** — Base class. Subclasses define PARAMS (list of Param), LOSS ('multinomial' or 'sse') and predict(design).
  - `loglik(self, data, design)` — Multinomial log-likelihood of outcome counts (constant terms omitted).
  - `predict(self, design)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
  - `sample(self, design, n, rng=None)` — Simulated outcome counts: n observations per condition.
  - `sse(self, data, design)` — Sum of squared errors between predicted and observed values (judgement models).
  - `to_vector(self)` — Unconstrained vector of the fitted parameters (inverse of from_vector).
- `normalize(v)` — Return v / ||v|| (raises on the zero vector).
- **class `Param(name: 'str', kind: 'str' = 'real', default: 'float' = 0.0, lo: 'float' = 0.0, hi: 'float' = 1.0) -> None`** — A model parameter. kind: 'prob' (0..1), 'angle' (free real, periodic), 'real', 'positive', 'bounded' (lo..hi) or 'fixed' (not fitted).
  - `from_free(self, x)` — Inverse of to_free: unconstrained value -> parameter value.
  - `random_free(self, rng)` — Random starting point (unconstrained) for multi-start fitting.
  - `to_free(self, v)` — Map a parameter value to the unconstrained real line used by the optimiser.
- `projector(basis)` — Orthogonal projector onto the span of the given vectors (rows or a single vector). The vectors are orthonormalised first, so any spanning set may be passed.
- `recovery(generators, candidates, design, n, reps=20, rng=None, criterion='bic', restarts=8)` — Model-recovery study: simulate data from each generator (dict name -> model instance), fit all candidates and count which one the criterion selects. Returns {generator: {candidate: count}}.
- `sequence_probabilities(psi, projectors, order)` — Probabilities of every yes/no answer sequence to the questions in `order`.
- `tvd(p, q)` — Total variation distance between two probability vectors.
- `unitary(H, t)` — exp(-i H t).

## `qlcog.families.order_effects`

Question-order effects (see README.md in this folder).

- **class `AnchoringOrderModel(**params)`** — First answer at its first-position rate; P(second yes | first yes) = p2 + w (1 - p2) and P(second yes | first no) = p2 (1 - w) for w >= 0 (mirror image for w < 0). w depends on the second question (wA, wB). Option tied=True forces wA = wB (one weight).
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
- **class `BayesOrderModel(**params)`** — Order-free joint distribution with marginals pA, pB and correlation rho in (-1, 1) (scaled between the bounds allowed by the marginals).
  - `joint(self)` — Order-free joint distribution [yy, yn, ny, nn] with A first.
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
- `ORDERS` (constant)
- **class `ProjectiveQuestionModel(psi, projectors)`** — General projective model: psi (unit vector, real or complex) and a dict of 'yes' projectors.
  - `answer_probs(self, order)` — Probabilities of all answer sequences for questions asked in `order` (Lueders rule).
  - `predict(self, design=None)` — Answer distribution [yy, yn, ny, nn] for each question order in the design (default AB, BA).
- `qq_statistic(cells_ab, cells_ba)` — [p_AB(yn) + p_AB(ny)] - [p_BA(yn) + p_BA(ny)]; zero for every projective model.
- `qq_test(counts)` — Two-sided z test of the QQ equality from answer counts {'AB': [...], 'BA': [...]}. Returns (q, z, p_value).
- **class `QuantumOrderModel(**params)`** — psi = e1; u_A = (cos a, sin a, 0); u_B = (cos b, sin b cos g, sin b sin g). Option ranks=(rA, rB): rank 1 -> 'yes' is the ray of u_q; rank 2 -> the plane orthogonal to u_q.
  - `as_projective(self)` — The same model as a ProjectiveQuestionModel (state and projectors).
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
  - `projectors(self)` — 'Yes' projectors of the two questions {'A': P_A, 'B': P_B}.
  - `vectors(self)` — Unit vectors u_A and u_B that define the two questions.
- **class `QuantumOrderModel4D(**params)`** — Four-dimensional real model in which the questions can be compatible or incompatible. Basis e1..e4 = (yes,yes), (yes,no), (no,yes), (no,no) of a classical 2 x 2 table; P_A = span(e1, e2). P_B = U span(e1, e3) U^T, where U rotates by the incompatibility angle phi in the planes (e1, e4) and (e2, e3). The state psi has three hyperspherical angles. With phi = 0 the projectors commute and the model is exactly the order-free Bayesian model (psi_k^2 is the joint table), so the model nests BayesOrderModel; phi != 0 produces order effects (and the QQ equality still holds).
  - `as_projective(self)` — The same model as a ProjectiveQuestionModel (state and projectors).
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
  - `projectors(self)` — 'Yes' projectors of the two questions {'A': P_A, 'B': P_B}.
- `RANK_STRUCTURES` (constant)
- `rates(pred)` — First- and second-position 'yes' rates [A first, B first, A second, B second].
- **class `SaturatedOrderModel(**params)`** — Reference model with a free answer distribution per order (6 parameters); the best any model can fit.
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.

## `qlcog.families.conjunction`

Conjunction and disjunction judgements (see README.md in this folder).

- **class `AveragingModel(**params)`** — Baseline: the conjunction is judged as a weighted average of the two event judgements.
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
- **class `ClassicalJointModel(**params)`** — Classical baseline: one joint distribution, so P(A and B) <= min(P(A), P(B)) (no fallacies).
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
- `fallacy_rate(judgements)` — Whether a set of judgements commits the conjunction fallacy (P(A&B) > min(P(A), P(B))) and the disjunction fallacy (P(A|B) < max(P(A), P(B))).
- **class `PTNModel(**params)`** — Probability theory plus noise baseline (Costello and Watts): classical probabilities read with random noise, which regresses judgements towards 0.5.
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
- **class `QuantumConjunctionModel(**params)`** — Quantum-like conjunction and disjunction judgements: two events as projectors in a real 3D space, with the conjunction judged by asking the more likely event first (Lueders rule); can produce conjunction and disjunction fallacies.
  - `judgements(self)` — Predicted judgements {A, B, A&B, A|B}.
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
  - `projectors(self)` — Projectors {'A': P_A, 'B': P_B} of the two events.
- `RANK_STRUCTURES` (constant)
- `TYPES` (constant)

## `qlcog.families.interference`

Interference in decisions under uncertainty: disjunction effect (see README.md in this folder).

- **class `ClassicalMixtureModel(**params)`** — Classical baseline: the unknown condition is a mixture of the known ones (law of total probability).
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
- `CONDITIONS` (constant)
- `interference_phase(p1, p2, p_unknown, c=0.5)` — Phase theta for which the unnormalised quantum-like law reproduces p_unknown (nan if impossible).
- **class `InterferenceModel(**params)`** — Quantum-like law of total probability with an interference term between the two paths (normalised by default) for disjunction effects.
  - `p_unknown(self)` — Probability of acting when the event is unknown.
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
- `total_probability_bounds(p1, p2)` — Classical range of P(act | unknown).

## `qlcog.families.qlbn`

Quantum-like Bayesian networks (see README.md in this folder).

- **class `BayesNet(variables, parents, cpt)`** — variables: {name: [values]}; parents: {name: [parent names]}; cpt: {name: function(value, parent_values_dict) -> probability} or nested dicts {name: {tuple(parent values): {value: prob}}}.
  - `configurations(self, names)` — All assignments of the variables in `names`.
  - `joint(self, assignment)` — Joint probability of a full assignment (product of the conditional tables).
  - `prob(self, var, value, assignment)` — Conditional probability of var = value given the parents in `assignment`.
- `classical_marginal(net, query, evidence=None)` — Classical Bayesian-network marginal of `query` given `evidence` (sum over hidden configurations).
- **class `ClassicalBNModel(**params)`** — Classical baseline for QLBNModel: the same network without interference.
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
- `for_network(net, query, conditions)` — QLBNModel subclass with exactly one free phase per extra hidden configuration (the largest number over the conditions), so that information criteria count only the phases in use.
- `n_hidden_configurations(net, query, evidence=None)` — Number of configurations of the hidden (non-query, non-evidence) variables, i.e. the number of interference phases.
- **class `QLBNModel(**params)`** — Options: net (BayesNet), query (variable), conditions {name: evidence dict}. Parameters: phases theta_1 .. theta_{H-1} of the hidden configurations (theta_0 = 0); H <= 8. Design: condition names. Prediction: distribution of the query in each condition.
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
- `quantum_like_marginal(net, query, evidence=None, phases=None)` — phases: sequence with one phase per hidden configuration (in itertools.product order of the hidden variables' values); default all zero.

## `qlcog.families.dynamics`

Belief dynamics: Markov, quantum and open-system models (see README.md in this folder).

- `final_yes(model, events, queries)` — P(yes) at the last query, marginalised over earlier answers.
- **class `MarkovBelief(**params)`** — Classical baseline: a yes-probability updated by each event (successes raise, failures lower); asking does not change it.
  - `answer_probs(self, events, queries)` — Probabilities of the answer sequences given at the query times, after the events.
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
  - `step(self, p, e)` — Updated yes-probability after one event (1 = success, 0 = failure).
- **class `MarkovWalk(**kw)`** — Classical baseline: evidence accumulation as a Markov random walk over N belief states (a judgement does not disturb the state).
  - `joint(self, t1, t2)` — Joint distribution of the judgements at times t1 and t2 (judgement at t1 included).
  - `state_probs(self, t)` — Distribution over belief states at time t.
- **class `OpenSystemBelief(**params)`** — Belief (e.g. trust) as a qubit: events rotate it, dephasing gamma pulls it towards the measurement axis, and asking collapses it (Lueders), so asking can change later answers.
  - `answer_probs(self, events, queries)` — Probabilities of the answer sequences given at the query times (rotation, dephasing, collapse).
- **class `OpenSystemWalk(**kw)`** — Quantum walk with Lindblad dephasing between the quantum walk (rate 0) and Markov-like behaviour (large rate).
  - `joint(self, t1, t2)` — Joint distribution of judgements at t1 and t2 with collapse at t1.
  - `state_probs(self, t)` — Distribution over belief states at time t under Lindblad dynamics.
- **class `QuantumWalk(**kw)`** — Quantum walk over N belief states (Busemeyer, Kvam and Pleskac): unitary evolution, so an intermediate judgement changes later judgements.
  - `U(self, t)` — Unitary exp(-i H t).
  - `joint(self, t1, t2)` — Joint distribution of the judgements at t1 and t2, with collapse at t1 (Lueders).
  - `state_probs(self, t)` — Distribution over belief states at time t (Born rule).
- `question_effect(model, events, last, intermediate)` — Change in P(yes at `last`) caused by also asking at `intermediate`.

## `qlcog.families.decision`

Risky choice: quantum decision theory and classical baselines (see README.md in this folder).

- `expected_utility(lottery, alpha)` — Expected utility of a lottery [(p, x), ...] with power utility x^alpha (sign-preserving).
- **class `ExpectedUtilityModel(**params)`** — Classical baseline: logit choice between two lotteries by expected utility.
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
- `outcome_variance(lottery)` — Variance of the outcomes of a lottery [(p, x), ...].
- **class `ProspectTheoryModel(**params)`** — Classical baseline: cumulative prospect theory value (loss aversion, probability weighting) with logit choice.
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
  - `value(self, lottery)` — Prospect-theory value of a lottery.
- **class `QDTModel(**params)`** — Quantum decision theory (Yukalov and Sornette): choice probability p = f + q, a utility factor f plus an attraction factor q that favours the less uncertain prospect.
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.

## `qlcog.families.contextuality`

Contextuality analysis: CHSH and Contextuality-by-Default for cyclic systems (see README.md).

- `chsh(E11, E12, E21, E22)` — CHSH value E11 + E12 + E21 - E22 (classical bound 2, quantum bound 2 sqrt 2).
- `correlations_from_data(pairs)` — pairs: dict context -> array of shape (m, 2) of +-1 outcomes (first, second variable). Returns {context: (<X Y>, <X>, <Y>)}.
- `cyclic_contextuality(product_expectations, means_pairs)` — product_expectations: [<R_1 R_2>_c1, <R_2 R_3>_c2, ..., <R_n R_1>_cn] (one per context); means_pairs: for each content q, the pair (<R_q> in its first context, <R_q> in its second context). Returns dict with s_odd, Delta, n, the contextuality measure (positive means contextual).
- `qubit_chsh_correlations(a0=0.0, a1=1.5707963267948966, b0=0.7853981633974483, b1=-0.7853981633974483)` — E(a, b) = cos(a - b) for the Bell state (|00> + |11>)/sqrt 2 measured in the x-z plane at angles a and b; the defaults give S = 2 sqrt 2.
- `s_odd(values)` — Maximum over sign patterns with an odd number of minus signs of sum_i s_i x_i (Kujala, Dzhafarov and Larsson).

## `qlcog.families.similarity`

Similarity judgements and asymmetry (see README.md in this folder).

- `asymmetry(pred, a, b)` — Sim(A, B) - Sim(B, A).
- **class `BiasedGeometricModel(**params)`** — Classical baseline with asymmetry (Nosofsky 1991): geometric similarity plus a bias per stimulus.
  - `bias(self, name)` — Bias parameter of a stimulus.
- `for_concepts(cls, concepts)` — Subclass of a similarity model whose parameters cover exactly the given concepts (so that information criteria count only the parameters in use). Pass the same concepts as an option.
- **class `GeometricModel(**params)`** — Classical baseline: similarity decreases with distance in a psychological space (symmetric).
  - `bias(self, name)` — Bias of a stimulus (zero for the symmetric model).
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
- **class `QuantumSimilarityModel(**params)`** — Options: concepts (list of names, at most 6), ranks {name: 1 or 2} (default 1). Use for_concepts(QuantumSimilarityModel, names) to get a class with exactly 2 angles per concept.
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
  - `projectors(self)` — Projector of each concept subspace.

## `qlcog.applications.robotics`

Human-robot interaction: order-aware human models for robots that ask questions and track trust.

- **class `AnchoringOrderModel(**params)`** — First answer at its first-position rate; P(second yes | first yes) = p2 + w (1 - p2) and P(second yes | first no) = p2 (1 - w) for w >= 0 (mirror image for w < 0). w depends on the second question (wA, wB). Option tied=True forces wA = wB (one weight).
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
- `ask_or_act(p_success, uncertainty=None, ask_cost=1.0, error_cost=5.0, max_disagreement_bits=0.05)` — Decision rule for a robot that may ask a person before acting.
- **class `BayesOrderModel(**params)`** — Order-free joint distribution with marginals pA, pB and correlation rho in (-1, 1) (scaled between the bounds allowed by the marginals).
  - `joint(self)` — Order-free joint distribution [yy, yn, ny, nn] with A first.
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
- `compare(candidates, data, design=None, restarts=8, rng=None, criterion='bic')` — Fit several models and rank them. candidates: list of model classes or (class, fit kwargs). Returns a list of FitResult sorted by the criterion (lower is better).
- `domain_models(domain)` — {'QL': ..., 'Anchoring': ..., 'Bayes': ...} for a domain. Bayes reproduces the QL model's A-first joint distribution without an order effect.
- `estimate_unprimed_rates(design, generator, n, rng=None, model=None, probe=0.1)` — Estimate the first-position 'yes' rates (pi_A, pi_B) from n people who answer both questions.
- `fit(model_cls, data, design=None, restarts=8, rng=None, structures=None, method='Nelder-Mead', **options)` — Fit a model class to data by multi-start optimisation.
- `HRI_DOMAINS` (constant)
- **class `HumanModelEnsemble(candidates=None, design=None)`** — Competing human models weighted by evidence.
  - `predict(self, condition)` — Mixture prediction for `condition` and its uncertainty {entropy_bits, model_disagreement_bits, total_bits}.
  - `summary(self)` — One dict per model: name, ensemble weight and BIC.
  - `update(self, data, restarts=8, rng=None)` — Fit every candidate model to `data` and set BIC weights; returns self.
- **class `HumanModelService(candidates=None, refit_every=20, min_answers=20, restarts=4, seed=0)`** — Middleware-independent service a robot can run next to its planner (the ROS 2 node in integrations/ros2 wraps it). It accumulates people's answers to two questions asked in either order, refits a HumanModelEnsemble every `refit_every` answers, and returns a prediction with its uncertainty and an ask-or-act decision. Messages are plain dicts (JSON-friendly).
  - `add_answer(self, msg)` — msg = {'order': 'AB' | 'BA', 'answers': [first, second]} with 1 = yes, 0 = no, in the order asked. Returns {'n_answers': ..., 'refitted': bool}.
  - `query(self, msg=None)` — msg = {'order': 'AB', 'ask_cost': 1.0, 'error_cost': 5.0, 'max_disagreement_bits': 0.05}. Returns the predicted answer distribution [yy, yn, ny, nn], P(yes to the first question), the uncertainty, the ensemble weights and the ask_or_act decision; before enough answers have arrived the decision is 'ask' with reason 'not enough data'.
- **class `MarkovBelief(**params)`** — Classical baseline: a yes-probability updated by each event (successes raise, failures lower); asking does not change it.
  - `answer_probs(self, events, queries)` — Probabilities of the answer sequences given at the query times, after the events.
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
  - `step(self, p, e)` — Updated yes-probability after one event (1 = success, 0 = failure).
- **class `OpenSystemBelief(**params)`** — Belief (e.g. trust) as a qubit: events rotate it, dephasing gamma pulls it towards the measurement axis, and asking collapses it (Lueders), so asking can change later answers.
  - `answer_probs(self, events, queries)` — Probabilities of the answer sequences given at the query times (rotation, dephasing, collapse).
- **class `QuantumOrderModel(**params)`** — psi = e1; u_A = (cos a, sin a, 0); u_B = (cos b, sin b cos g, sin b sin g). Option ranks=(rA, rB): rank 1 -> 'yes' is the ray of u_q; rank 2 -> the plane orthogonal to u_q.
  - `as_projective(self)` — The same model as a ProjectiveQuestionModel (state and projectors).
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
  - `projectors(self)` — 'Yes' projectors of the two questions {'A': P_A, 'B': P_B}.
  - `vectors(self)` — Unit vectors u_A and u_B that define the two questions.
- **class `QuantumOrderModel4D(**params)`** — Four-dimensional real model in which the questions can be compatible or incompatible. Basis e1..e4 = (yes,yes), (yes,no), (no,yes), (no,no) of a classical 2 x 2 table; P_A = span(e1, e2). P_B = U span(e1, e3) U^T, where U rotates by the incompatibility angle phi in the planes (e1, e4) and (e2, e3). The state psi has three hyperspherical angles. With phi = 0 the projectors commute and the model is exactly the order-free Bayesian model (psi_k^2 is the joint table), so the model nests BayesOrderModel; phi != 0 produces order effects (and the QQ equality still holds).
  - `as_projective(self)` — The same model as a ProjectiveQuestionModel (state and projectors).
  - `predict(self, design=None)` — Predictions for every condition of the design: {condition: probability vector (or predicted values)}.
  - `projectors(self)` — 'Yes' projectors of the two questions {'A': P_A, 'B': P_B}.
- `RANK_STRUCTURES` (constant)
- `TRUST_EVENTS` (constant)
- `TRUST_MODELS` (constant)
- `TRUST_PROTOCOL` (constant)

## `qlcog.quantum`

Quantum models: quantum machine learning and quantum algorithms (gate-model circuits).

- `angle_encoding(n_features, n_qubits=None, scale=1.0, circ=None)` — RY(scale * x_j) on qubit j (features cycle over qubits if there are more features).
- `apply_matrix(state, M, qubits, n)` — Apply a 2^k x 2^k matrix (same for the batch, or shape (B, 2^k, 2^k)) to `qubits` (first qubit = least significant bit of M's index).
- **class `Circuit(n)`** — Parameterised circuit. Build with gate methods, simulate with `state(w, X)`.
  - `compose(self, other)` — Append the gates of another circuit; returns self.
  - `cx(self, c, t)` — CNOT with control c and target t.
  - `cz(self, a, b)` — Controlled Z on qubits a and b.
  - `diagonal(self, phases, label='D')` — Diagonal unitary diag(exp(i phases)) on all qubits (e.g. a cost or oracle layer).
  - `h(self, q)` — Hadamard on qubit q.
  - `p(self, p, q)` — Phase gate P(p) on qubit q.
  - `probabilities(self, w=None, X_=None)` — Born-rule probabilities of the final states, shape (B, 2^n).
  - `rx(self, p, q)` — RX(p) on qubit q; p is a number, W(k), X(j) or XX(i, j).
  - `ry(self, p, q)` — RY(p) on qubit q.
  - `rz(self, p, q)` — RZ(p) on qubit q.
  - `rzz(self, p, a, b)` — RZZ(p) = exp(-i p Z_a Z_b / 2) on qubits a and b.
  - `s(self, q)` — S (phase pi/2) on qubit q.
  - `sdg(self, q)` — S dagger on qubit q.
  - `state(self, w=None, X_=None, init=None)` — Final states, shape (B, 2^n). X_: (B, n_features) or None; w: (K,) or (B, K).
  - `swap(self, a, b)` — Swap qubits a and b.
  - `to_qiskit(self, w=None, x=None, measure=True)` — Qiskit QuantumCircuit with all angles bound (x: one sample).
  - `unitary(self, M, qubits, label='U')` — Fixed unitary on `qubits`.
  - `value_and_grad(self, w, X_, loss, chunk=None)` — Loss and its exact gradient with respect to the trainable weights by the adjoint method.
  - `x(self, q)` — Pauli X on qubit q.
  - `y(self, q)` — Pauli Y on qubit q.
  - `z(self, q)` — Pauli Z on qubit q.
- `expectation_z(psi, q, n)` — <Z_q> for each state in the batch.
- `grover(n, marked, iterations=None)` — Grover search over n qubits. marked: list of basis indices, or a predicate on the bit tuple (x_0, ..., x_{n-1}). Simulated with diagonal phase layers; `to_qiskit()` exports a gate-level circuit (X and multi-controlled Z), or the diagonal form with style='diagonal'.
- **class `GroverResult(probabilities: 'np.ndarray', iterations: 'int', marked: 'list', success: 'float', circuit: 'Circuit' = None, n: 'int' = 0) -> None`** — Grover result: output distribution, number of iterations, marked states, success probability and the circuit.
  - `to_qiskit(self, measure=True, style='gates')` — Qiskit circuit of the search.
- **class `Hamiltonian(terms)`** — Sum of weighted Pauli strings, e.g. Hamiltonian([(1.0, 'ZZ'), (0.5, 'XI')]).
  - `expectation(self, psi)` — <psi|H|psi> for each state of a batch (B, 2^n) -> (B,).
  - `ground_energy(self)` — Exact ground-state energy (dense diagonalisation).
- `hardware_efficient(n_qubits, layers=2, start=0, circ=None, entangle='ring', rotations=('ry', 'rz'), entangler='cz')` — Layers of single-qubit rotations (one weight each) and two-qubit entanglers ('cz' or 'cx') on a 'ring' or 'linear' chain. Weights are numbered from `start`; returns (circuit, next free weight index).
- `parameter_shift(f, w, shift=1.5707963267948966)` — Exact gradient of f(w) (any array output) for circuits in which every weight sets one rotation gate exp(-i w G / 2) with G^2 = I:  df/dw_k = [f(w + s e_k) - f(w - s e_k)] / 2 with s = pi/2. Returns an array of shape (len(w),) + shape of f(w).
- `pauli_matrix(label)` — Matrix of a Pauli string in Qiskit order (the leftmost letter acts on the highest qubit).
- **class `QAOA(qubo, p=2, restarts=6, maxiter=300, seed=0)`** — Quantum Approximate Optimisation Algorithm (Farhi, Goldstone and Gutmann, 2014) for a `qlcog.problems.Qubo`. The cost layer is built from RZ and RZZ gates of the equivalent Ising model, so the simulated and exported circuits are identical.
  - `expectation(self, w)` — Expected QUBO energy of the QAOA state for angles w = (gamma_1, beta_1, ...).
  - `run(self)` — Schedule search: (1) grid search at depth 1, then layer-by-layer interpolation of the optimal angles to the next depth (INTERP, Zhou et al., PRX 10, 021067, 2020); (2) random restarts at full depth. The best schedule over both is kept.
  - `to_qiskit(self, result, measure=True)` — Qiskit circuit with the optimised angles of `result` bound.
- **class `QAOAResult(x: 'np.ndarray', energy: 'float', expectation: 'float', probabilities: 'np.ndarray', gammas: 'np.ndarray', betas: 'np.ndarray', circuit: 'Circuit' = None, optimum: 'float' = None, p_optimal: 'float' = 0.0, weights: 'np.ndarray' = None) -> None`** — QAOA result: best bit string among the most probable outcomes, its energy, the expectation, the output distribution, the angles, the exact optimum and P(optimal).
- **class `QuantumKernel(reps=2, scale=1.0)`** — Fidelity kernel k(x, x') = |<phi(x)|phi(x')>|^2 with the ZZ feature map (Havlicek et al. 2019). Use it with any kernel method, e.g. scikit-learn's SVC(kernel='precomputed').
  - `fit(self, X)` — Fit the feature scaling and build the feature map for X; returns self.
  - `states(self, X)` — Feature-map states of the rows of X, shape (samples, 2^features).
  - `to_qiskit(self, x1, x2)` — Compute-uncompute circuit: P(all zeros) = k(x1, x2).
- **class `QuantumKernelClassifier(reps=2, scale=1.0, ridge=0.01)`** — Kernel ridge classifier (one-versus-rest, targets +-1) on the quantum kernel.
  - `decision_function(self, X)` — Scores per class (one-versus-rest).
  - `fit(self, X, y)` — Solve the kernel ridge system for X and y; returns self.
  - `predict(self, X)` — Class with the highest score for each sample.
  - `score(self, X, y)` — Accuracy on (X, y).
- `reuploading_classifier_circuit(n_features, n_qubits, layers, reupload=True)` — Data re-uploading classifier (Perez-Salinas et al., Quantum 4, 226, 2020): encoding and trainable layers alternate, so the model is a trainable Fourier series of the inputs.
- **class `VariationalClassifier(layers=3, n_qubits=None, reupload=True, l2=0.001, maxiter=200, seed=0)`** — Variational quantum classifier with data re-uploading.
  - `fit(self, X, y)` — Train on X (samples x features) and labels y; returns self.
  - `predict(self, X)` — Most probable class for each sample.
  - `predict_proba(self, X)` — Class probabilities (Born rule, restricted to the class labels).
  - `score(self, X, y)` — Accuracy on (X, y).
  - `to_qiskit(self, x, measure=True)` — Circuit for one input sample, weights bound.
- **class `VQE(hamiltonian, layers=2, restarts=4, maxiter=400, seed=0, rotations=('ry', 'rz'), entangle='linear', entangler='cx')`** — Variational Quantum Eigensolver (Peruzzo et al., Nature Communications 5, 4213, 2014) with a hardware-efficient ansatz and parameter-shift gradients. rotations=('ry',) gives a real ansatz, enough for Hamiltonians with real ground states (e.g. transverse-field Ising models).
  - `energy(self, w)` — Energy expectation of the ansatz state for weights w.
  - `run(self)` — Optimise from several random starts; returns {energy, exact, weights, state}.
  - `to_qiskit(self, measure=True)` — Qiskit circuit of the optimised ansatz.
- **class `W(k: 'int', scale: 'float' = 1.0) -> None`** — Trainable weight k (times a constant scale).
- **class `X(j: 'int', scale: 'float' = 1.0, shift: 'float' = 0.0) -> None`** — Feature j of the input sample (times scale, plus shift).
- **class `XX(i: 'int', j: 'int', a: 'float' = 3.141592653589793, scale: 'float' = 1.0) -> None`** — Product feature  scale * (a - x_i)(a - x_j)  used by ZZ feature maps.
- `zz_feature_map(n_features, reps=2, circ=None)` — Second-order Pauli-Z feature map of Havlicek et al. (Nature 567, 209-212, 2019): H, RZ(2 x_i) and RZZ(2 (pi - x_i)(pi - x_j)) on every pair, repeated.

## `qlcog.inspired`

Quantum-inspired models: classical algorithms that borrow quantum ideas. No quantum computer is used.

- **class `MPSClassifier(bond=6, local_dim=2, epochs=60, lr=0.02, batch=32, seed=0, method='adam', sweeps=6, steps=40, sweep_lr=0.05, cutoff=1e-10)`** — Parameters: bond (maximum bond dimension D), local_dim (d), method ('adam' or 'sweep'), epochs, lr, batch (Adam training); sweeps (back-and-forth sweeps), steps (optimisation steps per bond), sweep_lr, cutoff (relative singular-value cutoff) (sweep training); seed.
  - `fit(self, X, y)` — Train on X (samples x features) and labels y; returns self.
  - `predict(self, X)` — Most probable class for each sample.
  - `predict_proba(self, X)` — Class probabilities (softmax of the MPS scores).
  - `score(self, X, y)` — Accuracy on (X, y).
- **class `OptimResult(x: 'np.ndarray', value: 'float', history: 'list' = <factory>, evaluations: 'int' = 0) -> None`** — Optimiser result: best solution x, its value, the best value per iteration (history) and the number of objective evaluations.
- **class `QIEA(problem, n=None, pop=20, generations=300, rotation='lookup', delta=0.031415926535897934, migrate_every=20, groups=4, seed=0)`** — Quantum-inspired evolutionary algorithm for binary minimisation.
  - `run(self)` — Run the evolutionary search; returns OptimResult.
- **class `QPSO(f, lower, upper, particles=30, iterations=300, beta0=1.0, beta1=0.5, seed=0)`** — Quantum-behaved particle swarm optimisation for continuous minimisation on a box.
  - `run(self)` — Run the swarm; returns OptimResult.
- `simulated_annealing(qubo, sweeps=400, T0=2.0, T1=0.01, seed=0)` — Classical simulated annealing baseline (single-spin Metropolis on the same Ising model).
- **class `SQA(qubo, replicas=16, sweeps=400, T=0.05, gamma0=3.0, gamma1=0.001, seed=0)`** — Simulated quantum annealing (path-integral Monte Carlo) for a Qubo or Ising model.
  - `run(self)` — Run the annealing schedule; returns OptimResult with the best bit string found in any replica.

## `qlcog.problems`

Binary optimisation problems shared by the quantum and quantum-inspired solvers.

- `from_ising(h, J, c=0.0)` — QUBO with the same energies as  h.s + sum_{i<j} J_ij s_i s_j + c,  s = 1 - 2x.
- `knapsack(values, weights, capacity, penalty=None)` — Maximise value subject to total weight <= capacity, using binary slack variables.
- `maxcut(edges, n=None, weights=None)` — Maximum cut of a graph as a minimisation (energy = -cut weight).
- `portfolio(mu, cov, budget, risk=0.5, penalty=None)` — Choose exactly `budget` assets: minimise risk * x' cov x - mu' x.
- **class `Qubo(Q: 'np.ndarray', offset: 'float' = 0.0, labels: 'list' = <factory>, name: 'str' = 'QUBO') -> None`** — Quadratic unconstrained binary optimisation problem: minimise x^T Q x + offset over x in {0, 1}^n.
  - `all_energies(self)` — Energies of all 2^n bit strings, index k <-> bits of k with x_0 the least significant bit (Qiskit ordering).
  - `brute_force(self)` — Exact minimum (x, energy) for small n.
  - `energy(self, x)` — Energy of one bit string (1-D) or of each row of a 2-D array of bit strings.
  - `normalised(self)` — Upper-triangular form with the same energies.
  - `to_ising(self)` — Spins s = 1 - 2x in {+1, -1}. Returns (h, J, c) with energy = h.s + sum_{i<j} J_ij s_i s_j + c.
- `task_allocation(costs, penalty=None)` — costs[a, t]: cost of agent a doing task t. Every task goes to exactly one agent. Variable x_{a,t} has index a * n_tasks + t.

## `qlcog.circuits`

Qiskit circuits for the model families and a single run() for simulators and cloud hardware (see README.md in this folder).

- `belief_circuit(model, events, queries, form='dynamic')` — OpenSystemBelief over a sequence of events. Dephasing of strength gamma is applied by an ancilla that triggers Z with probability gamma/2. decode -> P(yes) at the last query.
- `chsh_circuit(a, b)` — Bell pair measured at angles a (qubit 0) and b (qubit 1) in the x-z plane; decode -> E(a, b).
- `conjunction_circuit(model, form='dynamic')` — Circuit for a QuantumConjunctionModel: the two events asked in the model's order (more likely first); decode -> {'A&B': ..., 'A|B': ...}.
- `interference_circuit(p1, p2, c, theta, condition='unknown')` — Two-path interference (InterferenceModel, normalised form). Qubit 0: path (event), qubit 1: action (|0> = act). Known conditions prepare one path; 'unknown' superposes both with relative phase theta and erases the path with a Hadamard gate, post-selecting path = 0. decode -> [P(act), P(not act)].
- `n_qubits(d)` — Number of qubits needed for dimension d (at least 1).
- `order_effects_circuit(model, order='AB', form='dynamic')` — Circuit for a QuantumOrderModel (or ProjectiveQuestionModel); decode -> [yy, yn, ny, nn].
- `projective_sequence_circuit(psi, projectors, order, form='dynamic')` — Sequential yes/no questions. psi: state (length d); projectors: {name: d x d 'yes' projector}. decode(counts) -> {answer tuple: probability} with 1 = yes, in the order asked.
- `qlbn_circuit(net, query, evidence=None, phases=None)` — Quantum-like Bayesian network inference: amplitudes sqrt(P(v, e, h)) e^{i theta_h} are loaded with StatePreparation, the hidden register is put through Hadamard gates and post-selected on 0, which sums the amplitudes over the hidden configurations. decode -> distribution of the query.
- `run(qc, backend='aer', shots=10000, seed=7)` — Run a circuit on `backend` (see the module docstring for the backend strings) and return counts in Qiskit bit order.
- `similarity_circuit(model, a, b, form='dynamic')` — Circuit for Sim(a, b) of a QuantumSimilarityModel; decode -> similarity.
- `subspace_unitary(P, n)` — Unitary on n qubits whose first r rows are an orthonormal basis (conjugated) of range(P), so that it maps range(P) onto span(|0>, ..., |r-1>). P is d x d with d <= 2^n; padding dimensions are completed to a full unitary.
- `to_qasm3(qc)` — Transpile to {rz, sx, x, cx, measure} and export OpenQASM 3 in Braket's gate names (sx -> v, cx -> cnot; no stdgates include). Used for every Braket backend.
- `walk_circuit(model, t)` — Quantum walk at time t (QuantumWalk): state preparation, U(t) = exp(-iHt) as a unitary gate on the padded space, measurement of every qubit. decode -> distribution over the model's categories.

## `qlcog.viz`

Bloch-sphere visualisation: watch qubits evolve in 3D.

- `animate_bloch(traj, theme='dark', fps=30, trail=True, title=None, height=500, max_frames=400)` — Animated interactive figure (play / pause buttons and a slider labelled with the frame labels).
- `animate_trajectory(traj, theme='dark', interval=40, trail=True, save=None, fps=25, size=3.6, rotate=0.0, title=None, dpi=90)` — Animate a Trajectory (one sphere per qubit). save: path ending in .gif (Pillow) or .mp4 (ffmpeg). rotate: degrees of camera rotation per frame. Returns the FuncAnimation.
- `belief_trajectory(model, events, steps=12)` — Trajectory of the belief qubit of qlcog's OpenSystemBelief (trust) model over a sequence of events (1 = success, 0 = failure): each event rotates the belief about the y axis and dephasing of strength gamma pulls the vector towards the z axis. |0> (north pole) = 'yes, I trust the robot'.
- `bloch_figure(vectors, names=None, theme='dark', title=None, height=460)` — Interactive figure with one sphere per qubit. vectors: (qubits, 3) or a state.
- `bloch_tomography(qc, backend='aer', shots=4000, seed=7)` — Bloch vector of every qubit estimated from measurements on a simulator or quantum hardware (single-qubit state tomography). backend: any string accepted by qlcog.circuits.run.
- `bloch_vector(state)` — Bloch vector (x, y, z) of a qubit given as a 2-vector or a 2 x 2 density matrix.
- `bloch_vectors(state, n=None)` — Bloch vectors of every qubit, shape (n, 3).
- **class `BlochSphere(ax=None, title='', theme='dark', figsize=(5, 5), elev=20, azim=35)`** — One Bloch sphere on a Matplotlib 3D axis.
  - `add_points(self, points, color=None, size=12, alpha=0.9)` — Scatter points (n, 3) on or inside the sphere.
  - `add_state(self, state, **kw)` — Arrow for a qubit state (vector, density matrix or Qiskit object).
  - `add_trajectory(self, points, color=None, lw=1.8, alpha=0.8)` — Draw a path of Bloch vectors (n, 3).
  - `add_vector(self, r, color=None, label=None, lw=3)` — Arrow from the origin to Bloch vector r. Returns the artists (line, tip).
  - `save(self, path, dpi=150)` — Save the figure to `path`.
  - `show(self)` — Show the figure.
- `circuit_trajectory(qc, steps=12, initial=None)` — Smooth Bloch-sphere trajectory of every qubit of a Qiskit circuit, gate by gate.
- `concurrence(rho)` — Wootters concurrence of a two-qubit density matrix (0 = separable, 1 = maximally entangled).
- `entanglement_summary(state, n=None)` — Per-qubit Bloch-vector length (1 = not entangled with the rest, for a pure global state) and the matrix of pairwise concurrences.
- **class `LiveBloch(n_qubits=1, names=None, theme='dark', backend='auto', trail=True, title=None)`** — Real-time Bloch spheres.
  - `play(self, traj, fps=30)` — Replay a Trajectory in real time.
  - `reset(self)` — Clear the trails; returns self.
  - `show(self)` — Display the spheres (widget in Jupyter, window elsewhere); returns self.
  - `update(self, state, label='')` — state: Bloch vectors (n, 3), or a state vector / density matrix / Qiskit state.
- `plot_bloch(vectors, names=None, theme='dark', title=None, size=3.6)` — Static figure with one sphere per qubit. vectors: (qubits, 3) or a state (vector/density/Qiskit).
- `plot_entanglement(traj, pairs=None, theme='dark', size=(8, 3.2))` — Timeline of a circuit trajectory: the Bloch-vector length of every qubit (1 = not entangled with the others, for a pure global state) and the concurrence of qubit pairs (0 = separable, 1 = maximally entangled), with the gate labels on the x axis.
- `plot_qsphere(state, **kw)` — Q-sphere of a multi-qubit state (basis states placed by Hamming weight, amplitude as size and phase as colour), drawn by Qiskit's plot_state_qsphere.
- `reduced_density(state, q, n)` — Reduced density matrix of qubit q of an n-qubit state vector or density matrix (Qiskit order: qubit q = bit q of the basis index).
- `reduced_density_pair(state, a, b, n)` — Reduced 4 x 4 density matrix of qubits a and b of an n-qubit state vector (index bit 0 = qubit a).
- `rotation_trajectory(axis, angle, start=(0, 0, 1), steps=60)` — Rotation of a Bloch vector about `axis` by `angle` (useful for demonstrations).
- `save_html(fig, path, auto_open=False)` — Standalone HTML (Plotly JavaScript embedded from its CDN).
- `state_from_bloch(r)` — Density matrix of a Bloch vector.
- `THEMES` (constant)
- `tomography_circuits(qc)` — Three copies of a (measurement-free) circuit measured in the Z, X and Y bases on every qubit.
- **class `Trajectory(vectors: 'np.ndarray', labels: 'list' = <factory>, names: 'list' = <factory>, title: 'str' = '', states: 'list' = <factory>) -> None`** — Bloch vectors over time: vectors (frames, qubits, 3) and one label per frame.
  - `concurrence(self, a=0, b=1)` — Concurrence of qubits a and b per frame (needs the full states, i.e. a circuit trajectory).
  - `purity(self)` — |r| per frame and qubit (1 = pure, 0 = maximally mixed or maximally entangled).

## `qlcog.data`

Published aggregate data used in the examples and tests. These are the only human data shipped with the package; every value is taken from the source named and was checked against secondary reports.

- `CLINTON_GORE` (constant)
- `LINDA` (constant)
- `PRISONERS_DILEMMA` (constant)
- `proportions_to_counts(props, n)` — {condition: p} -> {condition: [round(n p), n - round(n p)]} (for fitting multinomial models when only proportions and a sample size are known).
- `TWO_STAGE_GAMBLE` (constant)
