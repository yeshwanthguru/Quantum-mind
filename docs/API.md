# API reference

Generated from the docstrings by `docs/make_api.py`; each entry shows the signature and the first
paragraph of the docstring. Full docstrings: `help(quantum_mind.<module>.<name>)`.

## `quantum_mind.core`

Core layer: linear algebra of quantum-like models, the Model base class, fitting and comparison.

- `angles_to_unit(angles)`: Unit vector from hyperspherical angles.
- `bootstrap(result, data, design=None, n_boot=200, restarts=2, rng=None)`: Parametric bootstrap of a fitted model.
- **class `BootstrapResult(fit: 'FitResult', samples: 'dict', predictions: 'dict') -> None`**: Refits of a model to data simulated from its own fit, returned by :func:`bootstrap`.
  - `ci(self, level=0.95)`: Percentile intervals of the parameters.
  - `predictive_interval(self, level=0.95)`: Percentile intervals of the predicted probabilities (or values) per condition.
  - `se(self)`: Bootstrap standard errors of the parameters.
- `compare(candidates, data, design=None, restarts=8, rng=None, criterion='bic')`: Fit several model classes to the same data and rank them.
- `compare_individuals(candidates, data_by_person, design=None, restarts=4, rng=None, criterion='bic')`: Fit every candidate to every person and compare them.
- `complement(P)`: Projector onto the orthogonal complement, :math:`I - P`.
- `density(psi)`: Density matrix of a pure state.
- `dephase(rho, strength, basis_projectors=None)`: Partial dephasing channel.
- `evolve_density(rho, superop, t)`: Evolve a density matrix under a Lindblad generator.
- `fit(model_cls, data, design=None, restarts=8, rng=None, structures=None, method='Nelder-Mead', x0=None, **options)`: Fit a model class to data by multi-start optimisation.
- `fit_individuals(model_cls, data_by_person, design=None, restarts=4, rng=None, **fit_kwargs)`: Fit a model class separately to each person's data.
- **class `FitResult(model: 'object', loss: 'float', loglik: 'float', k: 'int', n: 'int', options: 'dict' = <factory>) -> None`**: Result of :func:`fit`.
  - `summary(self)`: Summary of the fit.
- `givens_frame(angles, d)`: Orthonormal frame built from Givens rotations.
- **class `IndividualFits(model_name: 'str', results: 'dict') -> None`**: Per-person fits of one model class, returned by :func:`fit_individuals`.
  - `parameters(self)`: Fitted parameter values across people.
- `information_gain(models, condition, design=None, weights=None)`: Expected information about the model identity from one observation in a condition.
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
- `model_posterior(models, data, design=None, prior=None)`: Posterior probabilities of the models after observing outcome counts.
- `normalize(v)`: Scale a vector to unit length.
- **class `OnlinePersonModel(model_cls, prior=None, prior_scale=1.0, n_particles=500, design=None, resample_below=0.5, move_steps=3, rng=None, prior_cov=None, **options)`**: Posterior over one person's parameters, updated one observed outcome at a time.
  - `credible_interval(self, name, level=0.9)`: Weighted quantile interval of one parameter.
  - `observe(self, data)`: Condition on a batch of outcome counts.
  - `posterior_mean(self)`: Posterior mean of the parameters on their natural scale.
  - `predict(self, design=None)`: Posterior predictive distribution for every condition.
  - `update(self, condition, outcome, count=1)`: Condition on an observed outcome.
- **class `Param(name: 'str', kind: 'str' = 'real', default: 'float' = 0.0, lo: 'float' = 0.0, hi: 'float' = 1.0) -> None`**: Declaration of one model parameter.
  - `from_free(self, x)`: Inverse of :meth:`to_free`.
  - `random_free(self, rng)`: Random unconstrained starting value for multi-start fitting.
  - `to_free(self, v)`: Map a parameter value to the unconstrained real line used by the optimiser.
- `projector(basis)`: Orthogonal projector onto the span of a set of vectors.
- `rank_conditions(models, conditions, design=None, weights=None)`: Rank candidate conditions by :func:`information_gain`.
- `recovery(generators, candidates, design, n, reps=20, rng=None, criterion='bic', restarts=8)`: Model-recovery study.
- `sequence_probabilities(psi, projectors, order)`: Probabilities of every yes/no answer sequence to a sequence of questions.
- `tvd(p, q)`: Total variation distance.
- `unitary(H, t)`: Time-evolution operator :math:`e^{-iHt}`.

## `quantum_mind.evaluation`

Evaluation and reproducibility utilities for model comparisons.

- `bootstrap_ci(values, statistic=<function mean>, confidence=0.95, n_boot=2000, rng=None)`: Percentile bootstrap confidence interval for a statistic.
- `brier_score(y_true, y_prob)`: Binary Brier score; lower is better.
- `classification_report(y_true, y_prob, bins=10)`: Return a JSON-friendly probabilistic evaluation summary.
- `compare_models(y_true, predictions, metric, n_boot=2000, n_permutations=5000, seed=0)`: Compare every pair of named probability predictions with paired tests.
- `create_manifest(experiment, dataset, models, metrics, seed, split=None, dataset_version=None, dataset_hash=None, data_source=None, model_provenance=None, parameters=None, notes=None)`: Create a JSON-serialisable reproducibility manifest.
- `expected_calibration_error(y_true, y_prob, bins=10)`: Equal-width expected calibration error; lower is better.
- `holm_bonferroni(p_values, alpha=0.05)`: Apply Holm's step-down correction and return adjusted p-values.
- `log_loss(y_true, y_prob)`: Mean binary log loss; lower is better.
- **class `ModelProvenance(model_id: 'str', category: 'str', original_reference: 'Optional[str]' = None, doi: 'Optional[str]' = None, original_algorithm: 'Optional[str]' = None, quantum_mind_extension: 'Optional[str]' = None, classical_baseline: 'Optional[str]' = None, validated_against: 'Optional[str]' = None, simulation_only: 'bool' = True, hardware_validated: 'bool' = False, human_data_validated: 'bool' = False) -> None`**: Evidence and implementation metadata for one benchmark model.
  - `to_dict(self)`: Convert provenance metadata to a dictionary.
- `paired_bootstrap_difference(y_true, p_a, p_b, metric, n_boot=2000, confidence=0.95, rng=None, chunk_size=256)`: Estimate a paired bootstrap CI for metric(A) - metric(B).
- `paired_effect_size(y_true, p_a, p_b, metric)`: Return standardised paired loss difference for log loss or Brier score.
- `paired_permutation_test(y_true, p_a, p_b, metric, n_permutations=5000, rng=None)`: Run a paired randomisation test by swapping model labels per observation.
- `sha256_file(path)`: Return the SHA-256 digest of a dataset or result file.
- `validate_provenance(record)`: Validate a provenance mapping and return a normalised dictionary.
- `write_manifest(manifest, path)`: Write a manifest as formatted JSON and return its path.

## `quantum_mind.families.order_effects`

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

## `quantum_mind.families.qlbn`

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

## `quantum_mind.families.dynamics`

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

## `quantum_mind.families.perception`

Bistable perception: quantum Zeno model versus Markov switching and gamma renewal (see README.md).

- `dwell_counts(dwell_times, edges)`: Histogram of dwell times, with a final bin for times beyond the last edge.
- `dwell_time_design(dts, max_time=20.0, n_bins=20)`: Design with equal-width dwell-time bins for several observation intervals.
- **class `GammaRenewalModel(**params)`**: Classical baseline: gamma-distributed dwell times (the usual empirical description).
  - `mean_dwell(self, dt=None)`: Mean dwell time :math:`k\theta` (independent of the observation interval).
  - `predict(self, design=None)`: Dwell-time bin probabilities (gamma distribution) for each condition.
- **class `MarkovSwitchingModel(**params)`**: Classical baseline: switches at a constant rate (exponential dwell times, no Zeno effect).
  - `mean_dwell(self, dt=None)`: Mean dwell time ``1 / rate`` (independent of the observation interval).
  - `predict(self, design=None)`: Dwell-time bin probabilities (exponential distribution) for each condition.
- `mean_dwell_time(model, dt)`: Mean dwell time predicted by a bistable-perception model.
- **class `QuantumZenoBistableModel(**params)`**: Quantum Zeno model of bistable perception.
  - `mean_dwell(self, dt)`: Mean dwell time :math:`\Delta t / \sin^2(g \Delta t)`.
  - `predict(self, design=None)`: Dwell-time bin probabilities for each condition (geometric in units of ``dt``).
  - `switch_probability(self, dt)`: Probability of a switch at one observation.

## `quantum_mind.applications.robotics`

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
- `fit(model_cls, data, design=None, restarts=8, rng=None, structures=None, method='Nelder-Mead', x0=None, **options)`: Fit a model class to data by multi-start optimisation.
- `HAZARD_COSTS` (constant)
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
- `risk_aware_ask_or_act(p_success, hazard='medium', ask_cost=1.0, p_lower=None, uncertainty=None, max_disagreement_bits=0.05, max_risk=None)`: Ask-or-act rule that scales with the hazard of the action and uses a lower confidence bound.
- `TRUST_EVENTS` (constant)
- `TRUST_MODELS` (constant)
- `TRUST_PROTOCOL` (constant)

## `quantum_mind.applications.intent`

Intent resolution from ambiguous commands and context cues (robots, assistants, interfaces).

- **class `BayesIntentResolver(intents, prior=None)`**: Classical baseline: posterior proportional to the prior times the cue likelihoods.
  - `add_cue(self, name, likelihood, theta=0.0)`: Register a cue.
  - `order_effect(self, a, b)`: Size of the order effect between two cues.
  - `posterior(self, cues)`: Intent probabilities after a sequence of cues.
- **class `QuantumIntentResolver(intents, prior=None)`**: Quantum-like intent resolver; it reduces to :class:`BayesIntentResolver` when every angle is 0.
  - `add_cue(self, name, likelihood, theta=0.0)`: Register a cue as a Kraus operator.
  - `posterior(self, cues)`: Intent probabilities :math:`|\psi|^2` after applying the cues in order.
  - `resolve(self, cues)`: Most likely intent after the cues.

## `quantum_mind.applications.calibration`

Calibration of confidence signals: measure it, correct it, and bound it.

- **class `AdaptiveConformalSets(alpha=0.1, gamma=0.01, calibration_scores=None)`**: Prediction sets for a person's next answer that keep their coverage while the person changes.
  - `bound(self)`: float: the guaranteed bound on ``|miscoverage - alpha|`` after the answers so far.
  - `predict_set(self, prob)`: Answers in the prediction set.
  - `should_ask(self, prob)`: bool: True when more than one answer is in the set.
  - `update(self, prob, answer)`: Record the observed answer and correct the working level.
- `brier_score(prob, outcome)`: Brier score (mean squared error of probabilities).
- **class `CalibrationMonitor(bins=10, alpha=0.05, min_count=10)`**: Online calibration check of one module, with a guaranteed lower bound on its success rate.
  - `ece(self)`: Expected calibration error of the outcomes recorded so far (bin centres as confidences).
  - `estimate(self, conf)`: Recalibrated success estimate and lower bound for a reported confidence.
  - `update(self, conf, success)`: Record one outcome.
- `clopper_pearson(successes, trials, alpha=0.05)`: Exact (Clopper-Pearson) confidence interval for a success probability.
- `expected_calibration_error(conf, outcome, bins=10)`: Expected calibration error (ECE).
- **class `IsotonicCalibration(x_: 'np.ndarray' = None, y_: 'np.ndarray' = None) -> None`**: Monotone, non-parametric recalibration by the pool-adjacent-violators algorithm.
  - `fit(self, conf, outcome)`: Fit the monotone map from confidence to success rate.
  - `transform(self, conf)`: Recalibrated confidences.
- `max_calibration_error(conf, outcome, bins=10)`: Largest calibration gap over the non-empty bins.
- **class `PlattScaling(a: 'float' = 1.0, b: 'float' = 0.0) -> None`**: Logistic recalibration of a binary confidence: :math:`\sigma(a\,\mathrm{logit}(c) + b)`.
  - `fit(self, conf, outcome)`: Fit the slope and intercept.
  - `transform(self, conf)`: Recalibrated confidences.
- `reliability_curve(conf, outcome, bins=10)`: Data of a reliability diagram.
- **class `SplitConformalClassifier(alpha: 'float' = 0.1, qhat: 'float' = None) -> None`**: Split conformal prediction sets for any probabilistic classifier.
  - `fit(self, prob, labels)`: Compute the conformal threshold from calibration data.
  - `predict_sets(self, prob)`: Prediction sets.
- **class `TemperatureScaling(T: 'float' = 1.0) -> None`**: Multi-class recalibration by one temperature: :math:`\mathrm{softmax}(\log p / T)`.
  - `fit(self, prob, labels)`: Fit the temperature by minimising the negative log-likelihood.
  - `transform(self, prob)`: Recalibrated class probabilities.

## `quantum_mind.applications.orchestration`

Orchestration: choose which module a robot relies on, from calibrated confidence and cost.

- **class `ConfidenceGate(modules, lam=0.05, calibrators=None, budget=None, window=50)`**: Cost-penalised confidence gate.
  - `calibrated(self, conf)`: Apply the per-module recalibration.
  - `scores(self, conf)`: Gate scores :math:`\hat c_k - \lambda\,\text{cost}_k`.
  - `select(self, conf)`: Choose a module for one situation and record the choice.
  - `select_batch(self, conf)`: Choose modules for many situations at once (no budget, no history).
- `meta_fit_offsets(logs, residual_var=0.25)`: Meta-learn the prior of :class:`MetaCalibratedGate` from earlier tasks (empirical Bayes).
- **class `MetaCalibratedGate(modules, prior_mean=None, prior_n=1.0, lam=0.05, explore=0.05, seed=0)`**: Gate that learns each module's confidence offset online, from a meta-learned prior.
  - `select(self, conf)`: Choose a module (with a small exploration probability).
  - `update(self, k, conf_k, success)`: Learn from the outcome of a call.
- **class `Module(name: 'str', cost: 'float' = 1.0) -> None`**: One module the gate can call.
- **class `RoutingTask(offsets: 'np.ndarray', mix: 'np.ndarray', noise: 'float' = 0.08, p_good: 'float' = 0.85, p_bad: 'float' = 0.35, names: 'tuple' = ('Vision', 'RL', 'Imitation', 'Fault-recovery')) -> None`**: A synthetic orchestration task with four modules and four situation types.
  - `modules(self)`: The four modules with their default costs.
  - `step(self, rng)`: One situation.
- `simulate_routing(gate, task, steps, rng, learn=True)`: Run a gate on a synthetic task.

## `quantum_mind.applications.questioning`

Adaptive questioning: which question a robot should ask next, and when to stop asking.

- **class `IndependentAnswerModel(hypotheses, questions, table)`**: Order-free answers: a fixed yes-probability per hypothesis and question.
  - `likelihood(self, h, history)`: Probability of the answer history under one hypothesis (answers independent).
  - `p_yes(self, h, history, q)`: Probability of "yes" to a question (independent of the history).
- **class `ProjectiveAnswerModel(states, projectors)`**: Order-dependent answers: Lüders rule on a belief state per hypothesis.
  - `likelihood(self, h, history)`: Probability of the whole answer history under one hypothesis.
  - `order_free(self)`: The order-blind approximation of this model.
  - `p_yes(self, h, history, q)`: Probability of "yes" to the next question, given the history.
- **class `QuestionPlanner(answer_model, prior=None, ask_cost=1.0, error_cost=5.0, repeat=False)`**: Value-of-information question planner.
  - `decide(self, history)`: Ask or act.
  - `information_gain(self, q, history)`: Expected entropy reduction (bits) of the belief from asking a question.
  - `posterior(self, history)`: Belief over hypotheses after an answer history.
  - `run(self, respond, max_questions=10)`: Interactive loop: ask until acting is better.
  - `simulate(self, truth_model, true_h, rng, max_questions=10)`: Run the planner against a simulated person.
  - `value_of_information(self, q, history)`: Expected reduction of the error cost from asking one more question.

## `quantum_mind.applications.personalisation`

Personalised human models: start from the population, adapt to each person.

- `fit_map(model_cls, data, prior, design=None, restarts=4, rng=None, **options)`: Maximum a posteriori fit of one person's data under a population prior.
- **class `PersonalisedHumanModel(model_cls, prior, design=None, outcomes=4, method='map', n_particles=300)`**: Per-person human models with partial pooling.
  - `add(self, person, condition, answers)`: Record one answer pair and refit that person.
  - `credible_interval(self, person, name, level=0.9)`: Posterior interval of one of a person's parameters (``method='online'`` only).
  - `model(self, person)`: The person's model (the population mean model for a new person).
  - `predict(self, person, design=None)`: Predicted answer distribution for one person.
- `population_prior(model_cls, data_by_person, design=None, restarts=4, rng=None)`: Convenience: fit every person by maximum likelihood and build the empirical-Bayes prior.
- **class `PopulationPrior(model_cls, mean, cov)`**: Gaussian prior on a model's free (unconstrained) parameters.
  - `neg_log_prior(self, x)`: Negative log prior density, up to a constant.
  - `online(self, n_particles=500, design=None, rng=None, **options)`: A per-person posterior that starts from this prior.

## `quantum_mind.applications.handover`

Trust-aware hand-over: a robot policy built on the trust qubit.

- `simulate_handover_session(policy, person, events_rng, n_steps=20)`: Run a policy against a simulated person whose trust follows a model.
- **class `TrustAwareHandover(model, fail_cost=10.0, slow_cost=1.0, ask_cost=0.5, wait_cost=0.3, slow_factor=0.5)`**: Hand-over policy on a trust belief.
  - `answer(self, yes)`: Update the belief after the person answers the trust question.
  - `decide(self)`: Expected cost of each action and the best one.
  - `observe(self, event)`: Update the belief after a hand-over outcome.
  - `wait(self)`: Let one step pass without a hand-over (dephasing only, open-system model).

## `quantum_mind.applications.fusion`

Multimodal fusion for robots: from detector scores, speech and gaze to one belief over intents.

- `asr_likelihood(nbest, keywords, hypotheses, floor=0.02)`: Likelihood over hypotheses from a speech recogniser's n-best list.
- `bayes_fusion(likelihoods, prior=None)`: Posterior of independent cues: prior times the product of the likelihoods.
- `compare_fusion(intents, cue_likelihoods, train, test, prior=None)`: Compare Bayesian, Dempster-Shafer and fitted quantum-like fusion on held-out trials.
- `dempster_shafer_fusion(likelihoods, reliabilities=None)`: Dempster's rule for singleton masses with discounting.
- `detector_likelihood(scores, hypotheses, temperature=1.0, floor=0.001)`: Likelihood over hypotheses from an object detector's scores.
- `direction_likelihood(direction, targets, hypotheses, kappa=5.0, floor=0.001)`: Likelihood over hypotheses from a gaze or pointing direction (von Mises-Fisher).
- `fit_incompatibility(resolver_cls, intents, cue_likelihoods, trials, prior=None, bounds=(0.0, 1.5707963267948966))`: Fit the incompatibility angle of each cue of a quantum-like resolver from logged choices.

## `quantum_mind.belief`

Belief filters that keep a robot's belief about a person as a density matrix.

- **class `BeliefTrace(probs: 'np.ndarray', coherence: 'np.ndarray', rho: 'np.ndarray') -> None`**: Per-frame outputs of a COBRA forward pass.
- **class `COBRA(n_hyp=3, n_features=3, *, beta=2.0, steps=100, batch=160, lr=0.1, l2=0.0001, restarts=4, seed=0)`**: Context-rotated density-matrix belief filter (see the module docstring for the equations).
  - `coherence(self, x)`: l1 coherence of the final belief (0 for a classical belief), shape (N,).
  - `fit(self, x, y)`: Train on sequences ``x`` of shape (N, T, n_features) and integer labels ``y`` of shape (N,).
  - `predict_proba(self, x)`: Hypothesis probabilities after the last frame, shape (N, d).
  - `start(self)`: Begin an :class:`OnlineBelief` for streaming use.
  - `trace(self, x)`: Probabilities, coherence and density matrix after every frame (a :class:`BeliefTrace`).
- **class `GaussianBayesFilter(n_hyp=3)`**: Commutative naive-Bayes filter with independent Gaussian features per hypothesis.
  - `fit(self, x, y)`:
  - `predict_proba(self, x)`:
- `min_cost_action(probs, cost)`: Action with minimum expected cost under a belief.
- **class `OnlineBelief(model, rho, prev)`**: A COBRA belief updated one frame at a time, as a robot would run it.
  - `update(self, frame)`: Fold one sensor frame into the belief and return the new probabilities.

## `quantum_mind.cognition`

Quantum cognition as a trainable model: a density-matrix belief with context-dependent measurements, and its classical competitors.

- **class `BayesCore(*a, **kw)`**: The same evidence map without rotations or dephasing: a Bayes filter with a learned likelihood.
- `belief_from_parse(reference, cells, colors)`: Belief over the 16 cells implied by a parsed reference (uniform over the matching objects), or ``None`` for ``'none'``/invalid.
- **class `BridgeResult(belief: 'np.ndarray', lm_belief: 'np.ndarray', llm_belief: 'np.ndarray', sentence: 'str', ask: 'bool', reason: 'str') -> None`**: What the bridge decided.
- **class `ContextWeightedCore(n_hyp=3, n_features=3, n_context=2, *, steps=500, lr=0.03, batch=256, restarts=3, seed=0, val_fraction=0.15)`**: Classical context effect: each context scales how much its evidence counts, ``log p = a + sum_t lambda(c_t) (W x_t + b)``.
  - `init(self, key)`:
  - `probs(self, params, X, C, mask)`:
- **class `DialogueBelief(lm, fusion='quantum core', **kw)`**: Fuse the turns of a dialogue into a belief over the objects on the table.
  - `accuracy(self, dialogues)`:
  - `features(self, dialogues)`: Evidence ``X`` (log belief of each object slot, per turn) and context ``Cx`` (instruction or follow-up) for the dialogues.
  - `fit(self, dialogues)`:
  - `labels(dialogues)`:
  - `log_loss(self, dialogues)`:
  - `predict_proba(self, dialogues)`: Probabilities over the object slots (left to right), padded slots get probability near 0.
- **class `GRUCore(*a, hidden=6, **kw)`**: A small GRU over the concatenated evidence and context, the generic recurrent baseline.
  - `init(self, key)`:
  - `probs(self, params, X, C, mask)`:
- `hermitian(A)`: Hermitian matrix from a real square matrix: symmetric part plus ``i`` times the antisymmetric part.
- `language_plan(lm, act)`: A closed-loop plan for :func:`quantum_mind.embodied.run_episode`: ground the sentence, drive to the most likely cell.
- `llm_parse(text, generate)`: The LLM's reading of the final referent: ``'red'``, ``'green'``, ``'blue'``, ``'leftmost'``, ``'rightmost'`` or ``'none'``.
- **class `LLMBridge(lm, generate=None, disagreement=0.5)`**: Ground free-form language with the small model, checked by a local LLM.
  - `ground(self, text, image, cells, colors)`: Ground ``text`` for a scene given its image strip and the object cells and colours (left to right).
- `make_dialogues(n, seed, e_after_clear=0.25, e_after_unclear=0.05)`: Two-turn dialogues about one scene: an instruction, then a follow-up that sometimes misspeaks.
- **class `MarkovContextCore(n_hyp=3, n_features=3, n_context=2, *, steps=500, lr=0.03, batch=256, restarts=3, seed=0, val_fraction=0.15)`**: Classical analogue of a rotation: a context-dependent stochastic transition between hypotheses, then the evidence.
  - `init(self, key)`:
  - `probs(self, params, X, C, mask)`:
- `measurement(w, U)`: The generalised measurement operator ``U diag(w) U^dagger``.
- `normalise(text, generate)`: Rewrite ``text`` into the robot vocabulary with the LLM; ``None`` if the reply uses any other word.
- **class `QuantumCognitionCore(n_hyp=3, n_features=3, n_context=2, *, coherent=True, dephasing=True, rotation_penalty=0.01, **kw)`**: The quantum-cognition core (see the module docstring).
  - `coherence(self, X, C, mask=None)`: l1 norm of the off-diagonal part of the final state: how much interference the model uses (0 for a Bayes filter).
  - `init(self, key)`:
  - `penalty(self, params)`:
  - `probs(self, params, X, C, mask)`:
  - `state(self, params, X, C, mask)`: Final density matrix of one sequence (``X`` of shape ``(T, F)``).
- **class `RobotLanguageModel(d=32, n_layers=2, n_heads=2, encoder='transformer', lm_weight=1.0, steps=1500, lr=0.003, batch=128, seed=0, d_img=16)`**: A tiny language model with a grounding head.
  - `complete(self, prefix='', max_words=12)`: Greedy completion of an instruction from a prefix (for example ``'please go to the'``).
  - `fit(self, utterances)`: Train on a list of :class:`~quantum_mind.cognition.corpus.Utterance` by minibatch Adam with cosine decay.
  - `ground(self, utterances)`: Belief over the 16 cells for each utterance (a list, or one utterance), shape ``(N, 16)``.
  - `ground_text(self, text, image)`: Belief over cells for a sentence (a string of vocabulary words) and an image strip ``(16, 3)``.
  - `perplexity(self, utterances)`: Per-word perplexity of the sentences under the language model (``<eos>`` included).
- `update_state(rho, w, U)`: Unnormalised post-measurement state ``M rho M^dagger`` (its trace is the probability of the outcome).

## `quantum_mind.chat`

Chat with a local LLM: type a question, press Enter, read the streamed answer; in a terminal or in a browser tab on this machine.

- `agreement(a, b)`: Word-overlap agreement (F1 of the content words, stop words dropped) between two answers, in [0, 1].
- **class `ChatError`**: A problem the user can fix (no server, no model, no package), with the fix in the message.
- **class `ChatSession(backend, system="You are a helpful, honest assistant running on the user's own computer. Answer clearly and briefly. If you are not sure, or the question needs information you do not have, say so instead of guessing.", max_turns=20, check=None, max_tokens=None)`**: A conversation with a backend.
  - `answer(self, question)`: The whole answer to ``question`` as a string.
  - `ask(self, question)`: Stream the answer to ``question`` piece by piece. The turn is added to the history when the stream ends or is closed.
  - `generate(self, prompt, **options)`: One-off answer with no history, so a session can be used as the ``generate`` function of the library's LLM bridges.
  - `messages(self, question=None)`:
  - `reset(self)`: Forget the conversation.
- **class `ConsistencyCheck(n: 'int' = 3, temperature: 'float' = 0.8, threshold: 'float' = 0.5) -> None`**: Ask again ``n - 1`` times at a higher temperature and compare the answers with the first one.
  - `score(self, backend, messages, first, max_tokens=None)`: Mean agreement of ``first`` with ``n - 1`` fresh samples (``None`` when ``n < 2``).
- `list_ollama_models(host='http://localhost:11434', timeout=5.0, allow_remote=False)`: Names of the models an Ollama server has pulled.
- **class `LlamaCppBackend(model_path, n_ctx=4096, **kw)`**: A local GGUF model file run in-process by llama.cpp (``pip install llama-cpp-python``).
  - `stream(self, messages, temperature=None, max_tokens=None, **_)`:
- `main(argv=None, stdin=None, stdout=None)`:
- `make_server(session, port=8765)`: A threading HTTP server bound to 127.0.0.1 for ``session`` (port 0 picks a free port).
- **class `OllamaBackend(model='llama3.2', host='http://localhost:11434', timeout=300.0, allow_remote=False)`**: A model served by Ollama on this machine (``/api/chat`` with streaming).
  - `stream(self, messages, temperature=None, max_tokens=None, **_)`:
- **class `ScratchBackend(checkpoint, temperature=0.7, top_k=40, top_p=0.95, max_new_tokens=200)`**: A model you trained yourself with :mod:`quantum_mind.llm` (``quantum-mind-llm train``), loaded from its checkpoint directory.
  - `stream(self, messages, temperature=None, max_tokens=None, **_)`:
- **class `ScriptedBackend(replies=('(demo backend) I am a stand-in, not a language model. Install Ollama and a model to get real answers.',))`**: Canned replies, for tests and for trying the program with no model: ``ScriptedBackend(['first reply', 'second reply'])``.
  - `stream(self, messages, temperature=None, max_tokens=None, **_)`:
- `serve(session, port=8765)`: Serve the chat page until Ctrl-C.

## `quantum_mind.llm`

Train your own language model from scratch: no pretrained weights, no other model.

- **class `BPETokenizer(merges=())`**: Byte-pair encoding over bytes.
  - `decode(self, ids, skip_special=True)`: The text for ``ids`` (invalid UTF-8 produced by a model is replaced, never an error).
  - `encode(self, text, bos=False, eos=False)`: Token ids for ``text`` (optionally wrapped in ``<bos>`` / ``<eos>``).
  - `save(self, path)`:
- **class `ChatData(conversations, tokenizer, block_size, system=None, val_fraction=0.05, seed=0)`**: Question-answer examples in the chat format, trained on the answer only.
  - `sample(self, batch_size, rng)`:
  - `validation(self, n_batches, batch_size)`:
- `format_chat(tokenizer, messages, add_generation_prompt=True)`: Token ids for a chat: ``<bos>`` then, per message, ``<user>`` or ``<assistant>`` and the text; answers end with ``<eos>``.
- **class `GPTConfig(vocab_size: 'int' = 1024, d_model: 'int' = 128, n_layers: 'int' = 4, n_heads: 'int' = 4, block_size: 'int' = 128, dropout: 'float' = 0.0, compute_dtype: 'str' = 'float32') -> None`**: GPTConfig(vocab_size: 'int' = 1024, d_model: 'int' = 128, n_layers: 'int' = 4, n_heads: 'int' = 4, block_size: 'int' = 128, dropout: 'float' = 0.0, compute_dtype: 'str' = 'float32')
- **class `LanguageModel(cfg, tokenizer, params=None, seed=0)`**: A tokenizer, a config and the parameters of a GPT-style model.
  - `chat(self, messages, **kw)`: The model's answer to a list of ``{"role", "content"}`` messages, using the chat format it was trained on.
  - `generate(self, prompt, **kw)`: A continuation of ``prompt`` as one string.
  - `logits(self, tokens)`: Logits for a ``(B, T)`` batch of token ids.
  - `perplexity(self, text)`: Per-token perplexity of ``text`` (split into windows of ``block_size``).
  - `save(self, directory)`: Write ``config.json``, ``tokenizer.json`` and ``weights.npz`` to a directory.
  - `stream(self, prompt, **kw)`: Yield the text of a continuation of ``prompt`` piece by piece (multi-byte characters are held back until they are complete).
  - `stream_chat(self, messages, **kw)`:
  - `stream_ids(self, ids, max_new_tokens=100, temperature=0.8, top_k=40, top_p=0.95, seed=0, stop_ids=None)`: Yield new token ids one at a time after the prompt ``ids``. ``temperature=0`` is greedy decoding.
- `library_corpus(package='quantum_mind', held_out_fraction=0.2, seed=0)`: Documents and question-answer pairs built from the docstrings of ``package``.
- **class `MixedData(sources)`**: Draw each training example from one of several sources with the given weights (for example ``[(text, 0.7), (chat, 0.3)]``).
  - `sample(self, batch_size, rng)`:
  - `validation(self, n_batches, batch_size)`:
- `PRESETS` (constant)
- `read_corpus(paths, suffixes=('.txt', '.md', '.rst', '.py', '.tex', '.csv'), max_chars=None)`: The text of every file under ``paths`` (files or directories) with a text-like suffix, as a list of strings.
- `read_qa(path)`: Question-answer pairs from a JSON-lines file: ``{"question": ..., "answer": ...}`` or ``{"messages": [...]}`` per line.
- `SPECIAL_TOKENS` (constant)
- **class `TextData(texts, tokenizer, block_size, val_fraction=0.05, seed=0)`**: Random windows of ``block_size + 1`` tokens from a corpus of documents, split into training and held-out documents.
  - `sample(self, batch_size, rng)`:
  - `validation(self, n_batches, batch_size)`:
- `train(model, data, steps=1000, batch_size=32, lr=0.0003, warmup=100, weight_decay=0.1, grad_clip=1.0, accum=1, eval_every=200, eval_batches=8, log_every=50, seed=0, checkpoint_dir=None, checkpoint_every=0, resume=False, callback=None, quiet=False)`: Train ``model`` in place and return the history (a list of dicts with ``step``, ``loss``, ``val_loss``, ``tokens_per_s``).

## `quantum_mind.embodied`

Embodied AI: a tiny vision-language-action (VLA) prototype for one motor, as a framework of replaceable slots.

- `ALLOWED` (constant)
- `assign_targets(motor_positions, target_positions, solver='exact', balance=0.5, **kw)`: Assign each target to a motor. Returns ``(mapping, total_travel, problem)``; ``mapping[target] = motor``.
- **class `BayesSlot()`**:
  - `fit(self, batch)`:
  - `posterior(self, g0, L, ctx, mask)`:
- `bc_plan(model)`:
- **class `BCBaseline(n_cells=16, hidden=64, seed=0)`**: The usual end-to-end behaviour-cloning policy: image, instruction and state in, command out (one hidden layer).
  - `command(self, image, text, theta, omega)`:
  - `fit(self, scenes, steps=3000, batch=128, lr=0.01, seed=0)`:
- `calibrate_gate(model, scenes, alpha=0.1)`: Split-conformal gate fitted on held-out scenes: ask when the prediction set over cells has more than one cell.
- `cell_centres(n_cells)`:
- `clarification(scene, rng=None, n_cells=16, pixel_noise=0.1)`: The person's answer to "which one?": a word that singles out the intended target.
- `COLORS` (constant)
- **class `ControlRuntime(hardware, act, brain=None, safety=None, heal=True, dt=0.05)`**: Fixed-rate loop: read, estimate, control, supervise, filter for safety, write. One :meth:`step` is one control period.
  - `run(self, steps, schedule=None)`: Run ``steps`` periods; ``schedule`` maps a step index to a new target.
  - `start(self)`:
  - `step(self, target=None)`:
- **class `Decision(action: 'str', reason: 'str' = '', source: 'str' = 'rules') -> None`**: Decision(action: 'str', reason: 'str' = '', source: 'str' = 'rules')
- `expert_command(target, theta, omega, kp=3.0, kd=1.0)`: Saturated PD command toward ``target`` (used as the demonstrator; it sees the true target).
- `fuse(g_first, g_answer, eps=1e-06)`: Classical belief update after an answer: product of the two groundings, renormalised (Bayes with independent evidence).
- **class `Health`**: Health level a sub-agent reports, from fine (``OK``) to out of service (``FAILED``).
- **class `Heartbeat(timeout=0.5)`**: Watchdog: FAILED if no telemetry arrived within ``timeout`` seconds.
  - `update(self, ctx)`:
- **class `JointEstimator(mode, dt=0.05, stroke=0.2, mems=None, quantum=None, enc_std=0.005, vel_process=3.0)`**: Kalman filter for (position, velocity, MEMS bias) in workspace units; ``mode`` is one of :data:`MODES`.
  - `reset(self, z0)`:
  - `step(self, meas, u=0.0)`:
  - `uses_accel(self)`:
- **class `LocalLLMBrain(generate, fallback=None)`**: Ask a local language model to choose the action. ``generate(prompt) -> str`` is any local text generator.
  - `decide(self, mode, statuses, history=())`:
  - `prompt(self, mode, statuses, history)`:
- `make_dataset(n_scenes, seed, n_cells=16)`: Scenes plus, for each, one random motor state and the demonstrator's command.
- **class `Mode`**: Operating mode of the robot, the states of the supervisor's state machine.
- `MODES` (constant)
- **class `MotorAgent(heal=True, quantum=True, dt=0.05, stroke=0.2, window=16, persist=3)`**: Monitor, diagnose and heal one motor. ``heal=False`` only monitors; ``quantum=False`` leaves the quantum accelerometer out.
  - `control(self, u_pd)`: Turn the controller's command into the one sent to the motor (identity unless healing and a plant fault is diagnosed).
  - `reset(self, z0)`:
  - `step(self, meas, u)`: Feed one set of measurements (and the command that produced them). Returns the state estimate ``(theta, omega)``.
- **class `MotorHardware`**: What a dev-kit driver must provide.
  - `read(self) -> 'dict'`: ``{'t': seconds, 'enc': position in -1..1 or None, 'a_mems': m/s^2 or None, 'a_q': m/s^2 or None}`` (latest values).
  - `stop(self) -> 'None'`: Cut torque / apply the brake. Must be safe to call at any time, repeatedly.
  - `write(self, u: 'float') -> 'None'`: Send a normalised command in [-1, 1] (the driver maps it to voltage, PWM or current).
- **class `MotorSubAgent()`**: Reports the motor agent's view: healing a fault, failed (safe stop) or fine.
  - `update(self, ctx)`:
- `ollama_generate(model='llama3.2', host='http://localhost:11434', timeout=30.0)`: A ``generate`` function for a local Ollama server (needs Ollama running on this machine; not exercised in the test suite).
- **class `OneMotor(gain: 'float' = 8.0, damping: 'float' = 2.0, coulomb: 'float' = 0.3, dt: 'float' = 0.05, encoder_noise: 'float' = 0.005, velocity_filter: 'float' = 0.4, randomise: 'bool' = True) -> None`**: One position-controlled motor joint (simulated).
  - `measure(self, first=False)`: Noisy ``(theta, omega)`` as the controller sees them.
  - `reset(self, theta0, rng)`:
  - `step(self, u)`:
- `oracle_plan(scene, rng)`:
- **class `PerceptionSubAgent()`**: Grounding confidence: WARN while the instruction is ambiguous (the gate would ask).
  - `update(self, ctx)`:
- `person_answers(scene, rng, e_first=0.05, e_same=0.1, e_switch=0.35, n_answers=2)`: Simulated person who answers ``n_answers`` clarifying questions about the intended object.
- **class `QuantumRobotVLA(vla: 'TinyVLA', gate: 'object' = None, slot: 'object' = <factory>, state_mode: 'str' = 'kalman + MEMS + quantum', solver: 'str' = 'exact') -> None`**: Quantum-robotics VLA (QR-VLA): one grounding model and three slots.
  - `assign(self, motor_positions, target_positions, **kw)`: Which motor goes to which target (computing strand).
  - `plan(self, person=<function person_answers>, n_questions=2)`: Closed-loop plan: ground, ask through the conformal gate when ambiguous, fuse the answers with the belief slot.
  - `run(self, scene, rng, **kw)`: One episode with the state slot between the sensors and the controller.
- **class `QuantumSlot()`**:
  - `init(self, rng)`:
- **class `RuleBrain()`**: Deterministic policy; also the fallback when the LLM misbehaves.
  - `decide(self, mode, statuses, history=())`:
- `run_episode(scene, rng, plan, motor=None, steps=60, tol=0.0625, n_cells=16)`: Roll one episode out. ``plan(scene, rng)`` returns ``(controller, n_questions)``; ``controller(theta, omega)`` is called every control step with the noisy measurement.
- `run_fault_episode(act, rng, faults=(), policy='agent', steps=80, switch_at=30, tol=0.0625)`: Two-leg move with faults injected: go to target A, then (at ``switch_at``) to target B, which is where the score is taken.
- `run_sensed_episode(scene, rng, plan, mode, outage_at=1.0, disturbance=0.0, steps=60, tol=0.0625, n_cells=16)`: Like :func:`run_episode`, but the controller sees a state *estimate* (see :mod:`quantum_mind.embodied.state`).
- `run_supervised_episode(act, rng, faults=(), brain=None, steps=100, switch_at=30, tol=0.0625, heal=True, dropped_telemetry_at=None)`: Two-leg move under the main brain. Returns the final error, the mode trace and the supervisor.
- **class `SafetyLayer(limits=None)`**: Last line of defence between any command and the motor; independent of the AI.
  - `filter(self, u, position)`:
  - `latch(self, reason)`:
  - `release(self)`: Human action only.
- **class `SafetyLimits(u_max: 'float' = 0.8, slew: 'float' = 0.25, soft_limit: 'float' = 0.95) -> None`**: SafetyLimits(u_max: 'float' = 0.8, slew: 'float' = 0.25, soft_limit: 'float' = 0.95)
- `sample_scene(rng, n_cells=16, pixel_noise=0.1, kinds=('color', 'leftmost', 'rightmost', 'block'))`: Draw a scene with 2-3 objects, an instruction, and the intended target.
- **class `Scene(image: 'np.ndarray', text: 'np.ndarray', cells: 'list', colors: 'list', valid: 'list', target: 'int', words: 'tuple') -> None`**: One episode: an image strip, an instruction, and the ground truth behind them.
- **class `SensedMotor(motor=None, stroke=0.2, mems=None, quantum=None, enc_std=0.005, outage_at=None, disturbance=0.0, faults=())`**: A :class:`OneMotor` plus an encoder (which can drop out) and the MEMS and quantum accelerometers, all simulated.
  - `reset(self, theta0, rng)`:
  - `step(self, u)`: Advance one control step. Faults fire when time passes their ``t``: ``encoder_dropout`` (no reading), ``encoder_offset`` (reading shifted by ``mag``), ``gain_loss`` (motor gain times ``mag``), ``friction`` (Coulomb friction times ``mag``), ``push`` (extra constant external acceleration ``mag``).
  - `theta0_reading(self, theta0)`:
- **class `SensorSubAgent(quantum_timeout=3.0, bias_limit=0.15)`**: Encoder, MEMS and quantum accelerometer health.
  - `update(self, ctx)`:
- `sequence_batch(model, scenes, rng, person=<function person_answers>, **kw)`: Turn scenes into the arrays the slots use: instruction belief, answer likelihoods, contexts, mask, target index.
- **class `SimulatedHardware(faults=(), seed=0, theta0=0.0, drop_telemetry_at=None, **kw)`**: The simulator behind the :class:`MotorHardware` interface.
  - `read(self)`:
  - `stop(self)`:
  - `write(self, u)`:
- `SOLVERS` (constant)
- **class `StateMachine()`**:
  - `fire(self, event, t=0.0)`:
- **class `Status(name: 'str', health: 'Health', message: 'str' = '', data: 'dict' = <factory>) -> None`**: Status(name: 'str', health: 'Health', message: 'str' = '', data: 'dict' = <factory>)
- **class `Supervisor(subagents=None, brain=None, calibrate_steps=20, recover_steps=40, dt=0.05, slow_factor=0.5)`**: The main brain's loop: sub-agent statuses -> state-machine events -> brain decision -> command limits.
  - `human_reset(self)`: The only way out of SAFE_STOP (an operator has inspected the machine).
  - `limit(self, u)`: Apply the current action to a motor command.
  - `observe(self, ctx)`: Run one supervision step with the telemetry context; returns the current mode.
  - `report(self)`:
  - `reset(self)`:
- **class `TemperedSlot(shared=False)`**: Product rule with learned exponents. ``shared=True`` learns a single exponent (calibrated Bayes, no context).
  - `init(self, rng)`:
- **class `TinyVLA(n_cells=16, d=16, hidden=24, seed=0)`**: Grounding by attention plus a small action head (about 1.3k parameters at the defaults).
  - `act(self, target, theta, omega)`: Motor command in [-1, 1] for a target position and the measured state.
  - `expected_target(self, g)`:
  - `fit(self, scenes, steps=3000, batch=128, lr=0.01, seed=0, lam_g=1.0, lam_a=1.0)`:
  - `ground(self, image, text)`: Distribution over cells for the instruction in the image. Accepts one scene or a batch.
  - `loss_and_grads(self, img, txt, tgt, theta, omega, u_star, lam_g=1.0, lam_a=1.0, l2=0.0001)`:
- `TRANSITIONS` (constant)
- `vla_plan(model, gate=None, always_ask=False, slot=None, person=<function person_answers>, n_questions=2)`: Plan factory. No gate: act on the model's expected target. With a gate: ask when the set is not a single cell.
- `WORDS` (constant)

## `quantum_mind.computing`

Quantum computing for robot algorithms: robot problems as QUBOs, classical baselines and a Qiskit QAOA adapter.

- `compare_solvers(problem, solvers=None, seed=0)`: Solve a :class:`~quantum_mind.computing.RobotProblem` with several solvers.
- `exhaustive(qubo)`: Exact optimum by enumeration (at most 22 variables).
- `grasp_selection(quality, conflict, k, penalty=None)`: Choose exactly ``k`` grasps of high total quality that do not conflict.
- `greedy_descent(qubo, restarts=20, seed=0)`: Best-improvement single-bit descent from random starts (the simplest classical baseline).
- `multi_robot_allocation(costs, balance=0.0, penalty=None)`: Assign every task to exactly one robot.
- `qaoa(qubo, depth=2, restarts=3, shots=1000, maxiter=150, seed=0)`: QAOA through Qiskit's ``QAOAAnsatz``, simulated exactly (at most 16 qubits).
- `random_sampling(qubo, shots=1000, seed=0)`: Best of ``shots`` uniformly random bit strings: the null baseline for any sampling-based solver.
- **class `RobotProblem(name: 'str', qubo: 'Qubo', objective: 'Callable', feasible: 'Callable', decode: 'Callable') -> None`**: A robot problem with its QUBO encoding.
  - `optimum(self)`: Exhaustive optimum of the true objective over feasible bit strings (n at most 22).
- `simulated_annealing(qubo, sweeps=1000, restarts=10, t_start=None, t_end=None, seed=0)`: Single-bit-flip simulated annealing with a geometric schedule (classical baseline).
- `slot_scheduling(conflicts, n_slots, penalty=None)`: Give every robot a time slot so that conflicting robots never share one.
- **class `SolverResult(x: 'np.ndarray', energy: 'float', method: 'str', info: 'dict' = <factory>) -> None`**: What a solver returns.

## `quantum_mind.sensing`

Quantum sensing for robots: sensor models with their quantum limits, and fusion with classical sensors.

- `atom_interferometer_accelerometer(n_atoms=1000000.0, interrogation_time=0.1, cycle_time=1.0, contrast=0.5, bias_walk=1e-08, bias_std0=1e-06)`: A cold-atom interferometer accelerometer: slow, very sensitive, almost drift-free (illustrative values).
- `atom_interferometer_sensitivity(n_atoms, interrogation_time, contrast=0.5, wavelength=7.8024e-07, scaling='sql')`: Single-shot acceleration sensitivity of a light-pulse atom interferometer.
- `compare_sensor_suites(sensors, suites, duration=300.0, dt=0.01, accel_std=0.5, accel_tau=2.0, fix_period=None, fix_std=0.1, seeds=range(0, 10))`: Run the same simulated trajectories through several sensor suites.
- **class `DeadReckoningData(dt: 'float', pos: 'np.ndarray', accel: 'np.ndarray', readings: 'dict', fixes: 'np.ndarray', sensors: 'list', accel_std: 'float', accel_tau: 'float', fix_period: 'float', fix_std: 'float') -> None`**: A simulated trajectory with the measurements of every sensor.
- **class `InterrogationProblem(slot_times, contrast, weights, k, duration=120.0, dt=0.1, accel_std=0.5, accel_tau=2.0, mems=None, quantum=None, min_contrast=0.3)`**: Choose ``k`` interrogation slots of a quantum accelerometer on a mission, to minimise weighted position error.
  - `greedy(self)`: Forward selection on the exact objective: add the slot that lowers it most, k times.
  - `local_quadratic(self, base)`: Quadratic model of the weighted variance around the schedule ``base``, exact within two flips of it.
  - `monte_carlo(self, slots, seeds=range(0, 20))`: Weighted RMS position error of the actual filter on simulated trajectories (an independent check of :meth:`objective`).
  - `objective(self, slots)`: Weighted RMS position standard deviation (m) over the mission for a set of slot indices (lower is better).
  - `optimum(self)`: The exact optimum by enumerating every k-subset of the usable slots (small problems only).
  - `qubo(self, base)`: The slot-selection QUBO around ``base`` (a :class:`~quantum_mind.computing.RobotProblem` over the usable slots).
  - `random(self, seed=0)`:
  - `uniform(self)`: Evenly spaced usable slots (the strong, simple baseline).
  - `variance(self, slots)`: Weighted position variance (m^2) for a set of slot indices: the quantity the QUBO model expands (``objective`` is its root).
- `mems_accelerometer(rate_hz=100.0, noise_density_ug=100.0, bias_walk=0.0002, bias_std0=0.02)`: A MEMS accelerometer: fast, noisy, with a drifting bias (illustrative values).
- `phase_sensitivity(n_probes, repeats=1, scaling='sql')`: Smallest phase resolvable with ``n_probes`` probes repeated ``repeats`` times.
- `random_mission(seed, n_slots=14, k=4, duration=120.0, dt=0.1, unusable=0.25, **kw)`: A random mission: slots spread over the window, some with too little contrast, and two phases where accuracy matters most.
- `RB87_WAVELENGTH_M` (constant)
- `run_filter(data, use=None, use_fixes=False)`: Run the multi-rate Kalman filter on a simulated trajectory.
- **class `Schedule(slots: 'list', objective: 'float', method: 'str', info: 'dict' = <factory>) -> None`**: A chosen set of slots.
- `schedule_interrogations(problem, method='exact', start='uniform', rounds=3, polish=False, seed=0, **solver_kw)`: Schedule the quantum sensor: iterate a local quadratic QUBO solved by a thread 1 solver.
- **class `Sensor(name: 'str', rate_hz: 'float', noise_std: 'float', bias_walk: 'float', bias_std0: 'float') -> None`**: One accelerometer.
- `simulate_dead_reckoning(sensors, duration=300.0, dt=0.01, accel_std=0.5, accel_tau=2.0, fix_period=None, fix_std=0.1, seed=0)`: Simulate a trajectory and the readings of every sensor.

## `quantum_mind.quantum`

Quantum state-vector simulator used by the Bloch-sphere viewer and the circuit exports.

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
- `expectation_z(psi, q, n)`: Expectation of Pauli Z on one qubit.
- **class `W(k: 'int', scale: 'float' = 1.0) -> None`**: Trainable weight as a gate angle.
- **class `X(j: 'int', scale: 'float' = 1.0, shift: 'float' = 0.0) -> None`**: Input feature as a gate angle.
- **class `XX(i: 'int', j: 'int', a: 'float' = 3.141592653589793, scale: 'float' = 1.0) -> None`**: Product feature used by ZZ feature maps.

## `quantum_mind.envs`

Reinforcement-learning environments with simulated people (Gymnasium API).

- **class `ClarificationEnv(model=None, ask_cost=0.3, success_reward=1.0, error_cost=5.0, max_questions=6, people=None)`**: Resolve an ambiguous request by asking questions, then act.
  - `reset(self, seed=None, options=None)`: Start an episode with a new hidden intent.
  - `step(self, action)`: Ask a question or act.
- `register_envs()`: Register the environments with Gymnasium.
- `sample_people(source, n, rng=None)`: Draw simulated people whose parameters reflect what is known, and not known, about real people.
- **class `TrustHandoverEnv(model=None, n_steps=20, fail_cost=10.0, slow_cost=1.0, ask_cost=0.5, wait_cost=0.3, slow_factor=0.5, people=None)`**: Hand objects to a simulated person whose trust follows the trust qubit.
  - `reset(self, seed=None, options=None)`: Start an episode at the model's initial trust.
  - `step(self, action)`: Take one action.

## `quantum_mind.problems`

The binary optimisation container used by trust-aware task allocation.

- **class `Qubo(Q: 'np.ndarray', offset: 'float' = 0.0, labels: 'list' = <factory>, name: 'str' = 'QUBO') -> None`**: Quadratic unconstrained binary optimisation problem.
  - `all_energies(self)`: Energies of all :math:`2^n` bit strings.
  - `brute_force(self)`: Exact minimum by enumeration (small n).
  - `energy(self, x)`: Energy of bit strings.
  - `normalised(self)`: Upper-triangular form with the same energies.
  - `to_ising(self)`: Equivalent Ising model with spins :math:`s = 1 - 2x \in \{+1, -1\}`.

## `quantum_mind.circuits`

Qiskit circuits for the model families, and a single :func:`run` for simulators and cloud hardware (see README.md in this folder). Needs the ``qiskit`` extra.

- `belief_circuit(model, events, queries, form='dynamic')`: Circuit for an :class:`~quantum_mind.families.dynamics.OpenSystemBelief` over a sequence of events.
- `n_qubits(d)`: Number of qubits needed for a dimension.
- `order_effects_circuit(model, order='AB', form='dynamic')`: Circuit for a question-order model.
- `projective_sequence_circuit(psi, projectors, order, form='dynamic')`: Circuit for a sequence of yes/no questions.
- `qlbn_circuit(net, query, evidence=None, phases=None)`: Quantum-like Bayesian network inference as a circuit.
- `run(qc, backend='aer', shots=10000, seed=7)`: Run a circuit and return counts.
- `subspace_unitary(P, n)`: Unitary that maps the range of a projector onto the first basis states.
- `to_qasm3(qc)`: Export a circuit as OpenQASM 3 in Braket's gate names.
- `walk_circuit(model, t)`: Quantum-walk circuit at time t.

## `quantum_mind.viz`

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
- `use_mpl_style(name='dark')`: Apply a theme to every Matplotlib figure that follows (used by the tutorials).

## `quantum_mind.data`

Published aggregate data used in the examples and tests.

- `CLINTON_GORE` (constant)
- `proportions_to_counts(props, n)`: Turn proportions into binary counts.
