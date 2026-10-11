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
