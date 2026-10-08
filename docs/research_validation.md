# Research validation protocol

Quantum Mind now separates software correctness from scientific evidence.

## Central research question

Can quantum-probabilistic human models improve robot decisions under ambiguous,
contextual, or order-sensitive human interaction?

## Required comparison

For every HRI experiment, compare at least:
1. a simple frequency or majority baseline;
2. a Bayesian or Markov baseline appropriate to the task;
3. the quantum-like model;
4. an ensemble only when the ensemble itself is being tested;
5. a neural or LLM baseline when justified by the task and data.

Do not compare only quantum-like variants with each other.

## Required metrics

Report log loss, Brier score, expected calibration error (ECE), accuracy where
appropriate, mean decision cost, ask rate, failure/unsafe-action rate, and
latency/memory when deployment is claimed. Report 95% confidence intervals,
random seeds, paired differences, and an appropriate statistical test.

## Data hierarchy

Use the strongest available evidence: newly collected HRI data; public HRI
datasets; published aggregate data; then synthetic populations for debugging,
recovery tests, and sensitivity analysis.

The HRI defaults in applications.robotics are illustrative and must not be
described as estimates of a real HRI population.

## Reproducibility record

Record dataset/version/source, inclusion criteria, train/validation/test split,
model and parameterisation, optimisation settings, seeds, software/hardware,
metric definitions, confidence-interval method, baselines, and raw prediction
files or their checksums.

The benchmark harness in benchmarks/hri_decision_benchmark.py implements common
scoring and bootstrap reporting.


## Decision-policy assumptions

The benchmark uses an explicit binary act-vs-ask policy. The model probability is converted to an act/no-act decision at 0.5. Asking has a configurable cost and ask_resolution in [0, 1]. A value of 1.0 means the question resolves uncertainty perfectly; 0.0 means no residual uncertainty reduction. The perfect-resolution setting is an optimistic sensitivity-analysis case, not a claim about real robot questioning.

## Model provenance

Each reported model should identify its original reference, DOI where available, original algorithm, Quantum Mind extension, classical baseline, validation status, and whether evidence is simulation-only, hardware-validated, or human-data-validated.

See docs/research_claims.md for the claim-status registry.

## Quantum claims

A circuit executing successfully is not evidence of quantum advantage. Any such
claim needs a matched classical baseline, problem-size scaling, resource
accounting, noise/hardware description, and statistical uncertainty.

The built-in statevector simulator is a small-circuit reference implementation,
not a scalable substitute for quantum hardware.

## API stability

The project is organised conceptually as core/stable infrastructure,
applications for robotics/HRI, and experimental algorithms or hardware adapters.
Experimental APIs may change without the same compatibility guarantees as stable
interfaces.

## What the repository does not establish

It does not establish that the brain is quantum, that quantum-like models
universally outperform classical models, that illustrative HRI parameters
represent real people, or that current quantum hardware has an advantage over
classical simulation.
