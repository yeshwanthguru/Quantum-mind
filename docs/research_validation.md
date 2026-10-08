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
