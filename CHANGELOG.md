# Changelog

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
