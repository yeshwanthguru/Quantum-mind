# Contributing

Contributions of models, baselines, circuits, data sets and examples are welcome.

## Principles

1. **Every quantum-like model ships with the classical model it must beat.** A new family needs at
   least one classical baseline and a README section on how to tell them apart.
2. **Only published human data.** Data sets in `src/qlcog/data` must be aggregate values from a cited
   source; simulated data must be labelled as simulated wherever they appear.
3. **Circuits must match their model.** A new circuit needs a test that compares its output with the
   analytic prediction (statevector-exact where possible).
4. **One interface.** Quantum-like models subclass `qlcog.core.Model`, declare `PARAMS`, `LOSS` and
   `predict(design)`. Classifiers follow `fit / predict / predict_proba / score`; optimisers take a
   `qlcog.problems.Qubo` (or an objective) and return a result with `x`, `value` (or `energy`) and a
   history.
5. **Name the pillar honestly.** Quantum-like (a model of people), quantum (a circuit for a quantum
   computer) or quantum-inspired (a classical algorithm); see `docs/concepts.md`.
6. **Always compare.** Quantum and quantum-inspired solvers are reported next to the exact answer (for
   small instances) and a classical baseline with the same budget.

## Workflow

```bash
git clone https://github.com/yeshwanthguru/quantum-cognition-robotics
cd quantum-cognition-robotics
pip install -e ".[all,dev]"
pytest -q
ruff check src tests examples
```

Open a pull request against `main` with tests, a README entry for the model, and an example if the
change adds a new use case. If figures in the README change, regenerate them with
`python3 docs/make_assets.py`. Please describe how the model was checked against its original paper.

## Reporting problems

Use the issue templates (bug report; model or data request).
