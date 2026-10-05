# Contributing

Contributions of models, baselines, circuits, data sets and examples are welcome.

## Principles

1. **Every quantum-like model ships with the classical model it must beat.** A new family needs at
   least one classical baseline and a README section on how to tell them apart.
2. **Only published human data.** Data sets in `src/qlcog/data` must be aggregate values from a cited
   source; simulated data must be labelled as simulated wherever they appear.
3. **Circuits must match their model.** A new circuit needs a test that compares its output with the
   analytic prediction (statevector-exact where possible).
4. **One interface.** Models subclass `qlcog.core.Model`, declare `PARAMS`, `LOSS` and `predict(design)`.

## Workflow

```bash
git clone https://github.com/yeshwanthguru/quantum-cognition-robotics
cd quantum-cognition-robotics
pip install -e ".[all,dev]"
pytest -q
```

Open a pull request against `main` with tests, a README entry for the family, and an example if the
change adds a new use case. Please describe how the model was checked against its original paper.

## Reporting problems

Use the issue templates (bug report; model or data request).
