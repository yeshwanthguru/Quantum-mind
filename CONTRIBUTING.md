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
5. **Name the pillar correctly.** Quantum-like (a model of people), quantum (a circuit for a quantum
   computer) or quantum-inspired (a classical algorithm); see `docs/concepts.md`.
6. **Always compare.** Quantum and quantum-inspired solvers are reported next to the exact answer (for
   small instances) and a classical baseline with the same budget.

## Workflow

```bash
git clone https://github.com/yeshwanthguru/quantum-cognition-robotics
cd quantum-cognition-robotics
pip install -e ".[all,dev,docs]"
pytest -q
pytest -q --doctest-modules src/qlcog            # the examples in the docstrings
ruff check src tests examples docs site
QLCOG_DOCS_FAST=1 python -m sphinx -b html docs docs/_build/html   # documentation preview
```

## Documentation

Every public class, function and method has a NumPy-style docstring (a one-line summary, then
`Parameters`, `Returns`, `Raises` and, where useful, `Examples` that run as doctests). Formulas use
`:math:`. The API reference in the documentation and `docs/API.md` are generated from these
docstrings; regenerate the latter with `python3 docs/make_api.py`.

Examples in `examples/` are also the documentation's example gallery. Start each script with a
docstring whose first line is a title underlined with `=`, and split the code into steps with
`# %%` comment blocks that explain what each step does; the gallery shows them as text between the
code cells.

Open a pull request against `develop` with tests, a README entry for the model, and an example if the
change adds a new use case. If figures in the README change, regenerate them with
`python3 docs/make_assets.py`. Please describe how the model was checked against its original paper.

## Branches and releases

| Branch | Purpose |
|---|---|
| `develop` | All day-to-day work. Push here, or open pull requests against it. CI runs on every push. |
| `main` | Released code only. It changes only through a pull request from `develop`, and every merge is a release. The website and documentation are deployed from `main`. |

Release steps:

1. On `develop`, update the version in `pyproject.toml`, `src/qlcog/__init__.py` and `CITATION.cff`,
   and move the changelog entries under a new version heading.
2. Open a pull request from `develop` to `main`; merge it when CI is green.
3. Create a GitHub release from `main` with the tag `v<version>` (for example `v1.6.0`). The release
   workflow checks that the tag matches the package version, builds the package and publishes it to
   PyPI.
4. Bring `develop` up to date with `main` if anything was changed during the release.

## Reporting problems

Use the issue templates (bug report; model or data request).
