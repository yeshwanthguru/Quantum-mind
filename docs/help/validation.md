# Validation status

What has been checked, how, and what has not. This page exists so that nobody has to guess.

## What is validated

| Claim | How it is checked |
|---|---|
| The code computes what the equations on the [mathematics pages](../math/index.md) say | `tests/test_math_docs.py` checks the key equations numerically against the implementation |
| Every model is fitted and compared the same way | unit tests (`pytest`), 80% coverage threshold in CI |
| Docstring examples are correct | doctests in CI |
| Tutorials run and their stated claims hold | every tutorial is executed in CI; assertions check each claim made in the text |
| Every public model has a block diagram | `tests/test_docs_coverage.py` checks the atlas against the package |
| Circuits match their analytic models | tests compare circuit outputs with the model predictions |
| Policy gradients are exact | gradient tests against finite differences |
| Quantum-like models reproduce published aggregate effects | the Clinton-Gore order effect, Linda conjunction, Prisoner's Dilemma and two-stage gamble data in `quantum_mind.data` |

## What is not validated

| Gap | Consequence |
|---|---|
| No data from human-robot interaction studies | the robotics results (ask or act, question planning, hand-over, fusion) are simulations with illustrative parameters |
| Simulated people often follow the same model the robot uses | results such as the hand-over cost comparison favour the order-aware policy by construction |
| No runs on quantum hardware are included | hardware support is tested with simulators and fake back ends only |
| The ROS 2 node has not been run inside a ROS 2 installation in CI | its callbacks are tested with stand-in modules |
| References were compiled by hand | the reference-check workflow (below) compares them with Crossref; see its report before relying on a citation detail |

## How to close the main gap: a pilot study

A small within-subject study is enough to test the central robotics claim, that asking about trust
changes trust, and that question order changes answers in human-robot interaction.

1. **Participants.** 40 or more adults; ethics approval and informed consent first.
2. **Task.** A robot (or a video of one) hands over six objects with a fixed sequence of successes and
   failures (`applications.robotics.TRUST_PROTOCOL`).
3. **Conditions.** Ask "Do you trust the robot?" only at the end, or also half-way (between subjects);
   ask the two clarification questions of `HRI_DOMAINS` in both orders (counterbalanced).
4. **Analysis.** Fit `OpenSystemBelief` and `MarkovBelief` to the trust answers and the order models to
   the clarification answers with `quantum_mind.compare`; run `qq_test`; report the question effect
   with a confidence interval. Use `quantum_mind.core.recovery` beforehand to check that the planned
   sample size can tell the models apart (tutorial 2 shows how).
5. **Report.** Whatever the result, including a null result for the quantum-like model.

Data from such a study (anonymised aggregates) can be added to `quantum_mind.data` with a citation.

## Reference check

The workflow `.github/workflows/references.yml` extracts every reference from the documentation, looks
each one up in Crossref and uploads a report with the best match, its DOI and a similarity score.
Run it from the Actions tab (it needs internet access, which the documentation build does not).
