# quantum-cognition-robotics

[![tests](https://github.com/yeshwanthguru/quantum-cognition-robotics/actions/workflows/tests.yml/badge.svg)](https://github.com/yeshwanthguru/quantum-cognition-robotics/actions/workflows/tests.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-2.x-6929C4)

**Quantum-like models of human judgement, decision and trust for robots, AI agents and behavioural
research — with classical baselines and Qiskit circuits.**

`qlcog` (the Python package in this repository) is a library of **quantum-like (quantum-probability) models** of human judgement and
decision, each paired with the **classical models it must beat**, a common fitting and comparison
interface, and **Qiskit circuits** that run the same models on simulators and on IBM Quantum or Amazon
Braket hardware. It was built for robots and agents that ask people questions, elicit preferences and track trust,
and it is domain-agnostic: the same models apply to surveys, finance, medicine, consumer research,
human–computer interaction and AI evaluation.

> Quantum-like models use the mathematics of quantum probability (vectors, projections, interference)
> to describe judgements. They run on ordinary computers and make no claim that the brain is a quantum
> computer. The circuits are an optional second way to execute them.

## Contents

- [Model families](#model-families)
- [For robotics](#for-robotics)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Design of the package](#design-of-the-package)
- [Running on quantum hardware](#running-on-quantum-hardware)
- [Choosing a model](#choosing-a-model)
- [Data](#data)
- [Testing](#testing)
- [Limitations](#limitations)
- [Citing and licence](#citing-and-licence)

## Model families

| Family | Phenomenon | Quantum-like model(s) | Classical baselines | Circuit | README |
|---|---|---|---|---|---|
| `order_effects` | Answers depend on question order | `QuantumOrderModel4D` (nests Bayes), `QuantumOrderModel` (3D, ranks) | Bayes, anchoring, saturated | yes | [link](src/qlcog/families/order_effects/README.md) |
| `conjunction` | Conjunction and disjunction fallacies | `QuantumConjunctionModel` | classical joint, averaging, probability theory plus noise | yes | [link](src/qlcog/families/conjunction/README.md) |
| `interference` | Disjunction effect, sure-thing violations | `InterferenceModel` (normalised or not) | classical mixture | yes | [link](src/qlcog/families/interference/README.md) |
| `qlbn` | Inference with unresolved hidden causes | quantum-like Bayesian network | classical Bayesian network | yes | [link](src/qlcog/families/qlbn/README.md) |
| `dynamics` | Belief change over time; judgements that change later judgements | `QuantumWalk`, `OpenSystemWalk`, `OpenSystemBelief` | `MarkovWalk`, `MarkovBelief` | yes | [link](src/qlcog/families/dynamics/README.md) |
| `decision` | Risky choice | `QDTModel` (quantum decision theory) | expected utility, prospect theory | – | [link](src/qlcog/families/decision/README.md) |
| `contextuality` | Is there one joint distribution? | CHSH, Contextuality-by-Default criterion | (classical bounds) | yes | [link](src/qlcog/families/contextuality/README.md) |
| `similarity` | Asymmetric similarity | `QuantumSimilarityModel` | biased geometric (Nosofsky), geometric | yes | [link](src/qlcog/families/similarity/README.md) |

## For robotics

`qlcog.applications.robotics` packages the models for human–robot interaction
([README](src/qlcog/applications/README.md)):

- **Question domains**: object clarification, trust and hand-over, preference elicitation (for
  reward learning), each with quantum-like, anchoring and Bayesian populations.
- **Questioning designs**: how a robot should ask to recover what people think before its own
  questions influence them (fixed order, reversed-order probe, split order).
- **Trust protocol**: does asking "do you trust me?" change trust? Markov versus open-system dynamics.
- **`HumanModelEnsemble`**: competing human models weighted by evidence, returning a predicted answer
  distribution and its uncertainty (entropy plus model disagreement, in bits). Pass it to a planner or
  to a learned orchestrator as a confidence signal, e.g. to decide when to ask for help.

```python
from qlcog.applications.robotics import HumanModelEnsemble
from qlcog.families.order_effects import QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel
ens = HumanModelEnsemble([QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel]).update(counts)
p, uncertainty = ens.predict('AB')
```

The models are plain NumPy and run in microseconds, so they fit on a single-board computer next to
ROS 2 or LeRobot nodes; the circuits are optional.

Circuits and backends are documented in [src/qlcog/circuits/README.md](src/qlcog/circuits/README.md);
worked examples by domain in [examples/README.md](examples/README.md).

## Installation

```bash
pip install "qlcog @ git+https://github.com/yeshwanthguru/quantum-cognition-robotics"            # core
pip install "qlcog[all] @ git+https://github.com/yeshwanthguru/quantum-cognition-robotics"       # + Qiskit, IBM, Braket
```

From a clone:

```bash
git clone https://github.com/yeshwanthguru/quantum-cognition-robotics
cd quantum-cognition-robotics
pip install -e .                 # core: numpy, scipy
pip install -e ".[qiskit]"       # + Qiskit and Aer (circuits, local simulation, IBM fake-backend noise)
pip install -e ".[cloud]"        # + IBM Qiskit Runtime and Amazon Braket SDK (hardware)
pip install -e ".[all,dev]"      # everything + pytest
```

Python 3.10 or later. The examples and tests also run without installing (they add `src/` to the path).

## Quick start

```python
import numpy as np
from qlcog.core import compare
from qlcog.families.order_effects import QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel, qq_test

counts = {'AB': np.array([212, 48, 61, 179]),   # A asked first: yes-yes, yes-no, no-yes, no-no
          'BA': np.array([240, 33, 52, 175])}   # B asked first
print(qq_test(counts))                           # (q, z, p)
for r in compare([QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel], counts):
    print(type(r.model).__name__, round(r.bic, 1), r.model.params)
```

Run the same model as a circuit:

```python
from qlcog.circuits import order_effects_circuit, run
from qlcog.core import tvd
model = QuantumOrderModel4D(t1=0.7, t2=1.0, t3=0.6, phi=0.4)
qc, decode = order_effects_circuit(model, 'AB', form='dynamic')
cells = decode(run(qc, 'aer:FakeTorino', shots=8000))     # IBM Heron noise model
print(cells, tvd(cells, model.predict()['AB']))
```

## Design of the package

```
src/qlcog/
  core/        linalg.py   projectors, Lueders rule, sequences, density matrices, Lindblad
               model.py    Model base class and Param (kinds: prob, angle, real, positive, bounded)
               fit.py      fit, compare (AIC/BIC), recovery (model-recovery study), kl, tvd
  families/    one subpackage per family: models.py, __init__.py, README.md
  applications/robotics.py   human-robot question domains, questioning designs, trust, model ensemble
  circuits/    builders.py (circuits + decoders), backends.py (run on Aer, IBM, Braket)
  data/        published aggregate data sets
examples/      twelve domain examples
tests/         unit tests for core, families and circuits
```

Every model follows one interface:

| Member | Meaning |
|---|---|
| `PARAMS` | list of `Param(name, kind, default, lo, hi)`; kinds map to an unconstrained vector for fitting |
| `LOSS` | `'multinomial'` (counts) or `'sse'` (mean judgements) |
| `predict(design)` | dict condition → probability vector (or predicted values) |
| `loglik(data, design)`, `sse(data, design)`, `sample(design, n, rng)` | likelihood, squared error, simulation |
| options (keyword arguments that are not parameters) | structural choices, e.g. projector ranks or a Bayesian network |

`fit(Model, data, design, restarts, structures=[...])` searches discrete structures (e.g. projector
ranks) as well as continuous parameters; `compare` ranks models by BIC or AIC; `recovery` runs a
model-recovery study before data collection, which shows how many observations are needed to tell the
models apart.

## Running on quantum hardware

```bash
python3 examples/11_cloud_run.py --dry-run                                  # local, no account
python3 examples/11_cloud_run.py --backend ibm:least_busy --shots 4000      # IBM Quantum
python3 examples/11_cloud_run.py --backend braket:arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1 --shots 1000
```

`run(circuit, backend)` accepts `aer`, `aer:<FakeBackend>`, `braket_local`, `braket_dm:<p>`,
`ibm:<device>` and `braket:<ARN>`. Braket backends need the deferred-measurement form (no mid-circuit
reset). Hardware runs cost queue time or money; report their results exactly as measured, with backend,
date and job identifier.

## Choosing a model

1. **Check the phenomenon first.** Is there an order effect (rates change with position)? A violation
   of the law of total probability? Asymmetric similarity? If not, a classical model is enough.
2. **Fit the quantum-like model and its classical baselines together** with `compare`, and prefer the
   model with the lower BIC; report all of them.
3. **Use the distinctive test** of each family (QQ equality, total-probability bounds, the
   Contextuality-by-Default criterion, the effect of an intermediate judgement), not only goodness of fit.
4. **Run a recovery study** (`qlcog.core.recovery`) with your planned sample size before collecting
   data.
5. **Beware saturation:** a model with as many parameters as data points (e.g. the interference model on
   three proportions) can represent an effect without predicting it.

## Data

`qlcog.data` contains the only human data in the package, all published aggregates:
the Clinton–Gore order effect (Moore, 2002), the Prisoner's Dilemma disjunction effect (Shafir and
Tversky, 1992), the two-stage gamble (Tversky and Shafir, 1992) and the Linda problem rate (Tversky and
Kahneman, 1983). Every other data set in the examples is simulated and labelled as such.

## Testing

```bash
python3 -m pytest -q tests        # 19 tests: core, all eight families, robotics, circuits on Aer and Braket
```

The circuit tests check every builder against its analytic model (statevector-exact for the
post-selection circuits).

## Limitations

- Fitting aggregate counts assumes a homogeneous population; individual differences can hide or mimic
  quantum-like structure.
- Several quantum-like models have been challenged by further tests (Grand Reciprocity equations for
  order effects; conjunction-fallacy tests; response replicability); the family READMEs list these.
- Circuits reproduce the analytic models on simulators; current hardware noise distorts them
  (total-variation error of a few percent for small circuits). No hardware results ship with the package.

## Citing and licence

If you use `qlcog`, please cite the package (see `CITATION.cff`) and the original papers of the models
you use (listed in each family README). Released under the Apache License 2.0 (`LICENSE`, `NOTICE`),
which permits commercial and research use and includes an explicit patent grant. Contributions are
welcome: see `CONTRIBUTING.md`.

Author: Yeshwanth Guru (ORCID 0009-0007-6353-4033), Department of Mechanical Engineering, Amrita Vishwa Vidyapeetham, Chennai,
India. Companion manuscripts (in preparation): a systematic review of quantum-like cognition for
autonomous agents, a systematic review of meta-learning orchestration on resource-constrained robots,
and "Order-Aware Human Models for Robot Questioning and Trust", whose simulations use the robotics
application of this library.
