<p align="center">
  <img src="https://raw.githubusercontent.com/yeshwanthguru/Quantum-mind/main/docs/assets/banner.svg" alt="quantum-mind: quantum-probability cognition for robots that work with people" width="100%">
</p>

<p align="center">
  <a href="https://pypi.org/project/quantum-mind/"><img src="https://img.shields.io/pypi/v/quantum-mind?label=PyPI" alt="PyPI version"></a>
  <a href="https://pypistats.org/packages/quantum-mind"><img src="https://img.shields.io/pypi/dm/quantum-mind?label=downloads%2Fmonth" alt="PyPI downloads per month"></a>
  <a href="https://pepy.tech/projects/quantum-mind"><img src="https://static.pepy.tech/badge/quantum-mind" alt="Total PyPI downloads"></a>
  <a href="https://github.com/yeshwanthguru/Quantum-mind/actions/workflows/tests.yml"><img src="https://github.com/yeshwanthguru/Quantum-mind/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <a href="https://github.com/yeshwanthguru/Quantum-mind/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white" alt="Python 3.10-3.12">
  <a href="https://yeshwanthguru.github.io/Quantum-mind/docs/"><img src="https://img.shields.io/badge/docs-Sphinx-0969da" alt="Documentation"></a>
</p>

<p align="center"><b><a href="https://yeshwanthguru.github.io/Quantum-mind/">Website and live Bloch-sphere playground</a> · <a href="https://yeshwanthguru.github.io/Quantum-mind/docs/">Documentation</a> (user guide, example gallery, API reference)</b></p>

<p align="center">
  <b><a href="#idea">The idea</a></b> ·
  <b><a href="#cobra">COBRA</a></b> ·
  <b><a href="#layers">Three layers</a></b> ·
  <b><a href="#quick-start">Quick start</a></b> ·
  <b><a href="#viewer">Bloch viewer</a></b> ·
  <b><a href="#validation">Validation</a></b> ·
  <b><a href="#status">Status</a></b> ·
  <b><a href="https://yeshwanthguru.github.io/Quantum-mind/docs/tutorials/index.html">Tutorials</a></b> ·
  <b><a href="https://yeshwanthguru.github.io/Quantum-mind/docs/atlas/index.html">Model atlas</a></b> ·
  <b><a href="https://yeshwanthguru.github.io/Quantum-mind/docs/math/index.html">Mathematics</a></b>
</p>

---

**Quantum Mind** (`pip install quantum-mind`, `import quantum_mind`) is a Python library for **quantum-probability
cognition in robotics**: it models how people judge, decide and trust, and how a robot should read them, with the
mathematics of quantum probability (projections, interference, density matrices, generalised measurements), and
it turns every belief into a cost-aware decision: hand over, wait, or ask.

> It runs on ordinary computers and makes **no claim of quantum computational advantage**. "Quantum-like" means
> the probability rules are borrowed; nothing in the brain or the robot is assumed to be quantum. Every model is
> reported next to the classical model it has to beat, with the ablation that isolates what is new.

<a id="idea"></a>

## The idea

Classical filters treat evidence as arriving in neutral, independent pieces. People and robots do not live in that
world:

* a **look** changes what the next **hand movement** means;
* a **question** changes the answer to the next question (order effects);
* **asking** a person whether they trust the robot changes their trust.

Quantum probability represents exactly this: beliefs are vectors or density matrices, evidence is a measurement,
and measurements need not commute. Quantum Mind turns that into tested code in three layers: quantum-like models of
people, a belief filter for the robot (COBRA), and a decision layer that never reports a belief without a cost, a
calibration check and a classical baseline.

<a id="cobra"></a>

## COBRA: a context-rotated belief filter

COBRA keeps the robot's belief about a person as a **density matrix** over hypotheses (for example the person's
intent). Each sensor frame updates it with a generalised measurement whose basis is **rotated by the previous
frame**, followed by partial dephasing:

```
M_t = U(x_{t-1}) · diag(exp((W x_t + b)/2)) · U(x_{t-1})†        ρ ← M_t ρ M_t / tr(·)        ρ ← (1−g) ρ + g · diag(ρ)
```

The probability of hypothesis *z* is the diagonal entry ρ_zz (Born rule). It has two built-in classical limits, so
the effect of context and the effect of coherence can be separated: with no rotation it **is** a Bayes filter, and
with full dephasing it is a classical *incoherent* context filter (the tests check both exactly).

```python
import numpy as np
from quantum_mind.belief import COBRA, GaussianBayesFilter, min_cost_action

model = COBRA(n_hyp=3, n_features=3).fit(frames, intent)      # frames (N, T, 3), intent in {0, 1, 2}
belief = model.start()                                         # online, one frame at a time
for frame in sensor_stream:
    p = belief.update(frame)                                   # P(handover, retrieve, withdraw)

cost = np.array([[0, 6, 10], [1.5, 0, 0], [0.8, 0.8, 0.8]])    # hand over, wait, ask
action = min_cost_action(p, cost)[0]
```

<p align="center"><img src="https://raw.githubusercontent.com/yeshwanthguru/Quantum-mind/main/docs/assets/cobra_context_effect.png" alt="The same hand cue after opposite previous looks: the Bayes filter shifts by zero, COBRA flips sign like the oracle" width="62%"><br>
<sub>The same hand cue after opposite previous looks. A Bayes filter adds the same evidence either way; this trained COBRA flips its evidence like the oracle that knows the generator (simulated task).</sub></p>

### What the benchmark says

`benchmarks/context_intent_benchmark.py` compares COBRA with the oracle, a Gaussian Bayes filter, logistic
regression and an MLP on a **synthetic** intent task with a context effect, and on a **control** task with none. It
adds an ablation without coherence, noise and dropout stress tests, and a paired sign-flip test. Mean ± std over 5
seeds, 3000 test sequences per cell:

<<COBRA_RESULTS>>

<p align="center"><img src="https://raw.githubusercontent.com/yeshwanthguru/Quantum-mind/main/docs/assets/cobra_benchmark.png" alt="Expected decision cost against training size for COBRA and the baselines" width="100%"></p>

The honest reading is in [`benchmarks/results/context_intent/`](https://github.com/yeshwanthguru/Quantum-mind/tree/main/benchmarks/results/context_intent)
and on the [COBRA page](https://yeshwanthguru.github.io/Quantum-mind/docs/user_guide/belief.html): <<COBRA_READING>>

<a id="layers"></a>

## Three layers

<table>
<tr>
<th width="33%">1 · Quantum-like human models</th>
<th width="33%">2 · COBRA belief filter</th>
<th width="33%">3 · Robot decision layer</th>
</tr>
<tr valign="top">
<td>

How **people** judge, decide and trust, in quantum probability, each with classical baselines.

- order effects · conjunction · interference
- trust as an open quantum system
- quantum-like Bayesian networks
- decision theory · contextuality · similarity
- memory · concepts · bistable perception
- per-person fitting, comparison, recovery

`quantum_mind.families`, `quantum_mind.core`

</td>
<td>

A robot's belief as a **density matrix** read in a context-dependent basis.

- context-rotated generalised measurement
- learned dephasing, coherence diagnostic
- online streaming filter for robots
- Bayes filter and incoherent filter as limits
- minimum-cost action with an "ask" row

`quantum_mind.belief`

</td>
<td>

From belief to action under cost, with calibrated uncertainty.

- ask or act, with hazard levels
- question planner (value of information)
- trust-aware hand-over and task allocation
- calibration, conformal sets, fusion, orchestration
- Gymnasium environments with simulated people
- ROS 2 node

`quantum_mind.applications`, `quantum_mind.envs`

</td>
</tr>
</table>

Around them: `quantum_mind.circuits` (Qiskit circuits of the families, `run()` on Aer, IBM Quantum and Amazon
Braket), `quantum_mind.quantum` (the exact state-vector simulator) and `quantum_mind.viz` (the Bloch-sphere viewer).

**What is deliberately not here.** General-purpose quantum algorithms (VQE, QAOA, Grover, quantum kernels,
variational classifiers) and quantum-inspired optimisers are provided and maintained far better by Qiskit,
PennyLane and optimisation libraries. Version 3.0 removed the copies this library used to carry, so that it
contains only what is specific to quantum cognition in robotics.

<a id="quick-start"></a>

## Quick start

```bash
pip install quantum-mind              # core: numpy, scipy
pip install "quantum-mind[all]"       # everything (Qiskit, cloud back ends, viewer)
```

Is there a question-order effect, and which model explains it?

```python
import numpy as np
from quantum_mind import compare
from quantum_mind.families.order_effects import QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel, qq_test

counts = {'AB': np.array([212, 48, 61, 179]),   # A asked first: yes-yes, yes-no, no-yes, no-no
          'BA': np.array([240, 33, 52, 175])}   # B asked first
print(qq_test(counts))                           # parameter-free test of the quantum-like prediction
for r in compare([QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel], counts):
    print(type(r.model).__name__, round(r.bic, 1))
```

A human model with uncertainty, for a planner or a ROS 2 system:

```python
from quantum_mind.applications.robotics import HumanModelService

svc = HumanModelService()
svc.add_answer({'order': 'AB', 'answers': [1, 0]})          # one person's two answers (1 = yes)
svc.query({'order': 'AB', 'ask_cost': 1.0, 'error_cost': 3.0})
# -> prediction, uncertainty (entropy, model disagreement in bits), ensemble weights, 'ask' or 'act'
```

<a id="viewer"></a>

## See the qubits

<table>
<tr>
<td align="center" width="62%"><img src="https://raw.githubusercontent.com/yeshwanthguru/Quantum-mind/main/docs/assets/entanglement.gif" alt="Two qubits entangled and disentangled by a circuit" width="100%"><br>
<sub>A Qiskit circuit, gate by gate. While the qubits are entangled, their vectors leave the sphere.<br><code>circuit_trajectory(qc)</code> → <code>animate_trajectory</code> / <code>animate_bloch</code></sub></td>
<td align="center" width="38%"><img src="https://raw.githubusercontent.com/yeshwanthguru/Quantum-mind/main/docs/assets/trust_belief.gif" alt="Trust belief of a simulated person on the Bloch sphere" width="100%"><br>
<sub>Trust in a robot as a qubit, over successes and failures, with dephasing.<br><code>belief_trajectory(OpenSystemBelief(), events)</code></sub></td>
</tr>
</table>

```python
from qiskit import QuantumCircuit
from quantum_mind.viz import circuit_trajectory, animate_bloch, save_html, LiveBloch, bloch_tomography

qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1); qc.ry(0.6, 1)
traj = circuit_trajectory(qc, steps=15)
save_html(animate_bloch(traj), 'bell.html')        # interactive: rotate, zoom, play, slider
LiveBloch(2).show().play(traj)                     # real time in Jupyter or a Matplotlib window
bloch_tomography(qc, 'aer:FakeTorino')             # Bloch vectors measured under IBM device noise
```

The same views draw a person's trust belief, and the density matrix of a COBRA belief
(`docs/assets/cobra_density_matrix.png`). The [viewer README](https://github.com/yeshwanthguru/Quantum-mind/blob/main/src/quantum_mind/viz/README.md)
covers everything else.

<a id="validation"></a>

## Validation, as measured

Quantum Mind is a **research platform, not a claim of quantum advantage**. Its strongest scientific use is
evaluating quantum-like models of human judgement against classical alternatives and connecting calibrated
uncertainty to robot questioning, trust and ask-or-act decisions.

* **Evaluation layer.** Log loss, Brier score, expected calibration error, decision cost, bootstrap intervals,
  paired bootstrap and randomisation tests, effect sizes, Holm correction, dataset hashes, experiment manifests and
  model-provenance records (`quantum_mind.evaluation`), plus a reproducible HRI decision benchmark
  (`benchmarks/hri_decision_benchmark.py`) and the COBRA benchmark above.
* **Every new idea gets** a classical baseline, an ablation that isolates it, a control task where it should not
  help, and a paired test. Where the classical method wins the documentation says so.
* **What is not validated.** All robotics results use simulated people or published survey aggregates. No study
  with real people and a robot has been run; [`docs/help/validation.md`](docs/help/validation.md) lists the gaps
  and gives a pilot-study protocol. In several comparisons the simulated people follow the model the robot uses,
  which favours the order-aware method by construction; the tutorials say where.

For the workflow see [`docs/research_validation.md`](docs/research_validation.md) and
[`docs/research_claims.md`](docs/research_claims.md).

## Model catalogue

The [model atlas](https://yeshwanthguru.github.io/Quantum-mind/docs/atlas/index.html) gives a block diagram, the
API object, the classical baselines and the reference for every model, and labels each by origin:

| Label | Entries | Meaning |
|---|---|---|
| Unique to Quantum Mind | 15 | designed in this library, building on the cited methods: COBRA and its online mode, the robot decision layer (ensemble, ask or act, question planner, trust-aware hand-over and allocation), the environments, the circuits for the quantum-like families |
| Published model, extended | 16 | implemented as published, with capabilities the papers do not have (bootstrap intervals, a per-person posterior updated after every answer, informative-question design) |
| Published method | 11 | implemented as published (Lüders rule, Lindblad evolution, fitting, calibration, the Bayes filter, ...) |
| Software tool | 6 | infrastructure |

The extensions that apply to every people model are `bootstrap`, `OnlinePersonModel` and
`rank_conditions` / `model_posterior`. For robots: `AdaptiveConformalSets` (sets for a person's next answer that
keep their long-run coverage), online personalisation, `trust_aware_allocation` and `sample_people` (training
against populations of simulated people).

## Architecture

```
src/quantum_mind/
├── core/           Lüders rule, density matrices, Lindblad · Model/Param · fit, compare, recovery · bootstrap, online, design
├── families/       order_effects · conjunction · interference · qlbn · dynamics · decision · contextuality · similarity · memory · concepts · perception
├── belief/         COBRA · OnlineBelief · GaussianBayesFilter · min_cost_action
├── applications/   robotics: question domains, trust, intent resolution, ensemble, ask_or_act · calibration · orchestration · questioning · handover · allocation · fusion
├── envs/           Gymnasium environments with simulated people
├── circuits/       Qiskit circuits of the families · run() on Aer, IBM Quantum, Amazon Braket
├── quantum/        exact state-vector simulator (used by the viewer and the circuit exports)
├── problems/       the QUBO container used by trust-aware allocation
├── viz/            Bloch vectors, trajectories, tomography · Matplotlib and Plotly viewers · LiveBloch
├── evaluation/     metrics, paired statistics, manifests, provenance
└── data/           published aggregate data sets
benchmarks/         COBRA context-intent benchmark · HRI decision benchmark
examples/           21 scripts        tutorials/   15 tutorials + a packaged hand-over use case
docs/               Sphinx documentation, atlas, mathematics        notebooks/   2 Jupyter notebooks
integrations/ros2/  ROS 2 node (quantum_mind_ros)        site/   project website (built by CI)
```

## Running on quantum hardware

The circuits of the quantum-like families run on simulators and, with credentials, on hardware:

```bash
python3 examples/09_cloud_run.py --dry-run                                  # local, no account
python3 examples/09_cloud_run.py --backend ibm:least_busy --shots 4000      # IBM Quantum
python3 examples/09_cloud_run.py --backend braket:arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1 --shots 1000
```

`run(circuit, backend)` accepts `aer`, `aer:<FakeBackend>`, `braket_local`, `braket_dm:<p>`, `ibm:<device>` and
`braket:<ARN>`. Aer (ideal and with IBM noise models) and the Braket simulators are tested in CI; the IBM and
Braket **submission code is tested against local stand-ins and has not been run on hardware**. Report hardware
results exactly as measured, with backend, date and job identifier.

## A packaged use case: trust-aware hand-over

[`tutorials/trust_handover_package`](https://github.com/yeshwanthguru/Quantum-mind/tree/main/tutorials/trust_handover_package)
turns one robotics model into an installable package (`quantum-handover`): a robot arm decides before each
hand-over whether to hand over, hand over slowly, ask or wait, using the quantum-like trust model. It has a
benchmark against two populations of simulated people, a PyBullet simulation recorded as a GIF, a ROS 2 node, its
own tests and a `requirements.txt`; tutorial 15 walks through it. Results are for simulated people only.

<a id="status"></a>

## Status and scope

| Area | What exists | What does not (yet) |
|---|---|---|
| Quantum-like models | 11 families with classical baselines, fitting, BIC/AIC, recovery studies, per-person fitting and comparison, partial pooling; published aggregate data | full hierarchical Bayesian inference (only MAP partial pooling) |
| COBRA belief filter | context-rotated density-matrix filter, online mode, Bayes and incoherent limits, benchmark with ablations and paired tests, figures | real-person data; reliable training (the context mechanism is found on some restarts only); analytic gradients |
| Robotics | simulated populations for three question domains, questioning designs, trust protocol, intent resolution, `HumanModelEnsemble`, `ask_or_act`, risk-aware decisions, calibration and conformal sets, orchestration gates, question planner, trust-aware hand-over and allocation, fusion, environments, `HumanModelService`, a ROS 2 node | a run of the ROS 2 node inside a ROS 2 installation (its callbacks are tested with stand-in modules); data from human-robot studies |
| Simulator, circuits, viewer | exact simulator with adjoint gradients and Qiskit export, circuits of the families, trajectories, animations, interactive HTML, live widget, tomography, concurrence, Q-sphere | runs on hardware |

## Installation

| Extra | Adds | For |
|---|---|---|
| (core) | numpy, scipy | quantum-like families, COBRA, decision layer, simulator |
| `[qiskit]` | qiskit, qiskit-aer, qiskit-ibm-runtime | circuits, Aer, IBM fake-backend noise, `circuit_trajectory` |
| `[cloud]` | qiskit-ibm-runtime, amazon-braket-sdk | IBM Quantum and Amazon Braket hardware |
| `[viz]` | matplotlib, plotly, ipywidgets, anywidget, seaborn | Bloch-sphere viewer, animations, live widget |
| `[all]` | all of the above | |
| `[dev]` | pytest, pytest-cov, ruff, nbclient, build, twine | tests, coverage, linting, notebooks, packaging |

From a clone: `pip install -e ".[all,dev]"`. Python 3.10 or later.

```bash
python3 -m pytest -q --cov=quantum_mind     # <<TESTS>> tests (CI requires at least 80% line coverage)
python3 benchmarks/context_intent_benchmark.py    # the COBRA benchmark, about 10 to 20 minutes on 4 cores
python3 docs/make_cobra_figures.py                # redraw the COBRA figures
```

## Limitations

- Fitting aggregate counts assumes a homogeneous population. Individual differences can hide or mimic
  quantum-like structure: in example 14 a pooled fit picks the quantum-like model for a population that is half
  anchoring, and per-person comparison needs about 3,000 answers per person to tell the models apart.
- Several quantum-like models have been challenged by further tests, for example the Grand Reciprocity equations
  and conjunction-fallacy tests. The family READMEs list these.
- COBRA's advantage is shown on **synthetic data built to contain a context effect**. With little training data it
  loses to the Bayes filter, training can miss the mechanism, and its coherence is a weak error signal.
- Hardware noise distorts circuits by a few percent of total variation for small circuits, and more for deep ones.

## Data

`quantum_mind.data` contains the only human data in the package, all of it published aggregates: the Clinton–Gore
order effect (Moore, 2002), the Prisoner's Dilemma disjunction effect (Shafir and Tversky, 1992), the two-stage
gamble (Tversky and Shafir, 1992) and the Linda problem (Tversky and Kahneman, 1983). Every other data set in the
examples is simulated and labelled as such.

## Citing and licence

To cite Quantum Mind, use `CITATION.cff` (GitHub's "Cite this repository" button), together with the original paper
of every model you use, listed on the
[references by model](https://yeshwanthguru.github.io/Quantum-mind/docs/atlas/references.html) page and in
[`references.bib`](https://github.com/yeshwanthguru/Quantum-mind/blob/main/references.bib) (entries without a
matching Crossref record hold the text as cited and are marked `UNVERIFIED`). The package is released under the
[Apache License 2.0](https://github.com/yeshwanthguru/Quantum-mind/blob/main/LICENSE) (see also
[NOTICE](https://github.com/yeshwanthguru/Quantum-mind/blob/main/NOTICE)). For contributing see
[CONTRIBUTING.md](https://github.com/yeshwanthguru/Quantum-mind/blob/main/CONTRIBUTING.md); for release history see
[CHANGELOG.md](https://github.com/yeshwanthguru/Quantum-mind/blob/main/CHANGELOG.md).

**Author and maintainer:** Yeshwanth Guru (yeshwanth445@gmail.com; ORCID
[0009-0007-6353-4033](https://orcid.org/0009-0007-6353-4033)). Security reports: see
[SECURITY.md](https://github.com/yeshwanthguru/Quantum-mind/blob/main/SECURITY.md).
