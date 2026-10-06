<p align="center">
  <img src="docs/assets/banner.svg" alt="quantum-cognition-robotics: quantum-like, quantum and quantum-inspired models" width="100%">
</p>

<p align="center">
  <a href="https://github.com/yeshwanthguru/quantum-cognition-robotics/actions/workflows/tests.yml"><img src="https://github.com/yeshwanthguru/quantum-cognition-robotics/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/Qiskit-2.x-6929C4?logo=qiskit&logoColor=white" alt="Qiskit 2.x">
  <img src="https://img.shields.io/badge/coverage-%E2%89%A580%25%20enforced-2ea44f" alt="coverage enforced in CI">
  <a href="docs/API.md"><img src="https://img.shields.io/badge/docs-API%20reference-0969da" alt="API reference"></a>
</p>

<p align="center">
  <b><a href="#pillars">Three pillars</a></b> ·
  <b><a href="#viewer">Bloch viewer</a></b> ·
  <b><a href="#quick-start">Quick start</a></b> ·
  <b><a href="#catalogue">Catalogue</a></b> ·
  <b><a href="#examples">Examples</a></b> ·
  <b><a href="#hardware">Hardware</a></b> ·
  <b><a href="#status">Status</a></b> ·
  <b><a href="notebooks/">Notebooks</a></b> ·
  <b><a href="docs/API.md">API</a></b> ·
  <b><a href="docs/concepts.md">Concepts</a></b>
</p>

---

**quantum-cognition-robotics** (Python package **`qlcog`**) is one library for the three ways
"quantum" enters modelling today:

- **quantum-like** models of how people judge, decide and trust, each paired with the classical models
  it must beat;
- **quantum** machine learning and algorithms as Qiskit circuits;
- **quantum-inspired** classical optimisers and tensor networks.

All three share one problem format, one fitting and comparison workflow, and one 3D Bloch-sphere
viewer. Every circuit runs on Aer, IBM Quantum and Amazon Braket. The library was built for robots
and AI agents that work with people, and it applies equally to surveys, finance, medicine, consumer
research, physics and operations research.

> Quantum-like models use the mathematics of quantum probability to describe judgements. They run on
> ordinary computers and do not claim that the brain is a quantum computer. Quantum models are
> circuits for quantum hardware. Quantum-inspired models are classical algorithms. The
> [concepts page](docs/concepts.md) explains the difference and how to test each.

<a id="pillars"></a>

## 🧭 Three pillars

<table>
<tr>
<th width="33%">🧠 Quantum-like</th>
<th width="33%">⚛️ Quantum</th>
<th width="33%">✨ Quantum-inspired</th>
</tr>
<tr valign="top">
<td>

Models of **human** judgement, decision and trust in quantum probability. Each comes with its
classical baselines and distinctive tests.

- order effects · conjunction · interference
- quantum-like Bayesian networks
- quantum, Markov and open-system dynamics
- quantum decision theory · contextuality · similarity
- **robotics**: questioning designs, trust, human-model ensemble

`qlcog.families`, `qlcog.applications`

</td>
<td>

**Gate-model** machine learning and algorithms. They train on a fast built-in simulator and export
to Qiskit.

- variational (re-uploading) classifier
- quantum-kernel classifier (ZZ feature map)
- QAOA for any QUBO
- VQE for Pauli Hamiltonians
- Grover search
- circuits for all quantum-like families

`qlcog.quantum`, `qlcog.circuits`

</td>
<td>

**Classical** algorithms that borrow quantum ideas. No quantum computer is needed.

- quantum-inspired evolutionary algorithm (QIEA)
- quantum-behaved particle swarm (QPSO)
- simulated quantum annealing (path-integral Monte Carlo)
- tensor-network (MPS) classifier
- classical simulated-annealing baseline

`qlcog.inspired`

</td>
</tr>
</table>

Shared layers: `qlcog.core` (fitting, BIC/AIC comparison, model recovery), `qlcog.problems` (QUBO
builders for MaxCut, knapsack, multi-robot task allocation, portfolio and Ising problems) and
`qlcog.viz` (the 3D Bloch-sphere viewer).

<a id="viewer"></a>

## 🌐 See the qubits

<table>
<tr>
<td align="center" width="62%"><img src="docs/assets/entanglement.gif" alt="Two qubits entangled and disentangled by a circuit" width="100%"><br>
<sub>A Qiskit circuit, gate by gate. While the qubits are entangled, their vectors leave the sphere.<br><code>circuit_trajectory(qc)</code> → <code>animate_trajectory</code> / <code>animate_bloch</code></sub></td>
<td align="center" width="38%"><img src="docs/assets/trust_belief.gif" alt="Trust belief of a simulated person on the Bloch sphere" width="100%"><br>
<sub>Trust in a robot as a qubit, over successes and failures, with dephasing.<br><code>belief_trajectory(OpenSystemBelief(), events)</code></sub></td>
</tr>
</table>

```python
from qiskit import QuantumCircuit
from qlcog.viz import circuit_trajectory, animate_bloch, save_html, LiveBloch, bloch_tomography

qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1); qc.ry(0.6, 1)
traj = circuit_trajectory(qc, steps=15)
save_html(animate_bloch(traj), 'bell.html')        # interactive: rotate, zoom, play, slider
LiveBloch(2).show().play(traj)                     # real time in Jupyter or a Matplotlib window
bloch_tomography(qc, 'aer:FakeTorino')             # Bloch vectors measured under IBM device noise
traj.concurrence(0, 1)                              # entanglement of the pair along the circuit
```

<p align="center"><img src="docs/assets/entanglement_timeline.png" alt="Bloch-vector lengths and concurrence along the circuit" width="88%"><br>
<sub>The same circuit as a timeline: <code>plot_entanglement(traj)</code> shows each qubit's Bloch-vector length and the pair's concurrence (Wootters); <code>plot_qsphere(state)</code> draws the Q-sphere.</sub></p>

Interactive versions of both animations (rotate, zoom, play, slider) are written by
[`examples/19`](examples/19_bloch_sphere_viewer.py) and [`examples/20`](examples/20_trust_on_the_bloch_sphere.py),
and shown inline in the [notebooks](notebooks/). The [viewer README](src/qlcog/viz/README.md) covers everything else.

<a id="quick-start"></a>

## 🚀 Quick start

```bash
pip install "qlcog[all] @ git+https://github.com/yeshwanthguru/quantum-cognition-robotics"
```

<details open>
<summary><b>🧠 Quantum-like: is there a question-order effect, and which model explains it?</b></summary>

```python
import numpy as np
from qlcog.core import compare
from qlcog.families.order_effects import QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel, qq_test

counts = {'AB': np.array([212, 48, 61, 179]),   # A asked first: yes-yes, yes-no, no-yes, no-no
          'BA': np.array([240, 33, 52, 175])}   # B asked first
print(qq_test(counts))                           # parameter-free test of the quantum-like prediction
for r in compare([QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel], counts):
    print(type(r.model).__name__, round(r.bic, 1))
```
</details>

<details>
<summary><b>⚛️ Quantum: classify with a circuit, then run it under IBM device noise</b></summary>

```python
from qlcog.quantum import VariationalClassifier
from qlcog.circuits import run

clf = VariationalClassifier(layers=3).fit(X_train, y_train)
print(clf.score(X_test, y_test))
counts = run(clf.to_qiskit(X_test[0]), 'aer:FakeTorino', shots=4000)
```
</details>

<details>
<summary><b>✨ Quantum-inspired: one robot task-allocation problem, three kinds of solver</b></summary>

```python
from qlcog.problems import task_allocation
from qlcog.quantum import QAOA
from qlcog.inspired import SQA, QIEA, simulated_annealing

q = task_allocation(costs)                       # costs[robot, task]
print(q.brute_force()[1],                        # exact
      QAOA(q, p=3).run().energy,                 # quantum
      SQA(q).run().value, QIEA(q).run().value,   # quantum-inspired
      simulated_annealing(q).value)              # classical baseline
```
</details>

<details>
<summary><b>🤖 Robotics: a human model with uncertainty, for a planner or a ROS 2 system</b></summary>

```python
from qlcog.applications.robotics import HumanModelService

svc = HumanModelService()                                   # wraps HumanModelEnsemble + ask_or_act
svc.add_answer({'order': 'AB', 'answers': [1, 0]})          # one person's two answers (1 = yes)
svc.query({'order': 'AB', 'ask_cost': 1.0, 'error_cost': 3.0})
# -> prediction, uncertainty (entropy, model disagreement in bits), ensemble weights, 'ask' or 'act'
```

The same service runs as a ROS 2 node with standard `std_msgs/String` JSON topics:
[`integrations/ros2`](integrations/ros2/README.md).
</details>

<p align="center"><img src="docs/assets/optimisers.png" alt="Optimisers on MaxCut and the QAOA output distribution" width="92%"></p>
<p align="center"><img src="docs/assets/classifiers.png" alt="Decision regions of the variational quantum classifier and the MPS classifier" width="92%"></p>
<p align="center"><sub>Generated by <code>docs/make_assets.py</code> from the package itself (simulated data).</sub></p>

<a id="catalogue"></a>

## 📚 Model catalogue

<details>
<summary><b>🧠 Quantum-like families</b>: 8 families, each with classical baselines and a README</summary>

| Family | Phenomenon | Quantum-like model(s) | Classical baselines | Circuit | README |
|---|---|---|---|---|---|
| `order_effects` | Answers depend on question order | `QuantumOrderModel4D` (nests Bayes), `QuantumOrderModel` (3D, ranks) | Bayes, anchoring, saturated | ✓ | [link](src/qlcog/families/order_effects/README.md) |
| `conjunction` | Conjunction and disjunction fallacies | `QuantumConjunctionModel` | classical joint, averaging, probability theory plus noise | ✓ | [link](src/qlcog/families/conjunction/README.md) |
| `interference` | Disjunction effect, sure-thing violations | `InterferenceModel` (normalised or not) | classical mixture | ✓ | [link](src/qlcog/families/interference/README.md) |
| `qlbn` | Inference with unresolved hidden causes | quantum-like Bayesian network | classical Bayesian network | ✓ | [link](src/qlcog/families/qlbn/README.md) |
| `dynamics` | Belief change over time; judgements that change later judgements | `QuantumWalk`, `OpenSystemWalk`, `OpenSystemBelief` | `MarkovWalk`, `MarkovBelief` | ✓ | [link](src/qlcog/families/dynamics/README.md) |
| `decision` | Risky choice | `QDTModel` (quantum decision theory) | expected utility, prospect theory | – | [link](src/qlcog/families/decision/README.md) |
| `contextuality` | Is there one joint distribution? | CHSH, Contextuality-by-Default criterion | classical bounds | ✓ | [link](src/qlcog/families/contextuality/README.md) |
| `similarity` | Asymmetric similarity | `QuantumSimilarityModel` | biased geometric (Nosofsky), geometric | ✓ | [link](src/qlcog/families/similarity/README.md) |

Robotics application ([README](src/qlcog/applications/README.md)): question domains (object
clarification, trust and hand-over, preference elicitation), questioning designs (fixed, probe,
split), a trust protocol, and `HumanModelEnsemble`.
</details>

<details>
<summary><b>⚛️ Quantum models</b>: QML and algorithms (<a href="src/qlcog/quantum/README.md">README</a>)</summary>

| Model | Use | Reference |
|---|---|---|
| `VariationalClassifier` | multi-class classification with data re-uploading | Pérez-Salinas et al., *Quantum* 2020 |
| `QuantumKernel`, `QuantumKernelClassifier` | fidelity kernel (ZZ map), kernel ridge; works with scikit-learn | Havlíček et al., *Nature* 2019 |
| `QAOA` | any `Qubo`: allocation, scheduling, MaxCut, portfolios | Farhi et al., 2014 |
| `VQE`, `Hamiltonian` | ground states of Pauli Hamiltonians or QUBOs | Peruzzo et al., *Nat. Commun.* 2014 |
| `grover` | search with an oracle from a list or a predicate | Grover, 1996 |
| `qlcog.circuits` | circuits of the quantum-like families; `run()` on any backend | [circuits README](src/qlcog/circuits/README.md) |
</details>

<details>
<summary><b>✨ Quantum-inspired models</b>: optimisers and tensor networks (<a href="src/qlcog/inspired/README.md">README</a>)</summary>

| Model | Problem | Reference |
|---|---|---|
| `QIEA` | binary optimisation (Han and Kim's rotation table) | Han and Kim, *IEEE TEVC* 2002 |
| `QPSO` | continuous optimisation | Sun, Feng and Xu, *CEC* 2004 |
| `SQA` | Ising / QUBO by path-integral Monte Carlo | Martoňák, Santoro and Tosatti, *PRB* 2002 |
| `MPSClassifier` | supervised classification with a matrix product state; DMRG-style sweeps or Adam | Stoudenmire and Schwab, *NeurIPS* 2016 |
| `simulated_annealing` | classical baseline | Kirkpatrick et al., *Science* 1983 |
</details>

## 🏗️ Architecture

```mermaid
flowchart TB
    core["qlcog.core<br/>Model · Param · fit · compare · recovery"]
    fam["🧠 qlcog.families<br/>8 quantum-like families"]
    app["🤖 qlcog.applications<br/>robotics"]
    prob["qlcog.problems<br/>QUBO builders"]
    qm["⚛️ qlcog.quantum<br/>VQC · kernel · QAOA · VQE · Grover"]
    ins["✨ qlcog.inspired<br/>QIEA · QPSO · SQA · MPS"]
    circ["qlcog.circuits<br/>circuits + run()"]
    viz["🌐 qlcog.viz<br/>Bloch-sphere viewer"]
    hw[("Aer · IBM Quantum · Amazon Braket")]
    core --> fam --> app
    fam --> circ
    prob --> qm
    prob --> ins
    qm --> circ
    circ --> hw
    fam --> viz
    qm --> viz
    circ --> viz
```

```
src/qlcog/
├── core/           Lüders rule, density matrices, Lindblad · Model/Param · fit, compare, recovery
├── families/       🧠 order_effects · conjunction · interference · qlbn · dynamics · decision · contextuality · similarity
├── applications/   🤖 robotics: question domains, questioning designs, trust, human-model ensemble, ask_or_act, HumanModelService
├── quantum/        ⚛️ statevector simulator · ansatz · classifiers · QAOA · VQE · Grover
├── inspired/       ✨ QIEA · QPSO · SQA · simulated annealing · MPS classifier
├── problems/       QUBO builders shared by quantum and quantum-inspired solvers
├── circuits/       Qiskit circuits of the families · run() on Aer, IBM Quantum, Amazon Braket
├── viz/            🌐 Bloch vectors, trajectories, tomography · Matplotlib and Plotly viewers · LiveBloch
└── data/           published aggregate data sets
examples/           22 scripts across domains        docs/       concepts, API reference, assets
notebooks/          2 Jupyter notebooks              tests/      51 tests
integrations/ros2/  ROS 2 node (qlcog_ros)
.github/            CI, release, CODEOWNERS, issue and pull-request templates
```

<a id="examples"></a>

## 🧪 Examples across domains

| # | Domain | Pillar | Script |
|---|---|---|---|
| 01 | Surveys, market research | 🧠 | [question-order effects, QQ test](examples/01_survey_question_order.py) |
| 02 | Behavioural finance | 🧠 | [disjunction effect, interference, QLBN](examples/02_finance_disjunction_effect.py) |
| 03 | Medical decision support | 🧠 | [quantum-like Bayesian network](examples/03_medical_diagnosis_qlbn.py) |
| 04 | Consumer choice | 🧠 | [quantum decision theory vs EU and PT](examples/04_consumer_choice_qdt.py) |
| 05 | AI / LLM evaluation | 🧠 | [order effects in judgements](examples/05_llm_evaluation_order.py) |
| 06 | Human–computer interaction | 🧠 | [trust dynamics](examples/06_hci_trust_dynamics.py) |
| 07 | Perception, confidence | 🧠 | [Markov vs quantum walk](examples/07_evidence_accumulation.py) |
| 08 | Physics, psychology | 🧠⚛️ | [CHSH and Contextuality-by-Default](examples/08_contextuality_analysis.py) |
| 09 | Marketing, linguistics | 🧠 | [asymmetric similarity](examples/09_similarity_asymmetry.py) |
| 10 | – | ⚛️ | [every family as a circuit (Aer, FakeTorino, Braket)](examples/10_circuits_quickstart.py) |
| 11 | – | ⚛️ | [IBM Quantum / Amazon Braket hardware](examples/11_cloud_run.py) |
| 12 | Robotics, HRI | 🧠 | [robot questioning and trust](examples/12_robot_questioning_trust.py) |
| 13 | Medicine (simulated) | ⚛️✨ | [VQC, quantum kernel, MPS vs logistic regression](examples/13_quantum_classifiers_triage.py) |
| 14 | Robotics | ⚛️✨ | [multi-robot task allocation: QAOA, SQA, QIEA, SA](examples/14_qaoa_multi_robot_allocation.py) |
| 15 | Finance (simulated) | ⚛️✨ | [portfolio selection](examples/15_portfolio_selection.py) |
| 16 | Physics, materials | ⚛️ | [VQE on a transverse-field Ising chain](examples/16_vqe_ising_chain.py) |
| 17 | Scheduling | ⚛️ | [Grover search for valid schedules](examples/17_grover_schedule_search.py) |
| 18 | Control, robotics | ✨ | [PID tuning with QPSO](examples/18_qpso_controller_tuning.py) |
| 19 | Visualisation | 🌐 | [Bloch-sphere viewer, HTML, GIF, live, tomography](examples/19_bloch_sphere_viewer.py) |
| 20 | Robotics, HRI | 🧠🌐 | [trust on the Bloch sphere](examples/20_trust_on_the_bloch_sphere.py) |
| 21 | Robotics | 🧠 | [when to ask for help: ensemble uncertainty and `ask_or_act`](examples/21_robot_ask_for_help.py) |
| 22 | Surveys, HRI | 🧠 | [individual differences: pooled versus per-person model comparison](examples/22_individual_differences.py) |

Notebooks: [`01_bloch_sphere_live`](notebooks/01_bloch_sphere_live.ipynb) (interactive and live spheres,
tomography under device noise) and [`02_robot_questioning_and_trust`](notebooks/02_robot_questioning_and_trust.ipynb)
(order effects, ensemble, ask-or-act, trust on the sphere). Both run in CI.

Results printed by the examples (simulated data, this version). They illustrate the methods on small
problems; they are not a benchmark, and on the classification task a well-specified classical model wins:

| Task | Result |
|---|---|
| Triage classification, test accuracy, mean ± sd over 10 seeds | **classical logistic regression on quadratic features 0.926 ± 0.016** · MPS 0.915 ± 0.015 · VQC 0.870 ± 0.033 · linear logistic regression 0.814 ± 0.041 · quantum kernel 0.781 ± 0.040 |
| Robot task allocation (3 × 3) | QAOA, SQA, QIEA and SA all reach the optimum; QAOA P(optimal) 0.04 ideal, 0.03 under FakeTorino noise (sampled), against 0.002 uniform |
| Portfolio, 3 of 8 assets | QAOA (p = 2), SQA, QIEA and SA all optimal; QAOA P(optimal) 0.014 against 0.004 uniform |
| VQE, 4-spin Ising chain | error below 1e-8 against exact diagonalisation |
| Grover, 9 of 32 schedules valid | P(valid) 0.99 after one iteration (random 0.28) |
| Trust qubit | P(trust) from the Bloch vector equals the model's prediction (0.7247) |

<a id="hardware"></a>

## 🛰️ Running on quantum hardware

```bash
python3 examples/11_cloud_run.py --dry-run                                  # local, no account
python3 examples/11_cloud_run.py --backend ibm:least_busy --shots 4000      # IBM Quantum
python3 examples/11_cloud_run.py --backend braket:arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1 --shots 1000
```

`run(circuit, backend)` accepts `aer`, `aer:<FakeBackend>`, `braket_local`, `braket_dm:<p>`,
`ibm:<device>` and `braket:<ARN>`. Every `to_qiskit()` circuit in `qlcog.quantum` and every builder
in `qlcog.circuits` works with it. Hardware runs cost queue time or money. Report their results
exactly as measured, with backend, date and job identifier.

| Backend | Status |
|---|---|
| Aer (ideal), Aer with IBM device noise models, Braket local and density-matrix simulators | tested in CI |
| IBM Quantum hardware (`ibm:<device>`) | submission code tested in CI against a fake IBM device (same transpilation, SamplerV2 call and result parsing); **not yet run on hardware** |
| Amazon Braket QPUs (`braket:<ARN>`) | submission code tested in CI with the local simulator as the device (same OpenQASM 3 program and `device.run` call); **not yet run on hardware** |

No hardware results are included in the package. IBM jobs use the client-side Qiskit Runtime
Sampler (`qiskit_ibm_runtime.executor_sampler`), falling back to `SamplerV2` on releases before 0.50.

<a id="status"></a>

## 🔎 Status, scope and honest limits

| Area | What exists | What does not (yet) |
|---|---|---|
| Quantum-like models | 8 families with classical baselines, fitting, BIC/AIC, recovery studies, per-person fitting and comparison (`fit_individuals`, `compare_individuals`); published aggregate data | hierarchical (partial-pooling) models |
| Robotics | simulated human populations for three question domains, questioning designs, trust protocol, `HumanModelEnsemble`, `ask_or_act`, `HumanModelService`, a ROS 2 node (`integrations/ros2`) | a run of the ROS 2 node inside a ROS 2 installation (its callbacks are tested with stand-in modules); data from human–robot studies |
| Quantum models | VQC, quantum kernel, QAOA, VQE, Grover on an exact simulator with adjoint gradients; Qiskit export; IBM submission through the current Qiskit Runtime Sampler | quantum advantage (none claimed); noise-aware training; runs on hardware |
| Quantum-inspired | QIEA with Han and Kim's rotation table (or a simplified rule), QPSO, SQA, simulated-annealing baseline, MPS classifier with DMRG-style sweeps or Adam | – |
| Viewer | trajectories, animations, interactive HTML, live widget, tomography, pairwise concurrence and entanglement timeline, Q-sphere | – |

**Sizes.** The simulator holds 2ⁿ amplitudes per sample, at most 22 qubits. Measured on a laptop
CPU: the variational classifier trains 8 features × 400 samples in about 80 s (151 L-BFGS steps,
85 MB); QAOA on 9 variables takes seconds; brute-force QUBO solutions are limited to 22 variables.
Grover's gate-level export needs about 360 CNOTs per iteration at 8 qubits and 660 at 10, against
510 and 2,040 for the diagonal-gate export, which grows exponentially.

**Name.** `qlcog` began as the quantum-like cognition library and keeps that name for stability; the
quantum and quantum-inspired toolkits sit beside it so that all three can be compared on the same
problems. The repository name reflects the main application, robots that work with people.


## 📦 Installation

```bash
pip install "qlcog @ git+https://github.com/yeshwanthguru/quantum-cognition-robotics"          # core: numpy, scipy
pip install "qlcog[all] @ git+https://github.com/yeshwanthguru/quantum-cognition-robotics"     # everything
```

| Extra | Adds | For |
|---|---|---|
| (core) | numpy, scipy | quantum-like families, quantum simulator, quantum-inspired solvers, problems |
| `[qiskit]` | qiskit, qiskit-aer, qiskit-ibm-runtime | circuits, Aer, IBM fake-backend noise, `circuit_trajectory` |
| `[cloud]` | qiskit-ibm-runtime, amazon-braket-sdk | IBM Quantum and Amazon Braket hardware |
| `[viz]` | matplotlib, plotly, ipywidgets, anywidget, seaborn | Bloch-sphere viewer, animations, live widget |
| `[all]` | all of the above | |
| `[dev]` | pytest, pytest-cov, ruff, nbclient, build, twine | tests, coverage, linting, notebooks, packaging |

From a clone: `pip install -e ".[all,dev]"`. Python 3.10 or later. Examples and tests also run
without installing, because they add `src/` to the path.

## ✅ Choosing and testing a model

1. **Check the phenomenon first.** Look for an order effect, a violation of total probability, or
   asymmetric similarity. If there is none, a classical model is enough.
2. **Fit quantum-like models and classical baselines together** with `compare`. Report all of them.
3. **Use each family's distinctive test**: the QQ equality, total-probability bounds,
   Contextuality-by-Default, or the effect of an intermediate judgement.
4. **Run a recovery study** (`qlcog.core.recovery`) at the planned sample size before collecting
   data.
5. **For quantum and quantum-inspired solvers, always print the exact or classical baseline**, as
   every example does.

```bash
python3 -m pytest -q --cov=qlcog     # 51 tests, about 90% line coverage (CI requires at least 80%)
```

The tests check the simulator against Qiskit gate by gate, adjoint and parameter-shift gradients
against finite differences, exported circuits against simulation, circuit builders against the
analytic models, the IBM and Braket submission code against local stand-ins, Bloch trajectories
against Qiskit's partial trace and the trust model, input validation and size limits. CI also lints,
checks that the API reference is current, runs the examples and notebooks, and builds the wheel.

## ⚠️ Limitations

- Fitting aggregate counts assumes a homogeneous population. Individual differences can hide or mimic
  quantum-like structure: in example 22 a pooled fit picks the quantum-like model for a population that
  is half anchoring. Per-person comparison (`compare_individuals`) needs many answers per person
  (there, about 3,000) to tell the models apart.
- Several quantum-like models have been challenged by further tests, for example the Grand Reciprocity
  equations and conjunction-fallacy tests. The family READMEs list these.
- The quantum models are small and exactly simulable. No quantum advantage is claimed: on the
  example classification task a classical model with quadratic features is the most accurate.
  Shallow QAOA concentrates little probability on the optimum (a few percent on 9-variable problems).
- "Quantum-inspired" names where an idea came from, not a speed-up. Compare against the classical
  baseline.
- Hardware noise distorts circuits by a few percent of total variation for small circuits, and more
  for deep ones.

## 📊 Data

`qlcog.data` contains the only human data in the package. All of it is published aggregate data:

- the Clinton–Gore order effect (Moore, 2002);
- the Prisoner's Dilemma disjunction effect (Shafir and Tversky, 1992);
- the two-stage gamble (Tversky and Shafir, 1992);
- the Linda problem (Tversky and Kahneman, 1983).

Every other data set in the examples is simulated and labelled as such.

## 📝 Citing and licence

To cite `qlcog`, use `CITATION.cff` (GitHub's "Cite this repository" button), together with the
original papers of the models used, which are listed in each README. The package is released under the
[Apache License 2.0](LICENSE) (see also [NOTICE](NOTICE)). The licence permits commercial and research
use and includes an explicit patent grant. For contributing, see [CONTRIBUTING.md](CONTRIBUTING.md);
for release history, see [CHANGELOG.md](CHANGELOG.md).

**Author and maintainer:** Yeshwanth Guru (yeshwanth445@gmail.com; ORCID
[0009-0007-6353-4033](https://orcid.org/0009-0007-6353-4033)), Department of Mechanical Engineering,
Amrita Vishwa Vidyapeetham, Chennai, India. Security reports: see [SECURITY.md](SECURITY.md).

Companion manuscripts (in preparation): a systematic review of quantum-like cognition for autonomous
agents, a systematic review of meta-learning orchestration on resource-constrained robots, and
"Order-Aware Human Models for Robot Questioning and Trust", whose simulations use the robotics
application of this library.
