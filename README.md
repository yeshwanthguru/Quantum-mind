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
  <b><a href="#author">Author's view</a></b> ·
  <b><a href="#idea">The idea</a></b> ·
  <b><a href="#cobra">COBRA</a></b> ·
  <b><a href="#layers">Three threads</a></b> ·
  <b><a href="#quick-start">Quick start</a></b> ·
  <b><a href="#viewer">Bloch viewer</a></b> ·
  <b><a href="#validation">Validation</a></b> ·
  <b><a href="#status">Status</a></b> ·
  <b><a href="https://yeshwanthguru.github.io/Quantum-mind/docs/tutorials/index.html">Tutorials</a></b> ·
  <b><a href="https://yeshwanthguru.github.io/Quantum-mind/docs/atlas/index.html">Model atlas</a></b> ·
  <b><a href="https://yeshwanthguru.github.io/Quantum-mind/docs/math/index.html">Mathematics</a></b>
</p>

---

**Quantum Mind** (`pip install quantum-mind`, `import quantum_mind`) is a Python library for **quantum robotics**, the
whole umbrella: **quantum computing for robot algorithms** (robot problems as optimisation, with classical baselines and a Qiskit
QAOA adapter), **quantum sensing for robots** (quantum-limited sensor models and their fusion with classical sensors) and
**quantum-probability cognition** (how people judge, decide and trust, and how a robot should read them, with projections,
interference, density matrices and generalised measurements, ending in a cost-aware decision: hand over, wait, or ask).

> It runs on ordinary computers and makes **no claim of quantum computational advantage**. "Quantum-like" means
> the probability rules are borrowed; nothing in the brain or the robot is assumed to be quantum. Every model is
> reported next to the classical model it has to beat, with the ablation that isolates what is new, and the benchmarks
> report where the quantum version does no better.

<a id="author"></a>

## Author's view

*Yeshwanth Guru, author and maintainer.*

I started Quantum Mind from one belief: a robot that works with people needs a different kind of probability
than the one inside most of its filters. People are not neutral sensors. A look changes what a hand movement
means, a question changes the answer to the next question, and asking someone whether they trust a robot changes
their trust. Classical filters assume the evidence arrives in independent pieces. Quantum probability was built
for exactly the case where the order and the context of a measurement matter.

Quantum robotics is a young field with at least three threads, and I think it helps to keep them apart:

1. **Quantum computing for robot algorithms**: planning, optimisation and learning on quantum hardware. It is
   promising on paper, but no practical advantage has been shown on real robot problems with today's noisy devices.
2. **Quantum sensing for robots**: atomic and solid-state sensors for navigation, magnetometry and gravimetry. That is
   physical hardware, beyond what a software library can offer.
3. **Quantum-probability models of the beliefs a robot holds about people**: context, order and trust. These run on
   ordinary computers today and can be tested today.

Quantum Mind covers all three, because I want to work on the whole umbrella, and it holds each thread to the same
standard: a classical baseline, an ablation, and results that are allowed to say no.

* **Thread 1** (`quantum_mind.computing`): robot problems posed as binary optimisation, the classical baselines a quantum
  result must be reported against, and a thin adapter to Qiskit's QAOA. It does not re-implement quantum algorithms, which
  Qiskit and PennyLane do better.
* **Thread 2** (`quantum_mind.sensing`): quantum-limited sensor models and a Kalman filter that fuses a slow, stable quantum
  accelerometer with a fast, drifting classical one, in simulation. It does not claim to build a sensor; that is hardware.
* **Thread 3** (`quantum_mind.belief`, `families`, `applications`): quantum-probability models of the beliefs a robot holds about
  people, the part of the library I have developed furthest.

The circuits of the thread-3 models also run on quantum hardware (Aer, IBM Quantum, Amazon Braket), which is the bridge between the
threads. I claim no speed-up anywhere. In every thread I ask the same question: *does the quantum version beat the classical one, and
does the robot decide or move better with it?* For beliefs about people that becomes: does quantum probability describe human
context better than the classical alternative? Every model is therefore built next to the classical model it has to beat, with
the ablation that isolates what is new.

What I chose to build, and not to build:

* I did not re-implement general-purpose quantum algorithms or quantum-inspired optimisers. Qiskit, PennyLane and
  dedicated optimisation libraries do them better, and a library should not carry copies of other people's work; thread 1
  connects to Qiskit instead.
* I limited the scope to AI, machine learning, vision, robotics and embodied AI. The cognitive-psychology and
  economics topics that quantum cognition is usually known for are gone.
* The models taken from the literature are implemented from their papers and cited in the model atlas, which labels
  each entry as *published*, *extended* or *unique*. The parts I designed myself are the *unique* ones: the COBRA
  belief filter, the robot decision layer, the environments and the circuits for the families; the robot problems, the
  baselines and the sensing study are labelled as what they are.

What I do not claim: no quantum computational advantage (QAOA does no better than random sampling on the problems I tested);
no quantum-sensing hardware; no results with real people or a real robot yet; and COBRA's gain over a Bayes filter is modest,
shown only on synthetic data built to contain a context effect, and it loses with little data and on a control task. I would
rather state that than hide it.

Where I want it to go: a pilot study with real people and a robot, a more reliable way to train COBRA, runs on quantum hardware
and measured sensor data for threads 1 and 2, and independent replication by other groups. If you find a mistake or a claim that does not hold, please open an issue.

---

<a id="idea"></a>

## Get started

### 1. Install

```bash
pip install quantum-mind                  # from PyPI: the core (numpy, scipy)
pip install "quantum-mind[all]"           # plus Qiskit, cloud back ends, the viewer and JAX (for the cognition core)
```

PyPI can lag behind this repository (the motor agent, supervisor, runtime and quantum-cognition core are in 3.2.0). To work with the latest code, clone it:

```bash
git clone https://github.com/yeshwanthguru/Quantum-mind.git
cd Quantum-mind
pip install -e ".[all,dev]"               # Python 3.10 or later; "[cognition]" alone adds just JAX and Optax
```

Check that it works:

```bash
python -c "import quantum_mind; print(quantum_mind.__version__)"
python -m pytest -q tests/test_core.py    # a few seconds
```

### 2. Run your first model (one minute)

Two questions were asked in both orders and the answers differ. Is it an order effect, and which model explains it? (These counts are made up
for the example.)

```python
import numpy as np
from quantum_mind import compare
from quantum_mind.families.order_effects import QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel, qq_test

counts = {'AB': np.array([212, 48, 61, 179]),   # A asked first: yes-yes, yes-no, no-yes, no-no
          'BA': np.array([240, 33, 52, 175])}   # B asked first
q, z, p = qq_test(counts)                       # parameter-free test of the quantum-like (projective) prediction
print('QQ statistic %.3f, z = %.2f, p = %.3f' % (q, z, p))
for r in compare([QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel], counts):
    print('%-22s BIC %.1f' % (type(r.model).__name__, r.bic))
```

Output:

```text
QQ statistic 0.048, z = 1.92, p = 0.054
BayesOrderModel        BIC 2378.8
QuantumOrderModel4D    BIC 2382.0
AnchoringOrderModel    BIC 2382.3
```

How to read it: a small `p` would reject every quantum-like (projective) model, here it does not at the 5% level; the lowest BIC wins, and on these counts the
simple Bayes model is as good as the quantum-like one. The library reports this either way, because a quantum model is only worth using when it beats its
classical rivals.

### 3. Do a tutorial

The tutorials are Markdown files in [`tutorials/`](tutorials/) that read on GitHub and run as notebooks. Pick one of three ways:

```bash
# (a) read them on GitHub, or on the website where every number and figure is real output:
#     https://yeshwanthguru.github.io/Quantum-mind/docs/tutorials/index.html

# (b) run one as a notebook
python tutorials/_convert.py notebooks_out          # writes one .ipynb per tutorial (needs nbformat, included in "[dev]")
jupyter lab notebooks_out/02_question_order.ipynb   # run the cells top to bottom

# (c) run a tutorial's code straight from the shell (no Jupyter)
python - <<'PY'
import re
src = open('tutorials/02_question_order.md').read()
ns = {}
for block in re.findall(r"```python\n(.*?)```", src, re.S):
    exec(block, ns)
PY
```

A good path, from the basics to the newest parts (each takes from seconds to about two minutes):

| Step | Tutorial | You learn |
|---|---|---|
| 1 | [`01_introduction`](tutorials/01_introduction.md) | beliefs as vectors, questions as projectors, why order matters |
| 2 | [`02_question_order`](tutorials/02_question_order.md) | fit the Clinton-Gore poll, run the QQ test, compare against classical rivals |
| 3 | [`04_ask_or_act`](tutorials/04_ask_or_act.md) | decide when a robot should ask a person instead of guessing |
| 4 | [`10_context_belief_filter`](tutorials/10_context_belief_filter.md) | COBRA, a belief whose update depends on context |
| 5 | [`17_quantum_cognition_core`](tutorials/17_quantum_cognition_core.md) | the trainable core (needs `[cognition]`), when it helps and when it does not |
| 6 | [`18_qr_vla_motor_agent`](tutorials/18_qr_vla_motor_agent.md) | a one-motor VLA that asks, a self-healing agent, a supervisor and a safety layer |
| 7 | [`19_robot_language_model`](tutorials/19_robot_language_model.md) | a small language model trained from scratch that reads corrections, and a local LLM kept on a short leash (needs `[cognition]`) |
| 8 | [`15`](tutorials/15_quantum_computing_for_robots.md), [`16`](tutorials/16_quantum_sensing.md) | the other two threads: robot problems on QAOA and a quantum accelerometer |

### 4. Run an example script

```bash
python examples/20_one_motor_vla.py       # train the tiny VLA and run the closed loop (about a minute)
```

```text
policy                 success   questions    (ambiguous scenes only: success)
oracle                 1.000     0.00         1.000
BC baseline            0.205     0.00         0.029
TinyVLA, act           0.660     0.00         0.015
TinyVLA, ask gate      0.965     0.33         0.897
```

The other scripts are in [`examples/`](examples/): COBRA (17), robot problems on QAOA (18), a quantum accelerometer (19), and more.

### 5. Fit the quantum-cognition core to your own data

```python
import numpy as np
from quantum_mind.cognition import QuantumCognitionCore, BayesCore

# Replace these arrays with your own sequences (here: random placeholders, so both models will sit at chance).
rng = np.random.default_rng(0)
X = rng.standard_normal((600, 3, 3)).astype('float32')                 # (sequences, clues, evidence features)
C = np.eye(2, dtype='float32')[rng.integers(0, 2, (600, 3))]           # (sequences, clues, context): which context each clue came in
y = rng.integers(0, 3, 600)                                            # which of 3 hypotheses turned out true

train, test = slice(0, 450), slice(450, 600)
for model in (BayesCore(3, 3, 2, steps=150, restarts=1), QuantumCognitionCore(3, 3, 2, steps=150, restarts=1)):
    model.fit(X[train], C[train], y[train])
    print('%-22s held-out log-loss %.3f' % (type(model).__name__, model.log_loss(X[test], C[test], y[test])))
```

Chance for 3 hypotheses is 1.099. On real data the question is whether the quantum core beats `BayesCore` and the classical context models on held-out
log-loss; if its rotations stay near zero (`model.rotation_size`) the data show no quantum-like order effect. [Tutorial 17](tutorials/17_quantum_cognition_core.md) walks through it.

## The idea

Classical filters treat evidence as arriving in neutral, independent pieces. People and robots do not live in that
world:

* a **look** changes what the next **hand movement** means;
* a **question** changes the answer to the next question (order effects);
* **asking** a person whether they trust the robot changes their trust.

Quantum probability represents exactly this: beliefs are vectors or density matrices, evidence is a measurement,
and measurements need not commute. That is one thread of quantum robotics. The other two are quantum computing for robot
algorithms and quantum sensing for robots, and Quantum Mind has a module for each (below). All three are tested code that never
reports a quantum result without a classical baseline next to it.

<a id="layers"></a>

## Three threads of quantum robotics

<table>
<tr>
<th width="33%">1 · Quantum computing for robot algorithms</th>
<th width="33%">2 · Quantum sensing for robots</th>
<th width="33%">3 · Quantum-probability cognition</th>
</tr>
<tr valign="top">
<td>

Robot problems as binary optimisation, solved classically and with QAOA.

- multi-robot allocation, grasp selection, slot scheduling
- QUBO encodings with true objectives and decoders
- classical baselines, including random sampling with the same shots
- thin adapter to Qiskit's QAOA (exact simulation)
- solver comparison in the robot's terms

`quantum_mind.computing`

</td>
<td>

Quantum-limited sensors and their fusion with classical ones, in simulation.

- standard quantum limit and Heisenberg limit
- atom-interferometer acceleration sensitivity
- MEMS and atom-interferometer accelerometer models
- multi-rate Kalman dead reckoning, sensor suites compared
- where the quantum sensor stops helping

`quantum_mind.sensing`

</td>
<td>

How **people** answer, trust and perceive, and how a robot reads them.

- COBRA: a context-rotated density-matrix belief filter
- question order, trust, Bayesian inference, bistable perception
- ask or act, question planning, hand-over, allocation
- calibration, conformal sets, fusion, ROS 2
- environments with simulated people

`quantum_mind.belief`, `families`, `applications`, `envs`

</td>
</tr>
</table>

Around them: `quantum_mind.circuits` (Qiskit circuits of the thread-3 models, `run()` on Aer, IBM Quantum and Amazon Braket),
`quantum_mind.quantum` (the exact state-vector simulator) and `quantum_mind.viz` (the Bloch-sphere viewer).

**What is deliberately not here.** General-purpose quantum algorithms (VQE, Grover, quantum kernels, variational classifiers) and
quantum-inspired optimisers are provided and maintained far better by Qiskit, PennyLane and optimisation libraries; thread 1
connects to Qiskit instead of carrying copies.

### Thread 1 in numbers

Simulation, 10 random instances per problem (9 to 12 qubits), QAOA through Qiskit on an exact simulator. Instances solved to
the exact optimum of the robot's own objective:

| Solver | Allocation 3×3 | Grasp selection 3 of 10 | Slot scheduling 4×3 | **Total** |
|---|---|---|---|---|
| Greedy descent (20 restarts) | 10 | 9 | 10 | **29 / 30** |
| Random sampling (1000 shots) | 7 | 7 | 10 | **24 / 30** |
| Simulated annealing | 5 | 1 | 10 | **16 / 30** |
| QAOA, depth 1 | 5 | 9 | 10 | **24 / 30** |
| QAOA, depth 2 | 6 | 10 | 10 | **26 / 30** |
| QAOA, depth 3 | 7 | 9 | 10 | **26 / 30** |

QAOA is within noise of random sampling with the same shots and below greedy descent, although its probability on the optimum
grows with depth on scheduling (0.04, 0.10, 0.14). This is an exact simulation on problems small enough to enumerate; it says nothing
about larger problems or hardware. See
[`benchmarks/results/robot_quantum_computing/`](https://github.com/yeshwanthguru/Quantum-mind/tree/main/benchmarks/results/robot_quantum_computing).

### Thread 2 in numbers

Simulation, 10 trajectories of 300 s, illustrative sensor values (atom-interferometer noise from the standard quantum limit).
RMSE of the position estimate in metres:

| | MEMS only | Quantum only | **Fused** |
|---|---|---|---|
| No position fixes | 339.7 | 229.1 | **10.0** (paired p = 0.0025) |
| Fix every 10 s | 0.173 | 1.93 | **0.144** (p = 0.005) |

The benefit comes from the quantum sensor being **stable**: when its bias is as large as the MEMS sensor's, the benefit disappears
(fused 297 m against 340 m, p = 0.51). The sensor numbers are assumptions; a real instrument has to be measured. See
[`benchmarks/results/quantum_sensing/`](https://github.com/yeshwanthguru/Quantum-mind/tree/main/benchmarks/results/quantum_sensing).

<a id="cobra"></a>

## QR-VLA: a one-motor vision-language-action prototype

`quantum_mind.embodied` puts the three threads into one small, inspectable robot policy: an instruction and an image go in, a motor command
comes out. It is a prototype to scale from, not a competitor to large VLA models, and it is trained and run entirely in NumPy on simulated data.

```text
 image strip ──┐
               ├─► GROUNDING (TinyVLA attention, classical, 817 parameters)
 instruction ──┘            │
                            ▼
                 belief over the objects on the table
                            │
                 conformal gate: is the prediction set one object?
                   │ yes                          │ no: ask a question
                   │                              ▼
                   │                  person answers (simulated)
                   │                              │
                   │                              ▼
                   │        BELIEF SLOT (thread 3, cognition)
                   │        Bayes | classical context weights | quantum-probability slot
                   ▼                              │
                   └──────────────┬───────────────┘
                                  ▼ target position
       STATE SLOT (thread 2, sensing)           ACTION HEAD
       Kalman: encoder + MEMS accelerometer     (target - position, velocity) -> command
       + slow quantum accelerometer ──────────► ────────────┐
       (encoder can drop out)                                ▼
              ▲                                      ONE DC-MOTOR JOINT
              └──────── encoder, accelerometers ◄────────────┘

 several motors:  ASSIGNMENT SLOT (thread 1, computing)
                  which motor goes to which target: QUBO with exact | annealing | greedy | QAOA solvers

 MOTOR AGENT (supervises the loop):  monitor encoder vs accelerometer, diagnose gain / friction / push / encoder faults,
                  heal by state fallback, command re-linearisation or safe stop
```


| Slot | Thread | Quantum component | Measured against |
|---|---|---|---|
| belief | cognition | `QuantumSlot`: rotated measurement per answer context; a Qiskit circuit reproduces it exactly | Bayes, classical context weights |
| state | sensing | Kalman fusion of the MEMS and a slow quantum accelerometer when the encoder drops out | encoder, Kalman with the nominal motor model |
| assignment | computing | `assign_targets` on the robot QUBOs, optional QAOA | exact, annealing, greedy |

```python
import numpy as np
from quantum_mind.embodied import QuantumRobotVLA, QuantumSlot, make_dataset, sequence_batch

system = QuantumRobotVLA.fit(make_dataset(2000, 0), make_dataset(800, 1))      # grounding + conformal gate
ambiguous = [s for s in make_dataset(3000, 2) if s.ambiguous]
system.slot = QuantumSlot().fit(sequence_batch(system.vla, ambiguous, np.random.default_rng(0)))   # fit the belief slot
print(system.run(make_dataset(1, 3)[0], np.random.default_rng(0)))             # one closed-loop episode
```

What the benchmarks say (all simulated; see `src/quantum_mind/embodied/README.md` and `benchmarks/results/`):

* **Grounding and asking:** 0.96 success on unambiguous instructions; on ambiguous ones acting blindly gives 0.04 and the conformal ask gate 0.90 with a question in 32% of episodes.
* **Belief slot:** the quantum slot wins only against a person built to be non-commuting; against classical context noise the classical context weights are better.
* **State slot:** with the plant as modelled, the classical motor-model Kalman filter wins (0.96 against 0.43); under an unmodelled push the quantum-accelerometer fusion wins (0.43 against 0.14).
* **Motor agent:** `MotorAgent` monitors the joint, diagnoses encoder, friction, gain and push faults and heals them (estimator fallback, command re-linearisation, safe stop). In simulation it lifts success from 0 to 0.86 on an encoder offset and from 0.77 to 0.99 on a friction increase, with 0.3% false alarms on a healthy motor, but it is still weak on external pushes (0.52, diagnosis right about half the time).
* Asking for help with conformal prediction has prior art in LLM planning (KnowNo, Ren et al., 2023); the grounding and behaviour cloning here are standard. The open step is data from real people and a real joint.

## The quantum-cognition core

`quantum_mind.cognition` is the trainable model behind thread 3: a belief held as a density matrix, updated by context-dependent generalised measurements
`M = U(c) diag(w(x)) U(c)†`, trained by gradients (optional extra `cognition`, JAX and Optax), with four classical competitors on the same interface.
With zero rotations it is exactly a Bayes filter, and the rotations are penalised toward zero, so quantum-like effects appear only when the data ask for them.

In simulation (five generators with a known best-possible log-loss, 5 seeds, paired tests) it is a strong few-parameter generalist but not a winner everywhere:
it recovers most of the context effect with 37 parameters, stays near Bayes when there is nothing to find, and clearly beats the 237-parameter GRU (gap 0.013
against 0.064) when the context is non-commuting, but a classical model that matches the data exactly always beats it, and a GRU beats it on a random recurrent
network. The non-commuting generator is built from the core's own form, so that win shows capacity, not that people are quantum. See
[`src/quantum_mind/cognition/README.md`](src/quantum_mind/cognition/README.md) for the table and the reading.

## Chat with a local LLM (type a question, press Enter)

The library does not contain a general-purpose language model, and cannot train one: a model that answers any question is trained on web-scale text with
clusters of GPUs. What it gives you is the chat program for **a model you run on your own machine**, with your questions never leaving it:

```bash
# 1. Install Ollama (https://ollama.com), then pull a model; a 3-billion-parameter one needs roughly 2 to 4 GB of memory
ollama pull llama3.2

# 2. Chat in the terminal: type a question, press Enter, the answer streams in
pip install -e .                          # or: pip install quantum-mind (once a release with the chat is on PyPI)
quantum-mind-chat                         # the same as: python -m quantum_mind.chat
quantum-mind-chat --model qwen2.5 --check # another model; --check asks each question 3 times and says when the answers disagree
quantum-mind-chat --web                   # the same chat in a browser tab at http://127.0.0.1:8765, this machine only
quantum-mind-chat --list-models           # what Ollama has
quantum-mind-chat --backend llamacpp --model-path ./model.gguf   # a GGUF file in-process (pip install llama-cpp-python)
quantum-mind-chat --backend demo          # no model: canned answers, to see how it works
```

Inside the chat: `/help`, `/reset`, `/model NAME`, `/system TEXT`, `/check on|off`, `/save FILE`, `/quit`; Ctrl-C stops an answer being written.
The same backend is the `generate` function of the library's robot supervisor and LLM bridge (`ChatSession.generate`).

Limits to know: the quality of the answers is the quality of the model you pick (small models make things up); the `--check` note is a heuristic
(word overlap between repeated answers), so a model can be consistently wrong and a right answer can be worded differently each time; and the tests use a fake
Ollama server that speaks the same streaming protocol, because no model could be downloaded where this was written, so try it on your machine and tell me what breaks.

## A small robot language model

`quantum_mind.cognition` also holds a language model trained from scratch (about 29,000 parameters, on a CPU) that grounds robot instructions in a camera strip, a bridge
that lets a local general LLM (for example through Ollama) widen what the robot understands without being trusted, and a dialogue belief that fuses the turns of a
conversation. With the same size and head, a bag of words fails on corrections ("go to the red one, no, the blue one": 0.51 against 0.99, and 0.50 against 1.00 as the
success of the one-motor VLA), and the bridge turns an LLM that contradicts the small model, invents words or rambles into a question to the person. The sentences are
generated by a small grammar, the held-out combinations did not separate the models, the language-modelling loss did not improve grounding, and the local LLM is only
scripted in the tests. See [`src/quantum_mind/cognition/README.md`](src/quantum_mind/cognition/README.md) and [tutorial 19](tutorials/19_robot_language_model.md).

## Thread 3 in depth: COBRA, a context-rotated belief filter

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

| Context task, expected decision cost (lower is better) | 60 training sequences | 250 | 1000 |
|---|---|---|---|
| Oracle (knows the generator) | 0.147 | 0.147 | 0.147 |
| **COBRA** | 0.347 ± 0.061 | **0.221 ± 0.005** | **0.221 ± 0.028** |
| COBRA without coherence (ablation) | 0.345 ± 0.028 | 0.265 ± 0.014 | 0.249 ± 0.010 |
| Gaussian Bayes filter | **0.280 ± 0.035** | 0.247 ± 0.006 | 0.239 ± 0.014 |
| Logistic regression | 0.323 ± 0.036 | 0.276 ± 0.019 | 0.255 ± 0.009 |
| MLP (515 parameters) | 0.358 ± 0.083 | 0.291 ± 0.036 | 0.233 ± 0.014 |
| *Control task (no context effect), COBRA / Bayes filter* | *0.288 / 0.227* | *0.209 / 0.199* | *0.204 / 0.189* |


<p align="center"><img src="https://raw.githubusercontent.com/yeshwanthguru/Quantum-mind/main/docs/assets/cobra_benchmark.png" alt="Expected decision cost against training size for COBRA and the baselines" width="100%"></p>

The honest reading is in [`benchmarks/results/context_intent/`](https://github.com/yeshwanthguru/Quantum-mind/tree/main/benchmarks/results/context_intent)
and on the [COBRA page](https://yeshwanthguru.github.io/Quantum-mind/docs/user_guide/belief.html):

* With 250 to 1000 training sequences COBRA has the lowest cost of the learned models on the context task (about 0.02 below the Bayes filter). A paired sign-flip test on per-sequence cost is significant in only 2 of 5 seeds at each size, because training varies from seed to seed.
* With 60 training sequences COBRA is **worse** than the Bayes filter (0.347 against 0.280).
* On the **control** task, where context does not matter, COBRA is slightly **worse** than the Bayes filter (0.204 against 0.189 at 1000 sequences).
* Removing coherence (the ablation) loses about 0.03 to 0.04 in cost at 250 and 1000 sequences, a hint that phase interference helps; five seeds are too few to be sure.
* Under 1.5x test noise COBRA is the best calibrated model (cost 0.481 against 0.608 for the Bayes filter at 1000 sequences).
* Its l1 coherence flags its own errors barely better than chance (AUROC about 0.53); belief entropy (about 0.81) is far better.


## API reference

The complete reference is generated from the docstrings, so it cannot drift from the code:

* **[`docs/API.md`](docs/API.md)**: every public class, function and method of every subpackage, with signatures (renders on GitHub; regenerate with `python docs/make_api.py`).
* **[Website API reference](https://yeshwanthguru.github.io/Quantum-mind/docs/api/index.html)**: one page per module with the full NumPy-style docstrings (parameters, returns, examples).
* In Python: `help(quantum_mind.embodied.MotorAgent)`, or `python -m pydoc quantum_mind.sensing`.

| Package | What it is | Main entry points |
|---|---|---|
| `quantum_mind.core` | model base class, fitting, model comparison, bootstrap | `Model`, `fit`, `compare`, `bootstrap` |
| `quantum_mind.families` | the four families of quantum-like models | `order_effects`, `qlbn`, `dynamics`, `perception` |
| `quantum_mind.belief` | COBRA, the context-rotated belief filter | `COBRA`, `OnlineBelief`, `min_cost_action` |
| `quantum_mind.cognition` | the trainable quantum-cognition core, a small robot language model, an LLM bridge and dialogue fusion (extra `cognition`) | `QuantumCognitionCore`, `RobotLanguageModel`, `LLMBridge`, `DialogueBelief` |
| `quantum_mind.applications` | robot-facing tools: calibration, conformal sets, fusion, questioning, hand-over | `calibration.SplitConformalClassifier`, `calibration.TemperatureScaling`, `calibration.AdaptiveConformalSets` |
| `quantum_mind.computing` | thread 1: robot problems as QUBOs and their solvers | `multi_robot_allocation`, `simulated_annealing`, `qaoa`, `compare_solvers` |
| `quantum_mind.sensing` | thread 2: quantum-limited sensors and dead reckoning | `atom_interferometer_sensitivity`, `mems_accelerometer`, `compare_sensor_suites` |
| `quantum_mind.embodied` | QR-VLA: the one-motor VLA, the motor agent, the supervisor and the runtime | `TinyVLA`, `QuantumRobotVLA`, `MotorAgent`, `Supervisor`, `ControlRuntime` |
| `quantum_mind.chat` | a chat program for a local LLM (terminal and browser), session memory, a self-consistency check | `ChatSession`, `OllamaBackend`, `ConsistencyCheck`, `main` |
| `quantum_mind.evaluation` | metrics, paired statistics, run manifests | `paired_bootstrap_difference`, `paired_permutation_test` |
| `quantum_mind.envs`, `quantum_mind.quantum`, `quantum_mind.circuits`, `quantum_mind.viz`, `quantum_mind.data` | simulated people, the state-vector simulator, circuits, the Bloch-sphere viewer, published data | see [`docs/API.md`](docs/API.md) |

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
* **What is not validated.** All robotics results use simulated people or the published Clinton-Gore poll aggregate. No study
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
| Published model, extended | 12 | implemented as published, with capabilities the papers do not have (bootstrap intervals, a per-person posterior updated after every answer, informative-question design) |
| Published method | 13 | implemented as published (Lüders rule, Lindblad evolution, fitting, calibration, the Bayes filter, ...) |
| Software tool | 8 | infrastructure |

The extensions that apply to every people model are `bootstrap`, `OnlinePersonModel` and
`rank_conditions` / `model_posterior`. For robots: `AdaptiveConformalSets` (sets for a person's next answer that
keep their long-run coverage), online personalisation, `trust_aware_allocation` and `sample_people` (training
against populations of simulated people).

## Architecture

```
src/quantum_mind/
├── core/           Lüders rule, density matrices, Lindblad · Model/Param · fit, compare, recovery · bootstrap, online, design
├── families/       order_effects · qlbn · dynamics · perception
├── belief/         COBRA · OnlineBelief · GaussianBayesFilter · min_cost_action
├── computing/      thread 1: robot problems as QUBOs · classical baselines · Qiskit QAOA adapter · compare_solvers
├── sensing/        thread 2: quantum limits · accelerometer models · multi-rate Kalman dead reckoning
├── applications/   robotics: question domains, trust, intent resolution, ensemble, ask_or_act · calibration · orchestration · questioning · handover · allocation · fusion
├── envs/           Gymnasium environments with simulated people
├── circuits/       Qiskit circuits of the families · run() on Aer, IBM Quantum, Amazon Braket
├── quantum/        exact state-vector simulator (used by the viewer and the circuit exports)
├── problems/       the QUBO container used by trust-aware allocation
├── viz/            Bloch vectors, trajectories, tomography · Matplotlib and Plotly viewers · LiveBloch
├── evaluation/     metrics, paired statistics, manifests, provenance
└── data/           published aggregate data sets
benchmarks/         COBRA context-intent · robot quantum computing · quantum sensing · HRI decision
examples/           21 scripts        tutorials/   19 tutorials + a packaged hand-over use case
docs/               Sphinx documentation, atlas, mathematics        notebooks/   2 Jupyter notebooks
integrations/ros2/  ROS 2 node (quantum_mind_ros)        site/   project website (built by CI)
```

## Running on quantum hardware

The circuits of the quantum-like families run on simulators and, with credentials, on hardware:

```bash
python3 examples/06_cloud_run.py --dry-run                                  # local, no account
python3 examples/06_cloud_run.py --backend ibm:least_busy --shots 4000      # IBM Quantum
python3 examples/06_cloud_run.py --backend braket:arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1 --shots 1000
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
own tests and a `requirements.txt`; tutorial 14 walks through it. Results are for simulated people only.

<a id="status"></a>

## Status and scope

| Area | What exists | What does not (yet) |
|---|---|---|
| Quantum computing for robots | three robot problems, classical baselines, a Qiskit QAOA adapter, a solver comparison and a benchmark | runs on quantum hardware; problems too large to enumerate |
| Quantum sensing for robots | quantum limits, accelerometer models, a multi-rate Kalman filter and a sensor-suite study | measured sensor data; any hardware |
| Quantum-like models | 4 families with classical baselines, fitting, BIC/AIC, recovery studies, per-person fitting and comparison, partial pooling; published aggregate data | full hierarchical Bayesian inference (only MAP partial pooling) |
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
| `[cognition]` | jax, optax | the trainable quantum-cognition core (`quantum_mind.cognition`) |
| `[all]` | all of the above, including `[cognition]` | |
| `[dev]` | pytest, pytest-cov, ruff, nbclient, build, twine | tests, coverage, linting, notebooks, packaging |

From a clone: `pip install -e ".[all,dev]"`. Python 3.10 or later.

```bash
python3 -m pytest -q --cov=quantum_mind     # the test suite (CI requires at least 80% line coverage)
python3 benchmarks/context_intent_benchmark.py          # the COBRA benchmark, about 10 to 20 minutes on 4 cores
python3 benchmarks/robot_quantum_computing_benchmark.py # robot problems on QAOA and classical solvers
python3 benchmarks/quantum_sensing_benchmark.py         # the quantum accelerometer study, a few minutes
python3 docs/make_cobra_figures.py                # redraw the COBRA figures
```

## Limitations

- Fitting aggregate counts assumes a homogeneous population. Individual differences can hide or mimic
  quantum-like structure: in example 11 a pooled fit picks the quantum-like model for a population that is half
  anchoring, and per-person comparison needs about 3,000 answers per person to tell the models apart.
- Several quantum-like models have been challenged by further tests; the family READMEs list these.
- COBRA's advantage is shown on **synthetic data built to contain a context effect**. With little training data it
  loses to the Bayes filter, training can miss the mechanism, and its coherence is a weak error signal.
- Hardware noise distorts circuits by a few percent of total variation for small circuits, and more for deep ones.

## Data

`quantum_mind.data` contains the only human data in the package, a published aggregate: the Clinton–Gore order
effect (Moore, 2002). Every other data set in the examples is simulated and labelled as such.

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
