---
html_theme.sidebar_secondary.remove: true
---

# Quantum Mind

```{raw} html
<div class="qm-hero">
<img class="banner" src="_static/assets/banner.svg" alt="quantum-mind banner">
<h1>Quantum robotics: computing, sensing and cognition, each against its classical baseline.</h1>
<p class="lead"><strong>Quantum Mind</strong> is an open-source Python library for the three threads of quantum robotics: <strong>quantum
computing</strong> for robot algorithms (robot problems as optimisation, with classical baselines and a Qiskit QAOA adapter),
<strong>quantum sensing</strong> (quantum-limited sensor models and their fusion with classical sensors) and <strong>quantum-probability
cognition</strong> (how people judge, decide and trust, and how a robot should read them). Every model sits next to the classical
model it has to beat, and every belief ends in a cost-aware decision.</p>
</div>
```

```{raw} html
<div class="qm-badges">
<span>Version __VERSION__</span><span>Apache-2.0</span><span>Python 3.10 to 3.12</span>
<span>Qiskit 2.x</span><span>IBM Quantum</span><span>Amazon Braket</span><span>NumPy-style API</span>
</div>
<p class="qm-downloads">
<a href="https://pypistats.org/packages/quantum-mind"><img src="https://img.shields.io/pypi/dm/quantum-mind?label=downloads%2Fmonth" alt="PyPI downloads per month" height="20"></a>
<a href="https://pepy.tech/projects/quantum-mind"><img src="https://static.pepy.tech/badge/quantum-mind" alt="Total PyPI downloads" height="20"></a>
</p>
```

```bash
pip install "quantum-mind[all]"
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 1 · Quantum computing for robots
:class-card: pillar-quantum
:link: user_guide/computing
:link-type: doc

Multi-robot allocation, grasp selection and slot scheduling as QUBOs, classical baselines and a thin Qiskit QAOA adapter.
:::

:::{grid-item-card} 2 · Quantum sensing for robots
:class-card: pillar-quantum
:link: user_guide/sensing
:link-type: doc

Quantum limits, atom-interferometer and MEMS accelerometer models, and a Kalman filter that fuses them.
:::

:::{grid-item-card} 3 · Quantum-probability cognition
:class-card: pillar-like
:link: user_guide/belief
:link-type: doc

COBRA, a context-rotated density-matrix belief filter, with quantum-like human models and the robot decision layer.
:::

:::{grid-item-card} Quantum-like human models
:class-card: pillar-like
:link: user_guide/families/index
:link-type: doc

Question order, quantum Bayesian networks, trust dynamics and bistable perception, each with classical baselines. Four families.
:::

:::{grid-item-card} Robot decision layer
:class-card: pillar-inspired
:link: user_guide/robotics
:link-type: doc

Ask or act, question planning, calibration and conformal sets, fusion, trust-aware hand-over and allocation, ROS 2.
:::

:::{grid-item-card} Bloch-sphere viewer
:class-card: pillar-viz
:link: user_guide/viz
:link-type: doc

Gate-by-gate trajectories, animations, interactive HTML, live widgets and tomography on real back ends.
:::
::::

## Author's view

*Yeshwanth Guru, author and maintainer.*

I started Quantum Mind from one belief: a robot that works with people needs a different kind of probability than the
one inside most of its filters. People are not neutral sensors. A look changes what a hand movement means, a question
changes the answer to the next question, and asking someone whether they trust a robot changes their trust. Classical
filters assume the evidence arrives in independent pieces; quantum probability was built for the case where the order
and the context of a measurement matter.

Quantum robotics is a young field with at least three threads, and I think it helps to keep them apart: (1) **quantum
computing for robot algorithms** (planning, optimisation, learning), which is promising on paper but has shown no
practical advantage on real robot problems with today's noisy devices; (2) **quantum sensing for robots** (atomic and
solid-state sensors for navigation, magnetometry, gravimetry), which is physical hardware; and (3) **quantum-probability
models of the beliefs a robot holds about people** (context, order and trust), which run on ordinary computers and can be
tested today.

Quantum Mind covers all three, because I want to work on the whole umbrella, and it holds each thread to the same standard: a
classical baseline, an ablation, and results that are allowed to say no. Thread 1 ({mod}`quantum_mind.computing`) poses robot
problems as binary optimisation, supplies the classical baselines and a thin adapter to Qiskit's QAOA, and does not re-implement
what Qiskit and PennyLane do better. Thread 2 ({mod}`quantum_mind.sensing`) models quantum-limited sensors and fuses a slow,
stable quantum accelerometer with a fast, drifting classical one in simulation, and does not claim to build a sensor. Thread 3
({mod}`quantum_mind.belief`) is the quantum-probability model of the beliefs a robot holds about people, the part I have developed
furthest. The circuits of the thread-3 models also run on quantum hardware, which is the bridge between the threads. I claim no
speed-up anywhere.

In every thread I ask the same question: does the quantum version beat the classical one, and does the robot decide or move better
with it? For beliefs about people that becomes: does quantum probability describe human context better than the classical
alternative? Every model sits next to the classical model it has to beat, with the ablation that isolates what is new.

I did not re-implement general-purpose quantum algorithms or optimisers (Qiskit, PennyLane and optimisation libraries do them
better; thread 1 connects to Qiskit instead), and I limited the scope to AI, machine learning, vision, robotics and embodied AI.
Models from the literature are implemented from their papers and cited; the {doc}`model atlas <atlas/index>` labels each as
published, extended or unique, and the unique ones are my own designs.

I do not claim a quantum computational advantage (QAOA does no better than random sampling on the problems I tested), quantum
sensing hardware, or results with real people and robots, and COBRA's gain is modest and shown only on synthetic data. I want to
run a pilot study with a real robot, make COBRA's training more reliable, run threads 1 and 2 on quantum hardware and measured
sensor data, and see others replicate the results. If a claim does not hold, please open an issue.

## What is new in this library

Classical filters treat evidence as arriving in neutral, independent pieces. People and robots do not: a look
changes what a hand movement means, a question changes the answer to the next one, and asking about trust
changes trust. The quantum-probability formalism represents exactly this, and Quantum Mind turns it into
tested code:

* **COBRA**, a belief filter whose generalised measurement is rotated by the previous frame, with the Bayes
  filter and an incoherent filter as built-in limits so the effect of context and the effect of coherence can be
  separated. On a synthetic task with a context effect it lowers decision cost by about 0.02 against a Bayes
  filter given enough data (significant in 2 of 5 seeds), but it is worse with little data and slightly worse
  where context does not matter; the benchmark, the ablations and the cases where it loses are reported in full.
* **Trust as an open quantum system**, an **order-aware** hand-over policy and a **value-of-information
  question planner** whose answers depend on the order of the questions.
* A decision layer that never reports a belief without a cost, a calibration check and a classical baseline.

It runs on ordinary computers and claims no quantum computational advantage.

## See the qubits

A Qiskit circuit, gate by gate. While the two qubits are entangled their Bloch vectors move inside
the sphere. Drag to rotate; press play or use the slider.

```{raw} html
<iframe class="qm-frame" src="_static/demos/entanglement.html" loading="lazy" title="Entangling two qubits"></iframe>
```

```python
from qiskit import QuantumCircuit
from quantum_mind.viz import circuit_trajectory, animate_bloch, save_html

qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1); qc.ry(0.6, 1)
save_html(animate_bloch(circuit_trajectory(qc, steps=15)), 'bell.html')
```

The [two-qubit playground](https://yeshwanthguru.github.io/Quantum-mind/#playground-section)
runs the same simulation in the browser.

## Where to go next

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {fas}`rocket` Getting started
:link: getting_started
:link-type: doc

Install, then fit, compare and simulate in a few lines.
:::

:::{grid-item-card} {fas}`graduation-cap` Background
:link: learn/index
:link-type: doc

Quantum-like cognition from the basics: projections, order effects, interference.
:::

:::{grid-item-card} {fas}`person-chalkboard` Tutorials
:link: tutorials/index
:link-type: doc

Sixteen step-by-step tutorials, from the introduction to the three threads and a robot loop, with real outputs and figures.
:::

:::{grid-item-card} {fas}`diagram-project` Model atlas
:link: atlas/index
:link-type: doc

A block diagram for every model: inputs, internals, outputs, baselines and references.
:::

:::{grid-item-card} {fas}`square-root-variable` Mathematics
:link: math/index
:link-type: doc

The equations of every model, as the code computes them, from the Born rule to COBRA.
:::

:::{grid-item-card} {fas}`book` User guide
:link: user_guide/index
:link-type: doc

The layers, every model family, circuits, hardware and the viewer.
:::

:::{grid-item-card} {fas}`images` Example gallery
:link: auto_examples/index
:link-type: doc

20 runnable scripts on quantum-like models, COBRA, robotics, perception and the viewer.
:::

:::{grid-item-card} {fas}`laptop-code` Notebooks
:link: notebooks/index
:link-type: doc

Interactive Bloch spheres, robot questioning and trust.
:::

:::{grid-item-card} {fas}`code` API reference
:link: api/index
:link-type: doc

Every public class and function, with parameters, returns and examples.
:::

:::{grid-item-card} {fas}`circle-question` Help
:link: help/index
:link-type: doc

Which model do I need, glossary, FAQ and what is validated.
:::

:::{grid-item-card} {fas}`circle-info` About
:link: about/index
:link-type: doc

Changelog, citing, licence and contributing.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

getting_started
learn/index
tutorials/index
user_guide/index
atlas/index
math/index
api/index
auto_examples/index
notebooks/index
help/index
about/index
research_claims
research_validation
```
