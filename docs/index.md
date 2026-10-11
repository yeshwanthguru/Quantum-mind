---
html_theme.sidebar_secondary.remove: true
---

# Quantum Mind

```{raw} html
<div class="qm-hero">
<img class="banner" src="_static/assets/banner.svg" alt="quantum-mind banner">
<h1>Quantum-probability cognition for robots that work with people.</h1>
<p class="lead"><strong>Quantum Mind</strong> is an open-source Python library that models how people judge, decide and
trust, and how a robot should read them, with quantum probability: projections, interference, density matrices and
generalised measurements. Every model sits next to the classical model it has to beat, and every belief ends in a
cost-aware decision: hand over, wait, or ask.</p>
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

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} Quantum-like human models
:class-card: pillar-like
:link: user_guide/families/index
:link-type: doc

How people judge, decide and trust, in quantum probability, each with classical baselines and a
distinctive test. Four families, chosen for robots.
:::

:::{grid-item-card} COBRA belief filter
:class-card: pillar-quantum
:link: user_guide/belief
:link-type: doc

A density-matrix belief whose measurement basis is rotated by the previous sensor frame, so what a
reading means depends on what came before it.
:::

:::{grid-item-card} Robot decision layer
:class-card: pillar-inspired
:link: user_guide/robotics
:link-type: doc

Ask or act, question planning, calibration and conformal sets, fusion, trust-aware hand-over and
allocation, and a ROS 2 node.
:::

:::{grid-item-card} Bloch-sphere viewer
:class-card: pillar-viz
:link: user_guide/viz
:link-type: doc

Watch qubits move: gate-by-gate trajectories, animations, interactive HTML, live widgets and
tomography on real back ends.
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
solid-state sensors for navigation, magnetometry, gravimetry), which is physical hardware beyond what a software library
can offer; and (3) **quantum-probability models of the beliefs a robot holds about people** (context, order and trust),
which run on ordinary computers and can be tested today. Quantum Mind is the third thread, and it keeps a bridge to the
first: the circuits of its models run on simulators and on quantum hardware (Aer, IBM Quantum, Amazon Braket), so the same
belief model can be executed on a quantum device if hardware makes that worthwhile. I claim no speed-up, and for the first
thread I point people to Qiskit and PennyLane. So the library does not try to make robots faster with qubits. It asks one question and keeps
asking it: does quantum probability describe human context better than the classical alternative, and does a robot
decide better when it uses it? Every model sits next to the classical model it has to beat, with the ablation that
isolates what is new.

I removed the general-purpose quantum algorithms and optimisers (Qiskit, PennyLane and optimisation libraries do them
better), and I limited the scope to AI, machine learning, vision, robotics and embodied AI. Models from the literature
are implemented from their papers and cited; the {doc}`model atlas <atlas/index>` labels each as published, extended or
unique, and the unique ones are my own designs.

I do not claim a quantum computational advantage or results with real people and robots, and COBRA's gain is modest and
shown only on synthetic data. I want to run a pilot study with a real robot, make COBRA's training more reliable, and see
others replicate the results. If a claim does not hold, please open an issue.

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

Fourteen step-by-step tutorials, from the introduction to the COBRA filter and a robot loop, with real outputs and figures.
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

17 runnable scripts on quantum-like models, COBRA, robotics, perception and the viewer.
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
