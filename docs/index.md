---
html_theme.sidebar_secondary.remove: true
---

# Quantum Mind

```{raw} html
<div class="qm-hero">
<img class="banner" src="_static/assets/banner.svg" alt="quantum-mind banner">
<h1>Quantum-like, quantum and quantum-inspired models, compared on the same problems.</h1>
<p class="lead"><strong>Quantum Mind</strong> is an open-source Python library for robots and AI agents that work with
people. It models how people judge, decide and trust with quantum probability, always next to the classical
models it has to beat; runs quantum machine learning and algorithms as Qiskit circuits; provides
quantum-inspired optimisers and learners; and shows every qubit moving on a 3D Bloch sphere.</p>
</div>
```

```{raw} html
<div class="qm-badges">
<span>Version __VERSION__</span><span>Apache-2.0</span><span>Python 3.10 to 3.12</span>
<span>Qiskit 2.x</span><span>IBM Quantum</span><span>Amazon Braket</span><span>NumPy-style API</span>
</div>
```

```bash
pip install "quantum-mind[all]"
```

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} Quantum-like
:class-card: pillar-like
:link: user_guide/families/index
:link-type: doc

Models of human judgement, decision and trust in quantum probability, each with classical
baselines and a distinctive test. Eleven families.
:::

:::{grid-item-card} Quantum
:class-card: pillar-quantum
:link: user_guide/quantum
:link-type: doc

Gate-model machine learning and algorithms: classifiers, kernels, QAOA, VQE, Grover, QFT,
amplitude estimation and quantum walks.
:::

:::{grid-item-card} Quantum-inspired
:class-card: pillar-inspired
:link: user_guide/inspired
:link-type: doc

Classical algorithms that borrow quantum ideas: QIEA, QPSO, simulated quantum annealing, tensor
networks, reinforcement learning and ranking.
:::

:::{grid-item-card} Bloch-sphere viewer
:class-card: pillar-viz
:link: user_guide/viz
:link-type: doc

Watch qubits move: gate-by-gate trajectories, animations, interactive HTML, live widgets and
tomography on real back ends.
:::
::::

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

The [two-qubit playground](https://yeshwanthguru.github.io/quantum-mind/#playground-section)
runs the same simulation in the browser.

## Where to go next

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {fas}`rocket` Getting started
:link: getting_started
:link-type: doc

Install, then fit, compare and simulate in a few lines.
:::

:::{grid-item-card} {fas}`book` User guide
:link: user_guide/index
:link-type: doc

The three pillars, every model family, circuits, hardware and the viewer.
:::

:::{grid-item-card} {fas}`images` Example gallery
:link: auto_examples/index
:link-type: doc

31 runnable scripts across surveys, finance, medicine, robotics, physics and networks.
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
user_guide/index
auto_examples/index
notebooks/index
api/index
about/index
```
