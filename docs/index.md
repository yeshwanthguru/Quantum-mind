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

:::{grid-item-card} Quantum-like
:class-card: pillar-like
:link: user_guide/families/index
:link-type: doc

Models of human judgement, decision and trust in quantum probability, each with classical
baselines and a distinctive test. Twelve families.
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

The [two-qubit playground](https://yeshwanthguru.github.io/Quantum-mind/#playground-section)
runs the same simulation in the browser.

## Learn the foundations

New to some of these fields? Each tab is a short start; the full pages explain the terms from the
basics, with diagrams, runnable examples and references.

::::{tab-set}

:::{tab-item} Robotics
:sync: robotics

A robot **senses**, **decides** and **acts**, in a loop, in real time. Kinematics maps joint angles to
positions, controllers turn errors into motor commands, filters keep a belief about the robot's state,
planners choose actions, and middleware such as ROS 2 connects the parts. Robots that work with people
also need models of what people want and how much they trust the robot.

{bdg-primary}`kinematics` {bdg-primary}`control` {bdg-primary}`state estimation` {bdg-primary}`planning` {bdg-primary}`HRI` {bdg-primary}`ROS 2`

[Read: Robotics](learn/robotics.md) · [Tutorial: when to ask for help](tutorials/04_ask_or_act.ipynb)
:::

:::{tab-item} AI
:sync: ai

Artificial intelligence designs **agents** that choose actions to reach goals: search, reasoning with
probability, decision theory and the value of information. Machine learning, deep learning and
reinforcement learning are ways to build parts of an agent from data.

{bdg-secondary}`agents` {bdg-secondary}`Bayes' rule` {bdg-secondary}`expected utility` {bdg-secondary}`value of information` {bdg-secondary}`LLMs and VLMs`

[Read: Artificial intelligence](learn/ai.md) · [Tutorial: planning which question to ask](tutorials/07_question_planner.ipynb)
:::

:::{tab-item} Machine learning
:sync: ml

Fit a model's parameters to data and judge it on data it has not seen. Likelihood, model comparison by
AIC and BIC, partial pooling across people, calibration of confidence and kernel methods, all of which
the library uses.

{bdg-info}`likelihood` {bdg-info}`BIC` {bdg-info}`partial pooling` {bdg-info}`calibration` {bdg-info}`conformal sets` {bdg-info}`kernels`

[Read: Machine learning](learn/machine_learning.md) · [Tutorial: calibration](tutorials/05_calibration.ipynb)
:::

:::{tab-item} Deep learning
:sync: dl

Neural networks with many layers learn features from raw data, trained by backpropagation. Their
softmax confidences are often over-confident, and large networks must be compressed for robots. The
library recalibrates their outputs, fuses them, and offers tensor-train compression and quanvolution.

{bdg-warning}`neural networks` {bdg-warning}`backpropagation` {bdg-warning}`CNNs` {bdg-warning}`transformers` {bdg-warning}`compression`

[Read: Deep learning](learn/deep_learning.md) · [Tutorial: tensor trains](tutorials/12_tensor_train.ipynb)
:::

:::{tab-item} Reinforcement learning
:sync: rl

Learn what to do from reward: Markov decision processes, value functions, Q-learning, exploration and
policy gradients. The library ships Gymnasium environments with simulated people, amplitude
exploration and a variational quantum policy.

{bdg-success}`MDP` {bdg-success}`Q-learning` {bdg-success}`exploration` {bdg-success}`REINFORCE` {bdg-success}`Gymnasium`

[Read: Reinforcement learning](learn/reinforcement_learning.md) · [Tutorial: RL with simulated people](tutorials/10_rl_environments.ipynb)
:::

:::{tab-item} Perception
:sync: perception

Turn camera, microphone and gaze signals into objects, words and intentions, each with a trustworthy
confidence, and fuse several uncertain sources. Includes multistable perception and the quantum Zeno
model.

{bdg-danger}`detection` {bdg-danger}`calibration` {bdg-danger}`sensor fusion` {bdg-danger}`Dempster-Shafer` {bdg-danger}`bistable perception`

[Read: Perception and vision](learn/perception.md) · [Tutorial: fusing vision, speech and gaze](tutorials/08_fusion.ipynb)
:::

:::{tab-item} Embodied AI
:sync: embodied

Agents with bodies that perceive and act in the world, trained in simulation and transferred to real
robots, increasingly driven by foundation models. When people are in the loop, the agent's own
questions change the person; that is where quantum-like models of people come in.

{bdg-primary-line}`embodiment` {bdg-primary-line}`sim-to-real` {bdg-primary-line}`VLA models` {bdg-primary-line}`behaviour trees` {bdg-primary-line}`interaction`

[Read: Embodied AI](learn/embodied_ai.md) · [Tutorial: trust on a qubit](tutorials/03_trust_on_a_qubit.ipynb)
:::

:::{tab-item} Quantum
:sync: quantum

Quantum computing (qubits, gates, entanglement, variational algorithms), quantum-like cognition (the
mathematics of quantum probability applied to human judgement) and quantum-inspired algorithms
(classical heuristics that borrow quantum ideas).

{bdg-secondary-line}`qubits` {bdg-secondary-line}`QAOA` {bdg-secondary-line}`order effects` {bdg-secondary-line}`QQ test` {bdg-secondary-line}`tensor networks`

[Quantum computing](learn/quantum_computing.md) · [Quantum-like cognition](learn/quantum_cognition.md) · [Quantum-inspired algorithms](learn/quantum_inspired.md)
:::
::::

## Where to go next

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {fas}`rocket` Getting started
:link: getting_started
:link-type: doc

Install, then fit, compare and simulate in a few lines.
:::

:::{grid-item-card} {fas}`graduation-cap` Foundations
:link: learn/index
:link-type: doc

Robotics, AI, machine learning, deep learning, RL, perception, embodied AI and quantum, from the basics.
:::

:::{grid-item-card} {fas}`person-chalkboard` Tutorials
:link: tutorials/index
:link-type: doc

Eighteen step-by-step tutorials, from the introduction to every model, with real outputs and figures.
:::

:::{grid-item-card} {fas}`diagram-project` Model atlas
:link: atlas/index
:link-type: doc

A block diagram for every model: inputs, internals, outputs, baselines and references.
:::

:::{grid-item-card} {fas}`square-root-variable` Mathematics
:link: math/index
:link-type: doc

The equations of every model, as the code computes them, from the Born rule to QAOA.
:::

:::{grid-item-card} {fas}`book` User guide
:link: user_guide/index
:link-type: doc

The three pillars, every model family, circuits, hardware and the viewer.
:::

:::{grid-item-card} {fas}`images` Example gallery
:link: auto_examples/index
:link-type: doc

36 runnable scripts across surveys, finance, medicine, robotics, perception, learning, physics and networks.
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
```
