# User guide

Quantum Mind is a library for **quantum-probability cognition in robotics**. It has three layers, each next to
the classical model it has to beat, plus a Bloch-sphere viewer.

```{mermaid}
flowchart TB
    core["quantum_mind.core<br/>Model · Param · fit · compare · recovery"]
    fam["quantum_mind.families<br/>4 quantum-like families"]
    bel["quantum_mind.belief<br/>COBRA belief filter"]
    app["quantum_mind.applications<br/>robot decision layer"]
    circ["quantum_mind.circuits<br/>circuits + run()"]
    sim["quantum_mind.quantum<br/>state-vector simulator"]
    viz["quantum_mind.viz<br/>Bloch-sphere viewer"]
    hw[("Aer · IBM Quantum · Amazon Braket")]
    core --> fam --> app
    core --> bel --> app
    fam --> circ --> hw
    sim --> viz
    circ --> viz
    fam --> viz
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Concepts
:link: concepts
:link-type: doc

What separates quantum-like models from quantum computing, and how to test each.
:::

:::{grid-item-card} Quantum-like families
:class-card: pillar-like
:link: families/index
:link-type: doc

Order effects, quantum-like Bayesian networks, trust and belief dynamics, and bistable perception.
:::

:::{grid-item-card} COBRA belief filter
:class-card: pillar-quantum
:link: belief
:link-type: doc

A density-matrix belief whose measurement basis is rotated by the previous sensor frame.
:::

:::{grid-item-card} Robotics
:link: robotics
:link-type: doc

Questioning designs, trust, intent resolution, a human-model ensemble and a ROS 2 node.
:::

:::{grid-item-card} Bloch-sphere viewer
:class-card: pillar-viz
:link: viz
:link-type: doc

Trajectories, animations, interactive HTML, live updates and tomography.
:::

:::{grid-item-card} Simulator and circuits
:link: quantum
:link-type: doc

The exact state-vector simulator and the circuits of the quantum-like families.
:::
::::

## Status and scope

| Area | What exists | What does not (yet) |
|---|---|---|
| Quantum-like models | 4 families with classical baselines, fitting, BIC/AIC, recovery studies, per-person fitting and comparison, partial pooling; published aggregate data | full hierarchical Bayesian inference (only MAP partial pooling) |
| COBRA belief filter | context-rotated density-matrix filter, online mode, Bayes and incoherent limits, benchmark with paired tests | real-person data; reliable training (the context mechanism is found on some restarts only) |
| Robotics | simulated populations for three question domains, questioning designs, trust protocol, intent resolution, `HumanModelEnsemble`, `ask_or_act`, risk-aware decisions, calibration and conformal sets, orchestration gates, a question planner, trust-aware hand-over and allocation, multimodal fusion, Gymnasium environments, `HumanModelService`, a ROS 2 node | a run of the ROS 2 node inside a ROS 2 installation; data from human-robot studies |
| Simulator and circuits | exact state-vector simulator with adjoint gradients; Qiskit export; circuits for the families; IBM submission through the Qiskit Runtime Sampler | noise-aware training; runs on hardware |
| Viewer | trajectories, animations, interactive HTML, live widget, tomography, concurrence timeline, Q-sphere | |

## Limitations

- Fitting aggregate counts assumes a homogeneous population. Individual differences can hide or
  mimic quantum-like structure (example 14).
- Several quantum-like models have been challenged by further tests; the family pages list these.
- COBRA is a classical simulation of the quantum-probability formalism. No quantum advantage is claimed, and
  its advantage over classical filters is shown only on synthetic data built to contain a context effect.
- Hardware noise distorts circuits by a few percent of total variation for small circuits, and more
  for deep ones. No hardware results are included.

```{toctree}
:hidden:
:maxdepth: 2

concepts
extensions
families/index
belief
robotics
ros2
circuits
quantum
problems
viz
```
