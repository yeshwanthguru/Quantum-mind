# User guide

Quantum Mind covers the three ways "quantum" enters modelling today. They share one problem format, one
fitting and comparison workflow, and one Bloch-sphere viewer.

```{mermaid}
flowchart TB
    core["quantum_mind.core<br/>Model · Param · fit · compare · recovery"]
    fam["quantum_mind.families<br/>11 quantum-like families"]
    app["quantum_mind.applications<br/>robotics"]
    prob["quantum_mind.problems<br/>QUBO builders"]
    qm["quantum_mind.quantum<br/>classifiers · kernels · QAOA · VQE · Grover · QFT · QAE · walks"]
    ins["quantum_mind.inspired<br/>QIEA · QPSO · SQA · MPS · QRL · QLM"]
    circ["quantum_mind.circuits<br/>circuits + run()"]
    viz["quantum_mind.viz<br/>Bloch-sphere viewer"]
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

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Concepts
:link: concepts
:link-type: doc

What separates quantum-like, quantum and quantum-inspired models, and how to test each.
:::

:::{grid-item-card} Quantum-like families
:class-card: pillar-like
:link: families/index
:link-type: doc

Order effects, conjunctions, interference, Bayesian networks, dynamics, decisions, contextuality,
similarity, games, memory and concepts.
:::

:::{grid-item-card} Quantum models
:class-card: pillar-quantum
:link: quantum
:link-type: doc

Machine learning and algorithms on a batched simulator with exact gradients.
:::

:::{grid-item-card} Quantum-inspired models
:class-card: pillar-inspired
:link: inspired
:link-type: doc

Optimisers, tensor networks, reinforcement learning and document ranking.
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
::::

## Status and scope

| Area | What exists | What does not (yet) |
|---|---|---|
| Quantum-like models | 11 families with classical baselines, fitting, BIC/AIC, recovery studies, per-person fitting and comparison; published aggregate data | hierarchical (partial-pooling) models |
| Robotics | simulated populations for three question domains, questioning designs, trust protocol, intent resolution, `HumanModelEnsemble`, `ask_or_act`, `HumanModelService`, a ROS 2 node | a run of the ROS 2 node inside a ROS 2 installation; data from human-robot studies |
| Quantum models | classifiers, regressor, kernels, QAOA, VQE, Grover, QFT, amplitude estimation, quantum walks on an exact simulator with adjoint gradients; Qiskit export; IBM submission through the Qiskit Runtime Sampler | quantum advantage (none claimed); noise-aware training; runs on hardware |
| Quantum-inspired | QIEA, QPSO, SQA, simulated-annealing baseline, MPS classifier, quantum reinforcement learning, quantum language model | |
| Viewer | trajectories, animations, interactive HTML, live widget, tomography, concurrence timeline, Q-sphere | |

## Limitations

- Fitting aggregate counts assumes a homogeneous population. Individual differences can hide or
  mimic quantum-like structure (example 22).
- Several quantum-like models have been challenged by further tests; the family pages list these.
- The quantum models are small and exactly simulable. No quantum advantage is claimed: on the
  example classification, forecasting, anomaly-detection and clustering tasks a classical model is
  at least as accurate.
- "Quantum-inspired" names where an idea came from, not a speed-up.
- Hardware noise distorts circuits by a few percent of total variation for small circuits, and more
  for deep ones. No hardware results are included.

```{toctree}
:hidden:
:maxdepth: 2

concepts
families/index
quantum
inspired
problems
circuits
viz
robotics
ros2
```
