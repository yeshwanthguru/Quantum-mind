# Foundations

Short introductions, from the basics, to the fields Quantum Mind draws on. Each page explains the key
terms, shows a diagram and a small runnable example, points to where the idea lives in the library,
and lists textbooks and papers to read next. No background beyond school mathematics and a little
Python is assumed.

```{mermaid}
flowchart LR
    accTitle: Foundations, diagram 1
    accDescr: How the Foundations topics connect: robotics, AI, machine learning, deep learning, reinforcement learning and perception lead to embodied AI; quantum computing leads to quantum-inspired algorithms and quantum-like cognition.
    R["Robotics"]:::a --> E["Embodied AI"]:::c
    AI["Artificial intelligence"]:::a --> ML["Machine learning"]:::b --> DL["Deep learning"]:::b
    ML --> RL["Reinforcement learning"]:::b
    DL --> P["Perception and vision"]:::b
    P --> E
    RL --> E
    QC["Quantum computing"]:::q --> QI["Quantum-inspired algorithms"]:::q
    QC --> QL["Quantum-like cognition"]:::h
    QL --> E
    QI --> RL
    classDef a fill:#0b2a4a,stroke:#79c0ff,color:#e6edf3
    classDef b fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef c fill:#0f2e1a,stroke:#7ee787,color:#e6edf3
    classDef q fill:#1f2a3a,stroke:#79c0ff,color:#e6edf3,stroke-dasharray:4 3
    classDef h fill:#3d1414,stroke:#ff7b72,color:#e6edf3
```

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {fas}`robot` Robotics
:link: robotics
:link-type: doc

Sense, decide, act. Kinematics, control, state estimation, planning, HRI and ROS 2.
:::

:::{grid-item-card} {fas}`brain` Artificial intelligence
:link: ai
:link-type: doc

Agents, search, probability, decision theory and the value of information.
:::

:::{grid-item-card} {fas}`chart-line` Machine learning
:link: machine_learning
:link-type: doc

Fitting, likelihood, model comparison, partial pooling, calibration and kernels.
:::

:::{grid-item-card} {fas}`layer-group` Deep learning
:link: deep_learning
:link-type: doc

Neural networks, backpropagation, CNNs, transformers, confidence and compression.
:::

:::{grid-item-card} {fas}`gamepad` Reinforcement learning
:link: reinforcement_learning
:link-type: doc

MDPs, value functions, Q-learning, exploration, policy gradients and Gymnasium.
:::

:::{grid-item-card} {fas}`eye` Perception and vision
:link: perception
:link-type: doc

Detection, calibration, sensor fusion and multistable perception.
:::

:::{grid-item-card} {fas}`person-walking` Embodied AI
:link: embodied_ai
:link-type: doc

Agents with bodies, sim-to-real, foundation models and interaction with people.
:::

:::{grid-item-card} {fas}`atom` Quantum computing
:link: quantum_computing
:link-type: doc

Qubits, the Bloch sphere, gates, entanglement and variational algorithms.
:::

:::{grid-item-card} {fas}`lightbulb` Quantum-like cognition
:link: quantum_cognition
:link-type: doc

Order effects, projections, interference and the QQ equality.
:::

:::{grid-item-card} {fas}`wand-magic-sparkles` Quantum-inspired algorithms
:link: quantum_inspired
:link-type: doc

QIEA, QPSO, simulated quantum annealing and tensor networks.
:::
::::

```{toctree}
:hidden:

robotics
ai
machine_learning
deep_learning
reinforcement_learning
perception
embodied_ai
quantum_computing
quantum_cognition
quantum_inspired
```
