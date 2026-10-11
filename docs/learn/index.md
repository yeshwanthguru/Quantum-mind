# Background

Quantum Mind rests on one idea from cognitive science: **quantum probability** describes how people judge,
decide and trust better than classical probability when the order or the context of the evidence matters.
The library applies that idea to robots that work with people. This section explains it from the basics.

```{mermaid}
flowchart LR
    accTitle: Background, diagram 1
    accDescr: Quantum-like cognition provides the human models; COBRA turns the same mathematics into a robot belief filter; the decision layer turns beliefs into actions under cost.
    QL["Quantum-like cognition<br/>projections · interference · order effects"]:::h --> H["Human models<br/>trust qubit · answer models"]:::op
    QL --> C["COBRA belief filter<br/>context-rotated measurement"]:::op
    H --> D["Decision layer<br/>ask or act · hand-over"]:::out
    C --> D
    classDef op fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef out fill:#0f2e1a,stroke:#7ee787,color:#e6edf3
    classDef h fill:#3d1414,stroke:#ff7b72,color:#e6edf3
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {fas}`lightbulb` Quantum-like cognition
:link: quantum_cognition
:link-type: doc

Order effects, projections, interference and the QQ equality.
:::
::::

For general introductions to robotics, machine learning and quantum computing, textbooks and the
documentation of Qiskit, PennyLane and ROS 2 already do the job; this documentation only covers what is
specific to quantum cognition in robotics.

```{toctree}
:hidden:

quantum_cognition
```
