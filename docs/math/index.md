# Mathematics

The equations behind every model in Quantum Mind, written in the form the code computes them. Each page
names the class or function that implements each equation, so the mathematics, the
[model atlas](../atlas/index.md) diagrams and the [API reference](../api/index.rst) can be read side by side.

```{mermaid}
flowchart LR
    C["Core<br/>Born and Lüders rules · Lindblad · likelihood · AIC/BIC"]:::op --> QL["Quantum-like families<br/>projections · interference · walks · Zeno"]:::hum
    C --> R["Decision layer<br/>value of information · calibration · conformal · empirical Bayes"]:::out
    QL --> R
    P["Perception<br/>adapters · Bayes · Dempster-Shafer · Kraus fusion"]:::in --> R
    L["Learning<br/>MDPs · Q-learning · amplitude exploration · REINFORCE · TT-SVD"]:::out
    Q["Quantum<br/>circuits · kernels · QAOA · VQE · Grover · QFT · QAE · walks"]:::in --> L
    I["Quantum-inspired<br/>QUBO · QIEA · QPSO · SQA · MPS · QLM"]:::in --> L
    classDef in fill:#0b2a4a,stroke:#79c0ff,color:#e6edf3
    classDef op fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef out fill:#0f2e1a,stroke:#7ee787,color:#e6edf3
    classDef hum fill:#3d1414,stroke:#ff7b72,color:#e6edf3
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {fas}`square-root-variable` Core
:link: core
:link-type: doc

States, projectors, the Born and Lüders rules, open-system dynamics, parameter transforms, likelihood,
AIC, BIC, model recovery.
:::

:::{grid-item-card} {fas}`brain` Quantum-like families
:class-card: pillar-like
:link: quantum_like
:link-type: doc

Question order and the QQ equality, conjunction, interference, QLBN, walks, trust qubit, decision,
contextuality, similarity, games, memory, concepts, Zeno perception.
:::

:::{grid-item-card} {fas}`robot` Robot decision layer
:class-card: pillar-inspired
:link: robotics
:link-type: doc

Ensemble uncertainty, ask or act, calibration, conformal sets, Clopper-Pearson, gates, value of
information, partial pooling, hand-over costs.
:::

:::{grid-item-card} {fas}`eye` Perception and fusion
:link: perception
:link-type: doc

Detector, speech and gaze adapters; Bayesian, Dempster-Shafer and quantum-like (Kraus) fusion.
:::

:::{grid-item-card} {fas}`gamepad` Learning
:class-card: pillar-viz
:link: learning
:link-type: doc

MDPs of the environments, Q-learning, exploration rules, REINFORCE on a circuit, tensor trains,
quanvolution.
:::

:::{grid-item-card} {fas}`atom` Quantum models
:class-card: pillar-quantum
:link: quantum
:link-type: doc

Encodings, parameter shift and adjoint gradients, classifiers, kernels, QAOA, VQE, Grover, QFT,
amplitude estimation, quantum walks.
:::

:::{grid-item-card} {fas}`wand-magic-sparkles` Quantum-inspired algorithms
:link: inspired
:link-type: doc

QUBO formulations, QIEA, QPSO, simulated quantum annealing, MPS classifier, quantum language model.
:::
::::

```{toctree}
:hidden:

core
quantum_like
robotics
perception
learning
quantum
inspired
```
