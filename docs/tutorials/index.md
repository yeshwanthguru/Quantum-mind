# Tutorials

Step-by-step tutorials that start from the basics and then work through the models one by one. Each
tutorial is a notebook that runs top to bottom in a few seconds to a minute; the pages below were
executed when this documentation was built, so every number and figure is real output. The sources
are Markdown files in [`tutorials/`](https://github.com/yeshwanthguru/Quantum-mind/tree/main/tutorials)
and can be converted to notebooks with `python tutorials/_convert.py`.

Simulated data are labelled as simulated, and every model is compared with a classical baseline.
Where the classical method wins, the tutorial says so.

```{mermaid}
flowchart LR
    accTitle: Reading order of the tutorials
    accDescr: The introduction leads to four tracks: models of people (2, 3), the robot decision layer (4 to 7, then 12, 13 and the packaged hand-over use case 14), perception (8, 9) and the COBRA belief filter and viewer (10, 11).
    T1["1 · Introduction"]:::a --> T2["2 · Question order"]:::h --> T3["3 · Trust on a qubit"]:::h
    T1 --> T4["4 · Ask or act"]:::r --> T5["5 · Calibration"]:::r --> T6["6 · Orchestration"]:::r --> T7["7 · Question planner"]:::r
    T5 --> T8["8 · Fusion"]:::p --> T9["9 · Bistable perception"]:::p
    T1 --> T10["10 · COBRA belief filter"]:::q --> T11["11 · Bloch sphere"]:::q
    T7 --> T12["12 · End to end"]:::r
    T8 --> T12
    T3 --> T12 --> T13["13 · ROS 2"]:::r --> T14["14 · Hand-over package"]:::r
    classDef a fill:#0b2a4a,stroke:#79c0ff,color:#e6edf3
    classDef h fill:#3d1414,stroke:#ff7b72,color:#e6edf3
    classDef r fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef p fill:#3a2a0a,stroke:#ffa657,color:#e6edf3
    classDef q fill:#1f2a3a,stroke:#79c0ff,color:#e6edf3,stroke-dasharray:4 3
```

## Start here

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 1 · Introduction to Quantum Mind
:link: 01_introduction
:link-type: doc

Beliefs as vectors, questions as projectors, why order matters, and the fit-and-compare workflow.
:::
::::

## Models of people

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 2 · Question-order effects
:class-card: pillar-like
:link: 02_question_order
:link-type: doc

Fit the Clinton-Gore poll, run the QQ test, check model recovery.
:::

:::{grid-item-card} 3 · Trust on a qubit
:class-card: pillar-like
:link: 03_trust_on_a_qubit
:link-type: doc

Trust on the Bloch sphere, the effect of asking, and a trust-aware hand-over policy.
:::

::::

## Robot decision layer

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 4 · When to ask for help
:class-card: pillar-inspired
:link: 04_ask_or_act
:link-type: doc

Expected cost, model disagreement and hazard-aware decisions.
:::

:::{grid-item-card} 5 · Calibration and conformal sets
:class-card: pillar-inspired
:link: 05_calibration
:link-type: doc

Reliability diagrams, Platt, temperature and isotonic recalibration, coverage guarantees.
:::

:::{grid-item-card} 6 · Choosing between modules
:class-card: pillar-inspired
:link: 06_orchestration
:link-type: doc

A cost-aware gate and a meta-learned prior for new tasks.
:::

:::{grid-item-card} 7 · Planning which question to ask
:class-card: pillar-inspired
:link: 07_question_planner
:link-type: doc

Value of information with order-dependent answers.
:::

:::{grid-item-card} 12 · End to end: fetch and hand over
:class-card: pillar-inspired
:link: 12_end_to_end_robot
:link-type: doc

Calibration, fusion, questions and trust-aware hand-over in one robot loop.
:::

:::{grid-item-card} 13 · Running it on a robot (ROS 2)
:class-card: pillar-inspired
:link: 13_ros2_service
:link-type: doc

The JSON service and the ROS 2 node, run without a ROS installation.
:::

:::{grid-item-card} 14 · A robot package: trust-aware hand-over
:class-card: pillar-inspired
:link: 14_trust_handover_package
:link-type: doc

An installable package with a benchmark, a PyBullet arm simulation (GIF), a ROS 2 node and tests.
:::
::::

## Perception

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 8 · Fusing vision, speech and gaze
:link: 08_fusion
:link-type: doc

Adapters, Bayesian, Dempster-Shafer and quantum-like fusion.
:::

:::{grid-item-card} 9 · Bistable perception
:link: 09_bistable_perception
:link-type: doc

The quantum Zeno model against Markov and gamma-renewal baselines.
:::
::::

## COBRA belief filter and the viewer

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 10 · A context-aware belief filter
:class-card: pillar-quantum
:link: 10_context_belief_filter
:link-type: doc

A density-matrix belief whose measurement basis depends on the previous frame, against a Bayes filter.
:::

:::{grid-item-card} 11 · Seeing qubits: the Bloch-sphere viewer
:class-card: pillar-quantum
:link: 11_bloch_sphere
:link-type: doc

Trajectories, animations and tomography for circuits and for the trust qubit.
:::
::::

```{toctree}
:hidden:
:caption: Tutorials

01_introduction
02_question_order
03_trust_on_a_qubit
04_ask_or_act
05_calibration
06_orchestration
07_question_planner
08_fusion
09_bistable_perception
10_context_belief_filter
11_bloch_sphere
12_end_to_end_robot
13_ros2_service
14_trust_handover_package
```
