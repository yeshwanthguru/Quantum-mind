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
    accDescr: The introduction leads to five tracks: models of people (2, 3, 17), the robot decision layer (4 to 7, then 16, 18 and the packaged hand-over use case 19), perception (8, 9), learning (10 to 13) and quantum computing (14, 15).
    T1["1 · Introduction"]:::a --> T2["2 · Question order"]:::h --> T3["3 · Trust on a qubit"]:::h
    T1 --> T4["4 · Ask or act"]:::r --> T5["5 · Calibration"]:::r --> T6["6 · Orchestration"]:::r --> T7["7 · Question planner"]:::r
    T5 --> T8["8 · Fusion"]:::p --> T9["9 · Bistable perception"]:::p
    T1 --> T10["10 · RL environments"]:::l --> T11["11 · Quantum policy"]:::l
    T10 --> T12["12 · Tensor trains"]:::l --> T13["13 · Quanvolution"]:::l
    T1 --> T14["14 · Task allocation"]:::q --> T15["15 · Bloch sphere"]:::q
    T2 --> T17["17 · Judgement families"]:::h
    T7 --> T16["16 · End to end"]:::r
    T8 --> T16
    T3 --> T16 --> T18["18 · ROS 2"]:::r --> T19["19 · Hand-over package"]:::r
    classDef a fill:#0b2a4a,stroke:#79c0ff,color:#e6edf3
    classDef h fill:#3d1414,stroke:#ff7b72,color:#e6edf3
    classDef r fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef p fill:#3a2a0a,stroke:#ffa657,color:#e6edf3
    classDef l fill:#0f2e1a,stroke:#7ee787,color:#e6edf3
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

:::{grid-item-card} 17 · A tour of the judgement families
:class-card: pillar-like
:link: 17_judgement_families
:link-type: doc

Conjunction, interference, QLBN, decision, similarity, memory, concepts, contextuality and games.
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

:::{grid-item-card} 16 · End to end: fetch and hand over
:class-card: pillar-inspired
:link: 16_end_to_end_robot
:link-type: doc

Calibration, fusion, questions and trust-aware hand-over in one robot loop.
:::

:::{grid-item-card} 18 · Running it on a robot (ROS 2)
:class-card: pillar-inspired
:link: 18_ros2_service
:link-type: doc

The JSON service and the ROS 2 node, run without a ROS installation.
:::

:::{grid-item-card} 19 · A robot package: trust-aware hand-over
:class-card: pillar-inspired
:link: 19_trust_handover_package
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

## Learning

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 10 · RL with simulated people
:class-card: pillar-viz
:link: 10_rl_environments
:link-type: doc

Gymnasium environments and four exploration strategies.
:::

:::{grid-item-card} 11 · A variational quantum policy
:class-card: pillar-viz
:link: 11_quantum_policy
:link-type: doc

REINFORCE on CartPole against a linear policy.
:::

:::{grid-item-card} 12 · Tensor-train compression
:class-card: pillar-viz
:link: 12_tensor_train
:link-type: doc

Size, accuracy and speed, measured.
:::

:::{grid-item-card} 13 · Quanvolution
:class-card: pillar-viz
:link: 13_quanvolution
:link-type: doc

Quantum image filters against a random classical filter.
:::
::::

## Quantum computing

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 14 · Multi-robot task allocation
:class-card: pillar-quantum
:link: 14_task_allocation
:link-type: doc

QUBO, QAOA, simulated quantum annealing, QIEA and simulated annealing.
:::

:::{grid-item-card} 15 · The Bloch-sphere viewer
:class-card: pillar-quantum
:link: 15_bloch_sphere
:link-type: doc

States, circuits, entanglement and animations.
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
10_rl_environments
11_quantum_policy
12_tensor_train
13_quanvolution
14_task_allocation
15_bloch_sphere
16_end_to_end_robot
17_judgement_families
18_ros2_service
19_trust_handover_package
```
