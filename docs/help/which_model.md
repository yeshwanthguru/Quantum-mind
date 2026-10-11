# Which model do I need?

Start from your problem, follow the arrows, and open the tutorial or atlas entry at the end.

```{mermaid}
flowchart TD
    accTitle: Choosing a model
    accDescr: A decision tree from the user's problem to the matching module and tutorial in Quantum Mind.
    S{"What is your problem?"}:::op
    S -->|"people's answers or judgements<br/>look inconsistent"| J{"What kind?"}:::op
    J -->|"answers depend on question order"| OE["order effects<br/>tutorial 2"]:::out
    J -->|"inference with hidden causes"| IF["quantum-like Bayesian network<br/>atlas: quantum-like"]:::out
    S -->|"a robot works with a person"| R{"What decision?"}:::op
    R -->|"ask or act now"| AA["ask_or_act · risk-aware<br/>tutorial 4"]:::out
    R -->|"which question to ask"| QP["QuestionPlanner<br/>tutorial 7"]:::out
    R -->|"hand an object over"| HO["TrustAwareHandover<br/>tutorial 3"]:::out
    R -->|"whole loop on a robot"| E2E["end-to-end + ROS 2<br/>tutorials 12, 13"]:::out
    S -->|"perception outputs"| P{"What do you need?"}:::op
    P -->|"trustworthy confidence"| CA["calibration · conformal<br/>tutorial 5"]:::out
    P -->|"combine camera, speech, gaze"| FU["fusion<br/>tutorial 8"]:::out
    P -->|"pick between modules"| OR["orchestration gate<br/>tutorial 6"]:::out
    S -->|"a stream of sensor frames<br/>whose meaning depends on context"| B{"What do you need?"}:::op
    B -->|"belief about intent"| CB["COBRA belief filter<br/>tutorial 10"]:::out
    B -->|"hand over, wait or ask"| MC["min_cost_action<br/>tutorial 10"]:::out
    S -->|"train against simulated people"| EN["Gymnasium environments<br/>atlas: environments"]:::out
    S -->|"could a quantum computer help a robot team?"| QC["robot QUBOs · classical baselines · QAOA<br/>tutorial 15"]:::out
    S -->|"would a quantum sensor help a robot navigate?"| QS["quantum limits · sensor fusion<br/>tutorial 16"]:::out
    classDef op fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef out fill:#0f2e1a,stroke:#7ee787,color:#e6edf3
```

The same choices as a table:

| Problem | Module | Classical baseline to report | Tutorial |
|---|---|---|---|
| Answers depend on question order | `families.order_effects` | Bayes, anchoring | [2](../tutorials/02_question_order.ipynb) |
| Inference with hidden causes | `families.qlbn` | classical Bayesian network | atlas: quantum-like |
| Trust that changes when asked | `families.dynamics.OpenSystemBelief` | `MarkovBelief` | [3](../tutorials/03_trust_on_a_qubit.ipynb) |
| Act now or ask | `applications.robotics.ask_or_act` | expected-cost rule without model uncertainty | [4](../tutorials/04_ask_or_act.ipynb) |
| Over-confident detector | `applications.calibration` | raw scores | [5](../tutorials/05_calibration.ipynb) |
| Choose between perception modules | `applications.orchestration` | gate without a prior | [6](../tutorials/06_orchestration.ipynb) |
| Which question to ask | `applications.questioning` | `IndependentAnswerModel` | [7](../tutorials/07_question_planner.ipynb) |
| Fuse camera, speech and gaze | `applications.fusion` | Bayes, Dempster-Shafer | [8](../tutorials/08_fusion.ipynb) |
| Dwell times of ambiguous stimuli | `families.perception` | Markov, gamma renewal | [9](../tutorials/09_bistable_perception.ipynb) |
| Belief about intent from sensor frames whose meaning depends on context | `belief.COBRA` | `belief.GaussianBayesFilter`, incoherent limit | [10](../tutorials/10_context_belief_filter.ipynb) |
| Robot allocation, grasp or schedule problems on a quantum computer | `computing` | greedy descent, simulated annealing, random sampling with the same shots | [15](../tutorials/15_quantum_computing_for_robots.ipynb) |
| Quantum accelerometer for dead reckoning | `sensing` | MEMS accelerometer alone | [16](../tutorials/16_quantum_sensing.ipynb) |
| Train against simulated people | `envs` | any agent you bring; compare with a person-blind policy | atlas: environments |
| Whole robot loop | all of the above | | [12](../tutorials/12_end_to_end_robot.ipynb), [14](../tutorials/13_ros2_service.ipynb) |
| Assign tasks among people working with a robot | `applications.allocation` | trust-blind allocation, brute force | atlas: infrastructure |
