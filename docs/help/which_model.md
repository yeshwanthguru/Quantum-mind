# Which model do I need?

Start from your problem, follow the arrows, and open the tutorial or atlas entry at the end.

```{mermaid}
flowchart TD
    accTitle: Choosing a model
    accDescr: A decision tree from the user's problem to the matching module and tutorial in Quantum Mind.
    S{"What is your problem?"}:::op
    S -->|"people's answers or judgements<br/>look inconsistent"| J{"What kind?"}:::op
    J -->|"answers depend on question order"| OE["order effects<br/>tutorial 2"]:::out
    J -->|"P(A and B) > P(B)"| CJ["conjunction<br/>tutorial 17"]:::out
    J -->|"choice when an outcome is unknown"| IF["interference · QLBN<br/>tutorial 17"]:::out
    J -->|"risky choices"| DC["decision (QDT)<br/>tutorial 17"]:::out
    J -->|"similarity, memory, concepts"| SM["similarity · memory · concepts<br/>tutorial 17"]:::out
    S -->|"a robot works with a person"| R{"What decision?"}:::op
    R -->|"ask or act now"| AA["ask_or_act · risk-aware<br/>tutorial 4"]:::out
    R -->|"which question to ask"| QP["QuestionPlanner<br/>tutorial 7"]:::out
    R -->|"hand an object over"| HO["TrustAwareHandover<br/>tutorial 3"]:::out
    R -->|"whole loop on a robot"| E2E["end-to-end + ROS 2<br/>tutorials 16, 18"]:::out
    S -->|"perception outputs"| P{"What do you need?"}:::op
    P -->|"trustworthy confidence"| CA["calibration · conformal<br/>tutorial 5"]:::out
    P -->|"combine camera, speech, gaze"| FU["fusion<br/>tutorial 8"]:::out
    P -->|"pick between modules"| OR["orchestration gate<br/>tutorial 6"]:::out
    S -->|"learning a policy"| L{"Setting?"}:::op
    L -->|"simulated people"| EN["Gymnasium environments<br/>tutorial 10"]:::out
    L -->|"quantum policy"| VP["VariationalPolicy<br/>tutorial 11"]:::out
    L -->|"smaller networks"| TT["tensor-train layers<br/>tutorial 12"]:::out
    S -->|"optimisation or QML"| Q{"Which?"}:::op
    Q -->|"assignment, scheduling, Max-Cut"| QA["QUBO + QAOA / SQA / QIEA<br/>tutorial 14"]:::out
    Q -->|"classification, kernels"| QM["VariationalClassifier · QuantumKernel<br/>atlas: quantum"]:::out
    classDef op fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef out fill:#0f2e1a,stroke:#7ee787,color:#e6edf3
```

The same choices as a table:

| Problem | Module | Classical baseline to report | Tutorial |
|---|---|---|---|
| Answers depend on question order | `families.order_effects` | Bayes, anchoring | [2](../tutorials/02_question_order.ipynb) |
| Conjunction fallacy, disjunction effect, risky choice, similarity, memory, concepts | `families.*` | listed per family | [17](../tutorials/17_judgement_families.ipynb) |
| Trust that changes when asked | `families.dynamics.OpenSystemBelief` | `MarkovBelief` | [3](../tutorials/03_trust_on_a_qubit.ipynb) |
| Act now or ask | `applications.robotics.ask_or_act` | expected-cost rule without model uncertainty | [4](../tutorials/04_ask_or_act.ipynb) |
| Over-confident detector | `applications.calibration` | raw scores | [5](../tutorials/05_calibration.ipynb) |
| Choose between perception modules | `applications.orchestration` | gate without a prior | [6](../tutorials/06_orchestration.ipynb) |
| Which question to ask | `applications.questioning` | `IndependentAnswerModel` | [7](../tutorials/07_question_planner.ipynb) |
| Fuse camera, speech and gaze | `applications.fusion` | Bayes, Dempster-Shafer | [8](../tutorials/08_fusion.ipynb) |
| Dwell times of ambiguous stimuli | `families.perception` | Markov, gamma renewal | [9](../tutorials/09_bistable_perception.ipynb) |
| Train against simulated people | `envs` | ε-greedy, softmax, UCB | [10](../tutorials/10_rl_environments.ipynb) |
| Whole robot loop | all of the above | | [16](../tutorials/16_end_to_end_robot.ipynb), [18](../tutorials/18_ros2_service.ipynb) |
| Assignment and scheduling | `problems` + `quantum.QAOA`, `inspired.SQA` | simulated annealing, brute force | [14](../tutorials/14_task_allocation.ipynb) |
