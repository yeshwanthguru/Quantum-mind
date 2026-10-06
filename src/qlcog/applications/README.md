# Applications

## `qlcog.applications.robotics` — robots that ask people questions

| Name | Purpose |
|---|---|
| `HRI_DOMAINS`, `domain_models(domain)` | Three question pairs (object clarification, trust and hand-over, preference elicitation) with quantum-like, anchoring and Bayesian populations (illustrative parameters) |
| `estimate_unprimed_rates(design, generator, n, model, probe)` | Questioning designs: fixed order, reversed-order probe with a model, split order (model-free) |
| `TRUST_PROTOCOL`, `TRUST_MODELS` | Six hand-overs, with the trust question at the end only or also midway |
| `HumanModelEnsemble` | Competing human models weighted by BIC; predicted answer distribution plus uncertainty (entropy and model disagreement, in bits) for use as a confidence signal |
| `ask_or_act(p_success, uncertainty, ask_cost, error_cost, max_disagreement_bits)` | Decision rule: ask when the human models disagree or when the expected cost of a wrong action exceeds the cost of asking |
| `QuantumIntentResolver`, `BayesIntentResolver` (`qlcog.applications.intent`) | Intent from ambiguous commands and context cues; cues as Kraus operators rotated by an incompatibility angle, so cue order matters; every angle 0 gives exactly Bayes' rule |
| `HumanModelService` | Answers in, prediction + uncertainty + ask/act decision out (JSON-friendly dicts); wrapped by the ROS 2 node in [`integrations/ros2`](../../../integrations/ros2/README.md) |

Findings from the companion simulation study ("Order-Aware Human Models for Robot Questioning and Trust"):
splitting the question order beat model-based correction for recovering unprimed answers; the
quantum-like model was the best predictor only when people followed it and were alike; asking about
trust midway changed final trust under open-system dynamics (−0.076) and needed about 1,600 people per
condition to detect. Use the ensemble rather than a single model.
