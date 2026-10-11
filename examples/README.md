# Examples

Each script runs without installing the package (`python3 examples/<script>.py`). Most finish in seconds; 11 and 21 take about a minute. Only the data in `quantum_mind.data` are human data; every simulated data set is labelled as such.

| Script | What it shows |
|---|---|
| `01_survey_question_order.py` | **Question-order effects in a survey.** Survey research / market research: question-order effects. |
| `02_medical_diagnosis_qlbn.py` | **Medical diagnosis with a quantum-like Bayesian network.** Medical decision support: classical versus quantum-like inference in a Bayesian network. |
| `02_llm_evaluation_order.py` | **Order effects in human or LLM judgements.** AI evaluation: order effects in a language model's or a human rater's yes/no judgements. |
| `03_hci_trust_dynamics.py` | **Does asking about trust change trust?.** Human-computer interaction: does asking about trust change trust? |
| `04_evidence_accumulation.py` | **Evidence accumulation: Markov versus quantum walk.** Perception and judgement over time: Markov versus quantum walks (Busemeyer, Kvam and Pleskac, 2019). |
| `06_contextuality_analysis.py` | **Contextuality: CHSH and Contextuality-by-Default.** Contextuality analysis for any set of binary measurements (physics, psychology, survey items). |
| `07_similarity_asymmetry.py` | **Asymmetric similarity judgements.** Marketing, linguistics, perception: asymmetric similarity ('Korea is like China' rated higher than 'China is like Korea'). |
| `05_circuits_quickstart.py` | **Every family as a circuit.** Every family as a quantum circuit: compare circuit output with the analytic model on the ideal Aer simulator, on Aer with the FakeTorino (IBM Heron) noise model, and on the Braket local simulator. |
| `06_cloud_run.py` | **Running on IBM Quantum or Amazon Braket hardware.** Run any quantum_mind circuit on IBM Quantum or Amazon Braket hardware. |
| `07_robot_questioning_trust.py` | **A robot that asks questions and tracks trust.** Robotics: a robot that asks people questions and tracks their trust. |
| `08_bloch_sphere_viewer.py` | **Watching qubits on the Bloch sphere.** Visualisation: watch qubits move on the Bloch sphere. |
| `09_trust_on_the_bloch_sphere.py` | **Trust as a qubit.** Robotics / HRI: a person's trust in a robot as a qubit (quantum_mind OpenSystemBelief, simulated parameters). |
| `10_robot_ask_for_help.py` | **When should a robot ask for help?.** Robotics: when should a robot ask for help? |
| `11_individual_differences.py` | **Individual differences: pooled versus per-person fits.** Surveys, HRI: do people differ? |
| `15_memory_and_concepts.py` | **Memory overdistribution and concept combination.** Memory and concepts (simulated judgements): two effects classical probability cannot produce. |
| `12_intent_resolution_robot.py` | **Resolving an ambiguous command.** Robotics / assistants: resolving an ambiguous command from context cues. |
| `13_calibration_and_conformal.py` | **Calibrated confidence for a robot's detector.** Robotics, perception: a detector's scores are over-confident (simulated). |
| `14_multimodal_fusion.py` | **Fusing a detector, speech and gaze.** Robotics, perception: which object does the person mean? |
| `15_bistable_perception_zeno.py` | **Bistable perception and the quantum Zeno effect.** Perception: the quantum Zeno model predicts that the mean dwell time in one interpretation of an ambiguous figure grows as the percept is checked more often. |
| `16_trust_aware_handover.py` | **Trust-aware hand-over.** Robotics, human-robot interaction: before each hand-over the robot chooses to hand over, hand over slowly, ask about trust or wait, by expected cost under a trust model. |
| `17_context_belief_filter.py` | **A robot's belief that depends on context (COBRA).** The same hand cue after opposite looks: COBRA against a Bayes filter, online belief and decision cost (simulated) |

Scripts 11 and 12 write HTML (and GIF) files to `examples/output/` (ignored by git).
