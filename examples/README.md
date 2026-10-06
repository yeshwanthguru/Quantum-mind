# Examples

Each script runs without installing the package (`python3 examples/<script>.py`). Most finish in
seconds; 13 and 19 take about a minute. Only the data in `qlcog.data` are human data; every simulated data set is labelled as such.

| Script | Domain | Family or pillar | What it shows |
|---|---|---|---|
| `01_survey_question_order.py` | Survey / market research | order effects | Fits to the Clinton–Gore rates; QQ test and BIC comparison on a simulated split-ballot survey |
| `02_finance_disjunction_effect.py` | Behavioural finance | interference, QLBN | Two-stage gamble data; interference versus classical mixture; the same effect as a network |
| `03_medical_diagnosis_qlbn.py` | Medical decision support | QLBN | Classical versus quantum-like inference in a synthetic diagnostic network; fitting phases |
| `04_consumer_choice_qdt.py` | Consumer choice / economics | decision | QDT versus expected utility and prospect theory on simulated choices |
| `05_llm_evaluation_order.py` | AI evaluation | order effects | Workflow for order effects in human or LLM judgements |
| `06_hci_trust_dynamics.py` | Human–computer interaction | dynamics | Does asking about trust change trust? Markov versus open-system |
| `07_evidence_accumulation.py` | Perception, confidence | dynamics | Markov versus quantum walk; interference of an intermediate judgement |
| `08_contextuality_analysis.py` | Physics, psychology | contextuality | CHSH (analytic and on Aer) and the Contextuality-by-Default criterion |
| `09_similarity_asymmetry.py` | Marketing, linguistics | similarity | Asymmetric similarity: quantum versus biased and symmetric geometric models |
| `10_circuits_quickstart.py` | – | all | Circuit versus model on Aer, FakeTorino noise and Braket |
| `11_cloud_run.py` | – | order effects | Running on IBM Quantum or Amazon Braket hardware (`--dry-run` for local) |
| `12_robot_questioning_trust.py` | Robotics / HRI | robotics application | Questioning designs, human-model ensemble with uncertainty, trust question effect |
| `13_quantum_classifiers_triage.py` | Medicine (simulated) | quantum, quantum-inspired | Variational quantum classifier, quantum kernel and MPS classifier against linear and quadratic logistic regression, 10 seeds (mean ± sd); one circuit on Aer |
| `14_qaoa_multi_robot_allocation.py` | Robotics | quantum, quantum-inspired | Task allocation solved exactly and by QAOA, SQA, QIEA and simulated annealing; QAOA sampled on Aer and FakeTorino |
| `15_portfolio_selection.py` | Finance (simulated) | quantum, quantum-inspired | Choose 3 of 8 assets: QAOA, SQA, QIEA, simulated annealing against the exact optimum |
| `16_vqe_ising_chain.py` | Physics, materials | quantum | VQE ground-state energies of a transverse-field Ising chain against exact diagonalisation |
| `17_grover_schedule_search.py` | Scheduling | quantum | Grover search for valid schedules; sampled on Aer |
| `18_qpso_controller_tuning.py` | Control, robotics | quantum-inspired | PID gains for a robot joint by QPSO against random search with the same budget |
| `19_bloch_sphere_viewer.py` | Visualisation | viewer | Circuit trajectory on Bloch spheres, interactive HTML, GIF (`--gif`), live window (`--live`), tomography on Aer and FakeTorino |
| `20_trust_on_the_bloch_sphere.py` | Robotics / HRI | quantum-like, viewer | The trust belief qubit over successes and failures; P(trust) from the sphere equals the model |
| `21_robot_ask_for_help.py` | Robotics | robotics application | Ensemble uncertainty and `ask_or_act`: when model disagreement and error costs make a robot ask before acting |
| `22_individual_differences.py` | Surveys, HRI | order effects | Pooled versus per-person model comparison on a mixed simulated population; how many answers per person are needed |

Scripts 19 and 20 write HTML (and GIF) files to `examples/output/` (ignored by git).
