# Examples

Each script runs without installing the package (`python3 examples/<script>.py`) and finishes in under
a minute. Only the data in `qlcog.data` are human data; every simulated data set is labelled as such.

| Script | Domain | Family | What it shows |
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
