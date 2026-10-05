# Circuits and backends

Every builder returns `(circuit, decode)`; `decode(counts)` returns the same quantity the analytic
model predicts, so circuit output and model can be compared directly (`qlcog.core.tvd`).

| Builder | Family | Qubits | Forms |
|---|---|---|---|
| `projective_sequence_circuit(psi, projectors, order, form)` | any projective model, any dimension | ⌈log₂ d⌉ + 1 (dynamic) or + number of questions (deferred) | dynamic, deferred |
| `order_effects_circuit(model, order, form)` | order effects | 2 + ancillas | dynamic, deferred |
| `conjunction_circuit(model, form)` | conjunction | 2 + ancillas | dynamic, deferred |
| `similarity_circuit(model, a, b, form)` | similarity | 2 + ancillas | dynamic, deferred |
| `interference_circuit(p1, p2, c, theta, condition)` | interference | 2 | post-selection |
| `qlbn_circuit(net, query, evidence, phases)` | QLBN | query + hidden registers | post-selection |
| `walk_circuit(model, t)` | quantum walk | ⌈log₂ N⌉ | single time |
| `belief_circuit(model, events, queries, form)` | open-system belief | 1 + ancillas | dynamic, deferred |
| `chsh_circuit(a, b)` | contextuality | 2 | – |

**Binary projective measurement.** A unitary maps the "yes" subspace onto the first r computational
basis states, an ancilla is flipped for each of them (multi-controlled X), only the ancilla is
measured, and the unitary is undone. The system itself is never measured, so the post-measurement
state follows the Lüders rule exactly.

**Forms.** *Dynamic* circuits use mid-circuit measurement and reset (IBM hardware, Aer). *Deferred*
circuits copy each measurement onto a fresh ancilla and read all ancillas at the end; they run on
hardware without mid-circuit measurement and are required for every Braket backend.

## run(circuit, backend, shots)

| backend | What it does | Needs |
|---|---|---|
| `'aer'` | Qiskit Aer, ideal | qiskit-aer |
| `'aer:FakeTorino'` (any fake backend) | Aer with the noise model of an IBM device | qiskit-ibm-runtime |
| `'braket_local'` | Braket state-vector simulator (OpenQASM 3) | amazon-braket-sdk |
| `'braket_dm:0.005'` | Braket density-matrix simulator, depolarizing noise after every gate | amazon-braket-sdk |
| `'ibm:<device>'`, `'ibm:least_busy'` | IBM Quantum hardware via Qiskit Runtime SamplerV2 | saved IBM account |
| `'braket:<device ARN>'` | Amazon Braket managed simulator or QPU | AWS credentials; billed |

Counts are returned in Qiskit's bit-string convention for every backend.

**IBM Quantum setup:**
```
python3 -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; S.save_account(channel='ibm_quantum_platform', token='<API key>', instance='<instance>')"
```
**Amazon Braket setup:** enable Braket in the AWS console and run `aws configure`. QPU tasks are billed
per task and per shot; check current prices first.

Simulated noise levels observed in the examples (8,000 shots): order-effects circuit, TVD 0.003
(ideal) and about 0.07 (FakeTorino); quantum walk on 4 qubits, about 0.19 (FakeTorino), because the
general unitary compiles to many two-qubit gates. Treat hardware runs as tests of the representation,
not as a way to speed up the models, which run in microseconds on a CPU.
