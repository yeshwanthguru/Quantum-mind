# State-vector simulator (`quantum_mind.quantum`)

A small, exact NumPy state-vector simulator. It is infrastructure, not a model: the Bloch-sphere viewer
uses it to draw trajectories, and the circuits that belong to the quantum-like families (see
`quantum_mind.circuits`) can be simulated and checked with it before they are sent to a back end.

`Circuit` takes the gates `h x y z s sdg rx ry rz p cx cz swap rzz`, `unitary` and `diagonal`. Angles are
plain numbers, trainable weights `W(k)`, input features `X(j)` or products `XX(i, j)`; a batch of inputs is
simulated at once. `Circuit.value_and_grad` returns adjoint-method gradients (one forward and one backward
pass whatever the number of weights), and `Circuit.to_qiskit()` exports the circuit so it runs unchanged on
Aer, IBM Quantum and Amazon Braket through `quantum_mind.circuits.run`.

```python
from quantum_mind.quantum import Circuit, W

c = Circuit(2)
c.h(0).cx(0, 1).ry(W(0), 1)             # a Bell pair with a trainable rotation
state = c.state([0.4])                  # exact state vector
qc = c.to_qiskit([0.4])                 # the same circuit in Qiskit
```

**Sizes.** Up to 22 qubits are accepted. Cost grows as 2ⁿ.

**Verification.** The simulator is tested gate by gate against Qiskit's `Statevector`, and the adjoint
gradients against finite differences (`tests/test_quantum.py`, `tests/test_robustness.py`).

**What is deliberately not here.** General-purpose quantum algorithms (VQE, QAOA, Grover, quantum kernels,
variational classifiers, the quantum Fourier transform, amplitude estimation) are provided, maintained and
benchmarked far better by Qiskit and PennyLane. Version 3.0 removed the copies this library used to carry so
that it contains only what is specific to quantum cognition in robotics.
