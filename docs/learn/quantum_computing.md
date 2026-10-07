# Quantum computing

A quantum computer stores information in **qubits**. A qubit's state is a unit vector in a
two-dimensional complex space, $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$, and measuring it
gives 0 with probability $|\alpha|^2$ and 1 with probability $|\beta|^2$ (the **Born rule**).
Computation is a sequence of **gates** (unitary matrices) applied to qubits, followed by measurement.

```{mermaid}
flowchart LR
    I["|0…0⟩"]:::in --> E["encode data<br/>RY(x) rotations"]:::op --> V["variational layers<br/>RY(θ) · RZ(θ) · CNOT"]:::op --> MZ["measure<br/>Born-rule probabilities"]:::out
    MZ --> CL["classical optimiser<br/>updates θ"]:::base
    CL -.-> V
    classDef in fill:#0b2a4a,stroke:#79c0ff,color:#e6edf3
    classDef op fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef out fill:#0f2e1a,stroke:#7ee787,color:#e6edf3
    classDef base fill:#262626,stroke:#8b949e,color:#e6edf3,stroke-dasharray:4 3
```

## From the basics

Qubit and Bloch sphere
: Any pure qubit state can be drawn as a point on a sphere: $|0\rangle$ at the top, $|1\rangle$ at the
  bottom, superpositions in between. Mixed (noisy) states lie inside. The library's viewer animates
  these points gate by gate.

Superposition and interference
: Amplitudes, not probabilities, add. Two paths to the same outcome can reinforce or cancel, which is
  what quantum algorithms exploit and what quantum-like cognitive models use to describe violations of
  the law of total probability.

Entanglement
: A state of two qubits that is not a product of single-qubit states, such as
  $(|00\rangle + |11\rangle)/\sqrt{2}$. Measurement outcomes are correlated more strongly than any
  classical local model allows (the CHSH inequality).

Gates and circuits
: Single-qubit rotations (RX, RY, RZ), Hadamard (H) and two-qubit gates (CNOT, CZ). A circuit is a
  list of gates; Qiskit builds and runs them on simulators and on IBM Quantum hardware.

Variational algorithms
: Hybrid loops in which a classical optimiser tunes circuit angles: VQE (ground-state energy), QAOA
  (combinatorial optimisation), variational classifiers and policies. They suit today's noisy,
  intermediate-scale quantum (**NISQ**) devices.

Noise and hardware
: Real qubits decohere and gates are imperfect. Results from hardware must be compared with a
  simulator and with a classical method; small problems run faster classically.

## A small example: a Bell state

```python
from quantum_mind.quantum import Circuit
bell = Circuit(2).h(0).cx(0, 1)
print(bell.probabilities().round(3))    # [[0.5 0. 0. 0.5]]: only 00 and 11
```

## Where Quantum Mind fits

The **quantum** pillar runs machine learning and algorithms as circuits (classifiers, kernels, QAOA,
VQE, Grover, QFT, amplitude estimation, quantum walks, policies, quanvolution) on a built-in NumPy
simulator, with export to Qiskit for Aer, IBM Quantum or Amazon Braket. The quantum-like models of
people also compile to circuits.

- [Tutorial: introduction](../tutorials/01_introduction.ipynb)
- [Tutorial: robot task allocation with QAOA and annealing](../tutorials/14_task_allocation.ipynb)
- [Tutorial: Bloch-sphere visualisation](../tutorials/15_bloch_sphere.ipynb)
- [Model atlas: quantum](../atlas/quantum.md)

## References

- Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information* (10th
  anniversary ed.). Cambridge University Press.
- Preskill, J. (2018). Quantum computing in the NISQ era and beyond. *Quantum*, 2, 79.
- Schuld, M., & Petruccione, F. (2021). *Machine Learning with Quantum Computers* (2nd ed.). Springer.
- Farhi, E., Goldstone, J., & Gutmann, S. (2014). A quantum approximate optimization algorithm.
  arXiv:1411.4028.
- Peruzzo, A., et al. (2014). A variational eigenvalue solver on a photonic quantum processor. *Nature
  Communications*, 5, 4213.
- Havlíček, V., et al. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*, 567,
  209-212.
