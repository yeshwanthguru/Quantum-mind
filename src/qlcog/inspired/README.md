# Quantum-inspired models (`qlcog.inspired`)

Classical algorithms that borrow an idea from quantum mechanics. They run on ordinary CPUs, need no
quantum SDK and make no use of a quantum computer.

| Model | Quantum idea | Problem type | Reference |
|---|---|---|---|
| `QIEA` | Each bit is a "Q-bit" with an amplitude angle; observation samples bit strings; rotation gates steer the population | Binary optimisation (any `Qubo` or `f(x)`, x ∈ {0,1}ⁿ) | Han and Kim, *IEEE TEVC* 6(6), 580–593 (2002) |
| `QPSO` | Particles in a delta potential well, positions sampled from its wave function (no velocities) | Continuous optimisation on a box | Sun, Feng and Xu, *CEC* (2004) |
| `SQA` | Transverse-field Ising model mapped to coupled classical replicas (Suzuki–Trotter); tunnelling through thin barriers | Ising / QUBO problems | Martoňák, Santoro and Tosatti, *PRB* 66, 094203 (2002) |
| `MPSClassifier` | Weight tensor in an exponentially large tensor-product space, stored as a matrix product state | Supervised classification | Stoudenmire and Schwab, *NeurIPS* (2016) |
| `simulated_annealing` | (classical baseline) | Ising / QUBO | Kirkpatrick et al., *Science* 220 (1983) |

```python
from qlcog.problems import maxcut, knapsack, portfolio
from qlcog.inspired import QIEA, SQA, QPSO, MPSClassifier, simulated_annealing

q = maxcut(edges, n)
for solver in (SQA(q), QIEA(q)):
    r = solver.run()                       # r.x, r.value, r.history, r.evaluations
print(simulated_annealing(q).value, q.brute_force()[1])

best = QPSO(cost, lower=[0, 0, 0], upper=[50, 20, 5]).run()          # e.g. PID gains
clf = MPSClassifier(bond=6, local_dim=3).fit(X_train, y_train)      # local_dim > 2: polynomial features
```

**Shared problem layer.** QIEA, SQA and simulated annealing accept the same `qlcog.problems.Qubo`
objects as QAOA, so one instance can be solved by quantum, quantum-inspired and classical methods and
compared with the exact optimum (`brute_force()` for up to about 22 variables).

**Honest expectations.** "Quantum-inspired" names the origin of the idea, not a speed-up. Compare
against the classical baseline on the problem at hand (the examples always print it).
