# Quantum-inspired algorithms

Quantum-inspired algorithms are **classical** algorithms that borrow ideas from quantum mechanics:
probability amplitudes, superposition of candidate solutions, tunnelling through energy barriers, or
the tensor networks used to describe many-body quantum states. They run on ordinary CPUs today and
must be judged against strong classical heuristics, which they do not always beat.

```{mermaid}
flowchart TB
    accTitle: Quantum-inspired algorithms, diagram 1
    accDescr: Five quantum ideas and the classical algorithms they inspire (QIEA and amplitude exploration, simulated quantum annealing, QPSO, tensor networks, the quantum language model), each compared with a classical baseline.
    Q["quantum idea"]:::hum --> A1["amplitudes and the Born rule"]:::op --> R1["QIEA · amplitude exploration · QIRL"]:::out
    Q --> A2["tunnelling between solutions"]:::op --> R2["simulated quantum annealing"]:::out
    Q --> A3["delta-potential-well sampling"]:::op --> R3["QPSO"]:::out
    Q --> A4["tensor networks (MPS / MPO)"]:::op --> R4["MPS classifier · tensor-train layers"]:::out
    Q --> A5["density matrices"]:::op --> R5["quantum language model"]:::out
    R1 & R2 & R3 & R4 & R5 --> CB["classical baseline<br/>SA · GA · PSO · ε-greedy · dense layers · BM25"]:::base
    classDef op fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef out fill:#0f2e1a,stroke:#7ee787,color:#e6edf3
    classDef hum fill:#3d1414,stroke:#ff7b72,color:#e6edf3
    classDef base fill:#262626,stroke:#8b949e,color:#e6edf3,stroke-dasharray:4 3
```

## From the basics

Quantum-inspired evolutionary algorithm (QIEA)
: Each bit of a candidate solution is a "Q-bit" with an amplitude; solutions are sampled from the
  amplitudes and the amplitudes are rotated towards the best solution found.

Quantum-behaved particle swarm (QPSO)
: Particles move by sampling from a distribution inspired by a particle in a potential well, rather
  than with velocities, which needs fewer parameters than PSO.

Simulated quantum annealing (SQA)
: Path-integral Monte Carlo on several coupled copies of the problem; the transverse-field term lets
  the search pass through barriers that trap simulated annealing.

Tensor networks
: A matrix product state (MPS) or tensor train factorises a huge tensor into a chain of small ones.
  Used here for classification and for compressing neural-network layers.

## A small example: QIEA against simulated annealing on Max-Cut

```python
import itertools
import numpy as np
from quantum_mind.problems import maxcut
from quantum_mind.inspired import QIEA, simulated_annealing
rng = np.random.default_rng(0)
edges = [e for e in itertools.combinations(range(12), 2) if rng.random() < 0.4]
q = maxcut(edges, n=12)
qiea, sa = QIEA(q, seed=0).run().value, simulated_annealing(q, seed=0).value
print(qiea, sa)                  # both minimise minus the cut
assert sa <= qiea                # on this graph simulated annealing finds a cut at least as large
```

## Where Quantum Mind fits

The **quantum-inspired** pillar holds QIEA, QPSO, SQA, the MPS classifier, quantum-inspired
reinforcement learning, amplitude exploration, tensor-train layers and a density-matrix language
model, each with the classical method it has to beat.

- [Tutorial: robot task allocation](../tutorials/14_task_allocation.ipynb)
- [Tutorial: tensor-train compression](../tutorials/12_tensor_train.ipynb)
- [Model atlas: quantum-inspired](../atlas/inspired.md)

## References

- Han, K.-H., & Kim, J.-H. (2002). Quantum-inspired evolutionary algorithm for a class of combinatorial
  optimization. *IEEE Transactions on Evolutionary Computation*, 6(6), 580-593.
- Sun, J., Feng, B., & Xu, W. (2004). Particle swarm optimization with particles having quantum
  behavior. *Proceedings of the IEEE Congress on Evolutionary Computation*, 325-331.
- Martoňák, R., Santoro, G. E., & Tosatti, E. (2002). Quantum annealing by the path-integral Monte
  Carlo method: The two-dimensional random Ising model. *Physical Review B*, 66, 094203.
- Stoudenmire, E. M., & Schwab, D. J. (2016). Supervised learning with tensor networks. *Advances in
  Neural Information Processing Systems*, 29.
- Oseledets, I. V. (2011). Tensor-train decomposition. *SIAM Journal on Scientific Computing*, 33(5),
  2295-2317.
- Sordoni, A., Nie, J.-Y., & Bengio, Y. (2013). Modeling term dependencies with quantum language models
  for IR. *Proceedings of SIGIR*, 653-662.
