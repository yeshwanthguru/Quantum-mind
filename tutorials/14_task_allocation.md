# Multi-robot task allocation: QAOA, annealing and evolution

Three robots, three tasks, a cost for every robot-task pair: who does what? Assignment problems like
this (and harder versions with constraints) can be written as a **QUBO**, a quadratic function of
binary variables, which is the input format of quantum annealers and of the **QAOA** circuit. This
tutorial solves one instance five ways, from exact enumeration to a quantum circuit, and compares
them: on a problem this small, the classical methods are faster and at least as good.

```python
import numpy as np
import matplotlib.pyplot as plt
from quantum_mind.viz import use_mpl_style
from quantum_mind.problems import task_allocation
from quantum_mind.quantum import QAOA
from quantum_mind.inspired import SQA, QIEA, simulated_annealing
theme = use_mpl_style('dark')

costs = np.array([[2.0, 7.0, 4.0],      # robot 0 -> tasks 0, 1, 2
                  [6.0, 3.0, 5.0],      # robot 1
                  [5.0, 6.0, 2.5]])     # robot 2
q = task_allocation(costs)               # x[r, t] = 1 if robot r does task t, with one-hot penalties
print('binary variables:', q.n, '   possible assignments of the bits:', 2 ** q.n)
```

## 1. The exact answer

```python
x_opt, e_opt = q.brute_force()
print('optimal energy %.2f' % e_opt)
print(x_opt.reshape(3, 3))
fig, ax = plt.subplots(figsize=(3.8, 3.4))
ax.imshow(costs, cmap='magma_r'); ax.grid(False)
for r in range(3):
    for t in range(3):
        ax.text(t, r, '%.1f' % costs[r, t], ha='center', va='center', color='black', fontweight='bold')
        if x_opt.reshape(3, 3)[r, t]:
            ax.add_patch(plt.Rectangle((t - 0.45, r - 0.45), 0.9, 0.9, fill=False, ec=theme['palette'][3], lw=3))
ax.set_xticks(range(3), ['task %d' % t for t in range(3)]); ax.set_yticks(range(3), ['robot %d' % r for r in range(3)])
ax.set_title('Costs and the optimal assignment'); plt.show()
```

## 2. QAOA on the built-in simulator

QAOA alternates a cost layer (phases from the QUBO) and a mixer layer (X rotations) `p` times, then
tunes the angles classically. Success is measured by the probability of sampling the optimum.

```python
qa = QAOA(q, p=3, restarts=3).run()
print('QAOA p=3: expected energy %.2f, most likely sample %.2f, P(optimal) = %.3f (uniform guessing %.4f)'
      % (qa.expectation, qa.energy, qa.p_optimal, 1 / 2 ** q.n))
energies = q.all_energies()
order = np.argsort(energies)
fig, ax = plt.subplots(figsize=(7.0, 2.8))
ax.bar(range(40), qa.probabilities[order][:40], color=theme['palette'][1])
ax.axhline(1 / 2 ** q.n, color=theme['text'], ls='--', label='uniform')
ax.set_xlabel('bit strings, lowest energy first (40 best of 512)'); ax.set_ylabel('probability')
ax.set_title('QAOA concentrates probability on low-energy assignments'); ax.legend()
plt.show()
```

## 3. Classical and quantum-inspired heuristics

```python
results = {'exact': e_opt, 'QAOA (best sample)': q.energy(qa.x), 'SQA': SQA(q).run().value,
           'QIEA': QIEA(q).run().value, 'simulated annealing': simulated_annealing(q).value}
for k, v in results.items():
    print('%-20s energy %.2f %s' % (k, v, '(optimal)' if np.isclose(v, e_opt) else ''))
```

## 4. Run it on real quantum hardware (optional)

The same circuit exports to Qiskit; `quantum_mind.circuits.run` sends it to Aer, a fake IBM device
with realistic noise, IBM Quantum or Amazon Braket (credentials needed for the last two).

```python
try:
    from quantum_mind.circuits import run
    counts = run(QAOA(q, p=3).to_qiskit(qa), 'aer', 4000)
    good = sum(k for b, k in counts.items() if np.isclose(q.energy([int(c) for c in b.replace(' ', '')[::-1]]), e_opt)) / 4000
    print('Aer simulator: sampled P(optimal) = %.3f' % good)
except ImportError:
    print('install "quantum-mind[qiskit]" to sample the circuit with Aer')
```

The statements below are checked by this cell every time the documentation is built:

```python
assert all(np.isclose(v, e_opt) for v in results.values())
print('checked')
```

**Result.** All methods find the optimum of this 9-variable instance, and enumeration takes
milliseconds. The value of the QUBO formulation is that the same problem runs unchanged on annealers,
gate-model hardware and classical heuristics, so they can be compared as problems grow.

References: Farhi, Goldstone & Gutmann (2014), arXiv:1411.4028; Lucas (2014), *Frontiers in Physics* 2,
5 (Ising formulations); Gerkey & Matarić (2004), *International Journal of Robotics Research* 23,
939-954 (multi-robot task allocation).
