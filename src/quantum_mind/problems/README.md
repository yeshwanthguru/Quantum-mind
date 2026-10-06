# Optimisation problems (`quantum_mind.problems`)

One problem format, three families of solvers. Every builder returns a `Qubo`
(minimise xᵀQx + offset, x ∈ {0,1}ⁿ) that QAOA (`quantum_mind.quantum`), simulated quantum annealing and the
quantum-inspired evolutionary algorithm (`quantum_mind.inspired`), and classical simulated annealing all
accept.

| Builder | Domain | Encoding |
|---|---|---|
| `maxcut(edges, n, weights)` | Networks, clustering, circuit layout | energy = −(cut weight) |
| `knapsack(values, weights, capacity)` | Resource selection, payload planning | value + penalty with binary slack variables |
| `task_allocation(costs)` | Multi-robot task allocation, scheduling | cost + one-hot penalty per task; variable `a * n_tasks + t` |
| `portfolio(mu, cov, budget, risk)` | Finance, sensor selection | risk · xᵀΣx − μᵀx + cardinality penalty |
| `from_ising(h, J, c)` | Physics, any Ising model | exact change of variables s = 1 − 2x |

`Qubo` methods: `energy(x)` (one or many bit strings), `all_energies()` (Qiskit bit order),
`brute_force()` (exact optimum, n ≤ 22), `to_ising()`, `normalised()`.

Penalty weights default to values large enough that every optimum satisfies the constraints; pass
`penalty=` to tune them.
