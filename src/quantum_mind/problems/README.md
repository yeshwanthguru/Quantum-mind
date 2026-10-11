# QUBO container (`quantum_mind.problems`)

`Qubo` is the binary optimisation problem the trust-aware allocation layer produces: minimise
xᵀQx + offset over x ∈ {0, 1}ⁿ.

`Qubo` methods: `energy(x)` (one or many bit strings), `all_energies()` (Qiskit bit order),
`brute_force()` (exact optimum, n ≤ 22), `to_ising()` (so any external solver can take the instance) and
`normalised()`.

`quantum_mind.applications.allocation.trust_aware_allocation` builds a `Qubo` in which a task that needs the
robot's help costs more with a person who does not trust the robot, and heavy workloads are penalised.
General-purpose problem builders (max-cut, knapsack, portfolio) and solvers were removed in version 3.0;
dedicated optimisation libraries do that better.
