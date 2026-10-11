# Robot problems on quantum and classical solvers

10 random instances per problem; QAOA through Qiskit `QAOAAnsatz`, exact simulation, 3 restarts, best of 1000 samples (as is random sampling). "Optimal" = the returned solution reaches the exact optimum of the robot's own objective.

| Problem | Solver | Optimal | Feasible | Mean gap | QAOA P(optimal) | QAOA approximation ratio | Seconds / instance |
|---|---|---|---|---|---|---|---|
| allocation (3 robots x 3 tasks) | greedy descent | 10/10 | 10/10 | 0.000 | – | – | 0.00 |
| allocation (3 robots x 3 tasks) | simulated annealing | 5/10 | 10/10 | 0.677 | – | – | 0.25 |
| allocation (3 robots x 3 tasks) | random sampling (1000 shots) | 7/10 | 10/10 | 0.613 | – | – | 0.00 |
| allocation (3 robots x 3 tasks) | QAOA p=1 | 5/10 | 10/10 | 0.351 | 0.001 | 0.823 | 1.19 |
| allocation (3 robots x 3 tasks) | QAOA p=2 | 6/10 | 10/10 | 0.394 | 0.014 | 0.844 | 3.12 |
| allocation (3 robots x 3 tasks) | QAOA p=3 | 7/10 | 10/10 | 0.061 | 0.005 | 0.839 | 4.33 |
| grasp selection (3 of 10) | greedy descent | 9/10 | 10/10 | 0.000 | – | – | 0.00 |
| grasp selection (3 of 10) | simulated annealing | 1/10 | 10/10 | 0.425 | – | – | 0.29 |
| grasp selection (3 of 10) | random sampling (1000 shots) | 7/10 | 10/10 | 0.028 | – | – | 0.00 |
| grasp selection (3 of 10) | QAOA p=1 | 9/10 | 10/10 | 0.014 | 0.019 | 0.957 | 1.42 |
| grasp selection (3 of 10) | QAOA p=2 | 10/10 | 10/10 | 0.000 | 0.014 | 0.952 | 2.61 |
| grasp selection (3 of 10) | QAOA p=3 | 9/10 | 10/10 | 0.001 | 0.010 | 0.963 | 4.91 |
| slot scheduling (4 robots, 3 slots) | greedy descent | 10/10 | 10/10 | 0.000 | – | – | 0.00 |
| slot scheduling (4 robots, 3 slots) | simulated annealing | 10/10 | 10/10 | 0.000 | – | – | 0.35 |
| slot scheduling (4 robots, 3 slots) | random sampling (1000 shots) | 10/10 | 10/10 | 0.000 | – | – | 0.00 |
| slot scheduling (4 robots, 3 slots) | QAOA p=1 | 10/10 | 10/10 | 0.000 | 0.036 | 0.864 | 2.61 |
| slot scheduling (4 robots, 3 slots) | QAOA p=2 | 10/10 | 10/10 | 0.000 | 0.098 | 0.907 | 6.27 |
| slot scheduling (4 robots, 3 slots) | QAOA p=3 | 10/10 | 10/10 | 0.000 | 0.143 | 0.917 | 8.91 |
