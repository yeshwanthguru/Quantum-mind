# Quantum computing for robot algorithms (`quantum_mind.computing`)

The first of the three threads of quantum robotics. This module does **not** re-implement quantum algorithms (Qiskit and
PennyLane do that better). It supplies the robot side of the question "can a quantum computer help a robot team?":
realistic problems with their true objectives, the classical baselines a quantum result must be reported against, and a thin
adapter to Qiskit's QAOA.

| Object | What it does |
|---|---|
| `multi_robot_allocation(costs, balance)` | Assign every task to one robot at the lowest total cost, optionally balancing the load |
| `grasp_selection(quality, conflict, k)` | Choose exactly `k` high-quality grasps that do not conflict |
| `slot_scheduling(conflicts, n_slots)` | Give robots that share a workspace time slots so conflicting ones never move together |
| `exhaustive`, `greedy_descent`, `simulated_annealing`, `random_sampling` | Classical baselines; random sampling with the same shots is the null baseline for a sampling solver |
| `qaoa(qubo, depth)` | QAOA through Qiskit's `QAOAAnsatz`, simulated exactly (at most 16 qubits) |
| `compare_solvers(problem)` | Run several solvers on one problem and report them in the robot's terms (true objective, gap to the optimum) |

Each problem is a `RobotProblem` with a `Qubo` encoding (the penalty formulations of Lucas, 2014), the **true objective**
(without penalty terms), a feasibility check and a decoder, so a solver's answer is judged on the robot's terms and not only on
the QUBO energy.

```python
import numpy as np
from quantum_mind.computing import multi_robot_allocation, compare_solvers

problem = multi_robot_allocation(np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0], [3.0, 3.0, 1.0]]))
for row in compare_solvers(problem):
    print(row['solver'], row['feasible'], row['gap'])
```

## What the benchmark says

`benchmarks/robot_quantum_computing_benchmark.py` solves 10 random instances of each problem (9 to 12 qubits) with every
solver. See `benchmarks/results/robot_quantum_computing/` for the table; the reading is below.

* **Greedy descent with 20 restarts solves 29 of 30 instances** to the exact optimum and takes milliseconds.
* **Random sampling with the same 1000 shots solves 24 of 30.** On a search space of 512 to 4096 bit strings, drawing 1000 samples
  is already a strong baseline. Simulated annealing (single flips across a penalty barrier) solves only 16 of 30.
* **QAOA solves 24, 26 and 26 of 30 at depths 1, 2 and 3**, which is within noise of random sampling and below greedy descent. Its
  approximation ratio is 0.82 to 0.96, but the probability it puts on the optimum is small (1% to 14%).
* **QAOA does improve with depth on slot scheduling** (probability on the optimum 0.036, 0.098, 0.143 at depths 1 to 3), so the
  circuit is doing something; it just does not beat the classical baselines on problems this small.
* All of this is an exact simulation of a noiseless device on problems that can be enumerated. It says nothing about larger
  problems or about hardware.

## What is and is not claimed

* The quantum solver is Qiskit's QAOA run on an exact simulator, **not on hardware**. Problems of this size are exactly
  solvable, so the study measures solution quality, not speed, and says nothing about problems too large to enumerate.
* A sampling solver that returns the best of many samples has to be compared with random sampling using the same shots, which
  is why `random_sampling` is a baseline.
* No quantum advantage is claimed.
