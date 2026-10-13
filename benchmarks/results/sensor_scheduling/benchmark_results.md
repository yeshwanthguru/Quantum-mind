# Scheduling a quantum accelerometer with QUBO solvers

Simulation. Objective: weighted RMS position standard deviation over a 120 s mission (m), from the exact Kalman covariance; lower is better. Sensor values are the library's illustrative models.

## A. 14 slots, choose 4 (the exact optimum is known)

Regret is the mean relative excess over the exact optimum found by enumerating every schedule.

| Method | Mean objective (m) | Mean regret vs exact optimum | Missions where it matches or beats it | p vs uniform spacing |
|---|---|---|---|---|
| uniform spacing | 5.493 | +24.1% | 0 / 20 | – |
| random (mean of 5) | 8.299 | +89.8% | 0 / 20 | 0.0002 |
| greedy on the exact objective | 4.580 | +0.0% | 12 / 20 | 0.0002 |
| swap local search from uniform | 4.579 | +0.0% | 20 / 20 | 0.0002 |
| exact optimum | 4.579 | +0.0% | 20 / 20 | 0.0002 |
| QUBO from the empty schedule, 1 round | 5.845 | +26.9% | 0 / 20 | 0.1312 |
| QUBO from the empty schedule, iterated | 4.581 | +0.0% | 13 / 20 | 0.0002 |
| local QUBO + exact solver | 4.592 | +0.2% | 18 / 20 | 0.0002 |
| local QUBO + annealing | 4.806 | +5.4% | 1 / 20 | 0.0002 |
| local QUBO + greedy descent | 4.592 | +0.2% | 15 / 20 | 0.0002 |
| local QUBO + random sampling (1000 shots) | 4.596 | +0.5% | 6 / 20 | 0.0002 |
| local QUBO + QAOA (depth 2) | 4.619 | +0.8% | 5 / 20 | 0.0002 |

## B. 24 slots, choose 6 (too big to enumerate)

Regret is relative to greedy selection on the exact objective, the best reference available at this size.

| Method | Mean objective (m) | Mean regret vs greedy on the exact objective | Missions where it matches or beats it | p vs uniform spacing |
|---|---|---|---|---|
| uniform spacing | 4.044 | +19.5% | 0 / 10 | – |
| random (mean of 5) | 5.554 | +65.5% | 0 / 10 | 0.0025 |
| greedy on the exact objective | 3.410 | +0.0% | 10 / 10 | 0.0025 |
| swap local search from uniform | 3.408 | -0.1% | 10 / 10 | 0.0025 |
| QUBO from the empty schedule, 1 round | 6.139 | +80.0% | 0 / 10 | 0.0025 |
| QUBO from the empty schedule, iterated | 3.562 | +4.7% | 0 / 10 | 0.0025 |
| local QUBO + annealing | 3.595 | +5.9% | 0 / 10 | 0.0025 |
| local QUBO + greedy descent | 3.415 | +0.1% | 3 / 10 | 0.0025 |
| local QUBO + random sampling (1000 shots) | 3.522 | +3.0% | 0 / 10 | 0.0025 |

The local QUBO is a second-order model of the true objective, exact within two flips of its expansion point, so a solver
that finds its exact optimum can still be beaten by a method that uses the exact objective. Solvers of the QUBO, quantum or
classical, only trade accuracy of the QUBO solution for time.
