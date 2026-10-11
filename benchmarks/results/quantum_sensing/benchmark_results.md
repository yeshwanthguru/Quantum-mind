# Quantum accelerometer dead-reckoning study

Simulation, 10 trajectories of 300 s; position error in metres, mean ± std over trajectories. Sensor values are illustrative assumptions (see `quantum_mind.sensing.sensors`).

| Sensor | Rate (Hz) | White noise per sample (m/s²) | Bias walk ((m/s²)/√s) |
|---|---|---|---|
| MEMS accelerometer | 100 | 0.00981 | 0.0002 |
| atom interferometer | 1 | 1.24e-08 | 1e-08 |

## A. Sensor suites

| Position fixes | Suite | RMSE (m) | Final error (m) | Paired p vs MEMS only |
|---|---|---|---|---|
| none | MEMS only | 339.663 ± 199.175 | 757.490 ± 433.381 | – |
| none | quantum only | 229.084 ± 172.267 | 429.453 ± 406.288 | 0.0420 |
| none | fused (MEMS + quantum) | 9.952 ± 5.423 | 17.925 ± 11.228 | 0.0025 |
| every 10 s | MEMS only | 0.173 ± 0.023 | 0.153 ± 0.099 | – |
| every 10 s | quantum only | 1.933 ± 0.184 | 3.485 ± 1.485 | 0.0025 |
| every 10 s | fused (MEMS + quantum) | 0.144 ± 0.017 | 0.135 ± 0.090 | 0.0052 |

## B. Fused suite when the quantum sensor is less stable (no fixes)

The quantum sensor's bias walk and initial bias are set to a fraction of the MEMS values (its white noise stays at the quantum limit).

| Quantum bias as a fraction of the MEMS bias | MEMS only RMSE (m) | Fused RMSE (m) | Paired p |
|---|---|---|---|
| 0.0001 | 339.663 ± 199.175 | 9.948 ± 5.432 | 0.0025 |
| 0.001 | 339.663 ± 199.175 | 9.882 ± 5.589 | 0.0025 |
| 0.01 | 339.663 ± 199.175 | 10.929 ± 5.385 | 0.0025 |
| 0.1 | 339.663 ± 199.175 | 34.150 ± 28.759 | 0.0025 |
| 1 | 339.663 ± 199.175 | 297.312 ± 180.130 | 0.5071 |

## C. Fused suite when the quantum sensor is slower (no fixes)

| Quantum cycle time (s) | MEMS only RMSE (m) | Fused RMSE (m) | Paired p |
|---|---|---|---|
| 0.5 | 339.663 ± 199.175 | 7.345 ± 7.035 | 0.0025 |
| 1 | 339.663 ± 199.175 | 9.952 ± 5.423 | 0.0025 |
| 2 | 339.663 ± 199.175 | 10.081 ± 6.522 | 0.0025 |
| 5 | 339.663 ± 199.175 | 20.243 ± 7.812 | 0.0025 |
| 10 | 339.663 ± 199.175 | 32.661 ± 17.850 | 0.0025 |
