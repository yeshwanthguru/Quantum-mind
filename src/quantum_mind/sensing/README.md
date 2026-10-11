# Quantum sensing for robots (`quantum_mind.sensing`)

The second of the three threads of quantum robotics. Quantum sensors are physical instruments, so this module does not build
one. It lets you ask, in simulation and with the sensor's own numbers, whether a slow, almost drift-free quantum sensor changes a
robot's position estimate, and when it does not.

| Object | What it does |
|---|---|
| `phase_sensitivity(n, repeats, scaling)` | Standard quantum limit `1/√(Nν)` and Heisenberg limit `1/(N√ν)` of a phase measurement |
| `atom_interferometer_sensitivity(n_atoms, T, ...)` | Single-shot acceleration sensitivity of a light-pulse atom interferometer, `Δφ / (C k_eff T²)` |
| `Sensor`, `mems_accelerometer`, `atom_interferometer_accelerometer` | Accelerometer models: sample rate, white noise, drifting bias. The atom-interferometer noise comes from the quantum limit |
| `simulate_dead_reckoning`, `run_filter`, `compare_sensor_suites` | One-axis dead reckoning with a multi-rate Kalman filter that fuses any set of sensors and optional position fixes |

```python
from quantum_mind.sensing import mems_accelerometer, atom_interferometer_accelerometer, compare_sensor_suites

mems, quantum = mems_accelerometer(), atom_interferometer_accelerometer()
rows = compare_sensor_suites([mems, quantum], {'MEMS only': [mems.name], 'fused': [mems.name, quantum.name]},
                             duration=120, seeds=range(5))
for r in rows:
    print(r['suite'], r['rmse'])          # position error in metres, mean and std over trajectories
```

**The sensor values are illustrative assumptions, not measurements of any product.** The atom-interferometer white noise is
derived from the standard quantum limit for the atom number and interrogation time you pass; its bias walk is an assumption.
Set every number for the sensors you care about.

## What the benchmark says

`benchmarks/quantum_sensing_benchmark.py` simulates 10 trajectories of 300 s. See `benchmarks/results/quantum_sensing/` for the
tables.

* **Without any outside reference** the MEMS sensor alone drifts to an RMSE of about 340 m over 300 s. The quantum sensor alone
  (229 m) is stable but too slow to follow the acceleration. **Fused**, they reach about 10 m, roughly 34 times smaller than MEMS
  alone (paired sign-flip p = 0.0025 over 10 trajectories).
* **With a position fix every 10 s** the MEMS sensor alone is already good (0.173 m) and fusing helps only a little (0.144 m, 17%
  smaller, p = 0.005); the quantum sensor alone is worse (1.93 m).
* **The benefit comes from the quantum sensor being stable.** When its bias walk and initial bias are as large as the MEMS sensor's,
  the benefit disappears (fused 297 m against 340 m, p = 0.51). At a tenth of the MEMS bias the fused error is still about 10 times
  smaller than MEMS alone (34 m), and at a hundredth or less it is about 10 m.
* **A slower quantum sensor still helps in this setting:** with a 5 s or 10 s cycle the fused error is 20 m and 33 m, against 340 m.

## What is and is not claimed

* It is a **simulation** of one axis with illustrative sensor numbers and a matched Kalman filter (an upper bound on what a
  tuned filter gets). It shows how the benefit depends on the quantum sensor's stability and speed, not what a real instrument
  delivers.
* No claim is made about building or buying a quantum sensor; that is hardware.
