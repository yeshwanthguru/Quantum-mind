# State slot of the one-motor VLA: quantum accelerometer under encoder dropout

Simulation. 5 seeds x 200 unambiguous scenes. Encoder drops out at 1 s of a 3 s move. Success = final position within a quarter cell. p: paired sign-flip test on final error against "Kalman + MEMS + quantum".

## Unmodelled push up to 0 units/s^2

| Estimator | Success | Mean final error | p vs quantum fusion |
|---|---|---|---|
| encoder | 0.283 | 0.199 | 0.0002 |
| kalman, encoder only | 0.020 | 1.125 | 0.0002 |
| kalman + motor model | 0.959 | 0.025 | 0.0002 |
| kalman + MEMS | 0.128 | 0.241 | 0.0002 |
| kalman + MEMS + quantum | 0.432 | 0.079 | – |

## Unmodelled push up to 2 units/s^2

| Estimator | Success | Mean final error | p vs quantum fusion |
|---|---|---|---|
| encoder | 0.265 | 0.206 | 0.0002 |
| kalman, encoder only | 0.020 | 1.121 | 0.0002 |
| kalman + motor model | 0.135 | 0.423 | 0.0002 |
| kalman + MEMS | 0.139 | 0.238 | 0.0002 |
| kalman + MEMS + quantum | 0.426 | 0.086 | – |

## Unmodelled push up to 4 units/s^2

| Estimator | Success | Mean final error | p vs quantum fusion |
|---|---|---|---|
| encoder | 0.194 | 0.216 | 0.0002 |
| kalman, encoder only | 0.019 | 1.095 | 0.0002 |
| kalman + motor model | 0.067 | 0.711 | 0.0002 |
| kalman + MEMS | 0.125 | 0.239 | 0.0002 |
| kalman + MEMS + quantum | 0.287 | 0.119 | – |

The model-based Kalman filter uses the known command and the nominal plant; it is excellent when the plant is as modelled and fails when something unmodelled pushes the joint, which is where an accelerometer can help. The quantum accelerometer's edge over the MEMS one comes from its near-zero bias; with the encoder present, the encoder calibrates the MEMS bias and the gap is small.
