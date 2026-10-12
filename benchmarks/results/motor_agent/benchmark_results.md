# Monitoring and self-healing agent for the motor

Simulation. 5 seeds x 80 episodes per cell. One fault is injected at 1.5 s of a 4 s two-leg move; success = final error under a quarter cell. p: paired sign-flip test on final error against the full agent.

## Fault: none

| Policy | Success | Mean final error | Safe stops | p vs agent |
|---|---|---|---|---|
| raw encoder | 1.000 | 0.011 | 0.000 | 0.0535 |
| Kalman (no agent) | 1.000 | 0.009 | 0.000 | 0.0005 |
| Kalman + motor model (no agent) | 1.000 | 0.009 | 0.000 | 0.0005 |
| agent, monitor only | 1.000 | 0.009 | 0.000 | 0.0005 |
| agent, no quantum accelerometer | 0.950 | 0.014 | 0.000 | 0.0005 |
| agent (monitor + self-heal) | 0.963 | 0.012 | 0.000 | – |

False alarms (any event in a healthy episode): 0.043 of episodes.

## Fault: encoder dropout

| Policy | Success | Mean final error | Safe stops | p vs agent |
|---|---|---|---|---|
| raw encoder | 0.013 | 0.663 | 0.000 | 0.0002 |
| Kalman (no agent) | 0.435 | 0.109 | 0.000 | 0.0002 |
| Kalman + motor model (no agent) | 0.575 | 0.079 | 0.000 | 0.0002 |
| agent, monitor only | 0.435 | 0.109 | 0.000 | 0.0002 |
| agent, no quantum accelerometer | 0.805 | 0.042 | 0.000 | 0.0002 |
| agent (monitor + self-heal) | 0.848 | 0.037 | 0.000 | – |

Agent diagnosis: first event is the injected fault in 0.995 of episodes; the final diagnosis is in 0.995; any event in 1.000; median latency 0.10 s.

## Fault: encoder offset

| Policy | Success | Mean final error | Safe stops | p vs agent |
|---|---|---|---|---|
| raw encoder | 0.000 | 0.246 | 0.000 | 0.0002 |
| Kalman (no agent) | 0.000 | 0.267 | 0.000 | 0.0002 |
| Kalman + motor model (no agent) | 0.000 | 0.246 | 0.000 | 0.0002 |
| agent, monitor only | 0.000 | 0.267 | 0.025 | 0.0002 |
| agent, no quantum accelerometer | 0.805 | 0.042 | 0.000 | 0.0002 |
| agent (monitor + self-heal) | 0.843 | 0.037 | 0.000 | – |

Agent diagnosis: first event is the injected fault in 1.000 of episodes; the final diagnosis is in 1.000; any event in 1.000; median latency 0.10 s.

## Fault: actuator gain loss

| Policy | Success | Mean final error | Safe stops | p vs agent |
|---|---|---|---|---|
| raw encoder | 0.725 | 0.056 | 0.000 | 0.0002 |
| Kalman (no agent) | 0.765 | 0.053 | 0.000 | 0.0002 |
| Kalman + motor model (no agent) | 0.495 | 0.074 | 0.000 | 0.0002 |
| agent, monitor only | 0.765 | 0.053 | 0.043 | 0.0002 |
| agent, no quantum accelerometer | 0.887 | 0.036 | 0.010 | 0.0020 |
| agent (monitor + self-heal) | 0.897 | 0.035 | 0.013 | – |

Agent diagnosis: first event is the injected fault in 0.517 of episodes; the final diagnosis is in 0.510; any event in 0.675; median latency 0.45 s.

## Fault: friction increase

| Policy | Success | Mean final error | Safe stops | p vs agent |
|---|---|---|---|---|
| raw encoder | 0.770 | 0.043 | 0.000 | 0.0002 |
| Kalman (no agent) | 0.677 | 0.049 | 0.000 | 0.0002 |
| Kalman + motor model (no agent) | 0.590 | 0.056 | 0.000 | 0.0002 |
| agent, monitor only | 0.677 | 0.049 | 0.000 | 0.0002 |
| agent, no quantum accelerometer | 0.958 | 0.026 | 0.000 | 0.0095 |
| agent (monitor + self-heal) | 0.965 | 0.025 | 0.000 | – |

Agent diagnosis: first event is the injected fault in 0.925 of episodes; the final diagnosis is in 0.910; any event in 0.955; median latency 1.20 s.

## Fault: external push

| Policy | Success | Mean final error | Safe stops | p vs agent |
|---|---|---|---|---|
| raw encoder | 0.000 | 0.185 | 0.000 | 0.2222 |
| Kalman (no agent) | 0.000 | 0.186 | 0.000 | 0.2347 |
| Kalman + motor model (no agent) | 0.000 | 0.255 | 0.000 | 0.0002 |
| agent, monitor only | 0.000 | 0.186 | 0.412 | 0.2347 |
| agent, no quantum accelerometer | 0.318 | 0.210 | 0.098 | 0.1445 |
| agent (monitor + self-heal) | 0.345 | 0.201 | 0.100 | – |

Agent diagnosis: first event is the injected fault in 0.330 of episodes; the final diagnosis is in 0.250; any event in 0.998; median latency 1.40 s.

Limits: the agent knows the nominal plant and handles only these five fault kinds; faults are single and step-like; all data are simulated, with illustrative sensor values.
