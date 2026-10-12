# Monitoring and self-healing agent for the motor

Simulation. 5 seeds x 80 episodes per cell. One fault is injected at 1.5 s of a 4 s two-leg move; success = final error under a quarter cell. p: paired sign-flip test on final error against the full agent.

## Fault: none

| Policy | Success | Mean final error | Safe stops | p vs agent |
|---|---|---|---|---|
| raw encoder | 1.000 | 0.011 | 0.000 | 0.4101 |
| Kalman (no agent) | 1.000 | 0.009 | 0.000 | 0.1290 |
| Kalman + motor model (no agent) | 1.000 | 0.009 | 0.000 | 0.8983 |
| agent, monitor only | 1.000 | 0.009 | 0.000 | 0.1290 |
| agent, no quantum accelerometer | 0.993 | 0.010 | 0.000 | 0.3654 |
| agent (monitor + self-heal) | 0.998 | 0.010 | 0.000 | – |

False alarms (any event in a healthy episode): 0.003 of episodes.

## Fault: encoder dropout

| Policy | Success | Mean final error | Safe stops | p vs agent |
|---|---|---|---|---|
| raw encoder | 0.013 | 0.663 | 0.000 | 0.0002 |
| Kalman (no agent) | 0.435 | 0.109 | 0.000 | 0.0002 |
| Kalman + motor model (no agent) | 0.575 | 0.079 | 0.000 | 0.0002 |
| agent, monitor only | 0.435 | 0.109 | 0.000 | 0.0002 |
| agent, no quantum accelerometer | 0.805 | 0.042 | 0.000 | 0.0002 |
| agent (monitor + self-heal) | 0.848 | 0.037 | 0.000 | – |

Agent diagnosis: first event is the injected fault in 0.998 of episodes; the final diagnosis is in 0.998; any event in 1.000; median latency 0.10 s.

## Fault: encoder offset

| Policy | Success | Mean final error | Safe stops | p vs agent |
|---|---|---|---|---|
| raw encoder | 0.000 | 0.246 | 0.000 | 0.0002 |
| Kalman (no agent) | 0.000 | 0.267 | 0.000 | 0.0002 |
| Kalman + motor model (no agent) | 0.000 | 0.246 | 0.000 | 0.0002 |
| agent, monitor only | 0.000 | 0.267 | 0.000 | 0.0002 |
| agent, no quantum accelerometer | 0.825 | 0.041 | 0.000 | 0.0002 |
| agent (monitor + self-heal) | 0.863 | 0.036 | 0.000 | – |

Agent diagnosis: first event is the injected fault in 1.000 of episodes; the final diagnosis is in 0.998; any event in 1.000; median latency 0.15 s.

## Fault: actuator gain loss

| Policy | Success | Mean final error | Safe stops | p vs agent |
|---|---|---|---|---|
| raw encoder | 0.725 | 0.056 | 0.000 | 0.1710 |
| Kalman (no agent) | 0.765 | 0.053 | 0.000 | 0.0770 |
| Kalman + motor model (no agent) | 0.495 | 0.074 | 0.000 | 0.2927 |
| agent, monitor only | 0.765 | 0.053 | 0.000 | 0.0770 |
| agent, no quantum accelerometer | 0.855 | 0.059 | 0.000 | 0.0130 |
| agent (monitor + self-heal) | 0.853 | 0.066 | 0.000 | – |

Agent diagnosis: first event is the injected fault in 0.520 of episodes; the final diagnosis is in 0.512; any event in 0.675; median latency 0.45 s.

## Fault: friction increase

| Policy | Success | Mean final error | Safe stops | p vs agent |
|---|---|---|---|---|
| raw encoder | 0.770 | 0.043 | 0.000 | 0.0002 |
| Kalman (no agent) | 0.677 | 0.049 | 0.000 | 0.0002 |
| Kalman + motor model (no agent) | 0.590 | 0.056 | 0.000 | 0.0002 |
| agent, monitor only | 0.677 | 0.049 | 0.000 | 0.0002 |
| agent, no quantum accelerometer | 0.993 | 0.019 | 0.000 | 0.7101 |
| agent (monitor + self-heal) | 0.993 | 0.019 | 0.000 | – |

Agent diagnosis: first event is the injected fault in 0.948 of episodes; the final diagnosis is in 0.865; any event in 0.955; median latency 1.25 s.

## Fault: external push

| Policy | Success | Mean final error | Safe stops | p vs agent |
|---|---|---|---|---|
| raw encoder | 0.000 | 0.185 | 0.000 | 0.0002 |
| Kalman (no agent) | 0.000 | 0.186 | 0.000 | 0.0002 |
| Kalman + motor model (no agent) | 0.000 | 0.255 | 0.000 | 0.0002 |
| agent, monitor only | 0.000 | 0.186 | 0.092 | 0.0002 |
| agent, no quantum accelerometer | 0.547 | 0.115 | 0.000 | 0.0717 |
| agent (monitor + self-heal) | 0.515 | 0.124 | 0.000 | – |

Agent diagnosis: first event is the injected fault in 0.492 of episodes; the final diagnosis is in 0.500; any event in 0.995; median latency 1.45 s.

Limits: the agent knows the nominal plant and handles only these five fault kinds; faults are single and step-like; all data are simulated, with illustrative sensor values.
