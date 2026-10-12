# One-motor VLA prototype: closed-loop study

Simulation. 5 training seeds, 400 fresh test scenes per seed, mean ± std over seeds. Training scenes for the main table: 4000. TinyVLA has 817 parameters, the BC baseline 7809.

## A. Policies

"Ambiguous" = the instruction is true of more than one object, so the intended cell is a coin flip unless the robot asks. Success = final position within a quarter cell of the intended cell.

| Policy | Success, all | Success, unambiguous | Success, ambiguous | Questions, all | Success − 0.1·questions | Paired p vs act |
|---|---|---|---|---|---|---|
| oracle | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.00 | 1.000 | – |
| BC baseline | 0.215 ± 0.006 | 0.261 ± 0.014 | 0.126 ± 0.023 | 0.00 | 0.215 | – |
| TinyVLA, act | 0.650 ± 0.023 | 0.959 ± 0.017 | 0.036 ± 0.018 | 0.00 | 0.650 | – |
| TinyVLA, always ask | 0.981 ± 0.005 | 0.988 ± 0.006 | 0.967 ± 0.015 | 1.00 | 0.881 | 0.0002 |
| TinyVLA, ask gate a=0.1 | 0.819 ± 0.016 | 0.959 ± 0.017 | 0.538 ± 0.028 | 0.18 | 0.801 | 0.0002 |
| TinyVLA, ask gate a=0.05 | 0.897 ± 0.009 | 0.962 ± 0.017 | 0.769 ± 0.028 | 0.27 | 0.870 | 0.0002 |
| TinyVLA, ask gate a=0.02 | 0.945 ± 0.009 | 0.966 ± 0.016 | 0.903 ± 0.019 | 0.32 | 0.913 | 0.0002 |
| TinyVLA, ask gate a=0.01 | 0.957 ± 0.010 | 0.972 ± 0.014 | 0.926 ± 0.014 | 0.34 | 0.923 | 0.0002 |

Paired p: sign-flip test on per-scene (success − 0.1·questions) against "TinyVLA, act", scenes pooled over seeds. The gate level a=0.02 is the headline; the other levels are shown in full because the ask rate depends on it, not hidden.

## B. Data efficiency (unambiguous instructions only)

| Training scenes | TinyVLA success | BC baseline success |
|---|---|---|
| 250 | 0.851 ± 0.029 | 0.153 ± 0.019 |
| 1000 | 0.955 ± 0.009 | 0.157 ± 0.027 |
| 4000 | 0.959 ± 0.016 | 0.258 ± 0.016 |
| 16000 | 0.978 ± 0.012 | 0.329 ± 0.011 |

The BC baseline was given a wider layer and more optimiser steps than TinyVLA and tuned lightly on one seed; a different architecture or far more data may close the gap, so read B as "this prototype is more sample-efficient than this baseline", not as a general result.
