# Belief dynamics

**Phenomenon.** Beliefs, preferences and trust change over time and with evidence, and **asking** for
a judgement can change later judgements. Quantum and Markov dynamics make different predictions about
this (Busemeyer, Kvam and Pleskac, 2019; Kvam et al., 2015).

**Use it for:** confidence over time in perception or diagnosis, trust in automation and AI
assistants, customer satisfaction over a service episode, opinion change during deliberation.

## Models

| Class | Parameters | Intermediate judgement changes later ones | Notes |
|---|---|---|---|
| `MarkovWalk` | mu (drift), gamma (rate) | no (on average) | N ordered states, K response bins |
| `QuantumWalk` | mu, sigma | yes | H_jj = μ x_j, H_{j,j±1} = σ |
| `OpenSystemWalk` | mu, sigma, lam (dephasing) | yes, less with larger lam | Lindblad master equation |
| `MarkovBelief` | p0, up, down | no | Yes/no belief over discrete events |
| `OpenSystemBelief` | phi0, a_pos, a_neg, gamma | yes for gamma < 1 | Qubit with dephasing |

Options for walks: `n_states` (default 21), `n_categories` (3), `initial_width` (0.15).

## Data format

```python
walk_design = {'t2_only': ('single', 1.5), 'both': ('joint', 0.5, 1.5)}   # K or K*K outcome cells
belief_design = {'end_only': ((1, 1, 0, 1), (3,)), 'twice': ((1, 1, 0, 1), (1, 3))}   # events, query indices
```
Outcome order for beliefs: answer sequences in `itertools.product((1, 0), repeat=k)` order (yes = 1).

## Quick start

```python
from qlcog.families.dynamics import QuantumWalk, MarkovWalk, OpenSystemBelief, MarkovBelief, question_effect
question_effect(OpenSystemBelief(gamma=0.3), (1, 1, 0, 1, 0, 1), last=5, intermediate=2)   # -0.076
```

## Circuits

`walk_circuit(model, t)` (state preparation and exp(−iHt) as a unitary gate; single-time
distributions) and `belief_circuit(model, events, queries, form)` (rotations, ancilla-controlled
dephasing, mid-circuit or deferred measurement).

## Limitations

Walk circuits measure a single time point; two-time judgements on hardware need coarse-grained
(category) measurements, which are implemented analytically only.

## References

Busemeyer, Wang and Townsend (2006) *J. Math. Psych.* 50. Kvam, Pleskac, Yu and Busemeyer (2015)
*PNAS* 112(34). Busemeyer, Kvam and Pleskac (2019) *Scientific Reports* 9.
