# Quantum-like families

Each family describes one phenomenon in human judgement, trust or perception, gives the quantum-like model,
the classical baselines it must be compared with, the distinctive test, the circuit and the
references.

| Family | Phenomenon | Quantum-like model(s) | Classical baselines |
|---|---|---|---|
| [order effects](order_effects.md) | Answers depend on question order | `QuantumOrderModel4D`, `QuantumOrderModel` | Bayes, anchoring, saturated |
| [qlbn](qlbn.md) | Inference with unresolved hidden causes | quantum-like Bayesian network | classical Bayesian network |
| [dynamics](dynamics.md) | Belief change; judgements that change later judgements | `QuantumWalk`, `OpenSystemWalk`, `OpenSystemBelief` | `MarkovWalk`, `MarkovBelief` |
| [perception](perception.md) | Bistable perception (dwell times of ambiguous figures) | `QuantumZenoBistableModel` | Markov switching, gamma renewal |

```{toctree}
:hidden:

order_effects
qlbn
dynamics
perception
```
