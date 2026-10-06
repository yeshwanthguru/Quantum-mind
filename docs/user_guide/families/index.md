# Quantum-like families

Each family describes one phenomenon in human judgement or decision, gives the quantum-like model,
the classical baselines it must be compared with, the distinctive test, the circuit and the
references.

| Family | Phenomenon | Quantum-like model(s) | Classical baselines |
|---|---|---|---|
| [order effects](order_effects.md) | Answers depend on question order | `QuantumOrderModel4D`, `QuantumOrderModel` | Bayes, anchoring, saturated |
| [conjunction](conjunction.md) | Conjunction and disjunction fallacies | `QuantumConjunctionModel` | joint distribution, averaging, probability theory plus noise |
| [interference](interference.md) | Disjunction effect, sure-thing violations | `InterferenceModel` | classical mixture |
| [qlbn](qlbn.md) | Inference with unresolved hidden causes | quantum-like Bayesian network | classical Bayesian network |
| [dynamics](dynamics.md) | Belief change; judgements that change later judgements | `QuantumWalk`, `OpenSystemWalk`, `OpenSystemBelief` | `MarkovWalk`, `MarkovBelief` |
| [decision](decision.md) | Risky choice | `QDTModel` | expected utility, prospect theory |
| [contextuality](contextuality.md) | Is there one joint distribution? | CHSH, Contextuality-by-Default | classical bounds |
| [similarity](similarity.md) | Asymmetric similarity | `QuantumSimilarityModel` | biased geometric, geometric |
| [game theory](game_theory.md) | Strategic choice with shared quantum resources | `EWLGame` | classical Nash equilibria |
| [memory](memory.md) | Episodic overdistribution | `QuantumEpisodicModel` | additive (verbatim plus gist) |
| [concepts](concepts.md) | Overextension in concept combination | `FockSpaceConceptModel` | product, minimum, weighted average |
| [perception](perception.md) | Bistable perception (dwell times of ambiguous figures) | `QuantumZenoBistableModel` | Markov switching, gamma renewal |

```{toctree}
:hidden:

order_effects
conjunction
interference
qlbn
dynamics
decision
contextuality
similarity
game_theory
memory
concepts
perception
```
