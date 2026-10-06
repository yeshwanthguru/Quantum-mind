# Episodic memory: overdistribution (`quantum_mind.families.memory`)

In conjoint-recognition experiments the same test item is judged under three instructions: *was it
studied?* (V), *is it related to a studied item?* (G), and *was it studied or related?* (V or G).
People's P(V) + P(G) often exceeds P(V or G) (**overdistribution**), impossible for exclusive events in
classical probability.

| Model | Idea | Parameters |
|---|---|---|
| `QuantumEpisodicModel` | memory state per probe type in a real 3D space; V and G rank-1 subspaces at angle c; "V or G" is the plane they span (quantum disjunction), so P(V) + P(G) − P(V or G) > 0 unless V ⊥ G | 2 per probe type + c |
| `AdditiveMemoryModel` | classical baseline: exclusive V and G, P(V or G) = P(V) + P(G) | 2 per probe type |

Design: `(probe, question)` pairs with probes such as `target`, `related`, `unrelated` and questions
`V`, `G`, `V|G`; data: observed proportions (least squares). `overdistribution(values, probe)` reports
the effect for data or predictions.

The examples use simulated data (labelled as such). Reference: Brainerd, Wang and Reyna, *Topics in
Cognitive Science* 5, 773–799 (2013); Brainerd and Reyna, *Psychological Review* 115, 2008.
