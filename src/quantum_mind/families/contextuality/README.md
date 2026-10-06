# Contextuality analysis

**Question.** Can a set of binary measurements be described by one joint distribution, or does the
outcome of a measurement depend on which other measurements accompany it (contextuality)?

**Use it for:** checking survey items, concept combinations, psychophysics and physics data for
contextuality before fitting quantum-like models; teaching Bell/CHSH tests.

## Functions

| Function | Description |
|---|---|
| `chsh(E11, E12, E21, E22)` | CHSH value; classical bound 2, quantum bound 2√2 |
| `cyclic_contextuality(products, means_pairs)` | Contextuality-by-Default criterion for cyclic systems of rank n (Kujala, Dzhafarov and Larsson, 2015): contextual iff s_odd(products) − (n − 2) − Δ > 0 |
| `s_odd(values)` | Maximum signed sum with an odd number of minus signs |
| `correlations_from_data(pairs)` | Product expectations and means from ±1 observations |
| `qubit_chsh_correlations(a0, a1, b0, b1)` | Quantum prediction E(a, b) = cos(a − b) for a Bell pair |

Δ is the total inconsistency of the single-variable means across contexts; behavioural data are
almost always inconsistently connected, which is why the Contextuality-by-Default criterion, not the
plain CHSH bound, should be used for them.

## Circuit

`chsh_circuit(a, b)` prepares a Bell pair and measures at angles a and b; `decode(counts)` returns E.

## References

Clauser, Horne, Shimony and Holt (1969) *Phys. Rev. Lett.* 23. Kujala, Dzhafarov and Larsson (2015)
*Phys. Rev. Lett.* 115, 150401.
