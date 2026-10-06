# Concept combination: the guppy effect (`qlcog.families.concepts`)

How much does an item belong to "A and B"? People often rate a guppy as a better example of *pet
fish* than of *pet* or of *fish* (**overextension**, Hampton 1988). Fuzzy-set (minimum) and
probabilistic (product) conjunction rules cannot exceed the smaller single membership.

| Model | Rule for μ(A and B) | Parameters |
|---|---|---|
| `FockSpaceConceptModel` | Aerts' two-sector Fock space: m² μ<sub>A</sub>μ<sub>B</sub> + (1 − m²)[(μ<sub>A</sub> + μ<sub>B</sub>)/2 + κ · bound], with bound = min(√(μ<sub>A</sub>μ<sub>B</sub>), √((1 − μ<sub>A</sub>)(1 − μ<sub>B</sub>))) the largest possible interference | m², κ (shared by all items) |
| `ProductConceptModel` | μ<sub>A</sub> μ<sub>B</sub> | – |
| `MinConceptModel` | min(μ<sub>A</sub>, μ<sub>B</sub>) | – |
| `WeightedAverageModel` | w μ<sub>A</sub> + (1 − w) μ<sub>B</sub> | w |

Design: `{item: (mu_A, mu_B)}` (observed single memberships); data: observed membership in the
conjunction. `overextension(design, values)` gives μ(A and B) − min(μ<sub>A</sub>, μ<sub>B</sub>).
Sharing m² and κ across items (rather than one interference angle per item, which fits any data)
makes the model testable.

Example data are simulated. References: Aerts, *J. Math. Psych.* 53, 314–348 (2009); Hampton, *JEP:
LMC* 14, 12–32 (1988); Aerts and Gabora, *Kybernetes* 34 (2005).
