# Bistable perception (`qlcog.families.perception`)

An ambiguous stimulus (a Necker cube, a rotating silhouette, an object a robot's camera cannot resolve)
is perceived as one of two interpretations that alternate. The **dwell time** in one interpretation is
the standard measurement.

| Model | Idea | Parameters |
|---|---|---|
| `QuantumZenoBistableModel` | the percept is a qubit that rotates freely between internal observations every Δt; each observation collapses it, so P(switch) = sin²(g Δt) and the mean dwell time ≈ 1/(g² Δt): more frequent observation slows switching (quantum Zeno effect) | g |
| `MarkovSwitchingModel` | classical baseline: constant switching rate, exponential dwell times, no dependence on Δt | rate |
| `GammaRenewalModel` | classical baseline: gamma-distributed dwell times, the usual empirical description | shape, scale |

**Distinctive test.** With a single observation interval the Zeno and Markov models predict the same
geometric or exponential shape. The Zeno model's own prediction is that the mean dwell time scales as
1/Δt, so a design must vary the observation (presentation or sampling) interval across conditions.

Design: `dwell_time_design({condition: dt})` gives histogram bins per condition; `dwell_counts(times,
edges)` turns raw dwell times into counts for fitting (multinomial). For robots, the same models describe
how a perception module that re-samples an ambiguous scene flips between interpretations.

References: Atmanspacher, Filk and Römer, *Biological Cybernetics* 90, 33–40 (2004); Atmanspacher and
Filk, *Journal of Mathematical Psychology* 54, 314–321 (2010); Levelt, *On Binocular Rivalry* (1965).
Examples use simulated dwell times, labelled as such.
