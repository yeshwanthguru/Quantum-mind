# Interference in decisions (disjunction effect)

**Phenomenon.** People choose X when they know an earlier event went one way, and also when it went
the other way, but not when they do not know (violation of Savage's sure-thing principle). Two-stage
gamble: play again 69% after a win, 59% after a loss, 36% when the outcome is unknown (Tversky and
Shafir, 1992). Prisoner's Dilemma: defect 97% / 84% / 63% (Shafir and Tversky, 1992).

**Use it for:** investment and insurance decisions under pending information, negotiation, voting,
medical decisions while waiting for test results, consumer purchases before uncertain events.

## Models

| Class | Parameters | Law of total probability | Notes |
|---|---|---|---|
| `InterferenceModel` | 4 (p1, p2, c, theta) | violated for theta ≠ π/2 | Normalised form (default) or `normalized=False` |
| `ClassicalMixtureModel` | 3 (p1, p2, c) | holds | P(unknown) = c p1 + (1 − c) p2 |

## Mathematics

The unknown condition superposes both paths, with amplitudes √c and √(1 − c) and relative phase θ:

    A = c p1 + (1−c) p2 + 2 cos θ √(c p1 (1−c) p2)          (act)
    B = c q1 + (1−c) q2 + 2 cos θ √(c q1 (1−c) q2),  q = 1 − p   (not act)
    P(act | unknown) = A / (A + B)

## Data format

```python
counts = {'known_1': [n_act, n_not], 'known_2': [n_act, n_not], 'unknown': [n_act, n_not]}
from qlcog.data import TWO_STAGE_GAMBLE, proportions_to_counts
counts = proportions_to_counts(TWO_STAGE_GAMBLE['conditions'], TWO_STAGE_GAMBLE['n'])
```

## Quick start

```python
from qlcog.core import compare
from qlcog.families.interference import InterferenceModel, ClassicalMixtureModel, total_probability_bounds
compare([InterferenceModel, ClassicalMixtureModel], counts)
```

## Circuit

`qlcog.circuits.interference_circuit(p1, p2, c, theta, condition)`: a path qubit prepared in
√c|0⟩ + e^{iθ}√(1−c)|1⟩ controls the action qubit's rotation; for the unknown condition a Hadamard gate
erases the path and the path qubit is post-selected on 0. This implements the normalised form exactly
(verified against the statevector in the tests).

## Limitations

With three proportions and four parameters the interference model is saturated: it **represents**
the effect but does not **predict** it. Predictive tests need several gambles or games with shared
parameters, or a theory of the phase.

## References

Tversky and Shafir (1992) *Psychological Science* 3(5). Shafir and Tversky (1992) *Cognitive
Psychology* 24(4). Pothos and Busemeyer (2009) *Proc. R. Soc. B* 276. Moreira and Wichert (2014)
*Applied Soft Computing* 25.
