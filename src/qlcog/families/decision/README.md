# Risky choice: quantum decision theory

**Idea.** In quantum decision theory (Yukalov and Sornette) the probability of choosing a prospect is
a utility factor plus an attraction factor: p = f + q. The attraction factor captures how attractive
or repulsive a prospect feels beyond its utility; a "quarter law" suggests |q| ≈ 0.25 as a default.

**Use it for:** consumer choice, insurance and investment decisions, policy preferences, any set of
binary choices between lotteries.

## Models

| Class | Parameters | Notes |
|---|---|---|
| `QDTModel` | alpha (utility curvature), beta (choice sensitivity), q0 (attraction size) | Sign of q: negative for the more uncertain prospect, or set per problem with `signs={problem: ±1}` |
| `ExpectedUtilityModel` | alpha, beta | Logit expected utility (QDT with q0 = 0) |
| `ProspectTheoryModel` | alpha, lam, g, beta | Tversky–Kahneman (1992) value and weighting functions, applied outcome by outcome |

## Data format

```python
problems = {'p1': ([(30, 1.0)], [(45, 0.8), (0, 0.2)])}     # (lottery 1, lottery 2); outcome, probability
counts = {'p1': [n_choose_1, n_choose_2]}
```

## Quick start

```python
from qlcog.core import compare
from qlcog.families.decision import QDTModel, ExpectedUtilityModel, ProspectTheoryModel
compare([QDTModel, ExpectedUtilityModel, ProspectTheoryModel], counts, problems)
```

## Limitations

The default sign rule encodes uncertainty aversion and conflicts with risk seeking for losses
(reflection effect); set signs per problem when theory says otherwise. No circuit is provided: QDT's
quantum structure is in its derivation, not in a dynamic that maps naturally onto gates.

## References

Yukalov and Sornette (2011) *Theory and Decision* 70. Yukalov and Sornette (2016) *Phil. Trans. R.
Soc. A* 374 ("Quantum probability and quantum decision-making"). Tversky and Kahneman (1992) *J. Risk and Uncertainty* 5.
