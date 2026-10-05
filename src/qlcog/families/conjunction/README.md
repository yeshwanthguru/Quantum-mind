# Conjunction and disjunction judgements

**Phenomenon.** People judge a conjunction as more probable than one of its parts (conjunction
fallacy). In the succinct Linda problem, 85% rated "bank teller and active in the feminist movement"
as more probable than "bank teller" (Tversky and Kahneman, 1983). The mirror error occurs for
disjunctions.

**Use it for:** risk communication, forecasting, diagnosis, intelligence analysis, legal judgement,
any setting where people estimate probabilities of combined events.

## Models

| Class | Parameters | Conjunction fallacy possible | Notes |
|---|---|---|---|
| `QuantumConjunctionModel` | 3 angles + ranks | yes, for incompatible events | Conjunction judged sequentially, more likely event first (Busemeyer et al., 2011) |
| `ClassicalJointModel` | 3 (pA, pB, rho) | no | P(A&B) ≤ min(P(A), P(B)) always |
| `AveragingModel` | 3 (pA, pB, w) | yes | Weighted average of the marginals |
| `PTNModel` | 5 (pA, pB, rho, d, dd) | yes, through noise | Probability theory plus noise (Costello and Watts, 2014) |

## Mathematics

P(A&B) = ‖P_B P_A ψ‖² when P(A) ≥ P(B) (otherwise the order is swapped); P(A or B) = 1 − ‖P_¬B P_¬A ψ‖².
When P_A and P_B do not commute, the first projection can raise the probability of the second event,
so P(A&B) can exceed P(B).

## Data format

```python
design = ('A', 'B', 'A&B', 'A|B')                 # any subset
data = {'A': [0.12], 'B': [0.45], 'A&B': [0.31], 'A|B': [0.58]}   # mean judged probabilities, 0-1
```
All models in this family are fitted by least squares (`LOSS = 'sse'`). `fallacy_rate(judgements)`
flags conjunction and disjunction fallacies in a set of judgements.

## Quick start

```python
from qlcog.core import compare
from qlcog.families.conjunction import QuantumConjunctionModel, ClassicalJointModel, AveragingModel, PTNModel, RANK_STRUCTURES
compare([(QuantumConjunctionModel, {'structures': RANK_STRUCTURES}), ClassicalJointModel, AveragingModel, PTNModel], data)
```

## Circuit

`qlcog.circuits.conjunction_circuit(model)` asks the two events in the model's order on the
sequential-measurement circuit and returns P(A&B) and P(A or B).

## Limitations

Mean judgements hide individual strategies; the noise account (PTN) fits many of the same patterns;
Boyer-Kassem, Duchêne and Guerci (2016, *Theory and Decision* 81) argue that quantum-like models cannot
account for the conjunction fallacy once their further predictions are tested;
the 85% Linda figure is a choice rate, not a probability estimate, and is not fitted directly.

## References

Tversky and Kahneman (1983) *Psychological Review* 90(4). Busemeyer, Pothos, Franco and Trueblood
(2011) *Psychological Review* 118(2). Costello and Watts (2014) *Psychological Review* 121(3).
