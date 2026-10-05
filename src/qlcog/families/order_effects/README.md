# Order effects

**Phenomenon.** The answer to a question depends on what was asked before it. In a 1997 Gallup poll,
50% called Clinton honest when asked first and 57% when asked after Gore; Gore fell from 68% to 60%
(Moore, 2002).

**Use it for:** surveys and questionnaires, user research, A/B tests of question wording, evaluation
of human raters or language models, clinical interviews, any sequence of yes/no judgements.

## Models

| Class | Parameters | Order effect | QQ equality | Notes |
|---|---|---|---|---|
| `QuantumOrderModel4D` | 4 (state angles t1–t3, incompatibility phi) | yes | holds | **Recommended.** phi = 0 is exactly the Bayesian model, so it nests it |
| `QuantumOrderModel` | 3 (angles a, b, g) + ranks (1/2 per question) | yes | holds | Fits the Clinton–Gore rates with ranks (1, 2); cannot represent order-free data with intermediate correlation |
| `BayesOrderModel` | 3 (pA, pB, rho) | no | holds (trivially) | Classical baseline |
| `AnchoringOrderModel` | 4 (pA, pB, wA, wB); option `tied=True` → 3 | yes | violated unless wA = wB | Classical order-effect baseline |
| `SaturatedOrderModel` | 6 | any | any | Reference |
| `ProjectiveQuestionModel` | – | – | holds | Any state and projectors, any dimension (not fitted) |

## Mathematics

A belief is a unit vector ψ; a "yes" to question q is a projector P_q. Answering applies the Lüders
rule: p(yes) = ‖P_q ψ‖², then ψ ← P_q ψ / ‖P_q ψ‖. For two questions in either order every projective
model satisfies the parameter-free **QQ equality**

    p_AB(y,n) + p_AB(n,y) = p_BA(y,n) + p_BA(n,y)

(Wang and Busemeyer, 2013; supported in 70 national field experiments, Wang et al., 2014).

## Data format

```python
design = ('AB', 'BA')                    # or None (default)
counts = {'AB': [yy, yn, ny, nn],        # A asked first; first letter = answer to the first question
          'BA': [yy, yn, ny, nn]}        # B asked first
```

## Quick start

```python
from qlcog.core import compare
from qlcog.families.order_effects import (QuantumOrderModel4D, QuantumOrderModel, BayesOrderModel,
                                          AnchoringOrderModel, RANK_STRUCTURES, qq_test)
q, z, p = qq_test(counts)
results = compare([QuantumOrderModel4D, (QuantumOrderModel, {'structures': RANK_STRUCTURES}),
                   BayesOrderModel, AnchoringOrderModel], counts)
for r in results: print(type(r.model).__name__, r.bic)
```

## Circuit

`qlcog.circuits.order_effects_circuit(model, 'AB', form='dynamic' | 'deferred')`: the state is
loaded with `StatePreparation`; each question is a binary projective measurement (subspace rotated to
the computational basis, ancilla flagged, ancilla measured, rotation undone), so the post-measurement
state follows the Lüders rule. 3D models use 2 system qubits; the 4D model also uses 2.

## Limitations

- Aggregate fitting assumes a homogeneous population; individual differences can hide or mimic the
  structure (see the companion paper "Order-Aware Human Models for Robot Questioning and Trust").
- Standard projective models do not reproduce response replicability together with order effects
  (Khrennikov et al., 2014).
- The QQ equality alone does not prove quantum-like processing. Boyer-Kassem, Duchêne and Guerci (2016)
  derived further predictions (the Grand Reciprocity equations) that non-degenerate versions of these
  models fail on most existing data sets; check them before claiming support.

## References

Moore (2002) *Public Opinion Quarterly* 66(1). Wang and Busemeyer (2013) *Topics in Cognitive Science*
5(4). Wang, Solloway, Shiffrin and Busemeyer (2014) *PNAS* 111(26). Khrennikov, Basieva, Dzhafarov and
Busemeyer (2014) *PLoS ONE* 9. Boyer-Kassem, Duchêne and Guerci (2016) *Mathematical Social Sciences*
80, 33–46.
