# Similarity and asymmetry

**Phenomenon.** Similarity judgements are often asymmetric: "Korea is like China" is rated higher
than "China is like Korea" (Tversky, 1977). Classical geometric models are symmetric unless biases are
added.

**Use it for:** brand and product similarity, concept and category judgements, recommender
explanations, cross-cultural comparisons.

## Models

| Class | Parameters | Asymmetry | Notes |
|---|---|---|---|
| `QuantumSimilarityModel` | 2 angles per concept + ranks | yes | Sim(A,B) = ‖P_B P_A ψ‖²; a larger subspace (rank 2) = a better-known concept (Pothos, Busemeyer and Trueblood, 2013) |
| `BiasedGeometricModel` | 2 coordinates + bias per concept, c | yes | Sim(A,B) = b_B exp(−c d(A,B)) (Nosofsky, 1991) |
| `GeometricModel` | 2 coordinates per concept, c | no | Symmetric baseline |

Use `for_concepts(Model, names)` to get a class with exactly the parameters for your concepts (at
most six), so that information criteria are fair.

## Data format

```python
pairs = [('Korea', 'China'), ('China', 'Korea')]
data = {('Korea', 'China'): [0.62], ('China', 'Korea'): [0.51]}    # mean ratings rescaled to 0-1
```

## Circuit

`similarity_circuit(model, a, b)`: two sequential binary measurements; Sim(A, B) = P(yes, yes).

## References

Tversky (1977) *Psychological Review* 84(4). Nosofsky (1991) *Cognitive Psychology* 23, 91–140. Pothos,
Busemeyer and Trueblood (2013) *Psychological Review* 120(3).
