# Quantum-like Bayesian networks

**Idea.** Keep an ordinary Bayesian network, but marginalise hidden variables over amplitudes instead
of probabilities, so that unobserved causes can interfere (Moreira and Wichert, 2014, 2016).

**Use it for:** decision support where people (not the network) make the judgement under unresolved
uncertainty: diagnosis, credit and fraud assessment, judicial reasoning, expert elicitation.

## Models and functions

| Name | Description |
|---|---|
| `BayesNet(variables, parents, cpt)` | Discrete network; CPTs as nested dicts or functions |
| `classical_marginal(net, query, evidence)` | Exact classical inference by enumeration |
| `quantum_like_marginal(net, query, evidence, phases)` | Quantum-like inference, one phase per hidden configuration |
| `for_network(net, query, conditions)` | Returns a fit-able `QLBNModel` class with exactly the phases in use |
| `ClassicalBNModel` | Baseline with no free parameters |

## Mathematics

    P_q(Q = v | e) = α | Σ_h √P(v, e, h) · e^{iθ_h} |²,   α normalises over v

With two hidden configurations a phase difference of π/2 gives classical inference.

## Data format

```python
conditions = {'TestA+': {'TestA': 'pos'}, 'TestA-': {'TestA': 'neg'}}   # evidence per condition
counts = {'TestA+': [n_yes, n_no], 'TestA-': [n_yes, n_no]}            # judgements of the query
```

## Quick start

```python
from qlcog.core import compare
from qlcog.families.qlbn import BayesNet, for_network, ClassicalBNModel
QL = for_network(net, 'Disease', conditions)
compare([(QL, dict(net=net, query='Disease', conditions=conditions)),
         (ClassicalBNModel, dict(net=net, query='Disease', conditions=conditions))], counts)
```

## Circuit

`qlcog.circuits.qlbn_circuit(net, query, evidence, phases)`: the amplitudes are loaded with
`StatePreparation`, the hidden register goes through Hadamard gates and is post-selected on |0…0⟩,
which sums the amplitudes. Exact against the analytic model (tests).

## Limitations

Up to 8 hidden configurations in the fit-able model; phases are free parameters unless a heuristic
fixes them; enumeration is exponential in the number of hidden variables.

## References

Moreira and Wichert (2014) *Applied Soft Computing* 25. Moreira and Wichert (2016) *Frontiers in
Psychology* 7:11.
