# What Quantum Mind adds to published models

Most models in this library were published by other researchers, and each
{doc}`model atlas <../atlas/index>` entry cites its source. The atlas labels every entry by origin:
{bdg-success}`Unique to Quantum Mind`, {bdg-info}`Published model, extended`,
{bdg-secondary}`Published method` or {bdg-muted}`Software tool`.

The papers report a model as point estimates fitted once to a group of people. A robot works with
one person at a time, needs to know how sure it is, and has to choose what to ask next. Three tools
in {mod}`quantum_mind.core` give every people model those abilities, quantum-like or classical, which
is what the "extended" label refers to. The equations are on the
{doc}`core mathematics page <../math/core>`.

| Tool | What it gives | Works with |
|---|---|---|
| {func}`~quantum_mind.core.uncertainty.bootstrap` | Intervals for parameters and for predicted answer probabilities | every model |
| {class}`~quantum_mind.core.online.OnlinePersonModel` | A per-person posterior, updated after every answer, starting from the population fit | models fitted by likelihood |
| {func}`~quantum_mind.core.design.rank_conditions`, {func}`~quantum_mind.core.design.model_posterior` | The question, order or display that best tells competing models apart, and the model probabilities after each answer | models fitted by likelihood |

## Example: one person, live

```python
import numpy as np
from quantum_mind import fit, bootstrap, OnlinePersonModel
from quantum_mind.core import rank_conditions, model_posterior
from quantum_mind.families.order_effects import QuantumOrderModel4D, BayesOrderModel

group = {'AB': [40, 10, 20, 30], 'BA': [30, 20, 10, 40]}     # answers from earlier participants
q = fit(QuantumOrderModel4D, group)
b = fit(BayesOrderModel, group)
print(bootstrap(q, group, n_boot=50).predictive_interval()['AB'])   # how sure the robot is

person = OnlinePersonModel(QuantumOrderModel4D, prior=q, prior_scale=0.5)
print(rank_conditions([q.model, b.model], ['AB', 'BA']))      # which order to ask in
person.update('AB', 0)                                        # the person said yes, yes
print(model_posterior([q.model, b.model], {'AB': [1, 0, 0, 0]}))
print(person.predict()['BA'])                                 # this person's predicted answers
```

The bootstrap needs repeated fits, so it belongs before or between interactions. The online model
and the design functions cost a fraction of a second per answer and can run during one.
