# Question-order effects

When two questions are asked one after the other, the answers often depend on which came first. In a
1997 Gallup poll, 50% said Bill Clinton was honest when that question came first, but 57% when it
came after the same question about Al Gore. This tutorial fits that published data with a
quantum-like model and its classical rivals, runs the **QQ test** (a parameter-free prediction of
projective models) and checks whether model comparison can tell the models apart.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from quantum_mind.viz import use_mpl_style
from quantum_mind.data import CLINTON_GORE
from quantum_mind.families.order_effects import (QuantumOrderModel, AnchoringOrderModel, BayesOrderModel,
                                                 RANK_STRUCTURES, rates, qq_test)
theme = use_mpl_style('dark')
r = CLINTON_GORE['rates']
observed = np.array([r['A_first'], r['B_first'], r['A_second'], r['B_second']])
print('Observed "yes" rates (Moore 2002): A first, B first, A second, B second =', observed)
```

## 1. Fit three models to the four rates

The **quantum-like** model puts the belief and the two "yes" subspaces in a three-dimensional space,
with three angles. The **anchoring** model is the classical alternative that allows order effects:
the second answer is pulled towards the first. The **Bayesian** model gives each question a fixed
probability, so it cannot produce an order effect at all.

```python
best = None
for st in RANK_STRUCTURES:
    for seed in range(20):
        fit = least_squares(lambda x: rates(QuantumOrderModel(a=x[0], b=x[1], g=x[2], **st).predict()) - observed,
                            np.random.default_rng(seed).uniform(0, np.pi, 3))
        if best is None or fit.cost < best[0].cost:
            best = (fit, st)
ql = QuantumOrderModel(a=best[0].x[0], b=best[0].x[1], g=best[0].x[2], **best[1])
y = least_squares(lambda y: rates(AnchoringOrderModel(pA=y[0], pB=y[1], wA=y[2], wB=y[3]).predict()) - observed,
                  [0.5, 0.6, 0.2, 0.2], bounds=([0.01, 0.01, -0.99, -0.99], [0.99, 0.99, 0.99, 0.99])).x
anc = AnchoringOrderModel(pA=y[0], pB=y[1], wA=y[2], wB=y[3])
bay = BayesOrderModel(pA=observed[[0, 2]].mean(), pB=observed[[1, 3]].mean(), rho=0.0)
fits = {'quantum-like': rates(ql.predict()), 'anchoring': rates(anc.predict()), 'Bayes': rates(bay.predict())}
for k, v in fits.items():
    print('%-13s %s' % (k, v.round(3)))
```

```python
labels = ['Clinton\nfirst', 'Gore\nfirst', 'Clinton\nsecond', 'Gore\nsecond']
x = np.arange(4)
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.bar(x - 0.3, observed, 0.2, label='observed', color=theme['text'], alpha=0.85)
for i, (k, v) in enumerate(fits.items()):
    ax.bar(x - 0.1 + 0.2 * i, v, 0.2, label=k, color=theme['palette'][i])
ax.set_xticks(x, labels); ax.set_ylim(0.4, 0.75); ax.set_ylabel('P("yes, honest")')
ax.set_title('Clinton-Gore question order: data and fits'); ax.legend(ncol=4, loc='upper left')
plt.show()
```

The Bayesian model gives the same rate in both positions, so it misses the shift. Both order models
reproduce it with their parameters.

## 2. The QQ test

Projective models predict, with **no free parameters**, that the probability of answering the two
questions *the same way* (yes-yes or no-no) is the same in both orders:
$q = [p_{AB}(yy) + p_{AB}(nn)] - [p_{BA}(yy) + p_{BA}(nn)] = 0$. The anchoring model has no such
constraint. The published study only reports the four rates, so here the joint counts are
**simulated** from each fitted model (1000 people per order) to show what the test detects.

```python
for name, model in (('quantum-like', ql), ('anchoring', anc)):
    counts = model.sample(None, 1000, np.random.default_rng(7))
    q, z, p = qq_test(counts)
    print('%-13s data: q = %+.3f, z = %+.2f, p = %.3f' % (name, q, z, p))
```

## 3. Can the models be told apart?

**Model recovery**: simulate data from each model, fit both, and count how often the generating model
wins by BIC (10 simulated studies per cell here; use more for a real design). If recovery is poor at a given sample size, a study of that size cannot settle the
question.

```python
from quantum_mind.core import recovery
gens = {'quantum-like': ql, 'anchoring': anc}
out = {}
for n in (100, 400):
    rec = recovery(gens, [(QuantumOrderModel, {'structures': RANK_STRUCTURES}), AnchoringOrderModel],
                   None, n, reps=10, rng=np.random.default_rng(0), restarts=4)
    out[n] = rec
    print('n = %d per order:' % n, rec)
```

```python
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
for ax, (n, rec) in zip(axes, out.items()):
    M = np.array([[rec[g].get(c, 0) for c in ('QuantumOrderModel', 'AnchoringOrderModel')] for g in gens]) / 10
    ax.imshow(M, vmin=0, vmax=1, cmap='magma')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, '%.1f' % M[i, j], ha='center', va='center', color='white' if M[i, j] < 0.6 else 'black', fontsize=12)
    ax.set_xticks([0, 1], ['QL fit', 'anchoring fit']); ax.set_yticks([0, 1], ['QL data', 'anchoring data'])
    ax.set_title('n = %d per order' % n); ax.grid(False)
fig.suptitle('Model recovery: how often the generating model wins'); plt.tight_layout(); plt.show()
```

**Summary.** The order effect itself rules out the Bayesian model. Separating the quantum-like model
from anchoring needs the joint answers (the QQ test) and enough people per order; recovery shows how
many.

References: Moore (2002), *Public Opinion Quarterly* 66, 80-91; Wang & Busemeyer (2013), *Topics in
Cognitive Science* 5, 689-710; Wang et al. (2014), *PNAS* 111, 9431-9436.
