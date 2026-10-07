# Fusing vision, speech and gaze

The person says "the blue one" and looks to the right; the detector sees a red cup, a blue cup and a
bowl with different scores. Each source gives evidence about which object is meant. This tutorial
turns each source into a likelihood with an **adapter**, then fuses them three ways: **Bayesian**
(multiply), **Dempster-Shafer** (combine belief masses, with explicit ignorance) and **quantum-like**
(each cue is a basis in a shared space, so cues can be incompatible and the order they are considered
matters). The three are compared on held-out **simulated** trials.

```python
import numpy as np
import matplotlib.pyplot as plt
from quantum_mind.viz import use_mpl_style
from quantum_mind.applications.fusion import (detector_likelihood, asr_likelihood, direction_likelihood,
                                              bayes_fusion, dempster_shafer_fusion, fit_incompatibility, compare_fusion)
from quantum_mind.applications.intent import QuantumIntentResolver
theme = use_mpl_style('dark')
objects = ['red cup', 'blue cup', 'bowl']
```

## 1. Adapters: from raw outputs to likelihoods

```python
det = detector_likelihood({'red cup': 0.62, 'blue cup': 0.55, 'bowl': 0.20}, objects, temperature=1.5)
asr = asr_likelihood([('the blue one', 0.7), ('the bowl one', 0.2)],
                     {'red cup': ['red'], 'blue cup': ['blue'], 'bowl': ['bowl']}, objects)
gaze = direction_likelihood([0.9, 0.3], {'red cup': [-1, 0], 'blue cup': [1, 0.2], 'bowl': [0.2, 1]}, objects)
cues = {'detector': det, 'speech': asr, 'gaze': gaze}
for k, v in cues.items():
    print('%-9s %s' % (k, v.round(3)))
```

## 2. Fuse

```python
bayes = bayes_fusion(list(cues.values()))
ds, ignorance = dempster_shafer_fusion(list(cues.values()), reliabilities=[0.8, 0.9, 0.6])
print('Bayes            ', bayes.round(3))
print('Dempster-Shafer  ', ds.round(3), '  ignorance %.3f' % ignorance)
```

```python
fig, axes = plt.subplots(1, 5, figsize=(11, 2.6), sharey=True)
panels = list(cues.items()) + [('Bayes', bayes), ('Dempster-Shafer', ds)]
for ax, (name, v), c in zip(axes, panels, theme['palette'] + theme['palette']):
    ax.bar(range(3), v, color=c); ax.set_title(name); ax.set_xticks(range(3), ['red\ncup', 'blue\ncup', 'bowl'])
axes[0].set_ylabel('probability'); fig.suptitle('Three cues and two fusions', y=1.04)
plt.show()
```

## 3. Learn how incompatible the cues are, and compare on held-out trials

Simulate people whose choices follow a quantum-like resolver with two cues of known incompatibility,
in both cue orders. Fit the incompatibility angles on 1000 trials, then compare log loss on 500 new
ones (lower is better).

```python
intents = ['a', 'b', 'c']
cue_l = {'x': [0.7, 0.2, 0.1], 'y': [0.25, 0.6, 0.15]}
truth = QuantumIntentResolver(intents)
truth.add_cue('x', cue_l['x'], 0.7)
truth.add_cue('y', cue_l['y'], 0.7)
rng = np.random.default_rng(1)
orders = [['x', 'y'], ['y', 'x']]
trials = [(o, intents[rng.choice(3, p=truth.posterior(o))]) for o in [orders[i % 2] for i in range(1500)]]
_, angles = fit_incompatibility(QuantumIntentResolver, intents, cue_l, trials[:1000])
print('fitted angles:', {k: round(v, 2) for k, v in angles.items()}, '(true 0.7 each)')
res = compare_fusion(intents, cue_l, trials[:1000], trials[1000:])
res = {k: v for k, v in res.items() if isinstance(v, dict) and 'log_loss' in v}
for k, v in res.items():
    print('%-16s held-out log loss %.3f' % (k, v['log_loss']))
```

```python
fig, ax = plt.subplots(figsize=(5.6, 2.6))
names = list(res); vals = [res[k]['log_loss'] for k in names]
ax.barh(names, vals, color=theme['palette'][:len(names)])
ax.set_xlim(min(vals) - 0.05, max(vals) + 0.02); ax.set_xlabel('held-out log loss (lower is better)')
ax.set_title('Fusion rules on simulated order-dependent choices')
plt.show()
```

**Caveat.** These people are simulated from the quantum-like resolver, so it is expected to win; the
point is the workflow. The angles are only weakly identifiable from choices alone, so report them
with uncertainty.

References: Ernst & Banks (2002), *Nature* 415, 429-433; Shafer (1976), *A Mathematical Theory of
Evidence*; Busemeyer & Bruza (2024), ch. 4.
