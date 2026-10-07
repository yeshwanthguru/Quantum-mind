# When should a robot ask for help?

A robot is asked to fetch "the red cup on the left". It has a hypothesis, but it is not sure. It can
act now and risk a wrong action, or ask a clarifying question and spend the person's time. This
tutorial builds the decision rule step by step: the **expected cost** of each choice, the robot's
uncertainty about its **own models** of people, and the **hazard** of the action.

```python
import numpy as np
import matplotlib.pyplot as plt
from quantum_mind.viz import use_mpl_style
from quantum_mind.applications.robotics import (domain_models, HumanModelEnsemble, ask_or_act,
                                                risk_aware_ask_or_act, HAZARD_COSTS)
from quantum_mind.families.order_effects import QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel
theme = use_mpl_style('dark')
print(ask_or_act(0.9, ask_cost=1.0, error_cost=5.0))
```

## 1. The basic rule

Acting costs `error_cost × (1 - p)` on average; asking costs `ask_cost`. The robot asks when the
first is larger. The break-even confidence is `1 - ask_cost / error_cost`.

```python
p = np.linspace(0.5, 1.0, 200)
fig, ax = plt.subplots(figsize=(6.6, 3.2))
for err, c in ((2.0, theme['palette'][3]), (5.0, theme['palette'][1]), (20.0, theme['palette'][0])):
    ax.plot(p, err * (1 - p), color=c, lw=2, label='act, error cost %g' % err)
    ax.axvline(1 - 1.0 / err, color=c, ls=':', lw=1)
ax.axhline(1.0, color=theme['text'], ls='--', lw=1.5, label='ask (cost 1)')
ax.set_xlabel('robot confidence p'); ax.set_ylabel('expected cost'); ax.set_ylim(0, 6)
ax.set_title('Ask when the dashed line is below the curve'); ax.legend(loc='upper right')
plt.show()
```

## 2. Learn about people, and know when the models disagree

The robot's confidence comes from models of how people answer. Early on it has little data, and
competing models (quantum-like, Bayesian, anchoring) may predict different things. An ensemble weighs
them by evidence and reports how much they **disagree**; while they disagree, the robot asks
regardless of cost. The people here are **simulated** from a quantum-like population.

```python
rng = np.random.default_rng(3)
people = domain_models('object_clarification')['QL']
counts = {'AB': np.zeros(4, int), 'BA': np.zeros(4, int)}
rows = []
for batch in range(1, 7):
    new = people.sample(None, 25, rng)
    counts = {c: counts[c] + new[c] for c in counts}
    ens = HumanModelEnsemble([QuantumOrderModel4D, BayesOrderModel, AnchoringOrderModel]).update(counts, restarts=4, rng=rng)
    pred, unc = ens.predict('AB')
    p_confirm = pred[0] + pred[1]
    action = ask_or_act(p_confirm, unc, ask_cost=1.0, error_cost=1.5, max_disagreement_bits=0.01)['action']
    rows.append((50 * batch, p_confirm, unc['model_disagreement_bits'], action))
    print('%3d people  P(confirm) %.3f  disagreement %.4f bits  -> %s' % rows[-1])
print('ensemble weights:', [(s['model'], round(s['weight'], 2)) for s in ens.summary()])
```

```python
n, pc, dis, act = zip(*rows)
fig, ax1 = plt.subplots(figsize=(6.8, 3.2))
ax1.plot(n, dis, 'o-', color=theme['palette'][0], label='model disagreement (bits)')
ax1.axhline(0.01, color=theme['palette'][0], ls=':', lw=1)
ax1.set_xlabel('people observed'); ax1.set_ylabel('disagreement (bits)')
ax2 = ax1.twinx(); ax2.grid(False)
ax2.plot(n, pc, 's--', color=theme['palette'][1], label='P(confirm)'); ax2.set_ylabel('P(confirm)')
for x, a in zip(n, act):
    ax1.annotate(a, (x, max(dis) * 1.05), ha='center', color=theme['palette'][3] if a == 'act' else theme['text'])
ax1.set_title('The robot asks until its models agree'); fig.legend(loc='center right')
plt.show()
```

The ensemble gives most weight to the Bayesian model here even though the people are quantum-like:
for the question order the robot uses, both predict the same answers, and the Bayesian model has
fewer parameters. The models differ only on the reversed order, which is why the order matters when
designing what to ask (tutorial 7).

## 3. Hazard: some mistakes are worse than others

Handing over a cup is low risk; handing over a knife is not. `risk_aware_ask_or_act` uses a cost per
hazard level and can also use a lower confidence bound (`p_lower`) instead of the point estimate.

```python
print('hazard costs:', HAZARD_COSTS)
levels = list(HAZARD_COSTS)
grid = np.linspace(0.5, 0.999, 60)
M = np.array([[risk_aware_ask_or_act(pp, h)['action'] == 'act' for pp in grid] for h in levels])
fig, ax = plt.subplots(figsize=(6.8, 2.4))
ax.imshow(M, aspect='auto', cmap='viridis', extent=[grid[0], grid[-1], len(levels) - 0.5, -0.5])
ax.set_yticks(range(len(levels)), levels); ax.set_xlabel('robot confidence p'); ax.grid(False)
ax.set_title('Act (yellow) or ask (purple), by hazard')
plt.show()
```

The statements below are checked by this cell every time the documentation is built:

```python
assert rows[0][3] == 'ask' and rows[-1][3] == 'act'     # asks while the models disagree, then acts
assert rows[-1][2] < 0.01
print('checked')
```

**Summary.** The decision layer is small and framework-free: call it from a ROS 2 node, a behaviour
tree or a planner. What makes it work is reliable inputs: calibrated confidence (tutorial 5) and models
that know when they disagree.

References: Howard (1966), *IEEE Trans. Systems Science and Cybernetics* 2, 22-26; Kochenderfer et al.
(2022), *Algorithms for Decision Making*, MIT Press.
