# A context-aware belief filter (COBRA)

A person reaches toward a robot. The same hand movement can mean "give me that" or "I am taking my own tool",
and the difference is often a look a moment earlier. A filter that treats each sensor frame as independent
evidence cannot use that. **COBRA** keeps the robot's belief as a density matrix and lets the *previous frame*
rotate the basis in which the next frame is measured. This tutorial builds a small intent task, trains COBRA
and a classical Bayes filter on it, and checks what COBRA learned. The data are **simulated**, built to contain
a context effect, so the result shows what the model can do, not what people do.

```python
import numpy as np
import matplotlib.pyplot as plt
from quantum_mind.viz import use_mpl_style
from quantum_mind.belief import COBRA, GaussianBayesFilter, min_cost_action
theme = use_mpl_style('dark')
INTENTS = ['handover', 'retrieve', 'withdraw']
```

## 1. A task where context matters

Three hidden intents. A frame holds a gaze cue `a`, a hand cue `b` and a force cue `f`. The gaze cue leans
positive for a hand-over and negative for a retrieval. The hand cue carries the context effect: its mean is
`+sign(previous a)` for a hand-over and `-sign(previous a)` for a retrieval, so on its own `b` says almost
nothing, and its meaning flips with the previous look.

```python
MU_A, MU_F, KAPPA = np.array([0.7, -0.7, 0.0]), np.array([0.0, 0.0, 0.9]), np.array([1.0, -1.0, 0.0])

def simulate(n, rng, T=4):
    z = rng.integers(0, 3, n)
    a = MU_A[z][:, None] + rng.standard_normal((n, T))
    f = MU_F[z][:, None] + rng.standard_normal((n, T))
    prev = np.concatenate([np.zeros((n, 1)), a[:, :-1]], axis=1)
    b = KAPPA[z][:, None] * np.sign(prev) + rng.standard_normal((n, T))
    return np.stack([a, b, f], axis=2), z

rng = np.random.default_rng(11)
x_train, z_train = simulate(800, rng)
x_test, z_test = simulate(3000, rng)
print('training sequences', x_train.shape, '| marginal mean of b by intent:',
      [round(float(x_test[z_test == k][:, :, 1].mean()), 2) for k in range(3)])
```

## 2. A classical baseline and COBRA

`GaussianBayesFilter` treats the frames as conditionally independent, so reordering them cannot change its
belief. COBRA is trained on the same data. Training is slow (finite-difference gradients with several
restarts) and takes about a minute, and it does not always find the context mechanism, so the cell retrains with a new
seed if the check in section 3 shows it was missed.

```python
def shift(model, sign):
    """Log-odds shift (handover vs retrieve) from one hand cue b = +1.5 after a gaze cue of +-1.5 (see section 3)."""
    before = np.zeros((1, 1, 3)); before[0, 0, 0] = sign * 1.5
    after = np.zeros((1, 2, 3)); after[0, 0, 0] = sign * 1.5; after[0, 1, 1] = 1.5
    lo = lambda p: float(np.log(p[0, 0] / p[0, 1]))
    return lo(model.predict_proba(after)) - lo(model.predict_proba(before))

bayes = GaussianBayesFilter(3).fit(x_train, z_train)
for seed in range(6):                      # the context mechanism is found on some starts and not on others: retrain if missed
    cobra = COBRA(3, 3, steps=80, batch=100, restarts=6, seed=seed).fit(x_train, z_train)
    if shift(cobra, +1) > 1.0 and shift(cobra, -1) < -1.0:
        break
print('training seed that found the context mechanism: %d (of up to 6 tried)' % seed)
logloss = {}
for name, m in (('Bayes filter', bayes), ('COBRA', cobra)):
    p = m.predict_proba(x_test)
    logloss[name] = -np.log(p[np.arange(len(z_test)), z_test]).mean()
    print('%-13s accuracy %.3f   log loss %.3f' % (name, (p.argmax(1) == z_test).mean(), logloss[name]))
```

## 3. Did it learn to use context?

Feed the same hand cue `b = +1.5` after a gaze cue of `+1.5` or `-1.5`, and measure only how much the hand cue
moves the log-odds of "handover" against "retrieve" (the gaze cue is informative by itself, so it must be
subtracted out). The Bayes filter must add the same amount either way. A model that uses context adds an
amount whose sign depends on the previous look.

```python
shifts = {name: (shift(m, +1), shift(m, -1)) for name, m in (('Bayes filter', bayes), ('COBRA', cobra))}
fig, ax = plt.subplots(figsize=(5.6, 3.0))
for i, (name, (up, down)) in enumerate(shifts.items()):
    ax.bar(i - 0.18, up, 0.34, color=theme['palette'][0], label='previous a = +1.5' if i == 0 else None)
    ax.bar(i + 0.18, down, 0.34, color=theme['palette'][1], label='previous a = -1.5' if i == 0 else None)
ax.axhline(0, color=theme['text'], lw=1)
ax.set_xticks(range(len(shifts)), list(shifts)); ax.set_ylabel('log-odds shift from b = +1.5')
ax.set_title('Same hand cue, opposite previous look'); ax.legend(frameon=False)
plt.show()
print({k: tuple(round(v, 2) for v in s) for k, s in shifts.items()})
```

## 4. Online, and what the density matrix holds

On a robot the belief is updated one frame at a time. `start()` returns an online belief; its l1 coherence
says how much phase information the density matrix still carries.

```python
episode = x_test[np.flatnonzero(z_test == 0)[0]]
belief = cobra.start()
history, coherence = [], []
for frame in episode:
    history.append(belief.update(frame)); coherence.append(belief.coherence)
history = np.array(history)
fig, ax = plt.subplots(figsize=(5.6, 2.8))
for k in range(3):
    ax.plot(range(1, 5), history[:, k], 'o-', color=theme['palette'][k], label=INTENTS[k])
ax.set_xlabel('frame'); ax.set_ylabel('P(intent)'); ax.set_title('Online belief for a hand-over episode')
ax.legend(frameon=False)
plt.show()
print('l1 coherence per frame:', np.round(coherence, 3), '| learned dephasing g = %.2f' % cobra.dephasing)
assert np.allclose(history[-1], cobra.predict_proba(episode[None])[0])      # online equals batch
```

## 5. From belief to action

The robot can hand over, wait or ask. Handing over to someone who is not receiving costs a lot. Both filters
use the same cost matrix and the same rule.

```python
cost = np.array([[0, 6, 10], [1.5, 0, 0], [0.8, 0.8, 0.8]])      # hand over, wait, ask
costs = {}
for name, m in (('Bayes filter', bayes), ('COBRA', cobra)):
    act = min_cost_action(m.predict_proba(x_test), cost)
    costs[name] = cost[act, z_test]
    print('%-13s expected cost %.3f   ask rate %.2f' % (name, costs[name].mean(), (act == 2).mean()))
diff = costs['COBRA'] - costs['Bayes filter']
print('mean cost difference %.3f, standard error %.3f' % (diff.mean(), diff.std() / np.sqrt(len(diff))))
```

Better beliefs do not automatically mean cheaper actions: here the cost difference is within its standard
error, because the cost rule only changes the action when a belief crosses a threshold.

The statements below are checked by this cell every time the documentation is built:

```python
b_up, b_down = shifts['Bayes filter']
c_up, c_down = shifts['COBRA']
assert abs(b_up - b_down) < 0.2                      # the commutative filter adds the same evidence either way
assert c_up > 1.0 and c_down < -1.0                  # COBRA's evidence flips sign with the previous look
assert logloss['COBRA'] < logloss['Bayes filter'] - 0.02       # and its beliefs are better calibrated to the data
assert abs(diff.mean()) < 3 * diff.std() / np.sqrt(len(diff))    # but the cost gap here is within noise
print('checked')
```

**Summary.** On a task built to contain a context effect, a filter whose measurement basis depends on the
previous frame uses the hand cue the way the data require, and a commutative filter cannot, which shows up as a lower log
loss. The gain is modest and specific: the decision cost is not clearly lower; with little training data COBRA is
worse than the Bayes filter; training sometimes misses the context mechanism (it needs several restarts); the
coherence of the learned belief is small; and on a control task with no context effect it is slightly worse
than the classical filter. The full benchmark, with
ablations and paired tests, is in `benchmarks/context_intent_benchmark.py`.

References: Busemeyer & Bruza (2012), *Quantum Models of Cognition and Decision*, Cambridge University Press;
Kraus (1983), *States, Effects and Operations*, Springer.
