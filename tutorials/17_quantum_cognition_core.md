# The quantum-cognition core

When a person gives a robot a sequence of clues ("the red one", then "no, the left one"), the meaning of a clue can depend on what came
before it. A Bayes filter multiplies likelihoods, which cannot depend on order. The **quantum-cognition core** keeps its belief as a
density matrix and updates it with a measurement whose rotation depends on the *context* of each clue, so order can matter. This tutorial
trains it by gradients, compares it with classical models of context, checks that it reduces to Bayes when there is nothing to find, and says
when it does not help. All data are **simulated**; the "person" is a hidden model we choose, so a win only shows what the model can represent.

You need the optional extra: `pip install "quantum-mind[cognition]"` (JAX and Optax).

```python
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from quantum_mind.viz import use_mpl_style
from quantum_mind.cognition import (QuantumCognitionCore, BayesCore, ContextWeightedCore, GRUCore)
theme = use_mpl_style('dark')
D, F, K, T = 3, 3, 2, 3          # 3 hypotheses, 3 evidence features, 2 contexts, 3 clues per sequence
```

## 1. A simulated person whose answers depend on the order of the clues

Each sequence has three clues `x_t` (evidence features) and a context `c_t` (for example "asked about colour" or "asked about position").
We draw the true hypothesis from a hidden quantum-cognition core with random rotations. Nothing here is a real person. Some hidden people have a weak
order effect (in a quick check of six, one was matched by the classical context-weighted model); we use one with a clear effect, so read the
result below as what the core *can* find, not as how often it is needed.

```python
def clues(n, rng):
    X = rng.standard_normal((n, T, F)).astype(np.float32)
    C = np.eye(K, dtype=np.float32)[rng.integers(0, K, (n, T))]
    return X, C

def hidden_person(seed, rotation):
    core = QuantumCognitionCore(D, F, K)
    p = core.init(jax.random.PRNGKey(seed))
    p['W'] = p['W'] * 4
    p['H'] = rotation * jax.random.normal(jax.random.PRNGKey(seed + 1), p['H'].shape)
    p['phi'] = jax.random.uniform(jax.random.PRNGKey(seed + 2), (D,), maxval=6.28)
    p['g'] = jnp.asarray(-3.0)
    return core, p

def sample(core, p, n, rng):
    X, C = clues(n, rng)
    P = np.asarray(core.probs(p, jnp.asarray(X), jnp.asarray(C), jnp.ones((n, T), bool)))
    y = np.array([rng.choice(D, p=q / q.sum()) for q in P])
    return X, C, y, P

rng = np.random.default_rng(0)
person, params = hidden_person(seed=5, rotation=1.2)
Xa, Ca, ya, _ = sample(person, params, 2000, rng)          # training sequences
Xb, Cb, yb, Pb = sample(person, params, 4000, rng)         # fresh test sequences
oracle = -np.mean(np.log(Pb[np.arange(len(yb)), yb] + 1e-7))
print('best achievable log-loss (the oracle, known because we built the person): %.3f' % oracle)
```

## 2. Fit the core and its classical rivals

All models share the same evidence map; they differ only in how context enters. `BayesCore` has no rotations, `ContextWeightedCore` lets each
context scale how much its evidence counts, `GRUCore` is a small recurrent network with many more parameters.

```python
kw = dict(steps=250, restarts=2, seed=0)
models = {'Bayes': BayesCore(D, F, K, **kw), 'Context weights': ContextWeightedCore(D, F, K, **kw),
          'GRU': GRUCore(D, F, K, hidden=6, **kw), 'Quantum core': QuantumCognitionCore(D, F, K, **kw)}
gap = {}
for name, m in models.items():
    m.fit(Xa, Ca, ya)
    gap[name] = m.log_loss(Xb, Cb, yb) - oracle
    print('%-16s %4d parameters, gap to the oracle %.3f' % (name, m.n_params, gap[name]))
fig, ax = plt.subplots(figsize=(5.6, 2.8))
ax.barh(list(gap), list(gap.values()), color=[theme['palette'][i] for i in (1, 2, 4, 0)])
ax.set_xlabel('test log-loss above the oracle (lower is better)')
ax.set_title('Simulated person with non-commuting context')
plt.show()
```

**Result.** On this person the quantum core is close to the oracle and clearly better than Bayes and the context-weighted model. This is the
generator built from the core's own form, so it shows capacity, not that people behave this way.

```python
assert gap['Quantum core'] < 0.5 * gap['Bayes'] and gap['Quantum core'] < gap['Context weights']
print('quantum core gap %.3f against Bayes %.3f' % (gap['Quantum core'], gap['Bayes']))
```

## 3. What the rotations do: order matters

Take the same three clues and swap the order in which they arrive (each clue keeps its own context). A product of likelihoods cannot notice;
the quantum core can.

```python
core = models['Quantum core']
perm = [2, 0, 1]
p1 = core.predict_proba(Xb[:200], Cb[:200])
p2 = core.predict_proba(Xb[:200][:, perm], Cb[:200][:, perm])
b1 = models['Bayes'].predict_proba(Xb[:200], Cb[:200])
b2 = models['Bayes'].predict_proba(Xb[:200][:, perm], Cb[:200][:, perm])
print('largest change in a probability after reordering the clues: quantum core %.3f, Bayes %.3f'
      % (np.abs(p1 - p2).max(), np.abs(b1 - b2).max()))
print('rotation size the model learned: %.2f; mean interference (l1 coherence): %.3f'
      % (core.rotation_size, core.coherence(Xb[:200], Cb[:200]).mean()))
assert np.abs(p1 - p2).max() > 0.02 and np.abs(b1 - b2).max() < 1e-5
```

## 4. When there is nothing to find, it stays a Bayes filter

Now the person is a plain product of likelihoods (no order effect). The rotations are penalised toward zero, so the core should not invent one.

```python
bayes_person = BayesCore(D, F, K)
pb = bayes_person.init(jax.random.PRNGKey(5))
pb['W'] = pb['W'] * 4
Xa2, Ca2, ya2, _ = sample(bayes_person, pb, 2000, np.random.default_rng(1))
Xb2, Cb2, yb2, Pb2 = sample(bayes_person, pb, 4000, np.random.default_rng(2))
oracle2 = -np.mean(np.log(Pb2[np.arange(len(yb2)), yb2] + 1e-7))
ctrl = QuantumCognitionCore(D, F, K, **kw).fit(Xa2, Ca2, ya2)
print('control: gap to the oracle %.3f, rotation size %.2f' % (ctrl.log_loss(Xb2, Cb2, yb2) - oracle2, ctrl.rotation_size))
assert ctrl.log_loss(Xb2, Cb2, yb2) - oracle2 < 0.03 and ctrl.rotation_size < 0.35
```

## 5. What this does not tell you

**Summary.** The core never beats a classical model that is exactly right for the data (the library's benchmark shows the context-weighted,
Markov and GRU models each winning the data they match), it is a strong few-parameter generalist, and it clearly wins only when the context is
non-commuting. Whether real people are like that is an empirical question: fit all of these models to real sequences and let held-out
log-loss decide. If the rotations stay near zero, the data show no quantum-like order effect.

```python
print('benchmark: benchmarks/results/cognition_core/benchmark_results.md')
```
