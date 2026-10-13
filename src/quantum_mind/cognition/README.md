# `quantum_mind.cognition`: the quantum-cognition core

One trainable model for the library's third thread. A belief over hypotheses is a density matrix; each piece of evidence applies a generalised
measurement `M = U(c) diag(w(x)) U(c)†` whose rotation `U(c) = exp(i Σ c_k H_k)` depends on the **context** `c` (what was asked before, who is
answering, which sensor); the prediction is the Born rule. It replaces the separate COBRA filter and the VLA belief slot with one model trained
by gradients (JAX and Optax, optional extra `cognition`).

```python
import numpy as np
from quantum_mind.cognition import QuantumCognitionCore, BayesCore

# X: (N, steps, features) evidence, C: (N, steps, contexts) one-hot context per step, y: (N,) true hypothesis
core = QuantumCognitionCore(n_hyp=3, n_features=3, n_context=2).fit(X, C, y)
p = core.predict_proba(X_new, C_new)          # (N, 3)
core.rotation_size                            # 0 means the data asked for no quantum effect: it is a Bayes filter
```

What is guaranteed by tests: with zero rotations it is exactly a Bayes filter; the state is always a density matrix; projective measurements obey
the QQ equality while the order still matters; gradients match finite differences; padded steps are ignored.

Classical competitors on the same interface and the same evidence map: `BayesCore` (no rotations), `ContextWeightedCore` (each context scales how
much its evidence counts), `MarkovContextCore` (a context-dependent stochastic transition, the classical analogue of a rotation) and `GRUCore`.

## What the benchmark found (`benchmarks/results/cognition_core/`)

Five generators with a known best-possible log-loss; the table is each model's gap to it (0 is perfect), 5 seeds, 2500 training and 4000 test
sequences, paired sign-flip tests. Parameters: Bayes 37, context weights 17, Markov 33, GRU 237, quantum core 37.

| Truth | Bayes | Context weights | Markov | GRU | Quantum core |
|---|---|---|---|---|---|
| no context effect | **0.004** | 0.004 | 0.011 | 0.048 | 0.010 |
| context scales the evidence | 0.261 | **0.006** | 0.255 | 0.045 | 0.122 |
| context-dependent classical transitions | 0.245 | 0.235 | **0.008** | 0.046 | 0.137 |
| non-commuting (quantum) context | 0.167 | 0.103 | 0.154 | 0.064 | **0.013** |
| random recurrent network | 0.160 | 0.144 | 0.111 | **0.050** | 0.120 |

How to read it, plainly:

* **No single best model.** Each classical model wins the generator it matches exactly. The quantum core never beats a classical model that is
  exactly right for the data.
* **It is a strong generalist for its size.** With 37 parameters it recovers most of the context effect in every generator (Bayes is far worse
  whenever context matters), it stays close to Bayes when there is nothing to find (rotation size 0.08), and on non-commuting context it reaches
  0.013 from the optimum where the 237-parameter GRU reaches 0.064.
* **The non-commuting generator is built from the core's own form.** That win shows capacity, not that people behave this way.
* **It is not the most accurate where a matched or generic model fits.** On the random recurrent network the GRU is best (0.050 against 0.120).
* The decision that matters is empirical: fit all of them to real sequences from people and robots and let held-out log-loss decide. If the
  quantum core's rotations stay near zero or it loses to the context-weighted model, the data show no quantum-like order effect.

All data are simulated. No real people or robots were involved.
