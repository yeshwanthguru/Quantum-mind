# Quantum-cognition core against classical models of context and order

Simulation. 5 seeds; 2500 training and 4000 test sequences per seed; 3 steps, 3 hypotheses. Gap = test log-loss minus the oracle log-loss (0 is perfect; lower is better); mean ± std over seeds. p: paired sign-flip test on per-sequence log-loss, each model against the quantum core.

Parameters: Bayes 37, Context weights 17, Markov context 33, GRU 237, Quantum core 37.

## Generator: control

Oracle log-loss 0.467.

| Model | Gap to oracle | p vs quantum core |
|---|---|---|
| Bayes | 0.0041 ± 0.0007 | 0.0002 |
| Context weights | 0.0043 ± 0.0016 | 0.0002 |
| Markov context | 0.0113 ± 0.0050 | 0.1732 |
| GRU | 0.0475 ± 0.0046 | 0.0002 |
| Quantum core | 0.0099 ± 0.0063 | – |

Best other model: Bayes. Quantum core rotation size (rms): 0.078.

## Generator: context weights

Oracle log-loss 0.467.

| Model | Gap to oracle | p vs quantum core |
|---|---|---|
| Bayes | 0.2611 ± 0.0560 | 0.0002 |
| Context weights | 0.0055 ± 0.0027 | 0.0002 |
| Markov context | 0.2551 ± 0.0496 | 0.0002 |
| GRU | 0.0454 ± 0.0104 | 0.0002 |
| Quantum core | 0.1219 ± 0.0424 | – |

Best other model: Context weights. Quantum core rotation size (rms): 0.342.

## Generator: markov context

Oracle log-loss 0.557.

| Model | Gap to oracle | p vs quantum core |
|---|---|---|
| Bayes | 0.2448 ± 0.1038 | 0.0002 |
| Context weights | 0.2352 ± 0.1105 | 0.0002 |
| Markov context | 0.0076 ± 0.0020 | 0.0002 |
| GRU | 0.0458 ± 0.0076 | 0.0002 |
| Quantum core | 0.1368 ± 0.0771 | – |

Best other model: Markov context. Quantum core rotation size (rms): 0.279.

## Generator: quantum order

Oracle log-loss 0.811.

| Model | Gap to oracle | p vs quantum core |
|---|---|---|
| Bayes | 0.1674 ± 0.1169 | 0.0002 |
| Context weights | 0.1034 ± 0.0332 | 0.0002 |
| Markov context | 0.1539 ± 0.1232 | 0.0002 |
| GRU | 0.0643 ± 0.0202 | 0.0002 |
| Quantum core | 0.0132 ± 0.0055 | – |

Best other model: GRU. Quantum core rotation size (rms): 0.529.

## Generator: random GRU

Oracle log-loss 0.829.

| Model | Gap to oracle | p vs quantum core |
|---|---|---|
| Bayes | 0.1598 ± 0.0395 | 0.0002 |
| Context weights | 0.1439 ± 0.0361 | 0.0002 |
| Markov context | 0.1110 ± 0.0254 | 0.0002 |
| GRU | 0.0500 ± 0.0134 | 0.0002 |
| Quantum core | 0.1201 ± 0.0413 | – |

Best other model: GRU. Quantum core rotation size (rms): 0.428.

The "quantum order" generator is built from the quantum core's own form, so winning there shows the model can represent non-commuting context, not that people behave this way. A rotation size near 0 on the other generators means the penalty kept the model close to Bayes.
