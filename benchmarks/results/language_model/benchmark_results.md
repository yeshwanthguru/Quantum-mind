# A small robot-instruction language model

Simulation, sentences from a rule-based grammar of 32 words. 3 seeds; 8000 training sentences, 2000 test sentences, 1500 training steps; mean ± std over seeds. Every encoder has the same grounding head and 28992 parameters.

## A. Grounding accuracy on fresh sentences (unambiguous ones)

| Encoder | simple | polite | conjunction | correction | followup | held-out combinations | perplexity |
|---|---|---|---|---|---|---|---|
| bag of words | 0.939 ± 0.030 | 0.932 ± 0.034 | 0.997 ± 0.004 | 0.512 ± 0.005 | 0.920 ± 0.037 | 0.993 ± 0.004 | – |
| transformer, grounding only | 0.993 ± 0.001 | 0.993 ± 0.005 | 0.991 ± 0.013 | 0.988 ± 0.008 | 0.989 ± 0.004 | 0.993 ± 0.004 | – |
| transformer + language modelling | 0.994 ± 0.002 | 0.993 ± 0.007 | 0.994 ± 0.008 | 0.986 ± 0.009 | 0.993 ± 0.011 | 0.992 ± 0.005 | 3.34 |

Held-out combinations: conjunctions such as "the blue last block" whose colour-side pair never appears in training. Perplexity is per word on fresh sentences (a model that guessed uniformly among the 32 words would have 32).

## B. Closed loop: the one-motor VLA drives to the cell the model believes in

150 correction sentences per seed ("go to the red one, no, the blue one"); success = final position within a quarter cell.

| Encoder | Success | p vs bag of words |
|---|---|---|
| bag of words | 0.503 ± 0.032 | – |
| transformer, grounding only | 1.000 ± 0.000 | 0.0002 |
| transformer + language modelling | 1.000 ± 0.000 | 0.0002 |

## C. Fusing a two-turn dialogue on the language model's beliefs

1500 dialogues for fitting and 1500 for testing per seed: an instruction (simple or ambiguous), then a follow-up ("no, the left one") that misspeaks with probability 0.25 after a clear instruction and 0.05 after an ambiguous one. That person is a classical, context-dependent noise model chosen by the author, not data. Log-loss (lower is better) and accuracy of the most likely object.

| Fusion | Log-loss | Accuracy | p vs Bayes |
|---|---|---|---|
| bayes | 0.255 ± 0.012 | 0.879 ± 0.011 | – |
| context weights | 0.161 ± 0.012 | 0.945 ± 0.007 | 0.0002 |
| quantum core | 0.197 ± 0.010 | 0.926 ± 0.009 | 0.0002 |

The dialogue person is classical, so a quantum-like advantage is not expected here and none is claimed; the table shows that the language model's beliefs can feed any of the fusion rules.
