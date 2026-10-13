# A language model trained from scratch on this library's docstrings (CPU)

A 940800-parameter transformer from random weights, 3000 steps (16 minutes on a CPU). No pretrained model of any kind.

| Measure | Value |
|---|---|
| held-out document loss, the model | 2.299 (perplexity 10.0) |
| held-out document loss, a bigram model | 3.979 (perplexity 53.5) |

## Question answering (word-overlap F1 with the true docstring sentence; 1.0 is a perfect copy)

| Questions | Mean F1 | Share with F1 at least 0.8 |
|---|---|---|
| questions seen in training (100) | 0.99 | 99% |
| seen name, new phrasing (100) | 0.66 | 64% |
| names never seen (74) | 0.15 | 9% |

## Examples (greedy answers)

* **What does sensors.mems_accelerometer do?**
  * model: A MEMS accelerometer: fast, noisy, with a drifting bias (illustrative values).
  * truth: A MEMS accelerometer: fast, noisy, with a drifting bias (illustrative values).
* **What does session do?**
  * model: A conversation with a local LLM: history, streaming, and an optional "am I sure?" check.
  * truth: A conversation with a local LLM: history, streaming, and an optional "am I sure?" check.
* **What does statistics.compare_models do?**
  * model: Per-qubit density matrix:class:`QQQiskit clogory.
  * truth: Compare every pair of named probability predictions with paired tests.
* **What does personalisation.population_prior do?**
  * model: Gaussian prior on a model's free (unconstrained) parameters.
  * truth: Convenience: fit every person by maximum likelihood and build the empirical-Bayes prior.

Reading: the model memorises answers it trained on and cannot answer about names it never saw; a model this small with this little text learns the vocabulary and style of the library's documentation, not the knowledge in it.
