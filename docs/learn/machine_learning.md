# Machine learning

Machine learning (ML) builds a model from **data** instead of writing the rules by hand. A model has
**parameters**; learning chooses the parameters that make the model's predictions match the data, and
the real test is how well it predicts data it has **not** seen.

```{mermaid}
flowchart LR
    D[("data<br/>inputs x, labels y")]:::in --> S{"split"}:::op
    S -->|train| F["fit<br/>minimise loss over parameters θ"]:::op
    F --> M["model f(x; θ)"]:::out
    S -->|test| T["evaluate on unseen data<br/>accuracy · log-likelihood · calibration"]:::out
    M --> T
    T -->|compare| B["baseline model"]:::base
    classDef in fill:#0b2a4a,stroke:#79c0ff,color:#e6edf3
    classDef op fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef out fill:#0f2e1a,stroke:#7ee787,color:#e6edf3
    classDef base fill:#262626,stroke:#8b949e,color:#e6edf3,stroke-dasharray:4 3
```

## From the basics

Supervised, unsupervised and reinforcement learning
: **Supervised**: learn a map from inputs to known labels (classify images, predict an answer).
  **Unsupervised**: find structure without labels (clusters, anomalies). **Reinforcement**: learn
  actions from reward (see [reinforcement learning](reinforcement_learning.md)).

Loss and likelihood
: A loss measures how wrong the predictions are. For probabilistic models the usual choice is the
  negative **log-likelihood**; minimising it is **maximum likelihood** estimation. Quantum Mind fits
  every model this way.

Overfitting and model comparison
: A flexible model can fit noise. Use held-out data, or penalise parameters: **AIC** and **BIC** add a
  cost per parameter. A new model is only interesting if it beats a fair **baseline**.

Bias, variance and partial pooling
: With little data per person, a separate model for each person is noisy (high variance) and one model
  for everyone ignores differences (high bias). **Partial pooling** shrinks each person's estimate
  towards the population, which helps most when data are scarce.

Calibration
: A model is calibrated when events it gives 80% happen about 80% of the time. Modern classifiers are
  often over-confident; **temperature scaling**, **Platt scaling** and **isotonic regression** fix
  this, and **conformal prediction** turns scores into sets with guaranteed coverage.

Kernels
: A kernel measures similarity between inputs and lets linear methods learn non-linear boundaries
  (support vector machines). A **quantum kernel** computes the similarity as the overlap of two
  quantum states.

## A small example: maximum likelihood and BIC

Fit the probability of a "yes" answer, then compare a one-parameter model with a zero-parameter one.

```python
import numpy as np
yes, n = 62, 100
p_hat = yes / n                                            # maximum likelihood estimate
ll = yes * np.log(p_hat) + (n - yes) * np.log(1 - p_hat)   # log-likelihood at the estimate
ll0 = n * np.log(0.5)                                      # fixed p = 0.5, no parameters
bic, bic0 = -2 * ll + 1 * np.log(n), -2 * ll0
print(round(bic, 1), round(bic0, 1))                       # lower BIC wins
```

## A small example: is the model calibrated?

```python
from quantum_mind.applications.calibration import expected_calibration_error
rng = np.random.default_rng(0)
conf = rng.uniform(0.5, 1.0, 5000)
correct = rng.random(5000) < conf - 0.1            # over-confident by 0.1
print(round(expected_calibration_error(conf, correct), 3))   # about 0.1
```

## Where Quantum Mind fits

Every model in the library follows the same ML workflow: `fit` by maximum likelihood, `compare` by
AIC/BIC against classical baselines, check parameter `recovery`, and test on held-out data. The
quantum pillar adds variational classifiers and quantum kernels, each next to a classical baseline.

- [Tutorial: introduction (fit and compare)](../tutorials/01_introduction.ipynb)
- [Tutorial: calibration and conformal sets](../tutorials/05_calibration.ipynb)
- [Model atlas: quantum machine learning](../atlas/quantum.md)

## References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.).
  Springer.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks.
  *Proceedings of ICML*, 1321-1330.
- Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations
  and Trends in Machine Learning*, 16(4), 494-591.
- Gelman, A., & Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*.
  Cambridge University Press.
- Schuld, M., & Petruccione, F. (2021). *Machine Learning with Quantum Computers* (2nd ed.). Springer.
