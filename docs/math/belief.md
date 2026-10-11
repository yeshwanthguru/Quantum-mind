# COBRA belief filter

COBRA (`quantum_mind.belief.COBRA`) keeps a robot's belief about a person as a density matrix and lets the
previous sensor frame rotate the basis in which the next frame is read.

## The model

Hypotheses $z = 1, \dots, d$ (for example the person's intent) span a $d$-dimensional Hilbert space. The belief
is a density matrix $\rho$ (Hermitian, positive semidefinite, $\operatorname{tr}\rho = 1$), started at
$\rho_0 = \operatorname{diag}(\pi)$. A frame $x_t \in \mathbb{R}^m$ updates it in two steps.

**1. Context-rotated generalised measurement.** With evidence weights $w(x_t) = \exp\bigl(\tfrac12 (W x_t + b)\bigr)$
and a rotation driven by the *previous* frame,

$$
U(x_{t-1}) = \exp\Bigl(-i\bigl[H_0 + \textstyle\sum_{k=1}^m \tanh(\beta\,x_{t-1,k})\,H_k\bigr]\Bigr), \qquad
M_t = U(x_{t-1})\,\operatorname{diag}\bigl(w(x_t)\bigr)\,U(x_{t-1})^\dagger ,
$$

$$
\rho \;\leftarrow\; \frac{M_t\,\rho\,M_t}{\operatorname{tr}(M_t\,\rho\,M_t)} .
$$

$M_t$ is Hermitian and positive, so the update keeps $\rho$ a valid density matrix; $x_0 := 0$. The
$H_0, \dots, H_m$ are learned Hermitian matrices.

**2. Partial dephasing.**

$$
\rho \;\leftarrow\; (1 - g)\,\rho + g\,\operatorname{diag}(\rho), \qquad g \in (0, 1) \text{ learned.}
$$

**Born rule.** The probability of hypothesis $z$ after frame $t$ is $P_t(z) = \rho_{zz}$. The l1 coherence
$\sum_{i \ne j} \lvert\rho_{ij}\rvert$ measures how much phase information the belief still carries.

## Classical limits

* **$H_k = 0$.** Then $U = I$ and $M_t$ is diagonal, so the filter is the Bayes filter
  $P_t(z) \propto \pi_z \prod_{s \le t} e^{(W x_s + b)_z}$ (naive Bayes with log-linear likelihoods, which
  includes the shared-variance Gaussian). Reordering the frames cannot change the belief.
* **$g = 1$.** The off-diagonals are erased after every frame. The previous frame still changes the meaning
  of the next (through $U$), but through classical mixing only: the populations evolve by a stochastic map
  $P'_i \propto \sum_j \lvert (M_t)_{ij}\rvert^2 P_j$. This is a classical, incoherent context-dependent filter.

Comparing the full model with these two limits separates the effect of context from the effect of phase
coherence. The benchmark in `benchmarks/context_intent_benchmark.py` does so.

## Training

All parameters $\theta = (H_0..H_m, W, b, \log\pi, \text{logit}\,g)$ minimise the frame-weighted cross-entropy

$$
\mathcal{L}(\theta) = \frac{1}{N}\sum_n \sum_{t=1}^{T} \omega_t \bigl(-\log P_t(z_n)\bigr) + \lambda\lVert\theta\rVert^2,
\qquad \omega_t \propto t,
$$

by Adam with central finite-difference gradients. The evidence step starts at the classical Gaussian
log-likelihood. The rotation and the evidence weights must grow together (a product of two near-zero terms is
a saddle), so training uses several symmetry-breaking restarts and keeps the one with the best validation loss.

## Decision

`min_cost_action(P, C)` returns $\arg\min_a \sum_z C_{a,z}\,P(z)$ for a cost matrix $C$; including an "ask" row
with constant cost lets the robot hedge when the belief is spread out.
