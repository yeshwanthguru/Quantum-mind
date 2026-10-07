# Robot decision layer

## Human-model ensemble

Fit $M$ human models (quantum-like, Bayes, anchoring) to the answers so far and weight them by BIC,
$w_m \propto \exp\bigl(-\tfrac12(\mathrm{BIC}_m - \min\mathrm{BIC})\bigr)$. The predicted answer distribution is
the mixture $\bar p = \sum_m w_m p_m$ and its uncertainty has two parts, in bits:

$$
H(\bar p) = -\sum_o \bar p_o \log_2 \bar p_o, \qquad
\mathrm{JS}_w = H(\bar p) - \sum_m w_m H(p_m) \ \ \text{(model disagreement)} .
$$

`total_bits` $= H(\bar p) + \mathrm{JS}_w$. (`HumanModelEnsemble`)

**Unprimed rates.** The "yes" rate of a question before the other question can influence it is
estimated from the people who answered it first (`estimate_unprimed_rates`).

## Ask or act

With success probability $p$ for acting now, error cost $C_e$ and asking cost $C_a$ (asking is assumed to
remove the doubt):

$$
\mathbb{E}[\text{cost of acting}] = C_e\,(1 - p), \qquad
\text{ask} \iff C_e(1 - p) > C_a \ \ \text{or}\ \ \mathrm{JS}_w > \delta_{\max},
$$

so the break-even confidence is $p^* = 1 - C_a/C_e$. (`ask_or_act`)

**Risk-aware version** (`risk_aware_ask_or_act`): $C_e$ comes from the hazard level
($2, 10, 100, 1000$ for low, medium, high, critical), the confidence used is
$p = \min(\hat p, p_{\mathrm{lower}})$ when a lower bound is available, and the robot also asks whenever
$1 - p > r_{\max}$.

## Calibration measures

Bin the confidences $c_i$ into $B$ equal-width bins $\mathcal{B}_b$; with accuracy $\mathrm{acc}_b$ and mean
confidence $\mathrm{conf}_b$ in each bin,

$$
\mathrm{ECE} = \sum_b \frac{\lvert\mathcal{B}_b\rvert}{n}\,\bigl\lvert \mathrm{acc}_b - \mathrm{conf}_b \bigr\rvert, \qquad
\mathrm{MCE} = \max_b \bigl\lvert \mathrm{acc}_b - \mathrm{conf}_b \bigr\rvert, \qquad
\mathrm{Brier} = \frac1n\sum_i (c_i - y_i)^2 .
$$

## Recalibration

- **Platt scaling**: $\hat c = \sigma\bigl(a\,\mathrm{logit}(c) + b\bigr)$, with $a, b$ fitted by logistic
  regression on held-out data.
- **Temperature scaling**: $\hat p = \mathrm{softmax}(\log p / T)$, with one $T > 0$ that minimises the
  held-out negative log-likelihood; $T > 1$ softens over-confident outputs.
- **Isotonic regression**: the non-decreasing step function $g$ minimising $\sum_i (g(c_i) - y_i)^2$,
  found by the pool-adjacent-violators algorithm.

## Split conformal prediction sets

On calibration data $(x_i, y_i)_{i=1}^n$ the non-conformity score is $s_i = 1 - \hat p_{y_i}(x_i)$. With

$$
\hat q = s_{(\lceil (n + 1)(1 - \alpha)\rceil)}
$$

(the $\lceil (n+1)(1-\alpha)\rceil$-th smallest score), the set $C(x) = \{y : 1 - \hat p_y(x) \le \hat q\}$
satisfies $\Pr\{y \in C(x)\} \ge 1 - \alpha$ for exchangeable data, whatever the classifier. A set with
more than one label is a reason to ask.

**Adaptive sets for one person's answers.** Within an interaction the answers are not exchangeable,
because the person changes. Adaptive conformal inference keeps a working level $\alpha_t$: the set at
time $t$ uses the $\lceil (n_t+1)(1-\alpha_t)\rceil$-th smallest of the $n_t$ scores seen so far (every
answer if $\alpha_t \le 0$ or no score yet, none if $\alpha_t \ge 1$), and after the answer

$$
\alpha_{t+1} = \alpha_t + \gamma\,(\alpha - \mathrm{err}_t), \qquad \mathrm{err}_t = \mathbb 1\{y_t \notin C_t\}.
$$

For any sequence of answers, $\bigl|\tfrac1T\sum_{t=1}^T \mathrm{err}_t - \alpha\bigr| \le
\dfrac{\max(\alpha_1, 1-\alpha_1) + \gamma}{\gamma T}$ (Gibbs and Candès, 2021). (`AdaptiveConformalSets`)

## Exact bounds and online monitoring

For $k$ successes in $n$ trials, the Clopper-Pearson interval at level $1 - \alpha$ is

$$
\Bigl[\ \mathrm{Beta}^{-1}\bigl(\tfrac\alpha2;\ k,\ n - k + 1\bigr),\ \ \mathrm{Beta}^{-1}\bigl(1 - \tfrac\alpha2;\ k + 1,\ n - k\bigr)\Bigr].
$$

`CalibrationMonitor` keeps these counts per confidence bin and returns the bin's success rate and its
lower bound, which `risk_aware_ask_or_act` uses as $p_{\mathrm{lower}}$.

## Confidence gate

With calibrated confidences $\hat c_k$ and costs $\mathrm{cost}_k$,

$$
k^* = \arg\max_k\ \hat c_k - \lambda\,\mathrm{cost}_k .
$$

With a budget $\bar B$ over a window of $W$ calls, only modules whose cost fits the remaining headroom
$\bar B\,(w + 1) - \sum_{\text{last } w}\mathrm{cost}$ are eligible. (`ConfidenceGate`)

## Meta-calibrated gate

Each module reports $c_k = p_k + b_k + \varepsilon$ with a task-specific offset $b_k$. After a call with
outcome $y \in \{0, 1\}$, $c_k - y$ is an unbiased sample of $b_k$, and the gate keeps

$$
\hat b_k = \frac{n_0\,\mu_k + \sum (c_k - y)}{n_0 + n_k}, \qquad
k^* = \arg\max_k\ c_k - \hat b_k - \lambda\,\mathrm{cost}_k
$$

(with probability $\epsilon$ it explores at random). The prior is meta-learned from earlier tasks
(empirical Bayes, `meta_fit_offsets`): $\mu_k$ is the mean of the per-task offset estimates, and

$$
n_0 = \frac{\sigma^2_{\text{within}}}{\sigma^2_{\text{between}}}, \qquad
\sigma^2_{\text{between}} = \operatorname{Var}_{\text{tasks}}(\hat b_k) - \frac{\sigma^2_{\text{within}}}{\bar n_k},
$$

clipped to $[1, 200]$: many pseudo-observations when tasks agree, few when they differ.

## Question planning (value of information)

With a posterior $p(h)$ over hypotheses, an answer model $p(a \mid h, q, \text{history})$ and error cost
$C_e$,

$$
\mathrm{VOI}(q) = C_e\bigl(1 - \max_h p(h)\bigr) - \sum_{a\in\{0,1\}} p(a \mid q)\, C_e\bigl(1 - \max_h p(h \mid a)\bigr),
\qquad p(a \mid q) = \sum_h p(h)\,p(a \mid h, q).
$$

The planner asks $\arg\max_q \mathrm{VOI}(q)$ when it exceeds the asking cost and otherwise acts on the
MAP hypothesis; after each answer it updates $p(h) \propto p(h)\,p(a \mid h, q, \text{history})$. The
information gain $I(q) = H(h) - \mathbb{E}_a H(h \mid a)$ is also reported. (`QuestionPlanner`)

**Answer models.** *Projective*: hypothesis $h$ is a state $\psi_h$, questions are projectors, and the
likelihood of an answer sequence is $\lVert P^{(a_m)}_{q_m}\cdots P^{(a_1)}_{q_1}\psi_h\rVert^2$ (order
dependent). *Independent*: a fixed table $r_{hq} = P(\text{yes} \mid h, q)$ and
$\prod_k r_{hq_k}^{a_k}(1 - r_{hq_k})^{1 - a_k}$ (order free). `order_free()` builds the independent model
with each question's first-position rates.

## Personalisation (partial pooling)

Each person $i$ has free parameters $x_i$ (the unconstrained transforms of the model parameters). With a
population prior $\mathcal{N}(\mu, \Sigma)$,

$$
\hat x_i = \arg\min_x\ -\log L_i(x) + \tfrac12 (x - \mu)^T\Sigma^{-1}(x - \mu).
$$

With few answers $\hat x_i \approx \mu$; with many, it approaches the person's maximum-likelihood estimate.
The prior is estimated from per-person fits, $\hat\mu = \bar x$, $\hat\Sigma = \operatorname{Cov}(x_i)$ plus a
small ridge. (`PopulationPrior`, `fit_map`)

With `method='online'` the person's parameters keep a full posterior: particles
$x^{(j)} \sim \mathcal N(\hat\mu, \hat\Sigma)$ reweighted by every answer, as in
{class}`~quantum_mind.core.online.OnlinePersonModel` (equations on the {doc}`core page <core>`), which also
gives credible intervals for the person's parameters.

## Trust-aware hand-over

With trust belief $p_t = P(\text{trust})$ (from the open-system or Markov belief), failure cost $C_f$,
slow cost $C_s$, slow factor $s$, asking cost $C_a$ and waiting cost $C_w$:

$$
\begin{aligned}
\mathbb{E}C(\text{hand over}) &= C_f\,(1 - p_t), \qquad
\mathbb{E}C(\text{slow}) = C_s + s\,C_f\,(1 - p_t), \\
\mathbb{E}C(\text{ask}) &= C_a + p_t \cdot 0 + (1 - p_t)\,\min\{C_f,\ C_s + s\,C_f\}, \\
\mathbb{E}C(\text{wait}) &= C_w + \min\{\mathbb{E}C(\text{hand over}),\ \mathbb{E}C(\text{slow})\}.
\end{aligned}
$$

Asking is a one-step look-ahead: the answer collapses the belief to trust ($p_t = 1$) or distrust
($p_t = 0$), after which the better hand-over is chosen. The
policy takes the cheapest action; outcomes and answers update the belief (an answer collapses the
open-system belief). (`TrustAwareHandover`)

References: Howard (1966); Guo et al. (2017); Platt (1999); Zadrozny & Elkan (2002); Angelopoulos & Bates
(2023); Clopper & Pearson (1934); Efron & Morris (1975); Gelman & Hill (2007).

## Trust-aware task allocation

People $a$ and tasks $t$ with effort $C_{a,t}$, trust $\tau_a$ in the robot (from a trust model, or the lower
end of a credible interval for a cautious plan), reliance $r_t\in[0,1]$ of each task on the robot and cost
$F$ of a failed robot-assisted step. The expected cost is $\tilde C_{a,t} = C_{a,t} + F\,r_t\,(1-\tau_a)$, and
the allocation minimises

$$
E(x) = \sum_{a,t}\tilde C_{a,t}\,x_{a,t} + \lambda\sum_a\Bigl(\sum_t x_{a,t}\Bigr)^2
     + A\sum_t\Bigl(\sum_a x_{a,t} - 1\Bigr)^2, \qquad x_{a,t}\in\{0,1\},
$$

a QUBO (using $x^2 = x$, the workload square adds $\lambda$ to each diagonal entry and $2\lambda$ to each pair
of one person's tasks). For every valid assignment the penalty vanishes and $E$ is the expected cost plus
$\lambda$ times the sum of squared loads. (`trust_aware_allocation`, `expected_costs`)

References: Gibbs, I., & Candès, E. (2021). Adaptive conformal inference under distribution shift.
*Advances in Neural Information Processing Systems*, 34. Lucas, A. (2014). Ising formulations of many NP
problems. *Frontiers in Physics*, 2, 5.
