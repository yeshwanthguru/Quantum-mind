# Core: states, measurement, fitting and comparison

Everything else in the library is built from the definitions on this page.

## Notation

| Symbol | Meaning |
|---|---|
| $\psi \in \mathbb{R}^d$ or $\mathbb{C}^d$, $\lVert\psi\rVert = 1$ | belief state (pure) |
| $\rho = \lvert\psi\rangle\langle\psi\rvert$, $\rho \succeq 0$, $\operatorname{tr}\rho = 1$ | density matrix (pure or mixed) |
| $P = P^\dagger = P^2$ | orthogonal projector of a "yes" answer; $\bar P = I - P$ is "no" |
| $\theta$ | vector of model parameters |
| $\mathcal{D} = \{c: n_c\}$ | data: outcome counts $n_{c,o}$ per condition $c$ |

## Born rule and Lüders rule

Answering a yes/no question with projector $P$ gives

$$
p(\text{yes}) = \lVert P\psi\rVert^2 = \langle\psi\rvert P\lvert\psi\rangle,
\qquad
\psi \;\mapsto\; \psi' = \frac{P\psi}{\lVert P\psi\rVert}.
$$

For a sequence of questions $q_1, \dots, q_m$ with answers $a_k \in \{0, 1\}$ and projectors
$P_k^{(1)} = P_k$, $P_k^{(0)} = I - P_k$, applying the rule repeatedly gives

$$
p(a_1, \dots, a_m) = \bigl\lVert P_m^{(a_m)} \cdots P_2^{(a_2)} P_1^{(a_1)}\, \psi \bigr\rVert^2 .
$$

If two projectors do not commute, $P_B P_A \neq P_A P_B$, the joint probability depends on the order:
this single fact produces question-order effects, conjunction fallacies and interference.
(`core.linalg.luders`, `core.linalg.sequence_probabilities`)

The projector onto the span of vectors $v_1, \dots, v_r$ is $P = QQ^\dagger$, where the columns of $Q$ are
an orthonormal basis of the span (QR decomposition). (`core.linalg.projector`)

## Parameterising states and subspaces

A real unit vector in $\mathbb{R}^d$ is written with hyperspherical angles,

$$
v_i = \cos\theta_i \prod_{j<i}\sin\theta_j \quad (i < d), \qquad v_d = \prod_{j<d}\sin\theta_j ,
$$

and an orthonormal frame with $d(d-1)/2$ Givens rotations $G_{ij}(\phi_{ij})$ (rotation by $\phi_{ij}$ in
the $(i, j)$ plane), $F = \prod_{i<j} G_{ij}(\phi_{ij})$. (`angles_to_unit`, `givens_frame`)

## Mixed states, dephasing and open-system dynamics

Partial dephasing in a basis of projectors $\{P_k\}$ with strength $s \in [0, 1]$:

$$
\rho \mapsto (1 - s)\,\rho + s \sum_k P_k \rho P_k .
$$

The Lindblad (GKSL) equation with Hamiltonian $H$, jump operators $L_k$ and rates $\gamma_k \ge 0$:

$$
\dot\rho = \mathcal{L}\rho = -i[H, \rho] + \sum_k \gamma_k \Bigl(L_k \rho L_k^\dagger - \tfrac12\{L_k^\dagger L_k, \rho\}\Bigr),
\qquad \rho(t) = e^{t\mathcal{L}}\rho(0).
$$

The library builds $\mathcal{L}$ as a $d^2 \times d^2$ matrix acting on the row-stacked $\rho$ and uses the
matrix exponential. (`lindblad_superoperator`, `evolve_density`)

## Parameters and their transforms

Each parameter has a kind that maps it to an unconstrained value $z$ used by the optimiser:

| Kind | Range | Map from $z$ |
|---|---|---|
| `prob` | $(0, 1)$ | $p = 1/(1 + e^{-z})$ |
| `positive` | $(0, \infty)$ | $x = e^{z}$ |
| `bounded` | $(\ell, h)$ | $x = \ell + (h - \ell)/(1 + e^{-z})$ |
| `angle`, `real` | $\mathbb{R}$ | $x = z$ |

## Likelihood

For multinomial models (counts of outcomes), with predicted probabilities $p_{c,o}(\theta)$,

$$
\log L(\theta) = \sum_c \sum_o n_{c,o}\log p_{c,o}(\theta)
$$

(the multinomial coefficients are constant and omitted; probabilities are clipped at $10^{-12}$).

For judgement models (observed mean values $y_c$ against predictions $\hat y_c(\theta)$) the loss is the sum
of squared errors $\mathrm{SSE} = \sum_c (y_c - \hat y_c)^2$, and the Gaussian log-likelihood with the
variance profiled out is used for comparison,

$$
\log L = -\frac{n}{2}\Bigl[\log\Bigl(2\pi\,\frac{\mathrm{SSE}}{n}\Bigr) + 1\Bigr].
$$

## Fitting and comparison

$\hat\theta = \arg\max_\theta \log L(\theta)$ by Nelder-Mead from several random starts (and over all
discrete structures, such as projector ranks). Models are ranked by

$$
\mathrm{AIC} = 2k - 2\log L(\hat\theta), \qquad \mathrm{BIC} = k\log n - 2\log L(\hat\theta),
$$

with $k$ fitted parameters and $n$ observations (lower is better). The BIC weight of model $m$ is

$$
w_m = \frac{\exp(-\tfrac12\Delta_m)}{\sum_{m'}\exp(-\tfrac12\Delta_{m'})}, \qquad \Delta_m = \mathrm{BIC}_m - \min_{m'}\mathrm{BIC}_{m'} .
$$

**Model recovery.** For each generating model $g$, simulate $R$ data sets, fit every candidate, and record
$\Pr(\text{selected} = m \mid g)$. A design can tell the models apart when this matrix is close to the
identity.

**Individual fits.** For people $i = 1, \dots, I$, the group criterion of a model is
$\sum_i \mathrm{BIC}_i$ (each person has their own parameters), compared with the BIC of one pooled fit.

**Distances between distributions.**
$\mathrm{KL}(p\Vert q) = \sum_i p_i\log(p_i/q_i)$ and $\mathrm{TVD}(p, q) = \tfrac12\sum_i\lvert p_i - q_i\rvert$.

Reference: Burnham & Anderson (2002); Busemeyer & Bruza (2024), chapters 2-3.
