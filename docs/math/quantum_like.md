# Quantum-like families

Each family states the quantum-like model and the classical baselines it is compared with, in the
form the code computes. All of them are fitted and compared as on the [core page](core.md).

## Question-order effects

**Design.** Two yes/no questions $A$, $B$ asked in either order. Outcomes per order are
$(yy, yn, ny, nn)$, the first letter being the first answer.

**3D quantum-like model** (`QuantumOrderModel`). State $\psi = e_1 \in \mathbb{R}^3$, question vectors

$$
u_A = (\cos a, \sin a, 0), \qquad u_B = (\cos b,\ \sin b\cos g,\ \sin b\sin g).
$$

A rank-1 "yes" is the ray $P_q = u_q u_q^T$; a rank-2 "yes" is the plane $P_q = I - u_q u_q^T$. Then

$$
p_{AB}(yy) = \lVert P_B P_A\psi\rVert^2, \quad p_{AB}(yn) = \lVert (I - P_B)P_A\psi\rVert^2, \quad
p_{AB}(ny) = \lVert P_B (I - P_A)\psi\rVert^2, \quad p_{AB}(nn) = \lVert (I - P_B)(I - P_A)\psi\rVert^2,
$$

and the same with $A$ and $B$ exchanged for the order $BA$. Three parameters $(a, b, g)$; the ranks are a
structural choice fitted over $\{1, 2\}^2$.

**4D model that nests Bayes** (`QuantumOrderModel4D`). Basis $e_1..e_4$ = $(y,y), (y,n), (n,y), (n,n)$ of a
classical $2\times 2$ table; $\psi(t_1, t_2, t_3) \in \mathbb{R}^4$ from hyperspherical angles;
$P_A = \operatorname{span}(e_1, e_2)$ and

$$
P_B = U(\varphi)\,\operatorname{span}(e_1, e_3)\,U(\varphi)^T,
$$

where $U(\varphi)$ rotates by $\varphi$ in the planes $(e_1, e_4)$ and $(e_2, e_3)$. For $\varphi = 0$ the
projectors commute and $p(x) = \psi_x^2$ is an ordinary joint table, i.e. exactly the Bayesian model; for
$\varphi \neq 0$ the order matters.

**QQ equality.** For any state and any projectors,

$$
q = \bigl[p_{AB}(yn) + p_{AB}(ny)\bigr] - \bigl[p_{BA}(yn) + p_{BA}(ny)\bigr] = 0 .
$$

`qq_test` uses $z = \hat q / \widehat{\mathrm{se}}(\hat q)$ with the binomial standard errors of the two
"different answers" proportions, $\widehat{\mathrm{se}}^2 = \hat s_{AB}(1 - \hat s_{AB})/N_{AB} + \hat s_{BA}(1 - \hat s_{BA})/N_{BA}$,
and a two-sided normal $p$-value.

**Baselines.**

- *Bayes* (`BayesOrderModel`): marginals $p_A, p_B$ and a correlation $\rho \in (-1, 1)$ scaled between the
  Fréchet bounds; the same joint table in both orders.
- *Anchoring* (`AnchoringOrderModel`): the first answer has its first-position rate $p_1$; for weight
  $w \ge 0$, $P(\text{2nd yes} \mid \text{1st yes}) = p_2 + w(1 - p_2)$ and
  $P(\text{2nd yes} \mid \text{1st no}) = p_2(1 - w)$ (mirror image for $w < 0$), with a weight per
  question.
- *Saturated* (`SaturatedOrderModel`): a free softmax distribution per order (6 parameters), the best
  possible fit.

## Quantum-like Bayesian networks

A Bayesian network gives each full configuration $x$ the probability $P(x) = \prod_i P(x_i \mid \mathrm{pa}_i)$.
The quantum-like network gives it the amplitude $\sqrt{P(x)}\,e^{i\theta(x)}$ and sums amplitudes over
the hidden configurations $h$ (`QLBNModel`):

$$
P_q(v \mid e) = \frac{\bigl\lvert \sum_h \sqrt{P(v, e, h)}\, e^{i\theta_h} \bigr\rvert^2}
{\sum_{v'} \bigl\lvert \sum_h \sqrt{P(v', e, h)}\, e^{i\theta_h} \bigr\rvert^2}
= \frac{\sum_h P(v, e, h) + 2\sum_{h<h'} \sqrt{P(v,e,h)P(v,e,h')}\cos(\theta_h - \theta_{h'})}{\cdots}.
$$

The phases are the parameters; the classical network (`ClassicalBNModel`) has none.

## Belief dynamics

**Walks over $N$ ordered confidence states** $x_j = j/(N - 1)$, grouped into $K$ response categories; the
initial distribution is a Gaussian bump around $0.5$.

- *Markov walk*: $\dot\phi = K\phi$ with tridiagonal $K$, up rate $\alpha = \mu\gamma$, down rate
  $\beta = (1 - \mu)\gamma$, reflecting ends; $\phi(t) = e^{Kt}\phi(0)$.
- *Quantum walk*: $\psi(t) = e^{-iHt}\psi(0)$, $\psi(0) = \sqrt{\phi(0)}$, with
  $H_{jj} = \mu\,x_j$ and $H_{j,j+1} = H_{j+1,j} = \sigma$; $P(\text{state } j) = \lvert\psi_j(t)\rvert^2$.
- *Open-system walk*: the Lindblad equation with the same $H$ and dephasing jumps $L_j = \lvert j\rangle\langle j\rvert$
  at rate $\lambda$; $\lambda = 0$ is the quantum walk, large $\lambda$ behaves like a Markov walk.

A judgement at $t_1$ collapses the state onto the chosen category $C$: $\psi \mapsto P_C\psi/\lVert P_C\psi\rVert$
(or $\rho \mapsto P_C\rho P_C/\operatorname{tr}(P_C\rho)$). This is how an intermediate judgement changes a
later one; in the Markov walk the marginal at $t_2$ is unchanged on average.

**Trust (or any belief) updated by events and queried as yes/no.**

*Open-system belief* (`OpenSystemBelief`): a real qubit with $\lvert 0\rangle$ = "yes/trust",
$\rho_0 = vv^T$, $v = (\cos\tfrac{\phi_0}{2}, \sin\tfrac{\phi_0}{2})$. Each event applies
$R_y(\alpha) = \begin{pmatrix}\cos\frac\alpha2 & -\sin\frac\alpha2 \\ \sin\frac\alpha2 & \cos\frac\alpha2\end{pmatrix}$ with
$\alpha = -a_+$ after a success and $\alpha = a_-$ after a failure, then dephases:

$$
\rho \mapsto R\rho R^T, \qquad \rho_{01},\ \rho_{10} \mapsto (1 - \gamma)\,\rho_{01},\ (1 - \gamma)\,\rho_{10} .
$$

A query answers "yes" with probability $\rho_{00}$ and collapses $\rho$ to $\lvert 0\rangle\langle 0\rvert$ or
$\lvert 1\rangle\langle 1\rvert$. The model predicts the full distribution over answer sequences.

*Markov belief* (`MarkovBelief`): $p \mapsto p + (1 - p)\,u$ after a success and $p \mapsto p\,(1 - d)$ after
a failure; a query reads $p$ without changing it.

The **question effect** is $P(\text{yes at } t \mid \text{asked at } t' < t) - P(\text{yes at } t \mid \text{not asked})$,
marginalised over the intermediate answer; it is zero for the Markov belief.

## Bistable perception (quantum Zeno)

The percept is a qubit whose basis states are the two interpretations. Between observations, every
$\Delta t$, it rotates by $U = e^{-ig\Delta t\,\sigma_x}$; each observation collapses it. The switch
probability per observation and the dwell-time distribution are

$$
q = \sin^2(g\,\Delta t), \qquad P(T = k\,\Delta t) = (1 - q)^{k - 1} q, \qquad
\mathbb{E}[T] = \frac{\Delta t}{\sin^2(g\Delta t)} \approx \frac{1}{g^2\,\Delta t}\ \ (g\Delta t \ll 1).
$$

Halving $\Delta t$ doubles the mean dwell time. Baselines: *Markov switching* at rate $r$,
$P(T > t) = e^{-rt}$, independent of $\Delta t$; *gamma renewal*, $T \sim \mathrm{Gamma}(k, \theta)$. The
predictions are binned into dwell-time histograms (with a final open bin) and fitted as multinomials.

References: Wang & Busemeyer (2013); Busemeyer et al. (2011); Pothos & Busemeyer (2009); Moreira &
Wichert (2016); Busemeyer, Kvam & Pleskac (2019); Yukalov & Sornette (2011); Kujala, Dzhafarov & Larsson
(2015); Pothos, Busemeyer & Trueblood (2013); Eisert, Wilkens & Lewenstein (1999); Brainerd, Wang &
Reyna (2013); Aerts (2009); Atmanspacher, Filk & Römer (2004).
