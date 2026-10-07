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

## Conjunction and disjunction fallacies

State $\psi = e_1 \in \mathbb{R}^3$, events $A$, $B$ with vectors as above (rank 1 or 2). The conjunction is
judged by evaluating the **more likely event first** (`QuantumConjunctionModel`):

$$
P(A \wedge B) = \begin{cases} \lVert P_B P_A\psi\rVert^2 & P(A) \ge P(B) \\ \lVert P_A P_B\psi\rVert^2 & \text{otherwise}\end{cases},
\qquad
P(A \vee B) = 1 - \lVert \bar P_{2}\,\bar P_{1}\,\psi \rVert^2
$$

in the same order. When the projectors do not commute, $P(A \wedge B) > \min\{P(A), P(B)\}$ is possible
(the conjunction fallacy).

**Baselines.** *Classical joint*: $P(A \wedge B) \le \min(P(A), P(B))$ always. *Averaging*:
$P(A \wedge B) = w\min + (1 - w)\max$, $P(A \vee B) = w\max + (1 - w)\min$. *Probability theory plus
noise*: every estimate is $(1 - 2d)\,p + d$, with noise $d$ for single events and $d + d_d$ for
conjunctions and disjunctions.

## Interference and the law of total probability

A choice (act or not) when an earlier event is known to be $X$ ($p_1 = P(\text{act} \mid X)$), known to be
$Y$ ($p_2$), or unknown. Classically $P(\text{act} \mid \text{unknown}) = c\,p_1 + (1 - c)\,p_2$. The
quantum-like law adds interference terms (`InterferenceModel`):

$$
A = c\,p_1 + (1 - c)\,p_2 + 2\cos\theta\sqrt{c\,p_1(1 - c)\,p_2}, \qquad
B = c\,q_1 + (1 - c)\,q_2 + 2\cos\theta\sqrt{c\,q_1(1 - c)\,q_2},\ \ q_i = 1 - p_i,
$$

$$
P(\text{act} \mid \text{unknown}) = \frac{A}{A + B} .
$$

$\theta = \pi/2$ recovers the classical law; values outside $[\min(p_1, p_2), \max(p_1, p_2)]$ violate it.

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

## Decisions under risk

Lotteries $L = \{(x_i, p_i)\}$; power utility $u(x) = \operatorname{sign}(x)\lvert x\rvert^\alpha$;
expected utility $U(L) = \sum_i p_i u(x_i)$; logit choice $P(1) = 1/(1 + e^{-\beta[U(L_1) - U(L_2)]})$
(`ExpectedUtilityModel`).

**Quantum decision theory** (`QDTModel`): $p(L_n) = f(L_n) + q(L_n)$ with $f$ the logit utility factor
above and an attraction factor of size $q_0$, negative for the more uncertain prospect (larger outcome
variance), $q_1 + q_2 = 0$, clipped so that both probabilities stay in $[0, 1]$. $q_0 = 0.25$ is the
"quarter law" value.

**Prospect theory** (`ProspectTheoryModel`): value $v(x) = x^\alpha$ for gains and $-\lambda(-x)^\alpha$ for
losses, weighting $w(p) = p^g/(p^g + (1 - p)^g)^{1/g}$, $V(L) = \sum_i w(p_i)\,v(x_i)$, and logit choice.

## Contextuality

For $\pm1$ observables, $E_{ij} = \langle R_i R_j\rangle$.

$$
S_{\mathrm{CHSH}} = \lvert E_{11} + E_{12} + E_{21} - E_{22}\rvert \le 2 \ \text{(noncontextual)}, \qquad \le 2\sqrt2 \ \text{(quantum)}.
$$

For a cyclic system of rank $n$, Contextuality-by-Default declares the system contextual when

$$
s_{\mathrm{odd}}\bigl(\langle R_1R_2\rangle, \dots, \langle R_nR_1\rangle\bigr) - (n - 2) - \Delta > 0,
\qquad \Delta = \sum_q \bigl\lvert \langle R_q^{c}\rangle - \langle R_q^{c'}\rangle \bigr\rvert ,
$$

where $s_{\mathrm{odd}}(v) = \max \sum_i \pm v_i$ over sign patterns with an odd number of minus signs. For a
Bell pair measured at angles $a, b$ in the $x$-$z$ plane, $E(a, b) = \cos(a - b)$.

## Asymmetric similarity

**Quantum** (`QuantumSimilarityModel`): concepts are subspaces of $\mathbb{R}^3$ (each defined by two
angles), the neutral state is $\psi = e_1$, and

$$
\mathrm{Sim}(A, B) = \lVert P_B P_A\psi\rVert^2 \neq \mathrm{Sim}(B, A) \ \text{in general}.
$$

**Geometric**: points $z_A$ in a plane, $\mathrm{Sim}(A, B) = e^{-c\lVert z_A - z_B\rVert}$ (symmetric).
**Biased geometric**: $\mathrm{Sim}(A, B) = b_B\,e^{-c\lVert z_A - z_B\rVert}$ with a bias per concept
(Nosofsky, 1991).

## Quantum games (EWL)

Each player has a qubit ($\lvert 0\rangle$ = cooperate). With $D = i\sigma_y$,

$$
J = \exp\bigl(i\tfrac{\gamma}{2}\,D\otimes D\bigr), \qquad
\lvert\psi_f\rangle = J^\dagger (U_A \otimes U_B)\, J\, \lvert 00\rangle, \qquad
U(\theta, \phi) = \begin{pmatrix} e^{i\phi}\cos\frac\theta2 & \sin\frac\theta2 \\ -\sin\frac\theta2 & e^{-i\phi}\cos\frac\theta2\end{pmatrix}.
$$

The expected payoff of player $A$ is $\sum_{xy} \$_A(x, y)\,\lvert\langle xy\vert\psi_f\rangle\rvert^2$. $\gamma = 0$
is the classical game; at $\gamma = \pi/2$ in the Prisoner's Dilemma, $(Q, Q)$ with $Q = U(0, \pi/2)$ is a
Nash equilibrium of the two-parameter strategy set.

## Episodic memory (overdistribution)

For each probe type $k$ (target, related, unrelated), a memory state $\psi_k \in \mathbb{R}^3$ (two angles);
verbatim $V$ and gist $G$ are rays $v, g$ at a shared angle $c$; "$V$ or $G$" is the plane
$\operatorname{span}(v, g)$ (`QuantumEpisodicModel`):

$$
P(V) = (v^T\psi_k)^2, \quad P(G) = (g^T\psi_k)^2, \quad P(V \vee G) = \lVert P_{\mathrm{span}(v,g)}\psi_k\rVert^2,
$$

so the overdistribution $P(V) + P(G) - P(V \vee G)$ is positive unless $v \perp g$. The additive baseline
sets $P(V \vee G) = P(V) + P(G)$.

## Concept combination (Fock space)

For single memberships $\mu_A, \mu_B$ (`FockSpaceConceptModel`):

$$
\mu(A \wedge B) = m^2\,\mu_A\mu_B + (1 - m^2)\Bigl[\tfrac{\mu_A + \mu_B}{2}
+ \kappa\,\min\bigl(\sqrt{\mu_A\mu_B},\ \sqrt{(1 - \mu_A)(1 - \mu_B)}\bigr)\Bigr],
\qquad m^2 \in (0, 1),\ \kappa \in [-1, 1].
$$

Baselines: product $\mu_A\mu_B$, minimum $\min(\mu_A, \mu_B)$, weighted average $w\mu_A + (1 - w)\mu_B$.

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
