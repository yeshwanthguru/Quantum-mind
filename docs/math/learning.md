# Learning

## Environments as Markov decision processes

An environment is an MDP $(\mathcal S, \mathcal A, P, r, \gamma)$; an agent seeks a policy $\pi(a \mid s)$
maximising $\mathbb{E}\bigl[\sum_t \gamma^t r_{t+1}\bigr]$.

**Clarification** (`ClarificationEnv`). Hidden intent $h^\star$ drawn uniformly; the person answers
question $q$ through the projective answer model (Lüders rule on $\psi_{h^\star}$, so answers depend on
what was asked before). Observation: $o \in \{-1, 0, +1\}^Q$ (no, not asked, yes). Actions: ask $q$
(reward $-c_{\mathrm{ask}}$) or act on $h$ (reward $+1$ if $h = h^\star$, else $-C_e$, episode ends). After
$Q_{\max}$ questions the episode is truncated with $-C_e$.

**Trust hand-over** (`TrustHandoverEnv`). State: the trust qubit $\rho$ of the
[open-system belief](quantum_like.md#belief-dynamics). With $p_t = \rho_{00}$:

| Action | Failure probability | Reward | Effect on $\rho$ |
|---|---|---|---|
| hand over | $1 - p_t$ | $+1$ or $-C_f$ | rotate by the outcome, dephase |
| slow hand-over | $s\,(1 - p_t)$ | $(+1 \text{ or } -C_f) - C_s$ | rotate, dephase |
| ask | | $-C_a$ | collapse to $\lvert 0\rangle\langle 0\rvert$ w.p. $p_t$, else $\lvert 1\rangle\langle 1\rvert$ |
| wait | | $-C_w$ | off-diagonals $\times (1 - \gamma)$ |

**Grid world**: deterministic moves on a grid, reward $1$ at the goal.

## Tabular Q-learning

$$
\delta_t = r_{t+1} + \gamma\max_{a'} Q(s_{t+1}, a')\,[\text{not terminal}] - Q(s_t, a_t), \qquad
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\,\delta_t .
$$

(`TabularAgent`; continuous or vector observations are mapped to rows by `StateIndexer`.)

## Exploration strategies

- **ε-greedy**: a random action with probability $\epsilon$, else $\arg\max_a Q(s, a)$;
  $\epsilon \leftarrow \max(\epsilon_{\min}, \epsilon\cdot d)$ after each episode.
- **Boltzmann**: $\pi(a \mid s) \propto \exp\bigl((Q(s, a) - \max_b Q(s, b))/\tau\bigr)$, $\tau$ decayed per episode.
- **UCB**: untried actions first, then $\arg\max_a Q(s, a) + c\sqrt{\ln N(s)/N(s, a)}$.
- **Amplitude exploration** (`AmplitudeExploration`): each state keeps amplitudes $\psi_s \in \mathbb{R}_{\ge 0}^{\lvert\mathcal A\rvert}$,
  $\lVert\psi_s\rVert = 1$, and samples $a$ with probability $\psi_{s,a}^2$ (Born rule). After the update of
  $Q$, the advantage of the chosen action over the Born-rule average drives a rotation:

$$
\Delta = Q(s, a) - \sum_b \psi_{s,b}^2\,Q(s, b), \qquad
\varphi = \operatorname{clip}\bigl(\arcsin\psi_{s,a} + \operatorname{clip}(k\Delta, \pm\phi_{\max}),\ \varphi_{\mathrm{lo}},\ \varphi_{\mathrm{hi}}\bigr),
$$

$$
\psi_{s,a} \leftarrow \sin\varphi, \qquad \psi_{s,b} \leftarrow \cos\varphi\,\frac{\psi_{s,b}}{\lVert\psi_{s,\neg a}\rVert}\ \ (b \neq a),
$$

  then every amplitude is floored at $f$ and the vector renormalised, so every action keeps probability at
  least about $f^2$. The bounds are $\varphi_{\mathrm{lo}} = \arcsin f$ and
  $\varphi_{\mathrm{hi}} = \arcsin\sqrt{1 - (\lvert\mathcal A\rvert - 1) f^2}$.

## Quantum-inspired Q-learning (Dong et al.)

Action amplitudes per state, action chosen with probability $\lvert\psi_a\rvert^2$, state values learnt by
$\delta = r + \gamma V(s') - V(s)$, and the chosen amplitude rotated (a partial Grover rotation in the plane
of $\lvert a\rangle$ and the rest) by an angle proportional to $\delta$ and bounded by a maximum step.
(`QuantumInspiredQLearning`; tabular `QLearning` is the baseline.)

## Variational quantum policy

A data re-uploading circuit $U(s; w)$ on $n$ qubits gives Born-rule probabilities
$P(x \mid s) = \lvert\langle x\rvert U(s; w)\lvert 0\rangle\rvert^2$. The action probabilities are the marginals of the
first $m = \lceil\log_2\lvert\mathcal A\rvert\rceil$ qubits, restricted to the $\lvert\mathcal A\rvert$ valid actions and
renormalised:

$$
\pi_w(a \mid s) = \frac{M_a(s)}{\sum_{a'} M_{a'}(s)}, \qquad M_a(s) = \sum_{x:\ x \bmod 2^m = a} P(x \mid s).
$$

**REINFORCE.** After a batch of episodes with discounted returns $G_t = \sum_k \gamma^k r_{t+k+1}$ and a
running baseline $b \leftarrow 0.9\,b + 0.1\,\bar G$,

$$
\mathcal{L}(w) = -\frac1n\sum_t (G_t - b)\,\log\pi_w(a_t \mid s_t), \qquad w \leftarrow w - \eta\,\nabla_w\mathcal{L}.
$$

The gradient is exact: $\partial\mathcal{L}/\partial M_k = -\frac{G_t - b}{n}\bigl(\tfrac{[k = a_t]}{M_{a_t}} - \tfrac{1}{\sum_{a'}M_{a'}}\bigr)$
is pushed through $P = \lvert\psi\rvert^2$ and back through the circuit with the adjoint method.
(`VariationalPolicy`, `reinforce`)

## Tensor-train layers

A matrix $W \in \mathbb{R}^{M\times N}$ with $M = \prod_k m_k$, $N = \prod_k n_k$ is written as

$$
W\bigl[(i_1, \dots, i_d), (j_1, \dots, j_d)\bigr] = G_1[i_1, j_1]\,G_2[i_2, j_2]\cdots G_d[i_d, j_d],
\qquad G_k[i_k, j_k] \in \mathbb{R}^{r_{k-1}\times r_k},\ r_0 = r_d = 1 .
$$

**TT-SVD.** Reorder $W$ into a tensor with index pairs $(i_k, j_k)$, then for $k = 1, \dots, d - 1$ reshape
the remainder to $(r_{k-1}m_kn_k) \times (\cdots)$, take the SVD $U\Sigma V^T$, keep the
$r_k \le r_{\max}$ largest singular values (and those above $\mathrm{tol}\cdot\sigma_1$), set
$G_k = U_{:, :r_k}$ and continue with $\Sigma_{:r_k}V^T_{:r_k}$. By Eckart-Young each step is the best
rank-$r_k$ approximation of its unfolding, and the total error obeys
$\lVert W - \hat W\rVert_F^2 \le \sum_k \varepsilon_k^2$ (Oseledets, 2011).

**Storage** $\sum_k r_{k-1}m_kn_kr_k$ numbers instead of $MN$. **Product** $y = Wx$ is computed by
contracting the input with the cores one by one, never forming $W$. (`TTMatrix`, `compress_layers`)

## Quanvolutional filter

For each $2\times 2$ patch with pixels $x_1..x_4 \in [0, 1]$:

$$
\lvert\phi(x)\rangle = V\prod_{j=1}^{4} R_y^{(j)}(\pi x_j)\,\lvert 0000\rangle, \qquad
f_j(x) = \langle\phi(x)\rvert Z_j\lvert\phi(x)\rangle \in [-1, 1],
$$

where $V$ is a fixed random circuit (layers of random single-qubit rotations and a ring of CNOTs). With no
random layer, $f_j = \cos(\pi x_j)$. The four values form four output channels. The classical baseline
uses $\tanh(W^T x + b)$ with random $W$, $b$ of the same shape. (`QuanvolutionFilter`, `RandomConvFilter`)

References: Sutton & Barto (2018); Watkins & Dayan (1992); Auer, Cesa-Bianchi & Fischer (2002) for UCB;
Dong et al. (2008); Williams (1992); Jerbi et al. (2021); Oseledets (2011); Novikov et al. (2015);
Henderson et al. (2020).
