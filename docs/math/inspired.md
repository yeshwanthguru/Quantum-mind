# Quantum-inspired algorithms and problems

## QUBO problems

Minimise $E(x) = x^TQx + \mathrm{offset}$ over $x \in \{0, 1\}^n$, with $Q$ upper triangular. With
$x_i = (1 - z_i)/2$ the same energies are an Ising model in $z_i = \pm1$.

- **Max-Cut**: $E(x) = -\sum_{(i,j)} w_{ij}\,(x_i + x_j - 2x_ix_j)$ (minus the cut weight).
- **Task allocation** (`task_allocation`): variables $x_{a,t} = 1$ if agent $a$ does task $t$,
  $$
  E(x) = \sum_{a,t} C_{a,t}\,x_{a,t} + A\sum_t\Bigl(\sum_a x_{a,t} - 1\Bigr)^2 ,
  $$
  and a large enough penalty $A$ makes every task go to exactly one agent.
- **Knapsack**: maximise value under a weight capacity, with binary slack variables turning the inequality
  into a penalised equality.
- **Portfolio**: $\min\ q\,x^T\Sigma x - \mu^Tx + A\bigl(\sum_i x_i - B\bigr)^2$ (exactly $B$ assets).

## QIEA

Each bit is a Q-bit with angle $\theta_i$ and $P(x_i = 1) = \sin^2\theta_i$. Observing an individual
samples a bit string. Each generation compares every observed $x$ with the best solution $b$ and rotates

$$
\theta_i \leftarrow \theta_i + s(x_i, b_i, f(x) \le f(b))\cdot\Delta\theta(x_i, b_i, f(x) \le f(b)),
$$

with angles and signs from Han and Kim's lookup table (the comparison read for minimisation), which moves
the amplitude towards the better bit. An $H_\epsilon$ bound keeps $P(x_i = 1)$ away from 0 and 1, and
migration exchanges best solutions within groups and globally.

## QPSO

Particle $i$ has personal best $p_i$, the swarm best is $g$, and $m = \frac1N\sum_i p_i$ is the mean best.
With $\varphi \sim U(0, 1)$ and $u \sim U(0, 1)$ per coordinate,

$$
P_i = \varphi\,p_i + (1 - \varphi)\,g, \qquad x_i \leftarrow P_i \pm \beta\,\lvert m - x_i\rvert\,\ln(1/u),
$$

the sampled position of a particle in a delta potential well centred on $P_i$. The contraction-expansion
coefficient $\beta$ decreases linearly from $\beta_0$ to $\beta_1$.

## Simulated quantum annealing

The transverse-field Ising Hamiltonian $H = H_{\mathrm{problem}}(\sigma^z) - \Gamma\sum_i\sigma_i^x$ is mapped by
Suzuki-Trotter onto $P$ coupled classical replicas at temperature $T$:

$$
E_{\mathrm{eff}} = \frac1P\sum_{k=1}^{P}\Bigl[H_{\mathrm{problem}}(z^{(k)}) - J_\perp\sum_i z_i^{(k)}z_i^{(k+1)}\Bigr], \qquad
J_\perp = -\frac{PT}{2}\ln\tanh\Bigl(\frac{\Gamma}{PT}\Bigr),
$$

with periodic replicas ($z^{(P+1)} = z^{(1)}$) and Boltzmann weight $e^{-E_{\mathrm{eff}}/T}$.

$\Gamma$ is lowered from $\Gamma_0$ to $\Gamma_1$ over the sweeps with single-spin Metropolis moves, so
$J_\perp$ grows and the replicas merge. The baseline is **simulated annealing**: Metropolis acceptance
$\min(1, e^{-\Delta E/T})$ with $T$ lowered geometrically from $T_0$ to $T_1$.

## MPS classifier

Each feature $x_j \in [0, 1]$ is mapped to a local vector; for $d = 2$,
$\phi(x_j) = (\cos\frac{\pi x_j}{2}, \sin\frac{\pi x_j}{2})$, and for $d > 2$ the spin-coherent state with
components $\sqrt{\binom{d-1}{s}}\cos^{d-1-s}(\tfrac{\pi x_j}{2})\sin^{s}(\tfrac{\pi x_j}{2})$. The class scores are

$$
f_\ell(x) = \sum_{s_1..s_n} A^{(1)}_{s_1}A^{(2)}_{s_2}\cdots A^{(n)\,\ell}_{s_n}\ \phi_{s_1}(x_1)\cdots\phi_{s_n}(x_n),
$$

a weight tensor in a $d^n$-dimensional space stored as a matrix product state with bond dimension at most
$D$ (cost linear in $n$). Training minimises the softmax cross-entropy, either with Adam on all cores or with
DMRG-style two-site sweeps truncated by SVD. The baseline is logistic regression.

## Quantum language model

A document is a density matrix $\rho_d$ over the vocabulary, the maximum-likelihood estimate from
projectors $\Pi_k$: one per term occurrence ($\lvert e_w\rangle\langle e_w\rvert$) and one per co-occurring pair within a
window ($\lvert v\rangle\langle v\rvert$, $v = (e_a + e_b)/\sqrt2$). The estimate maximises $\sum_k\log\operatorname{tr}(\rho\Pi_k)$ and is
found by the $R\rho R$ iteration,

$$
R(\rho) = \sum_k\frac{\Pi_k}{\operatorname{tr}(\rho\Pi_k)}, \qquad \rho \leftarrow \frac{R(\rho)\,\rho\,R(\rho)}{\operatorname{tr}\bigl(R(\rho)\,\rho\,R(\rho)\bigr)},
$$

then smoothed with the collection matrix and a little of $I/V$. Documents are ranked by the negative von
Neumann divergence from the query matrix,
$-\Delta(\rho_q\Vert\rho_d) = -\operatorname{tr}\bigl(\rho_q(\log\rho_q - \log\rho_d)\bigr)$. The baseline is unigram query
likelihood with Dirichlet smoothing,
$\log P(q \mid d) = \sum_{w\in q}\log\frac{c(w, d) + \mu P(w \mid C)}{\lvert d\rvert + \mu}$.

## Tensor-train layers and quantum-inspired exploration

See the [learning page](learning.md).

References: Lucas (2014); Han & Kim (2002); Sun, Feng & Xu (2004); Martoňák, Santoro & Tosatti (2002);
Kirkpatrick, Gelatt & Vecchi (1983); Stoudenmire & Schwab (2016); Sordoni, Nie & Bengio (2013); Lvovsky
(2004); Zhai & Lafferty (2004).
