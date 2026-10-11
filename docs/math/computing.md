# Quantum computing for robot algorithms

A QUBO minimises $E(x) = x^TQx + \mathrm{offset}$ over $x \in \{0, 1\}^n$. Each robot problem below is encoded with penalty
terms (Lucas, 2014); `RobotProblem.objective` is the *true* objective without the penalties, and the penalty weight is chosen
large enough that every minimiser of $E$ is feasible.

## Multi-robot allocation (`multi_robot_allocation`)

Variables $x_{r,t} = 1$ if robot $r$ does task $t$ (index $r\,n_t + t$), costs $C_{r,t}$, balance weight $\lambda$:

$$
E(x) = \sum_{r,t} C_{r,t}\,x_{r,t} + \lambda \sum_r \Bigl(\sum_t x_{r,t}\Bigr)^2 + A \sum_t \Bigl(\sum_r x_{r,t} - 1\Bigr)^2 .
$$

On feasible strings the last term is zero and $E$ equals the objective $\sum C_{r,t}x_{r,t} + \lambda\sum_r(\sum_t x_{r,t})^2$.

## Grasp selection (`grasp_selection`)

Variables $x_i = 1$ if grasp $i$ is chosen, qualities $q_i$, conflict costs $c_{ij}$ ($c_{ii} = 0$), exactly $k$ grasps:

$$
E(x) = -\sum_i q_i x_i + \sum_{i<j} c_{ij}\,x_i x_j + A\Bigl(\sum_i x_i - k\Bigr)^2 .
$$

## Slot scheduling (`slot_scheduling`)

Variables $x_{i,s} = 1$ if robot $i$ moves in slot $s$, conflict pairs $\mathcal{C}$:

$$
E(x) = \sum_{(i,j) \in \mathcal{C}} \sum_s x_{i,s}\,x_{j,s} + A \sum_i \Bigl(\sum_s x_{i,s} - 1\Bigr)^2 ;
$$

the objective is the number of conflicting pairs that share a slot (0 is a valid schedule).

## QAOA (`qaoa`)

With $x_i = (1 - z_i)/2$ the QUBO is the Ising energy $H_C = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_iZ_j + c$ (`Qubo.to_ising`). QAOA of
depth $p$ prepares

$$
\lvert\gamma, \beta\rangle = \prod_{l=1}^{p} e^{-i\beta_l H_M}\, e^{-i\gamma_l H_C}\,\lvert +\rangle^{\otimes n}, \qquad H_M = \sum_i X_i ,
$$

(Farhi et al., 2014) and the classical optimiser minimises $\langle\gamma,\beta\rvert H_C\lvert\gamma,\beta\rangle$. The circuit is
Qiskit's `QAOAAnsatz`; the adapter reports the probability mass on the exact optimum, the approximation ratio
$(E_{\max} - \langle H_C\rangle)/(E_{\max} - E_{\min})$ and the best of $S$ sampled strings. A uniformly random sampler with the same
$S$ finds the optimum with probability $1 - (1 - 2^{-n})^S$ (`random_sampling`), which is the number a sampling solver has to beat.
