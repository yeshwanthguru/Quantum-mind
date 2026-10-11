# Infrastructure

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

## The state-vector simulator

An $n$-qubit state $\lvert\psi\rangle \in \mathbb{C}^{2^n}$ evolves under gates; rotations are
$R_\sigma(\theta) = e^{-i\theta\sigma/2}$ for $\sigma \in \{X, Y, Z\}$, and qubit $q$ is bit $q$ of the basis
index (Qiskit ordering). Batches of inputs are simulated at once, and a circuit exports to Qiskit with
`to_qiskit()`.

**Adjoint method**: for $f = \mathcal{L}(\lvert\psi(w)\rangle)$ with
$\lvert\psi\rangle = U_L\cdots U_1\lvert 0\rangle$, propagate $\lvert\lambda\rangle = \partial\mathcal L/\partial\langle\psi\rvert$
backwards; each gate contributes $\partial f/\partial w_k = 2\,\mathrm{Re}\,\langle\lambda_k\rvert\,\partial_{w_k}U_k\,\lvert\psi_{k-1}\rangle$,
so all gradients cost about as much as three forward simulations.

## QUBO container and trust-aware allocation

A `Qubo` minimises $E(x) = x^TQx + \mathrm{offset}$ over $x \in \{0, 1\}^n$, with $Q$ upper triangular. With
$x_i = (1 - z_i)/2$ the same energies are an Ising model in $z_i = \pm1$ (`Qubo.to_ising`), so an instance can
be handed to any external solver; small instances are solved exactly with `Qubo.brute_force`.

**Trust-aware task allocation** (`trust_aware_allocation`): variables $x_{a,t} = 1$ if person $a$ does
task $t$,

$$
E(x) = \sum_{a,t} x_{a,t}\,\bigl(C_{a,t} + F\,r_t\,(1 - \tau_a)\bigr)
     + \lambda \sum_a \Bigl(\sum_t x_{a,t}\Bigr)^2
     + A \sum_t \Bigl(\sum_a x_{a,t} - 1\Bigr)^2 ,
$$

where $\tau_a$ is person $a$'s probability of trusting the robot, $r_t \in [0, 1]$ how much task $t$ depends
on the robot, $F$ the cost of a failed robot-assisted step, $\lambda$ the workload weight and $A$ the
one-agent-per-task penalty.
