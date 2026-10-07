# Reinforcement learning

Reinforcement learning (RL) learns **what to do** from **reward**. The agent tries actions, sees what
happens and gradually prefers actions that lead to more reward over time. Nobody tells it the right
action; it has to balance **exploring** new actions against **exploiting** what already works.

```{mermaid}
flowchart LR
    accTitle: Reinforcement learning, diagram 1
    accDescr: An agent with a policy acts on an environment, which returns a new state and a reward; a value estimate is updated from the temporal-difference error.
    AG["agent<br/>policy π(a | s)"]:::op -->|action aₜ| EN(("environment")):::hum
    EN -->|state sₜ₊₁| AG
    EN -->|reward rₜ₊₁| AG
    AG --> V["value estimate<br/>Q(s, a)"]:::in
    V -->|update from TD error| AG
    classDef in fill:#0b2a4a,stroke:#79c0ff,color:#e6edf3
    classDef op fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef hum fill:#3d1414,stroke:#ff7b72,color:#e6edf3
```

## From the basics

Markov decision process (MDP)
: States $s$, actions $a$, transition probabilities $P(s' \mid s, a)$, rewards $r$ and a discount
  factor $\gamma < 1$. The goal is a **policy** that maximises the expected discounted return
  $G_t = \sum_k \gamma^k r_{t+k+1}$.

Value functions
: $V(s)$ is the expected return from state $s$; $Q(s, a)$ the expected return after taking action $a$.
  The **Bellman equation** relates the value of a state to the values of the next states.

Q-learning and temporal difference (TD)
: Update $Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,]$. The term in
  brackets is the **TD error**: how much better or worse things went than expected.

Exploration
: **ε-greedy** acts randomly with probability ε; **Boltzmann** (softmax) prefers higher values;
  **UCB** adds a bonus for rarely tried actions. **Amplitude exploration** keeps an amplitude per
  action and samples with probability $|\psi_a|^2$ (the Born rule), rotating amplitudes towards good
  actions.

Policy gradient
: Learn the policy directly: increase the log-probability of actions that did better than average
  (REINFORCE). A **variational quantum policy** uses the Born-rule probabilities of a parameterised
  circuit as $\pi(a \mid s)$.

Gymnasium
: The standard Python interface for RL environments: `reset()` and `step(action)`. Quantum Mind's
  environments follow it, so any RL library can train on them.

## A small example: Q-learning on the grid world

```python
from quantum_mind.envs import GridWorldEnv
from quantum_mind.inspired.exploration import TabularAgent, EpsilonGreedy, run_episodes
env = GridWorldEnv()
agent = TabularAgent(env.observation_space.n, env.action_space.n, EpsilonGreedy(seed=0))
out = run_episodes(agent, env, 200, max_steps=100)
print(out['lengths'][:20].mean(), '->', out['lengths'][-20:].mean())   # steps to the goal: about 39 -> 11
```

## Where Quantum Mind fits

The library provides RL environments with **simulated people** whose answers show order effects and
whose trust changes when asked about, amplitude-based exploration next to ε-greedy, softmax and UCB,
and a variational quantum policy trained with REINFORCE. Measured results: amplitude exploration is
competitive with, but not better than, UCB and softmax on FrozenLake.

- [Tutorial: environments and exploration](../tutorials/10_rl_environments.ipynb)
- [Tutorial: a quantum policy](../tutorials/11_quantum_policy.ipynb)
- [Model atlas: learning](../atlas/learning.md)

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8, 279-292.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist
  reinforcement learning. *Machine Learning*, 8, 229-256.
- Dong, D., Chen, C., Li, H., & Tarn, T.-J. (2008). Quantum reinforcement learning. *IEEE Transactions
  on Systems, Man, and Cybernetics, Part B*, 38(5), 1207-1220.
- Jerbi, S., Gyurik, C., Marshall, S., Briegel, H., & Dunjko, V. (2021). Parametrized quantum policies
  for reinforcement learning. *Advances in Neural Information Processing Systems*, 34.
- Towers, M., et al. (2024). Gymnasium: A standard interface for reinforcement learning environments.
  arXiv:2407.17032.
