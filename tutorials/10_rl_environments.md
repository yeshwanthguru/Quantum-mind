# Reinforcement learning with simulated people

Quantum Mind ships three Gymnasium environments:

| Environment | Task | Actions |
|---|---|---|
| `ClarificationEnv` | resolve an ambiguous request by asking yes/no questions, then act | ask question *i*, or act on object *j* |
| `TrustHandoverEnv` | hand objects to a person whose trust follows the trust qubit | hand over, slow hand-over, ask, wait |
| `GridWorldEnv` | reach a goal on a grid | up, down, left, right |

The people are simulated with the library's quantum-like models: answers depend on question order,
and asking about trust changes trust. This tutorial trains tabular agents with four exploration
strategies, including **amplitude exploration**, which samples actions with Born-rule probabilities
$|\psi_a|^2$ and rotates the amplitudes towards actions that did better than expected.

```python
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from quantum_mind.viz import use_mpl_style
from quantum_mind.envs import ClarificationEnv, TrustHandoverEnv, GridWorldEnv, register_envs
from quantum_mind.inspired.exploration import (EpsilonGreedy, Boltzmann, UCB, AmplitudeExploration, TabularAgent,
                                               StateIndexer, run_episodes)
theme = use_mpl_style('dark')
print(register_envs())
env = gym.make('quantum_mind/Clarification-v0')
obs, info = env.reset(seed=0)
print('observation', obs, '  actions', env.action_space)
```

## 1. Four explorers on the grid world

```python
def explorers(n_states, n_actions, seed):
    return {'ε-greedy': EpsilonGreedy(seed=seed), 'softmax': Boltzmann(seed=seed), 'UCB': UCB(seed=seed),
            'amplitude': AmplitudeExploration(n_states, n_actions, seed=seed)}

def train(make_env, episodes, seeds=5, state_fn=None, capacity=None, max_steps=100, key='returns'):
    curves = {}
    for seed in range(seeds):
        env = make_env()
        nS = capacity or env.observation_space.n
        for name, ex in explorers(nS, env.action_space.n, seed).items():
            agent = TabularAgent(nS, env.action_space.n, ex)
            sf = StateIndexer(capacity) if capacity else state_fn
            out = run_episodes(agent, make_env(), episodes, state_fn=sf, max_steps=max_steps, seed=100 * seed)
            curves.setdefault(name, []).append(out[key])
    return {k: np.array(v) for k, v in curves.items()}

grid = train(GridWorldEnv, 150, key='lengths')      # every agent reaches the goal; speed is what differs
for k, v in grid.items():
    print('%-10s steps to the goal: first 10 episodes %.1f, last 30 episodes %.1f' % (k, v[:, :10].mean(), v[:, -30:].mean()))
```

```python
def plot(curves, title, w=10, ylabel='return (moving average)'):
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    for (k, v), c in zip(curves.items(), theme['palette']):
        m = np.convolve(v.mean(0), np.ones(w) / w, mode='valid')
        ax.plot(np.arange(len(m)) + w, m, color=c, lw=2, label=k)
    ax.set_xlabel('episode'); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    plt.show()
plot(grid, 'Grid world', ylabel='steps to the goal (moving average)')
```

## 2. The clarification task with a simulated person

The state is the vector of answers so far, mapped to a table row by `StateIndexer`.

```python
clar = train(ClarificationEnv, 400, capacity=64, max_steps=10)
for k, v in clar.items():
    print('%-10s mean return, last 100 episodes: %.3f' % (k, v[:, -100:].mean()))
plot(clar, 'Clarification with a simulated person', w=25)
```

## 3. FrozenLake, a standard benchmark

```python
lake = train(lambda: gym.make('FrozenLake-v1', is_slippery=True), 1500, seeds=4, max_steps=100)
for k, v in lake.items():
    print('%-10s success rate, last 300 episodes: %.3f' % (k, v[:, -300:].mean()))
plot(lake, 'FrozenLake (slippery)', w=100)
```

**Result.** All four explorers learn the grid world. On the clarification task amplitude
exploration ends slightly behind the classical explorers, and on slippery FrozenLake the ranking
depends on seeds and settings: in this run softmax leads, amplitude exploration is second and UCB
last, while an earlier run with other seeds had UCB first (0.368), softmax second (0.331) and
amplitude third (0.297). Amplitude exploration is a quantum-inspired heuristic that is competitive
with, not better than, standard exploration; it is not a quantum speed-up.

References: Sutton & Barto (2018); Dong et al. (2008), *IEEE TSMC-B* 38, 1207-1220; Towers et al.
(2024), arXiv:2407.17032.
