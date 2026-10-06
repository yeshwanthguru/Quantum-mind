"""Quantum-inspired reinforcement learning (QRL) on a grid world, with tabular Q-learning as the
classical baseline.

Dong, Chen, Li and Tarn, "Quantum reinforcement learning", IEEE Transactions on Systems, Man, and
Cybernetics B 38, 1207-1220 (2008). Each state holds a vector of action amplitudes; an action is chosen
by "measuring" it (probability |amplitude|^2) and state values are learnt by temporal differences.
The chosen action's amplitude is then rotated (a partial Grover rotation in the plane of |a> and the
rest of the state) by an angle proportional to the temporal-difference error delta = r + gamma V(s') -
V(s), bounded by max_step: better-than-expected actions are amplified, worse ones suppressed, so no
explicit exploration rate is needed. This bounded form replaces the integer number of Grover
iterations of the original paper, which overshoots and, when the amplification follows V(s') alone,
also amplifies actions that leave the state unchanged. Runs on a CPU (quantum-inspired)."""
from __future__ import annotations

import numpy as np

__all__ = ['GridWorld', 'QuantumInspiredQLearning', 'QLearning']


class GridWorld:
    """Deterministic grid: start (0, 0), goal at the opposite corner (reward 1, episode ends), walls
    as a set of cells; actions 0 up, 1 down, 2 left, 3 right; every other step gives `step_reward`."""
    MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self, size=6, walls=frozenset({(1, 1), (2, 3), (3, 1), (4, 4)}), step_reward=0.0, max_steps=200):
        self.size, self.walls, self.step_reward, self.max_steps = size, set(walls), step_reward, max_steps
        self.goal = (size - 1, size - 1)
        self.n_states, self.n_actions = size * size, 4

    def reset(self):
        """Return to the start cell; returns the start state (0)."""
        self.pos, self.t = (0, 0), 0
        return 0

    def step(self, a):
        """Returns (next state, reward, done)."""
        r, c = self.pos[0] + self.MOVES[a][0], self.pos[1] + self.MOVES[a][1]
        if 0 <= r < self.size and 0 <= c < self.size and (r, c) not in self.walls:
            self.pos = (r, c)
        self.t += 1
        s = self.pos[0] * self.size + self.pos[1]
        if self.pos == self.goal:
            return s, 1.0, True
        return s, self.step_reward, self.t >= self.max_steps

    def shortest_path(self):
        """Length of the shortest path from start to goal (breadth-first search)."""
        from collections import deque
        q, seen = deque([((0, 0), 0)]), {(0, 0)}
        while q:
            (r, c), d = q.popleft()
            if (r, c) == self.goal:
                return d
            for dr, dc in self.MOVES:
                n = (r + dr, c + dc)
                if 0 <= n[0] < self.size and 0 <= n[1] < self.size and n not in self.walls and n not in seen:
                    seen.add(n); q.append((n, d + 1))
        return None


class QuantumInspiredQLearning:
    """QRL agent. Parameters: alpha (value learning rate), gamma (discount), k (rotation per unit of TD
    error), max_step (largest rotation per update, radians), floor (smallest amplitude kept for every
    action), seed."""

    def __init__(self, n_states, n_actions, alpha=0.2, gamma=0.95, k=2.0, max_step=0.3, floor=0.02, seed=0):
        self.amp = np.full((n_states, n_actions), 1 / np.sqrt(n_actions))
        self.V = np.zeros(n_states)
        self.alpha, self.gamma, self.k, self.max_step, self.floor = alpha, gamma, k, max_step, floor
        self.rng = np.random.default_rng(seed)

    def act(self, s):
        """Sample an action from the Born-rule probabilities of state s."""
        p = self.amp[s] ** 2
        return int(self.rng.choice(len(p), p=p / p.sum()))

    def _rotate(self, s, a, angle):
        """Rotate the amplitude of action a by `angle` in the plane of |a> and the other actions."""
        psi = self.amp[s]; n = len(psi)
        lo, hi = np.arcsin(self.floor), np.arcsin(np.sqrt(max(1 - self.floor ** 2 * (n - 1), 0)))
        phi = float(np.clip(np.arcsin(np.clip(psi[a], 0, 1)) + angle, lo, hi))
        rest = np.delete(np.arange(n), a); norm_rest = np.linalg.norm(psi[rest])
        new = np.empty_like(psi); new[a] = np.sin(phi)
        new[rest] = np.cos(phi) * (psi[rest] / norm_rest if norm_rest > 0 else 1 / np.sqrt(n - 1))
        new = np.maximum(new, self.floor); self.amp[s] = new / np.linalg.norm(new)

    def update(self, s, a, r, s2, done):
        """Temporal-difference value update, then rotate the chosen action's amplitude by k times the
        TD error (bounded by max_step): better-than-expected actions grow, worse ones shrink."""
        delta = r + (0.0 if done else self.gamma * self.V[s2]) - self.V[s]
        self.V[s] += self.alpha * delta
        self._rotate(s, a, float(np.clip(self.k * delta, -self.max_step, self.max_step)))

    def greedy(self, s):
        """Most probable action in state s."""
        return int(np.argmax(self.amp[s]))


class QLearning:
    """Classical baseline: tabular Q-learning with epsilon-greedy exploration (decaying)."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.2, decay=0.995, seed=0):
        self.Q = np.zeros((n_states, n_actions)); self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.decay = epsilon, decay; self.rng = np.random.default_rng(seed)

    def act(self, s):
        """Epsilon-greedy action (ties broken at random)."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.Q.shape[1]))
        q = self.Q[s]; return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def update(self, s, a, r, s2, done):
        """Q-learning update; epsilon decays after each terminal step."""
        target = r + (0.0 if done else self.gamma * self.Q[s2].max())
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        if done:
            self.epsilon *= self.decay

    def greedy(self, s):
        """Greedy action."""
        return int(np.argmax(self.Q[s]))


def train(agent, env, episodes=300):
    """Run episodes; returns the number of steps per episode."""
    steps = []
    for _ in range(episodes):
        s = env.reset(); done = False; n = 0
        while not done:
            a = agent.act(s); s2, r, done = env.step(a); agent.update(s, a, r, s2, done); s = s2; n += 1
        steps.append(n)
    return np.array(steps)


__all__.append('train')
