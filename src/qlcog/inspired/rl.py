r"""Quantum-inspired reinforcement learning (QRL) on a grid world, with tabular Q-learning as the
classical baseline.

Dong, Chen, Li and Tarn, "Quantum reinforcement learning", IEEE Transactions on Systems, Man, and
Cybernetics B 38, 1207-1220 (2008). Each state holds a vector of action amplitudes; an action is chosen
by "measuring" it (probability :math:`|a|^2`) and state values are learnt by temporal differences.
The chosen action's amplitude is then rotated (a partial Grover rotation in the plane of
:math:`|a\rangle` and the rest of the state) by an angle proportional to the temporal-difference error
:math:`\delta = r + \gamma V(s') - V(s)`, bounded by ``max_step``: better-than-expected actions are
amplified and worse ones suppressed, so no explicit exploration rate is needed.

This bounded form replaces the integer number of Grover iterations of the original paper, which
overshoots and, when the amplification follows :math:`V(s')` alone, also amplifies actions that leave
the state unchanged. Everything runs on a CPU (quantum-inspired).

Examples
--------
>>> from qlcog.inspired import GridWorld, QuantumInspiredQLearning, train
>>> env = GridWorld()
>>> agent = QuantumInspiredQLearning(env.n_states, env.n_actions, seed=0)
>>> steps = train(agent, env, episodes=300)
>>> env.shortest_path(), int(steps[-50:].mean().round())
(10, 10)
"""
from __future__ import annotations

import numpy as np

__all__ = ['GridWorld', 'QuantumInspiredQLearning', 'QLearning']


class GridWorld:
    """Deterministic grid world.

    The agent starts at (0, 0); the goal is the opposite corner (reward 1, the episode ends). Actions are
    0 up, 1 down, 2 left and 3 right; moves into a wall or off the grid leave the agent in place. Every
    other step gives ``step_reward``.

    Parameters
    ----------
    size : int, optional
        Side length.
    walls : set of tuple, optional
        Blocked cells.
    step_reward : float, optional
        Reward of a non-goal step.
    max_steps : int, optional
        Episode length limit.

    Attributes
    ----------
    n_states, n_actions : int
        Sizes for the agents (``size**2`` and 4).
    """
    MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self, size=6, walls=frozenset({(1, 1), (2, 3), (3, 1), (4, 4)}), step_reward=0.0, max_steps=200):
        self.size, self.walls, self.step_reward, self.max_steps = size, set(walls), step_reward, max_steps
        self.goal = (size - 1, size - 1)
        self.n_states, self.n_actions = size * size, 4

    def reset(self):
        """Return to the start cell.

        Returns
        -------
        int
            The start state (0).
        """
        self.pos, self.t = (0, 0), 0
        return 0

    def step(self, a):
        """Take one action.

        Parameters
        ----------
        a : int
            Action (0 up, 1 down, 2 left, 3 right).

        Returns
        -------
        state : int
            Next state.
        reward : float
        done : bool
            True at the goal or when ``max_steps`` is reached.
        """
        r, c = self.pos[0] + self.MOVES[a][0], self.pos[1] + self.MOVES[a][1]
        if 0 <= r < self.size and 0 <= c < self.size and (r, c) not in self.walls:
            self.pos = (r, c)
        self.t += 1
        s = self.pos[0] * self.size + self.pos[1]
        if self.pos == self.goal:
            return s, 1.0, True
        return s, self.step_reward, self.t >= self.max_steps

    def shortest_path(self):
        """Length of the shortest path from start to goal (breadth-first search).

        Returns
        -------
        int or None
            None if the goal is unreachable.
        """
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
    """Quantum-inspired reinforcement learning agent (Dong et al., 2008, with a bounded rotation).

    Parameters
    ----------
    n_states, n_actions : int
        Problem size.
    alpha : float, optional
        Value learning rate.
    gamma : float, optional
        Discount factor.
    k : float, optional
        Rotation (radians) per unit of TD error.
    max_step : float, optional
        Largest rotation per update (radians).
    floor : float, optional
        Smallest amplitude kept for every action, so no action is ever ruled out.
    seed : int, optional
        Seed.

    Attributes
    ----------
    amp : numpy.ndarray
        Action amplitudes, shape ``(n_states, n_actions)``; each row has unit norm.
    V : numpy.ndarray
        State values.
    """

    def __init__(self, n_states, n_actions, alpha=0.2, gamma=0.95, k=2.0, max_step=0.3, floor=0.02, seed=0):
        self.amp = np.full((n_states, n_actions), 1 / np.sqrt(n_actions))
        self.V = np.zeros(n_states)
        self.alpha, self.gamma, self.k, self.max_step, self.floor = alpha, gamma, k, max_step, floor
        self.rng = np.random.default_rng(seed)

    def act(self, s):
        """Choose an action by "measuring" the state's action amplitudes (Born rule).

        Parameters
        ----------
        s : int
            State.

        Returns
        -------
        int
        """
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
        """Learn from one transition.

        Temporal-difference value update, then a rotation of the chosen action's amplitude by k times the
        TD error (bounded by ``max_step``): better-than-expected actions grow, worse ones shrink.

        Parameters
        ----------
        s : int
            State.
        a : int
            Action taken.
        r : float
            Reward.
        s2 : int
            Next state.
        done : bool
            Whether the episode ended.
        """
        delta = r + (0.0 if done else self.gamma * self.V[s2]) - self.V[s]
        self.V[s] += self.alpha * delta
        self._rotate(s, a, float(np.clip(self.k * delta, -self.max_step, self.max_step)))

    def greedy(self, s):
        """Most probable action in a state.

        Parameters
        ----------
        s : int

        Returns
        -------
        int
        """
        return int(np.argmax(self.amp[s]))


class QLearning:
    """Classical baseline: tabular Q-learning with decaying epsilon-greedy exploration.

    Parameters
    ----------
    n_states, n_actions : int
        Problem size.
    alpha : float, optional
        Learning rate.
    gamma : float, optional
        Discount factor.
    epsilon : float, optional
        Initial exploration rate.
    decay : float, optional
        Factor applied to epsilon after each episode.
    seed : int, optional
        Seed.
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.2, decay=0.995, seed=0):
        self.Q = np.zeros((n_states, n_actions)); self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.decay = epsilon, decay; self.rng = np.random.default_rng(seed)

    def act(self, s):
        """Epsilon-greedy action (ties broken at random).

        Parameters
        ----------
        s : int

        Returns
        -------
        int
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.Q.shape[1]))
        q = self.Q[s]; return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def update(self, s, a, r, s2, done):
        """Q-learning update; epsilon decays at the end of each episode.

        Parameters
        ----------
        s, a, r, s2, done
            Transition, as in :meth:`QuantumInspiredQLearning.update`.
        """
        target = r + (0.0 if done else self.gamma * self.Q[s2].max())
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        if done:
            self.epsilon *= self.decay

    def greedy(self, s):
        """Greedy action.

        Parameters
        ----------
        s : int

        Returns
        -------
        int
        """
        return int(np.argmax(self.Q[s]))


def train(agent, env, episodes=300):
    """Run training episodes.

    Parameters
    ----------
    agent : QuantumInspiredQLearning or QLearning
        Agent with ``act`` and ``update``.
    env : GridWorld
        Environment.
    episodes : int, optional
        Number of episodes.

    Returns
    -------
    numpy.ndarray
        Number of steps in each episode.
    """
    steps = []
    for _ in range(episodes):
        s = env.reset(); done = False; n = 0
        while not done:
            a = agent.act(s); s2, r, done = env.step(a); agent.update(s, a, r, s2, done); s = s2; n += 1
        steps.append(n)
    return np.array(steps)


__all__.append('train')
