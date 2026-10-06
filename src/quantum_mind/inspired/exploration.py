r"""Exploration strategies for tabular reinforcement learning, quantum-inspired and classical.

All agents learn action values with Q-learning; they differ only in how they choose actions:

* :class:`EpsilonGreedy`: a random action with probability :math:`\varepsilon` (decaying);
* :class:`Boltzmann`: softmax of the action values at a temperature;
* :class:`UCB`: the action with the highest upper confidence bound (optimism in the face of uncertainty);
* :class:`AmplitudeExploration` (quantum-inspired, after Dong et al., IEEE TSMC-B 2008): each state
  keeps a unit vector of action amplitudes; actions are sampled with the Born rule
  :math:`P(a) = |\psi_a|^2`. After each update the chosen action's amplitude is rotated (a partial
  Grover rotation) by an angle proportional to its *advantage* over the current policy,
  :math:`Q(s,a) - \sum_b |\psi_b|^2 Q(s,b)`, so actions better than the policy's average are amplified
  and worse ones suppressed, while a floor keeps every action possible.

:class:`TabularAgent` combines an explorer with Q-learning, and :func:`run_episodes` trains it on any
environment with the Gymnasium interface (discrete or hashable observations; :class:`StateIndexer`
maps observations to table rows). No quantum computer is used.

Examples
--------
>>> from quantum_mind.envs import GridWorldEnv
>>> from quantum_mind.inspired.exploration import TabularAgent, AmplitudeExploration, run_episodes
>>> env = GridWorldEnv()
>>> agent = TabularAgent(36, 4, AmplitudeExploration(36, 4, seed=0))
>>> steps = run_episodes(agent, env, 300)['lengths']
>>> int(round(steps[-50:].mean()))
10
"""
from __future__ import annotations

import numpy as np

__all__ = ['EpsilonGreedy', 'Boltzmann', 'UCB', 'AmplitudeExploration', 'TabularAgent', 'StateIndexer', 'run_episodes']


class EpsilonGreedy:
    """Random action with probability epsilon, otherwise greedy; epsilon decays per episode.

    Parameters
    ----------
    epsilon : float, optional
        Initial exploration probability.
    decay : float, optional
        Factor applied at the end of each episode.
    min_epsilon : float, optional
        Floor of epsilon.
    seed : int, optional
    """

    def __init__(self, epsilon=0.3, decay=0.99, min_epsilon=0.01, seed=0):
        self.epsilon, self.decay, self.min_epsilon = epsilon, decay, min_epsilon
        self.rng = np.random.default_rng(seed)

    def choose(self, q, s, agent):
        """Choose an action from the value row ``q`` of state ``s``.

        Parameters
        ----------
        q : numpy.ndarray
            Action values of the state.
        s : int
            State index.
        agent : TabularAgent
            The agent (for counts).

        Returns
        -------
        int
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(q)))
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def end_episode(self):
        """Decay epsilon."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

    def update(self, s, a, td_error, q=None):
        """No exploration state to update."""


class Boltzmann(EpsilonGreedy):
    """Softmax (Boltzmann) exploration with a decaying temperature.

    Parameters
    ----------
    temperature : float, optional
        Initial temperature.
    decay : float, optional
        Factor per episode.
    min_temperature : float, optional
    seed : int, optional
    """

    def __init__(self, temperature=0.5, decay=0.99, min_temperature=0.01, seed=0):
        self.temperature, self.decay, self.min_temperature = temperature, decay, min_temperature
        self.rng = np.random.default_rng(seed)

    def choose(self, q, s, agent):
        """Sample from the softmax of the action values."""
        z = (q - q.max()) / self.temperature
        p = np.exp(z)
        return int(self.rng.choice(len(q), p=p / p.sum()))

    def end_episode(self):
        """Decay the temperature."""
        self.temperature = max(self.min_temperature, self.temperature * self.decay)


class UCB(EpsilonGreedy):
    """Upper-confidence-bound exploration: :math:`\\arg\\max_a Q(s,a) + c\\sqrt{\\ln N(s) / N(s,a)}`.

    Parameters
    ----------
    c : float, optional
        Exploration weight.
    seed : int, optional
    """

    def __init__(self, c=0.5, seed=0):
        self.c = c
        self.rng = np.random.default_rng(seed)

    def choose(self, q, s, agent):
        """Untried actions first, then the highest upper bound."""
        n = agent.counts[s]
        if np.any(n == 0):
            return int(self.rng.choice(np.flatnonzero(n == 0)))
        bonus = self.c * np.sqrt(np.log(n.sum()) / n)
        v = q + bonus
        return int(self.rng.choice(np.flatnonzero(v == v.max())))

    def end_episode(self):
        """Nothing to decay."""


class AmplitudeExploration(EpsilonGreedy):
    """Quantum-inspired exploration by action amplitudes (Born-rule sampling, TD-driven rotation).

    Parameters
    ----------
    n_states, n_actions : int
        Table size.
    k : float, optional
        Rotation angle per unit of temporal-difference error.
    max_step : float, optional
        Largest rotation per update (radians).
    floor : float, optional
        Smallest amplitude kept for every action.
    seed : int, optional

    Attributes
    ----------
    amp : numpy.ndarray
        Amplitudes, shape ``(n_states, n_actions)``; each row has unit norm.
    """

    def __init__(self, n_states, n_actions, k=2.0, max_step=0.3, floor=0.02, seed=0):
        self.amp = np.full((n_states, n_actions), 1 / np.sqrt(n_actions))
        self.k, self.max_step, self.floor = k, max_step, floor
        self.rng = np.random.default_rng(seed)

    def choose(self, q, s, agent):
        """Sample an action with probability :math:`|\\psi_a|^2`."""
        p = self.amp[s] ** 2
        return int(self.rng.choice(len(p), p=p / p.sum()))

    def update(self, s, a, td_error, q=None):
        """Rotate the chosen action's amplitude by ``clip(k * advantage, ±max_step)``.

        Parameters
        ----------
        s : int
        a : int
        td_error : float
            Temporal-difference error (used when the action values are not given).
        q : numpy.ndarray, optional
            Action values of the state; the advantage is ``q[a]`` minus their Born-rule average.
        """
        psi = self.amp[s]
        n = len(psi)
        drive = td_error if q is None else float(q[a] - (psi ** 2) @ q)
        angle = float(np.clip(self.k * drive, -self.max_step, self.max_step))
        lo, hi = np.arcsin(self.floor), np.arcsin(np.sqrt(max(1 - self.floor ** 2 * (n - 1), 0)))
        phi = float(np.clip(np.arcsin(np.clip(psi[a], 0, 1)) + angle, lo, hi))
        rest = np.delete(np.arange(n), a)
        norm_rest = np.linalg.norm(psi[rest])
        new = np.empty_like(psi)
        new[a] = np.sin(phi)
        new[rest] = np.cos(phi) * (psi[rest] / norm_rest if norm_rest > 0 else 1 / np.sqrt(n - 1))
        new = np.maximum(new, self.floor)
        self.amp[s] = new / np.linalg.norm(new)

    def end_episode(self):
        """Nothing to decay: exploration fades as amplitudes concentrate."""


class TabularAgent:
    """Q-learning agent with a pluggable exploration strategy.

    Parameters
    ----------
    n_states, n_actions : int
        Table size.
    explorer : EpsilonGreedy, Boltzmann, UCB or AmplitudeExploration
        How actions are chosen.
    alpha : float, optional
        Learning rate.
    gamma : float, optional
        Discount factor.

    Attributes
    ----------
    Q : numpy.ndarray
        Action values.
    counts : numpy.ndarray
        Visit counts per state and action.
    """

    def __init__(self, n_states, n_actions, explorer, alpha=0.2, gamma=0.95):
        self.Q = np.zeros((n_states, n_actions))
        self.counts = np.zeros((n_states, n_actions), int)
        self.explorer, self.alpha, self.gamma = explorer, alpha, gamma

    def act(self, s):
        """Choose an action in state s.

        Parameters
        ----------
        s : int

        Returns
        -------
        int
        """
        return self.explorer.choose(self.Q[s], s, self)

    def update(self, s, a, r, s2, done):
        """Q-learning update, then the explorer's update.

        Parameters
        ----------
        s, a : int
            State and action.
        r : float
            Reward.
        s2 : int
            Next state.
        done : bool
            Whether the episode ended (no bootstrap).
        """
        target = r + (0.0 if done else self.gamma * self.Q[s2].max())
        td = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td
        self.counts[s, a] += 1
        self.explorer.update(s, a, td, self.Q[s])

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


class StateIndexer:
    """Map hashable observations (tuples, rounded arrays) to table rows on first sight.

    Parameters
    ----------
    capacity : int
        Largest number of distinct states.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.index = {}

    def __call__(self, obs):
        """Row of an observation.

        Parameters
        ----------
        obs : int, tuple or array_like

        Returns
        -------
        int

        Raises
        ------
        ValueError
            When more distinct observations than ``capacity`` are seen.
        """
        key = int(obs) if np.ndim(obs) == 0 else tuple(np.round(np.asarray(obs, float), 3).ravel())
        if key not in self.index:
            if len(self.index) >= self.capacity:
                raise ValueError('more than %d distinct observations' % self.capacity)
            self.index[key] = len(self.index)
        return self.index[key]


def run_episodes(agent, env, episodes, state_fn=None, max_steps=1000, seed=0):
    """Train an agent on a Gymnasium-style environment.

    Parameters
    ----------
    agent : TabularAgent
    env : environment
        With ``reset(seed=...)`` and ``step(action)`` (Gymnasium interface).
    episodes : int
    state_fn : callable, optional
        Observation to table row (default: the observation itself, for discrete spaces).
    max_steps : int, optional
        Step limit per episode.
    seed : int, optional
        Seed of the first episode (episode i uses ``seed + i``).

    Returns
    -------
    dict
        ``returns`` and ``lengths`` per episode (numpy arrays).
    """
    state_fn = state_fn or int
    returns, lengths = np.zeros(episodes), np.zeros(episodes, int)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        s = state_fn(obs)
        total, n = 0.0, 0
        for n in range(1, max_steps + 1):
            a = agent.act(s)
            obs, r, terminated, truncated, _ = env.step(a)
            s2 = state_fn(obs)
            agent.update(s, a, r, s2, terminated)
            total += r
            s = s2
            if terminated or truncated:
                break
        agent.explorer.end_episode()
        returns[ep], lengths[ep] = total, n
    return {'returns': returns, 'lengths': lengths}
