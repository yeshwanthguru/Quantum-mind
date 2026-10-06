r"""Human-in-the-loop environments for reinforcement learning (Gymnasium API).

The simulated people follow the package's quantum-like models, so an agent trained here faces the
effects that classical simulators leave out: answers that depend on the order of questions, and trust
that changes when it is asked about. All people are simulated; the parameters are illustrative.

Every environment follows the Gymnasium interface: ``reset(seed=None) -> (observation, info)`` and
``step(action) -> (observation, reward, terminated, truncated, info)``.

Examples
--------
>>> from quantum_mind.envs import ClarificationEnv
>>> env = ClarificationEnv()
>>> obs, info = env.reset(seed=0)
>>> obs.shape, int(env.action_space.n)
((3,), 6)
>>> obs, reward, terminated, truncated, info = env.step(0)      # ask question 0
>>> reward, terminated
(-0.3, False)
"""
from __future__ import annotations

import numpy as np
from ..core import projector, angles_to_unit
from ..applications.questioning import ProjectiveAnswerModel
from ..families.dynamics import OpenSystemBelief

try:                                           # Gymnasium is optional
    import gymnasium as _gym
    from gymnasium import spaces as _spaces
    _Base = _gym.Env
except ImportError:                            # pragma: no cover - exercised only without gymnasium
    _gym = None
    _spaces = None

    class _Base:
        """Minimal stand-in for gymnasium.Env."""
        metadata = {}

        def reset(self, seed=None, options=None):
            if seed is not None:
                self.np_random = np.random.default_rng(seed)

__all__ = ['ClarificationEnv', 'TrustHandoverEnv', 'GridWorldEnv', 'register_envs']


class _Discrete:
    """Stand-in for gymnasium.spaces.Discrete when Gymnasium is not installed."""

    def __init__(self, n):
        self.n = n

    def sample(self):
        return int(np.random.default_rng().integers(self.n))

    def contains(self, x):
        return 0 <= int(x) < self.n


def _discrete(n):
    """A Discrete space (Gymnasium's when available)."""
    return _spaces.Discrete(n) if _spaces is not None else _Discrete(n)


def _box(low, high, shape):
    """A Box space (Gymnasium's when available), otherwise just the shape."""
    if _spaces is not None:
        return _spaces.Box(low, high, shape, dtype=np.float32)
    return type('Box', (), {'shape': shape, 'low': low, 'high': high})()


def default_domain(seed=7):
    """An object-clarification domain: three objects, three yes/no questions, order-dependent answers.

    Parameters
    ----------
    seed : int, optional
        Seed of the random belief states and question subspaces.

    Returns
    -------
    ProjectiveAnswerModel
    """
    rng = np.random.default_rng(seed)
    states = {h: angles_to_unit(rng.uniform(0, np.pi, 3)) for h in ('red cup', 'blue cup', 'red bowl')}
    projs = {q: projector(angles_to_unit(rng.uniform(0, np.pi, 3))) for q in ('red?', 'cup?', 'left?')}
    return ProjectiveAnswerModel(states, projs)


class ClarificationEnv(_Base):
    """Resolve an ambiguous request by asking questions, then act.

    Actions ``0 .. Q-1`` ask a question; actions ``Q .. Q+H-1`` act on a hypothesis (ending the
    episode). The observation has one entry per question: 0 if not asked yet, +1 for "yes", -1 for
    "no". The person's answers follow a :class:`~quantum_mind.applications.questioning.ProjectiveAnswerModel`,
    so they depend on the questions asked before.

    Parameters
    ----------
    model : ProjectiveAnswerModel or IndependentAnswerModel, optional
        How the simulated person answers (default: :func:`default_domain`).
    ask_cost : float, optional
        Reward penalty per question.
    success_reward : float, optional
        Reward for acting on the right hypothesis.
    error_cost : float, optional
        Penalty for acting on a wrong hypothesis.
    max_questions : int, optional
        After this many questions the episode is truncated with the error cost.
    """
    metadata = {'render_modes': []}

    def __init__(self, model=None, ask_cost=0.3, success_reward=1.0, error_cost=5.0, max_questions=6):
        self.model = model or default_domain()
        self.Q, self.H = len(self.model.questions), len(self.model.hypotheses)
        self.ask_cost, self.success_reward, self.error_cost, self.max_questions = ask_cost, success_reward, error_cost, max_questions
        self.action_space = _discrete(self.Q + self.H)
        self.observation_space = _box(-1.0, 1.0, (self.Q,))
        self.np_random = np.random.default_rng()
        self.history, self.target = [], None

    def _obs(self):
        o = np.zeros(self.Q, np.float32)
        for q, a in self.history:
            o[self.model.questions.index(q)] = 1.0 if a else -1.0
        return o

    def reset(self, seed=None, options=None):
        """Start an episode with a new hidden intent.

        Parameters
        ----------
        seed : int, optional
        options : dict, optional
            ``{'target': hypothesis}`` fixes the hidden intent.

        Returns
        -------
        observation : numpy.ndarray
        info : dict
            ``target`` (for evaluation only).
        """
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.history = []
        self.target = (options or {}).get('target') or self.model.hypotheses[int(self.np_random.integers(self.H))]
        return self._obs(), {'target': self.target}

    def step(self, action):
        """Ask a question or act.

        Parameters
        ----------
        action : int

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        action = int(action)
        if action < self.Q:
            q = self.model.questions[action]
            a = int(self.np_random.random() < self.model.p_yes(self.target, self.history, q))
            self.history.append((q, a))
            truncated = len(self.history) >= self.max_questions
            reward = -self.ask_cost - (self.error_cost if truncated else 0.0)
            return self._obs(), reward, False, truncated, {'question': q, 'answer': a}
        h = self.model.hypotheses[action - self.Q]
        correct = h == self.target
        return self._obs(), self.success_reward if correct else -self.error_cost, True, False, {'correct': correct}


class TrustHandoverEnv(_Base):
    """Hand objects to a simulated person whose trust follows the trust qubit.

    Actions: 0 hand over, 1 hand over slowly, 2 ask about trust, 3 wait. Observation: the last
    hand-over outcome (+1 success, -1 failure, 0 none), the last answer to the trust question (+1, -1,
    0) and the fraction of the episode elapsed. A hand-over fails with probability
    1 - P(trust); asking collapses the person's trust as in
    :class:`~quantum_mind.families.dynamics.OpenSystemBelief`.

    Parameters
    ----------
    model : OpenSystemBelief, optional
        Trust model of the simulated person.
    n_steps : int, optional
        Episode length.
    fail_cost, slow_cost, ask_cost, wait_cost : float, optional
        Costs (rewards are their negatives); a successful hand-over gives +1.
    slow_factor : float, optional
        Failure probability of a slow hand-over relative to a normal one.
    """
    metadata = {'render_modes': []}

    def __init__(self, model=None, n_steps=20, fail_cost=10.0, slow_cost=1.0, ask_cost=0.5, wait_cost=0.3, slow_factor=0.5):
        self.model = model or OpenSystemBelief(phi0=1.6, a_pos=0.8, a_neg=1.2, gamma=0.3)
        self.n_steps, self.fail_cost, self.slow_cost = n_steps, fail_cost, slow_cost
        self.ask_cost, self.wait_cost, self.slow_factor = ask_cost, wait_cost, slow_factor
        self.action_space = _discrete(4)
        self.observation_space = _box(-1.0, 1.0, (3,))
        self.np_random = np.random.default_rng()

    def _obs(self):
        return np.array([self.last_outcome, self.last_answer, self.t / self.n_steps], np.float32)

    def _rotate(self, event):
        a = -self.model.a_pos if event else self.model.a_neg
        c, s = np.cos(a / 2), np.sin(a / 2)
        U = np.array([[c, -s], [s, c]])
        r = U @ self.rho @ U.T
        f = 1 - self.model.gamma
        self.rho = np.array([[r[0, 0], f * r[0, 1]], [f * r[1, 0], r[1, 1]]])

    @property
    def p_trust(self):
        """float: the simulated person's current P(trust)."""
        return float(np.clip(self.rho[0, 0], 0, 1))

    def reset(self, seed=None, options=None):
        """Start an episode at the model's initial trust.

        Parameters
        ----------
        seed : int, optional
        options : dict, optional

        Returns
        -------
        observation : numpy.ndarray
        info : dict
        """
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        v = np.array([np.cos(self.model.phi0 / 2), np.sin(self.model.phi0 / 2)])
        self.rho = np.outer(v, v)
        self.t, self.last_outcome, self.last_answer = 0, 0.0, 0.0
        return self._obs(), {'p_trust': self.p_trust}

    def step(self, action):
        """Take one action.

        Parameters
        ----------
        action : int
            0 hand over, 1 slow hand-over, 2 ask, 3 wait.

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        action = int(action)
        reward = 0.0
        if action in (0, 1):
            p_fail = (1 - self.p_trust) * (self.slow_factor if action == 1 else 1.0)
            ok = self.np_random.random() >= p_fail
            reward = (1.0 if ok else -self.fail_cost) - (self.slow_cost if action == 1 else 0.0)
            self._rotate(int(ok))
            self.last_outcome = 1.0 if ok else -1.0
        elif action == 2:
            yes = self.np_random.random() < self.p_trust
            self.rho = np.diag([1.0, 0.0]) if yes else np.diag([0.0, 1.0])
            self.last_answer = 1.0 if yes else -1.0
            reward = -self.ask_cost
        else:
            f = 1 - self.model.gamma
            self.rho = np.array([[self.rho[0, 0], f * self.rho[0, 1]], [f * self.rho[1, 0], self.rho[1, 1]]])
            reward = -self.wait_cost
        self.t += 1
        return self._obs(), reward, False, self.t >= self.n_steps, {'p_trust': self.p_trust}


class GridWorldEnv(_Base):
    """The grid world of :class:`quantum_mind.inspired.rl.GridWorld` with the Gymnasium interface.

    Parameters
    ----------
    **kwargs
        Passed to :class:`~quantum_mind.inspired.rl.GridWorld`.
    """
    metadata = {'render_modes': []}

    def __init__(self, **kwargs):
        from ..inspired.rl import GridWorld
        self.world = GridWorld(**kwargs)
        self.action_space = _discrete(self.world.n_actions)
        self.observation_space = _discrete(self.world.n_states)
        self.np_random = np.random.default_rng()

    def reset(self, seed=None, options=None):
        """Return to the start cell.

        Returns
        -------
        observation : int
        info : dict
        """
        super().reset(seed=seed)
        return self.world.reset(), {}

    def step(self, action):
        """Move one cell.

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        s, r, done = self.world.step(int(action))
        reached = self.world.pos == self.world.goal
        return s, r, bool(reached), bool(done and not reached), {}


def register_envs():
    """Register the environments with Gymnasium.

    Returns
    -------
    list of str
        The registered ids (empty when Gymnasium is not installed).
    """
    if _gym is None:
        return []
    ids = {'quantum_mind/Clarification-v0': ClarificationEnv, 'quantum_mind/TrustHandover-v0': TrustHandoverEnv,
           'quantum_mind/GridWorld-v0': GridWorldEnv}
    for i, cls in ids.items():
        if i not in _gym.registry:
            _gym.register(id=i, entry_point=cls)
    return list(ids)
