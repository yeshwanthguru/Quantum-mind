r"""Variational quantum policy for reinforcement learning (policy gradient on a simulated circuit).

The state's features are angle-encoded into a data re-uploading circuit; the action probabilities are
the Born-rule probabilities of the first :math:`\lceil\log_2 A\rceil` qubits, restricted to the A
actions and renormalised (as in :class:`~quantum_mind.quantum.VariationalClassifier`). Training uses
REINFORCE with a running baseline,

.. math:: \nabla J = \mathbb{E}\Bigl[\sum_t (G_t - b)\,\nabla \log \pi_w(a_t \mid s_t)\Bigr],

with exact gradients from the adjoint method of :meth:`~quantum_mind.quantum.Circuit.value_and_grad`
(Jerbi et al., NeurIPS 2021, studied such policies). It runs on the built-in simulator and exports to
Qiskit; it is meant for small tasks and teaching, not for onboard control.

Examples
--------
>>> from quantum_mind.quantum.policy import VariationalPolicy
>>> pi = VariationalPolicy(n_features=2, n_actions=4, layers=2, seed=0)
>>> pi.probabilities([[0.1, 0.5]]).shape
(1, 4)
"""
from __future__ import annotations

import numpy as np
from .ansatz import reuploading_classifier_circuit

__all__ = ['VariationalPolicy', 'reinforce']


class VariationalPolicy:
    """Softmax-free quantum policy: Born-rule action probabilities of a re-uploading circuit.

    Parameters
    ----------
    n_features : int
        Number of state features (scaled to angles by ``scale``).
    n_actions : int
        Number of discrete actions.
    layers : int, optional
        Circuit layers.
    n_qubits : int, optional
        Qubits (default: number of features, at least :math:`\\lceil\\log_2 A\\rceil`).
    scale : float, optional
        Angle per unit of feature value.
    lr : float, optional
        Learning rate of the gradient-ascent step.
    seed : int, optional

    Attributes
    ----------
    weights : numpy.ndarray
        Circuit weights.
    circuit : Circuit
    """

    def __init__(self, n_features, n_actions, layers=2, n_qubits=None, scale=1.0, lr=0.1, seed=0):
        self.n_actions = n_actions
        self.n_out = max(1, int(np.ceil(np.log2(n_actions))))
        nq = n_qubits or max(n_features, self.n_out)
        self.circuit = reuploading_classifier_circuit(n_features, nq, layers)
        self.scale, self.lr = scale, lr
        self.rng = np.random.default_rng(seed)
        self.weights = self.rng.normal(0, 0.3, self.circuit.n_weights)

    def _marginals(self, P):
        """Probabilities of the read-out qubits restricted to the actions, renormalised."""
        idx = np.arange(P.shape[1]) & (2 ** self.n_out - 1)
        m = np.stack([P[:, idx == k].sum(1) for k in range(self.n_actions)], 1)
        return m / np.clip(m.sum(1, keepdims=True), 1e-12, None), idx

    def probabilities(self, states):
        """Action probabilities.

        Parameters
        ----------
        states : array_like
            Shape ``(n, n_features)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, n_actions)``.
        """
        X = np.atleast_2d(np.asarray(states, float)) * self.scale
        return self._marginals(self.circuit.probabilities(self.weights, X))[0]

    def act(self, state):
        """Sample an action.

        Parameters
        ----------
        state : array_like
            One state.

        Returns
        -------
        int
        """
        p = self.probabilities([state])[0]
        return int(self.rng.choice(self.n_actions, p=p))

    def update(self, states, actions, advantages):
        """One policy-gradient step on a batch of (state, action, advantage).

        Parameters
        ----------
        states : array_like
            Shape ``(n, n_features)``.
        actions : array_like
            Actions taken.
        advantages : array_like
            Return minus baseline for each step.

        Returns
        -------
        float
            The surrogate loss :math:`-\\frac1n\\sum A_t \\log \\pi(a_t \\mid s_t)` before the step.
        """
        X = np.atleast_2d(np.asarray(states, float)) * self.scale
        a = np.asarray(actions, int)
        adv = np.asarray(advantages, float)
        n = len(a)

        def loss(psi, rows):
            P = np.abs(psi) ** 2
            m_raw = np.stack([P[:, (np.arange(P.shape[1]) & (2 ** self.n_out - 1)) == k].sum(1)
                              for k in range(self.n_actions)], 1)
            S = np.clip(m_raw.sum(1, keepdims=True), 1e-12, None)
            q = np.clip(m_raw, 1e-12, None)
            ar, wr = a[rows], adv[rows]
            L = -np.sum(wr * np.log(q[np.arange(len(rows)), ar] / S[:, 0])) / n
            # dL/dm_k = -(w/n) (1[k=a]/m_a - 1/S); dm_k/d conj(psi_i) = psi_i for basis states of action k
            dm = np.repeat((wr / n)[:, None] / S, self.n_actions, axis=1)
            dm[np.arange(len(rows)), ar] -= (wr / n) / q[np.arange(len(rows)), ar]
            idx = np.arange(P.shape[1]) & (2 ** self.n_out - 1)
            g = np.zeros_like(P)
            valid = idx < self.n_actions
            g[:, valid] = dm[:, idx[valid]]
            return L, g * psi
        L, grad = self.circuit.value_and_grad(self.weights, X, loss)
        self.weights -= self.lr * grad
        return float(L)

    def to_qiskit(self, state, measure=True):
        """Qiskit circuit of the policy for one state.

        Parameters
        ----------
        state : array_like
        measure : bool, optional

        Returns
        -------
        qiskit.QuantumCircuit
        """
        return self.circuit.to_qiskit(self.weights, np.asarray(state, float) * self.scale, measure)


def reinforce(policy, env, episodes, features, gamma=0.99, batch=5, seed=0):
    """Train a policy with REINFORCE on an environment with the Gymnasium interface.

    Parameters
    ----------
    policy : VariationalPolicy
    env : environment
        ``reset(seed=...)`` and ``step(action)``.
    episodes : int
    features : callable
        Observation to feature vector.
    gamma : float, optional
        Discount factor.
    batch : int, optional
        Episodes per gradient step.
    seed : int, optional

    Returns
    -------
    numpy.ndarray
        Return of every episode.
    """
    returns = np.zeros(episodes)
    baseline = 0.0
    S, A, G = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        states, actions, rewards = [], [], []
        done = False
        while not done:
            x = features(obs)
            a = policy.act(x)
            obs, r, terminated, truncated, _ = env.step(a)
            states.append(x)
            actions.append(a)
            rewards.append(r)
            done = terminated or truncated
        g = 0.0
        disc = []
        for r in reversed(rewards):                       # discounted return from each step
            g = r + gamma * g
            disc.append(g)
        disc = disc[::-1]
        returns[ep] = sum(rewards)
        S += states
        A += actions
        G += disc
        if (ep + 1) % batch == 0:
            G_arr = np.array(G)
            baseline = 0.9 * baseline + 0.1 * G_arr.mean()
            policy.update(np.array(S), np.array(A), G_arr - baseline)
            S, A, G = [], [], []
    return returns
